import json
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def _load_learning_service_class():
    if "astrbot.api" not in sys.modules:
        astrbot_module = types.ModuleType("astrbot")
        api_module = types.ModuleType("astrbot.api")

        class _DummyLogger:
            def info(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

            def error(self, *args, **kwargs):
                return None

            def debug(self, *args, **kwargs):
                return None

        api_module.logger = _DummyLogger()
        astrbot_module.api = api_module
        sys.modules["astrbot"] = astrbot_module
        sys.modules["astrbot.api"] = api_module

    if "webui" not in sys.modules:
        webui_module = types.ModuleType("webui")
        webui_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["webui"] = webui_module

    if "webui.services" not in sys.modules:
        services_module = types.ModuleType("webui.services")
        services_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["webui.services"] = services_module

    root = Path(__file__).resolve().parents[2]
    service_path = root / "webui" / "services" / "learning_service.py"
    spec = importlib.util.spec_from_file_location(
        "webui.services.learning_service", service_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LearningService


LearningService = _load_learning_service_class()


def _build_service():
    return LearningService(Mock(database_manager=None))


def _review(
    *,
    review_id=1,
    group_id="g1",
    status="pending",
    timestamp=1000.0,
    learned_patterns=None,
):
    return SimpleNamespace(
        id=review_id,
        group_id=group_id,
        status=status,
        timestamp=timestamp,
        review_time=None,
        type="style_learning",
        learned_patterns=learned_patterns,
    )


def test_statistics_from_mixed_reviews_keep_semantics():
    service = _build_service()
    reviews = [
        _review(
            review_id=1,
            status="pending",
            timestamp=1000.0,
            learned_patterns=json.dumps(
                {
                    "language_patterns": [
                        {"name": "warm tone", "confidence": 0.8, "count": 3},
                        {"name": "brief reply", "count": 2},
                    ],
                    "topic_preferences": [{"name": "music", "score": 0.6, "count": 2}],
                    "processed_messages": 12,
                }
            ),
        ),
        _review(
            review_id=2,
            status="approved",
            timestamp=2000.0,
            learned_patterns={
                "emotion": [{"pattern": "calm", "confidence": 0.7, "count": 4}],
                "language": ["concise"],
                "source_message_count": 8,
            },
        ),
        _review(
            review_id=3,
            group_id="g2",
            status="rejected",
            timestamp=3000.0,
            learned_patterns=[{"category": "topic", "label": "games", "weight": 0.9}],
        ),
    ]

    result = service._aggregate_style_learning_statistics(reviews)

    assert result["unique_styles"] == 3
    assert result["total_samples"] == 20
    assert result["raw_message_count_source"] == "normalized_payload_fallback"
    assert result["style_feature_total_count"] == 5
    assert result["avg_confidence"] == 80.0
    assert result["confidence_value_count"] == 3
    assert result["confidence_fallback_used"] is False
    assert result["latest_update"] == 3000.0


def test_normalize_payload_accepts_string_json():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        json.dumps(
            {
                "emotion_patterns": [{"name": "calm", "confidence": 0.7}],
                "processed_messages": 11,
            }
        )
    )

    assert normalized.raw_message_count == 11
    assert normalized.emotion_patterns[0]["name"] == "calm"
    assert normalized.confidence == 70.0


def test_normalize_payload_accepts_dict():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        {
            "language_style": [{"name": "short reply", "count": 2}],
            "source_message_count": 4,
        }
    )

    assert normalized.raw_message_count == 4
    assert normalized.language_patterns[0]["numeric_value"] == 2


def test_normalize_payload_accepts_list():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        [{"category": "topic", "label": "games", "weight": 0.8}]
    )

    assert normalized.topic_preferences[0]["name"] == "games"
    assert normalized.topic_preferences[0]["numeric_value"] == 80.0


def test_payload_with_only_emotion_keeps_language_empty():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        {"emotion": [{"pattern": "friendly", "confidence": 0.9}]}
    )

    assert normalized.emotion_patterns
    assert normalized.language_patterns == []


def test_payload_with_only_language_keeps_emotion_empty():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        {"language_patterns": [{"name": "concise", "count": 3}]}
    )

    assert normalized.language_patterns
    assert normalized.emotion_patterns == []


def test_dirty_values_are_excluded_from_chart_patterns():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        {
            "language_patterns": [
                {"name": "good", "count": "3"},
                {"name": "bad-none", "count": None},
                {"name": "bad-object", "count": {"x": 1}},
            ]
        }
    )

    assert [item["name"] for item in normalized.language_patterns] == ["good"]
    assert normalized.language_patterns[0]["numeric_value"] == 3.0


def test_missing_raw_message_count_stays_zero():
    service = _build_service()
    result = service._aggregate_style_learning_statistics(
        [_review(learned_patterns={"language_patterns": [{"name": "warm"}]})]
    )

    assert result["total_samples"] == 0
    assert result["raw_message_count_source"] == "unavailable"


def test_missing_confidence_does_not_fallback_to_fifty():
    service = _build_service()
    result = service._aggregate_style_learning_statistics(
        [
            _review(
                learned_patterns={"language_patterns": [{"name": "warm", "count": 2}]}
            )
        ]
    )

    assert result["avg_confidence"] == 0.0
    assert result["confidence_value_count"] == 0
    assert result["confidence_fallback_used"] is True


def test_many_reviews_do_not_inflate_style_type_count():
    service = _build_service()
    reviews = [
        _review(
            review_id=index,
            timestamp=1000.0 + index,
            learned_patterns={
                "language_patterns": [
                    {"name": f"style-{index}-a", "count": 1},
                    {"name": f"style-{index}-b", "count": 1},
                ],
                "emotion_patterns": [{"name": f"emotion-{index}", "confidence": 0.8}],
                "topic_preferences": [{"name": f"topic-{index}", "count": 1}],
            },
        )
        for index in range(27)
    ]

    result = service._aggregate_style_learning_statistics(reviews)

    assert result["unique_styles"] == 3


def test_progress_item_keeps_real_quality_value():
    service = _build_service()

    normalized = service._normalize_progress_item(
        {
            "group_id": "g1",
            "label": "batch_1",
            "timestamp": 1713318300,
            "quality_score": 0.82,
            "processed_messages": 12,
        }
    )

    assert normalized["quality_score"] == 0.82
    assert normalized["quality_percent"] == 82.0
    assert normalized["quality_value_source"] == "quality_score"
    assert normalized["sample_count"] == 12
    assert normalized["sample_count_source"] == "processed_messages"


def test_progress_item_missing_quality_stays_null():
    service = _build_service()

    normalized = service._normalize_progress_item(
        {
            "group_id": "g1",
            "filtered_count": 9,
            "message_count": 12,
            "processed_messages": 15,
        }
    )

    assert normalized["quality_score"] is None
    assert normalized["quality_percent"] is None
    assert normalized["quality_value_source"] == "missing"
    assert normalized["sample_count"] == 9
    assert normalized["sample_count_source"] == "filtered_count"


async def test_build_statistics_prefers_learning_batch_message_totals():
    service = _build_service()
    service._resolve_raw_message_count = AsyncMock(
        return_value=(15, "learning_batches.filtered_count_sum")
    )
    service._resolve_latest_update = AsyncMock(return_value=2000.0)

    dto = await service._build_statistics_from_reviews(
        [
            _review(
                learned_patterns={
                    "language_patterns": [{"name": "warm", "count": 2}],
                    "source_message_count": 4,
                }
            )
        ]
    )

    assert dto.raw_message_count == 15
    assert dto.raw_message_count_source == "learning_batches.filtered_count_sum"
    assert dto.last_updated_at == 2000.0


def test_mojibake_pattern_label_is_repaired():
    service = _build_service()
    normalized = service.normalize_pattern_payload(
        {"topic_preferences": [{"name": "???", "weight": 0.5}]}
    )

    assert normalized.topic_preferences == []


def test_ingest_compatibility_populates_pattern_buckets():
    service = _build_service()
    pattern_type_map = {
        "emotion": "emotion_patterns",
        "sentiment": "emotion_patterns",
        "language": "language_patterns",
        "topic": "topic_patterns",
    }
    patterns_data = {
        "emotion_patterns": [],
        "language_patterns": [],
        "topic_patterns": [],
    }
    debug_info = {"emotion_keys": set(), "language_value_sources": []}

    service._ingest_learned_pattern_payload(
        {
            "emotion_patterns": [{"name": "calm", "confidence": 0.7}],
            "language_patterns": [{"pattern": "short replies", "count": 2}],
            "topic_preferences": [{"label": "music", "weight": 0.6}],
        },
        patterns_data,
        pattern_type_map,
        debug_info=debug_info,
    )

    assert patterns_data["emotion_patterns"]
    assert patterns_data["language_patterns"]
    assert patterns_data["topic_patterns"]
    assert debug_info["emotion_keys"]
