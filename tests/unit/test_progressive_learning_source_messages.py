import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def _load_progressive_learning_class():
    if "astrbot.api" not in sys.modules:
        astrbot_module = types.ModuleType("astrbot")
        api_module = types.ModuleType("astrbot.api")
        star_module = types.ModuleType("astrbot.api.star")

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
        star_module.Context = object
        astrbot_module.api = api_module
        sys.modules["astrbot"] = astrbot_module
        sys.modules["astrbot.api"] = api_module
        sys.modules["astrbot.api.star"] = star_module

    root_package = "astrbot_plugin_self_learning"
    for package_name in (
        root_package,
        f"{root_package}.services",
        f"{root_package}.services.core_learning",
        f"{root_package}.utils",
    ):
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = module

    config_module = types.ModuleType(f"{root_package}.config")
    config_module.PluginConfig = object
    sys.modules[config_module.__name__] = config_module

    constants_module = types.ModuleType(f"{root_package}.constants")
    constants_module.UPDATE_TYPE_PROGRESSIVE_PERSONA_LEARNING = "progressive_learning"
    sys.modules[constants_module.__name__] = constants_module

    exceptions_module = types.ModuleType(f"{root_package}.exceptions")
    exceptions_module.LearningError = RuntimeError
    sys.modules[exceptions_module.__name__] = exceptions_module

    json_utils_module = types.ModuleType(f"{root_package}.utils.json_utils")
    json_utils_module.safe_parse_llm_json = lambda value, fallback_result=None: fallback_result or {}
    json_utils_module.clean_llm_json_response = lambda value: value
    sys.modules[json_utils_module.__name__] = json_utils_module

    database_module = types.ModuleType(f"{root_package}.services.database")
    database_module.DatabaseManager = object
    sys.modules[database_module.__name__] = database_module

    root = Path(__file__).resolve().parents[2]
    module_path = root / "services" / "core_learning" / "progressive_learning.py"
    spec = importlib.util.spec_from_file_location(
        f"{root_package}.services.core_learning.progressive_learning", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ProgressiveLearningService


ProgressiveLearningService = _load_progressive_learning_class()


def _build_service(db_manager):
    config = SimpleNamespace(
        max_messages_per_batch=50,
        learning_interval_hours=1,
        style_update_threshold=0.5,
        refine_recent_message_limit=20,
    )
    return ProgressiveLearningService(
        config=config,
        context=SimpleNamespace(),
        db_manager=db_manager,
        message_collector=Mock(),
        multidimensional_analyzer=Mock(),
        style_analyzer=Mock(),
        quality_monitor=Mock(),
        persona_manager=Mock(),
        ml_analyzer=Mock(),
        prompts=SimpleNamespace(),
    )


async def test_source_message_helper_prefers_raw_records():
    db_manager = Mock()
    db_manager.get_recent_raw_messages = AsyncMock(
        return_value=[{"message": "raw-1"}, {"message": "raw-2"}]
    )
    service = _build_service(db_manager)

    result = await service._resolve_source_messages(
        "g1", [{"message": "filtered-1"}], session_id="session_g1"
    )

    assert result.fetch_path == "raw_recent_messages"
    assert result.fetch_count == 2
    assert result.degraded_mode is False
    assert len(result.messages) == 2


async def test_source_message_helper_falls_back_to_filtered_messages():
    db_manager = Mock()
    db_manager.get_recent_raw_messages = AsyncMock(return_value=[])
    db_manager.get_recent_filtered_messages = AsyncMock(return_value=[])
    service = _build_service(db_manager)

    result = await service._resolve_source_messages(
        "g1", [{"message": "filtered-1"}, {"message": "filtered-2"}], session_id=""
    )

    assert result.fetch_path == "filtered_messages_fallback"
    assert result.fetch_count == 2
    assert result.degraded_reason == "missing_raw_chat_records_fallback"
    assert len(result.messages) == 2


async def test_batch_quality_uses_aggregate_metrics_when_consistency_is_zero():
    db_manager = Mock()
    service = _build_service(db_manager)
    service.quality_monitor.evaluate_learning_batch = AsyncMock(
        return_value=SimpleNamespace(
            consistency_score=0.0,
            style_stability=0.8,
            vocabulary_diversity=0.7,
            emotional_balance=0.9,
            coherence_score=0.6,
        )
    )

    result = await service._evaluate_batch_quality(
        group_id="g1",
        current_persona={"prompt": "old"},
        updated_persona={"prompt": "new"},
        learning_messages=[{"message": "hello"}],
        analysis_data={},
        write_path="test.path",
    )

    assert result.quality_monitor_invoked is True
    assert result.batch_quality_write_source == "quality_monitor.aggregate_metrics"
    assert result.batch_quality_raw_value == 0.0
    assert result.batch_quality_score == 0.6


async def test_batch_quality_falls_back_to_analysis_confidence_when_monitor_missing():
    db_manager = Mock()
    service = _build_service(db_manager)
    service.quality_monitor = None

    result = await service._evaluate_batch_quality(
        group_id="g1",
        current_persona={"prompt": "old"},
        updated_persona={"prompt": "new"},
        learning_messages=[{"message": "hello"}],
        analysis_data={"confidence": 0.8},
        write_path="test.path",
    )

    assert result.quality_monitor_invoked is False
    assert result.quality_monitor_skipped_reason == "quality_monitor_unavailable"
    assert result.batch_quality_write_source == "style_analysis.confidence_fallback"
    assert result.batch_quality_score == 0.8


async def test_batch_quality_treats_zero_score_as_valid_value():
    db_manager = Mock()
    service = _build_service(db_manager)
    service.quality_monitor.evaluate_learning_batch = AsyncMock(
        return_value=SimpleNamespace(
            consistency_score=0.0,
            style_stability=0.0,
            vocabulary_diversity=0.0,
            emotional_balance=0.0,
            coherence_score=0.0,
        )
    )

    result = await service._evaluate_batch_quality(
        group_id="g1",
        current_persona={"prompt": "old"},
        updated_persona={"prompt": "new"},
        learning_messages=[{"message": "hello"}],
        analysis_data={},
        write_path="test.path",
    )

    assert result.batch_quality_score == 0.0
    assert result.batch_quality_raw_value == 0.0
    assert result.batch_quality_write_source == "quality_monitor.aggregate_metrics"
    assert result.quality_monitor_skipped_reason == ""
