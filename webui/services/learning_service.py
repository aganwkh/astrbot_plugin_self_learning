"""
Learning service for style-learning related web UI data.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from astrbot.api import logger


@dataclass
class StyleLearningStatisticsDTO:
    style_type_count: int = 0
    average_confidence: Optional[float] = None
    raw_message_count: int = 0
    last_updated_at: Optional[float] = None
    total_reviews: int = 0
    pending_reviews: int = 0
    approved_reviews: int = 0
    rejected_reviews: int = 0
    style_feature_total_count: int = 0
    raw_message_count_source: str = "unavailable"
    raw_message_count_value: int = 0
    confidence_value_count: int = 0
    confidence_fallback_used: bool = False

    def to_api_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "unique_styles": self.style_type_count,
                "avg_confidence": self.average_confidence,
                "total_samples": self.raw_message_count,
                "latest_update": self.last_updated_at,
            }
        )
        return payload


@dataclass
class StyleLearningResultsDTO:
    statistics: StyleLearningStatisticsDTO = field(
        default_factory=StyleLearningStatisticsDTO
    )
    progress: List[Dict[str, Any]] = field(default_factory=list)
    emotion_patterns: List[Dict[str, Any]] = field(default_factory=list)
    language_patterns: List[Dict[str, Any]] = field(default_factory=list)
    topic_preferences: List[Dict[str, Any]] = field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any]:
        statistics = self.statistics.to_api_dict()
        return {
            "statistics": statistics,
            "progress": list(self.progress),
            "style_progress": list(self.progress),
            "emotion_patterns": list(self.emotion_patterns),
            "language_patterns": list(self.language_patterns),
            "topic_preferences": list(self.topic_preferences),
            "topic_patterns": list(self.topic_preferences),
            "style_types_count": statistics["style_type_count"],
            "avg_confidence": statistics["avg_confidence"],
            "total_samples": statistics["total_samples"],
            "latest_update": statistics["latest_update"],
        }


@dataclass
class NormalizedStyleLearningPayload:
    emotion_patterns: List[Dict[str, Any]] = field(default_factory=list)
    language_patterns: List[Dict[str, Any]] = field(default_factory=list)
    topic_preferences: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    raw_message_count: Optional[int] = None
    style_categories: List[str] = field(default_factory=list)
    payload_detected_keys: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "emotion_patterns": len(self.emotion_patterns),
            "language_patterns": len(self.language_patterns),
            "topic_preferences": len(self.topic_preferences),
            "confidence": self.confidence,
            "raw_message_count": self.raw_message_count,
            "style_categories": self.style_categories,
            "payload_detected_keys": self.payload_detected_keys[:12],
        }


class LearningService:
    _results_cache: Dict[str, Any] = {"signature": None, "statistics": None}
    _patterns_cache: Dict[str, Any] = {"signature": None, "payload": None}

    _EMOTION_KEYS = {
        "emotion_patterns",
        "emotional_patterns",
        "emotion",
        "emotions",
        "sentiment",
    }
    _LANGUAGE_KEYS = {
        "language_patterns",
        "language",
        "language_style",
        "style_features",
        "linguistic_features",
        "expression",
        "vocabulary",
        "habit",
    }
    _TOPIC_KEYS = {
        "topic_preferences",
        "topic_patterns",
        "topic",
        "topics",
        "topic_preference",
    }
    _MOJIBAKE_MARKERS = (
        "鏆",
        "鍓",
        "妯",
        "璇",
        "椋",
        "鍙",
        "瀛",
        "缁",
        "鎯",
        "鍐",
        "鍔",
        "鐗",
        "鏍",
        "杩",
        "闆",
        "鍠",
        "閺",
        "鐨",
    )
    _RAW_MESSAGE_KEYS = {
        "source_message_count",
        "raw_message_count",
        "processed_messages",
        "filtered_messages",
        "message_count",
        "raw_messages",
    }
    _CONFIDENCE_KEYS = {"confidence", "confidence_score", "score", "weight"}
    _VALUE_KEYS = ("value", "count", "usage_count", "score", "confidence", "weight")

    def __init__(self, container):
        self.container = container
        self.database_manager = container.database_manager
        self.db_manager = container.database_manager
        self.persona_updater = getattr(container, "persona_updater", None)

    async def get_style_learning_results(self) -> Dict[str, Any]:
        has_db_manager = bool(self.db_manager)
        statistics_fetch_path = "none"
        statistics_raw_value: Any = None
        statistics_error: Optional[str] = None
        dto = StyleLearningResultsDTO()

        logger.info(
            f"[LearningService] get_style_learning_results has_db_manager={has_db_manager}"
        )
        if not has_db_manager:
            logger.warning(
                "[LearningService] style learning results requested without db_manager"
            )
            return dto.to_api_dict()

        progress = await self._get_progress_data()
        dto.progress = progress

        try:
            if hasattr(self.db_manager, "get_session"):
                statistics_fetch_path = "Repository"
                signature = await self._get_statistics_cache_signature()
                cached = self._get_cached_statistics(signature)
                if cached is None:
                    review_rows = await self.collect_reviews()
                    cached = (
                        await self._build_statistics_from_reviews(review_rows)
                    ).to_api_dict()
                    self._set_cached_statistics(signature, cached)
                statistics_raw_value = cached
            elif hasattr(self.db_manager, "get_style_learning_statistics"):
                statistics_fetch_path = "Facade"
                statistics_raw_value = await self.db_manager.get_style_learning_statistics()
            else:
                statistics_error = "no_supported_statistics_path"
                logger.error(
                    "[LearningService] no supported statistics path on db_manager"
                )
        except Exception as exc:
            statistics_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[LearningService] style statistics fetch failed "
                f"statistics_fetch_path={statistics_fetch_path} "
                f"statistics_error={statistics_error}",
                exc_info=True,
            )

        if isinstance(statistics_raw_value, dict):
            dto.statistics = self._statistics_from_payload(statistics_raw_value)
        else:
            statistics_error = statistics_error or "statistics_query_returned_empty"
            logger.warning(
                "[LearningService] style statistics unavailable "
                f"statistics_fetch_path={statistics_fetch_path} "
                f"statistics_raw_value={self._summarize_for_log(statistics_raw_value)} "
                f"statistics_error={statistics_error}"
            )

        payload = dto.to_api_dict()
        logger.info(
            "[LearningService] get_style_learning_results "
            f"has_db_manager={has_db_manager} "
            f"statistics_fetch_path={statistics_fetch_path} "
            f"statistics_raw_value={self._summarize_for_log(statistics_raw_value)} "
            f"statistics_error={statistics_error} "
            f"progress_count={len(progress)} "
            f"final_results_data={payload}"
        )
        return payload

    async def get_style_learning_reviews(self, limit: int = 50) -> Dict[str, Any]:
        if not self.database_manager:
            raise ValueError("Database manager is not initialized")

        pending_reviews = await self.database_manager.get_pending_style_reviews(
            limit=limit
        )
        formatted_reviews = []
        for review in pending_reviews:
            formatted_reviews.append(
                {
                    "id": review["id"],
                    "type": "dialog_style_learning",
                    "group_id": review["group_id"],
                    "description": review["description"],
                    "timestamp": review["timestamp"],
                    "created_at": review["created_at"],
                    "status": review["status"],
                    "learned_patterns": review["learned_patterns"],
                    "few_shots_content": review["few_shots_content"],
                }
            )
        return {"reviews": formatted_reviews, "total": len(formatted_reviews)}

    async def approve_style_learning_review(self, review_id: int) -> Tuple[bool, str]:
        if not self.database_manager:
            raise ValueError("Database manager is not initialized")

        pending_reviews = await self.database_manager.get_pending_style_reviews()
        target_review = next(
            (review for review in pending_reviews if review["id"] == review_id), None
        )
        if not target_review:
            return False, "Review record not found"

        success = await self.database_manager.update_style_review_status(
            review_id, "approved", target_review["group_id"]
        )
        if not success:
            return False, "Approval failed, please check the review record status"

        if target_review["few_shots_content"] and self.persona_updater:
            try:
                style_analysis = {
                    "enhanced_prompt": target_review["few_shots_content"],
                    "style_features": [],
                    "style_attributes": {},
                    "confidence": 0.8,
                    "source": f"style_learning_review_{review_id}",
                }
                success_apply = await self.persona_updater.update_persona_with_style(
                    target_review.get("group_id", "default"), style_analysis, []
                )
                if success_apply:
                    return True, f"Review {review_id} approved and applied"
                return True, f"Review {review_id} approved, but persona apply failed"
            except Exception as exc:
                logger.error(
                    f"[LearningService] apply style learning failed: {exc}",
                    exc_info=True,
                )
                return False, f"Approval succeeded, but application failed: {exc}"

        if target_review["few_shots_content"] and not self.persona_updater:
            logger.warning(
                "[LearningService] persona_updater unavailable, skip applying style review"
            )
            return True, f"Review {review_id} approved, but persona updater is unavailable"
        return True, f"Review {review_id} approved (no content to apply)"

    async def reject_style_learning_review(self, review_id: int) -> Tuple[bool, str]:
        if not self.database_manager:
            raise ValueError("Database manager is not initialized")
        success = await self.database_manager.update_style_review_status(
            review_id, "rejected"
        )
        if success:
            logger.info(f"[LearningService] style review rejected review_id={review_id}")
            return True, f"Review {review_id} rejected"
        return False, "Rejection failed, please check the review record status"

    async def get_style_learning_patterns(self) -> Dict[str, Any]:
        if not self.db_manager or not hasattr(self.db_manager, "get_session"):
            return {
                "emotion_patterns": [],
                "language_patterns": [],
                "topic_preferences": [],
                "topic_patterns": [],
            }

        try:
            signature = await self._get_patterns_cache_signature()
            cached = self._get_cached_patterns(signature)
            if cached is not None:
                return cached

            review_rows = await self.collect_reviews()
            normalized_payloads = self._normalize_reviews(review_rows)
            db_patterns = await self._collect_db_patterns()
            emotion_patterns = self.aggregate_emotion_patterns(
                normalized_payloads, db_patterns
            )
            language_patterns = self.aggregate_language_patterns(
                normalized_payloads, db_patterns
            )
            topic_preferences = self.aggregate_topic_preferences(
                normalized_payloads, db_patterns
            )
            emotion_keys_detected = sorted(
                {
                    key
                    for payload in normalized_payloads
                    for key in payload.payload_detected_keys
                    if key in self._EMOTION_KEYS
                }
            )
            language_value_sources = sorted(
                {
                    item.get("value_source")
                    for item in language_patterns
                    if item.get("value_source")
                }
            )
            logger.info(
                "[LearningService] "
                f"emotion_pattern_keys_detected={emotion_keys_detected}"
            )
            logger.info(
                "[LearningService] "
                f"language_pattern_value_source={language_value_sources}"
            )
            if not emotion_keys_detected:
                sample_keys = sorted(
                    {
                        key
                        for payload in normalized_payloads
                        for key in payload.payload_detected_keys
                    }
                )[:12]
                logger.warning(
                    "[LearningService] emotion_pattern_missing_keys "
                    f"raw_keys_summary={sample_keys}"
                )
            payload = {
                "emotion_patterns": emotion_patterns,
                "language_patterns": language_patterns,
                "topic_preferences": topic_preferences,
                "topic_patterns": topic_preferences,
            }
            self._set_cached_patterns(signature, payload)
            logger.info(
                "[LearningService] get_style_learning_patterns final_counts "
                f"emotion={len(emotion_patterns)} "
                f"language={len(language_patterns)} "
                f"topic={len(topic_preferences)}"
            )
            return payload
        except Exception as exc:
            logger.warning(
                "[LearningService] get_style_learning_patterns failed "
                f"error={type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return {
                "emotion_patterns": [],
                "language_patterns": [],
                "topic_preferences": [],
                "topic_patterns": [],
            }

    async def collect_reviews(self, limit: int = 1000) -> List[Any]:
        if not hasattr(self.db_manager, "get_session"):
            return []

        from sqlalchemy import desc, select

        from ...models.orm.learning import StyleLearningReview

        async with self.db_manager.get_session() as session:
            stmt = (
                select(StyleLearningReview)
                .order_by(desc(StyleLearningReview.timestamp))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def _build_statistics_from_reviews(
        self, reviews: Sequence[Any]
    ) -> StyleLearningStatisticsDTO:
        dto = self.aggregate_statistics_from_reviews(reviews)
        raw_message_count, raw_message_source = await self._resolve_raw_message_count(
            fallback_value=dto.raw_message_count,
            fallback_source=dto.raw_message_count_source,
        )
        dto.raw_message_count = raw_message_count
        dto.raw_message_count_value = raw_message_count
        dto.raw_message_count_source = raw_message_source
        dto.last_updated_at = await self._resolve_latest_update(
            fallback_value=dto.last_updated_at
        )
        return dto

    async def _get_statistics_cache_signature(
        self,
    ) -> Tuple[int, float, int, float]:
        if not hasattr(self.db_manager, "get_session"):
            return await self._get_review_cache_signature()

        from sqlalchemy import func, select

        from ...models.orm.learning import LearningBatch, StyleLearningReview

        async with self.db_manager.get_session() as session:
            review_stmt = select(
                func.count(StyleLearningReview.id),
                func.max(StyleLearningReview.timestamp),
            )
            batch_stmt = select(
                func.count(LearningBatch.id),
                func.max(func.coalesce(LearningBatch.end_time, LearningBatch.start_time)),
            )
            review_result = await session.execute(review_stmt)
            batch_result = await session.execute(batch_stmt)
            review_count, review_latest = review_result.one()
            batch_count, batch_latest = batch_result.one()
        return (
            int(review_count or 0),
            float(review_latest or 0.0),
            int(batch_count or 0),
            float(batch_latest or 0.0),
        )

    async def _resolve_raw_message_count(
        self,
        *,
        fallback_value: int,
        fallback_source: str,
    ) -> Tuple[int, str]:
        if not hasattr(self.db_manager, "get_session"):
            return fallback_value, fallback_source

        from sqlalchemy import func, select

        from ...models.orm.learning import LearningBatch
        from ...models.orm.message import FilteredMessage, RawMessage

        try:
            async with self.db_manager.get_session() as session:
                batch_stmt = select(
                    func.count(LearningBatch.id),
                    func.sum(LearningBatch.filtered_count),
                    func.sum(LearningBatch.message_count),
                    func.sum(LearningBatch.processed_messages),
                )
                batch_row = (await session.execute(batch_stmt)).one()
                batch_count = int(batch_row[0] or 0)
                if batch_count > 0:
                    batch_sources = (
                        ("learning_batches.filtered_count_sum", batch_row[1]),
                        ("learning_batches.message_count_sum", batch_row[2]),
                        ("learning_batches.processed_messages_sum", batch_row[3]),
                    )
                    for source, value in batch_sources:
                        if value is not None and int(value) > 0:
                            return int(value), source
                    for source, value in batch_sources:
                        if value is not None:
                            return int(value or 0), source

                filtered_table_count = (
                    await session.execute(select(func.count()).select_from(FilteredMessage))
                ).scalar()
                if filtered_table_count is not None:
                    return int(filtered_table_count or 0), "filtered_messages_table_count"

                raw_table_count = (
                    await session.execute(select(func.count()).select_from(RawMessage))
                ).scalar()
                if raw_table_count is not None:
                    return int(raw_table_count or 0), "raw_messages_table_count"
        except Exception as exc:
            logger.warning(
                "[LearningService] reliable raw_message_count lookup failed "
                f"error={type(exc).__name__}: {exc}",
                exc_info=True,
            )

        return fallback_value, fallback_source

    async def _resolve_latest_update(
        self,
        *,
        fallback_value: Optional[float],
    ) -> Optional[float]:
        if not hasattr(self.db_manager, "get_session"):
            return fallback_value

        from sqlalchemy import func, select

        from ...models.orm.learning import LearningBatch
        from ...models.orm.performance import LearningPerformanceHistory

        try:
            async with self.db_manager.get_session() as session:
                batch_latest = (
                    await session.execute(
                        select(
                            func.max(
                                func.coalesce(
                                    LearningBatch.end_time, LearningBatch.start_time
                                )
                            )
                        )
                    )
                ).scalar()
                performance_latest = (
                    await session.execute(
                        select(func.max(LearningPerformanceHistory.timestamp))
                    )
                ).scalar()
        except Exception as exc:
            logger.warning(
                "[LearningService] latest_update lookup failed "
                f"error={type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return fallback_value

        candidates = [
            self._coerce_timestamp(fallback_value),
            self._coerce_timestamp(batch_latest),
            self._coerce_timestamp(performance_latest),
        ]
        valid_candidates = [value for value in candidates if value is not None]
        if not valid_candidates:
            return fallback_value
        return max(valid_candidates)

    def aggregate_statistics_from_reviews(
        self, reviews: Sequence[Any]
    ) -> StyleLearningStatisticsDTO:
        normalized_payloads = self._normalize_reviews(reviews)
        status_counts = Counter(
            (
                str(self._extract_review_field(review, "status") or "unknown")
                .strip()
                .lower()
            )
            for review in reviews
        )
        style_categories = {
            category
            for payload in normalized_payloads
            for category in payload.style_categories
        }
        confidence_values = [
            payload.confidence
            for payload in normalized_payloads
            if payload.confidence is not None
        ]
        raw_message_values = [
            payload.raw_message_count
            for payload in normalized_payloads
            if payload.raw_message_count is not None
        ]
        raw_message_count = int(sum(raw_message_values)) if raw_message_values else 0
        raw_message_count_source = (
            "normalized_payload_fallback" if raw_message_values else "unavailable"
        )
        if raw_message_count == 0:
            logger.warning(
                f"[LearningService] raw_message_count_unavailable total_reviews={len(reviews)}"
            )
        average_confidence = (
            round(sum(confidence_values) / len(confidence_values), 1)
            if confidence_values
            else None
        )
        confidence_fallback_used = not bool(confidence_values)
        style_feature_total_count = sum(
            len(payload.emotion_patterns)
            + len(payload.language_patterns)
            + len(payload.topic_preferences)
            for payload in normalized_payloads
        )
        logger.info(
            "[LearningService] style statistics aggregate "
            f"style_type_distinct_count={len(style_categories)} "
            f"style_feature_total_count={style_feature_total_count} "
            f"raw_message_count_source={raw_message_count_source} "
            f"raw_message_count_value={raw_message_count} "
            f"confidence_value_count={len(confidence_values)} "
            f"confidence_fallback_used={confidence_fallback_used}"
        )
        return StyleLearningStatisticsDTO(
            style_type_count=len(style_categories),
            average_confidence=average_confidence,
            raw_message_count=raw_message_count,
            last_updated_at=self._get_last_updated_timestamp(reviews),
            total_reviews=len(reviews),
            pending_reviews=status_counts.get("pending", 0),
            approved_reviews=status_counts.get("approved", 0),
            rejected_reviews=status_counts.get("rejected", 0),
            style_feature_total_count=style_feature_total_count,
            raw_message_count_source=raw_message_count_source,
            raw_message_count_value=raw_message_count,
            confidence_value_count=len(confidence_values),
            confidence_fallback_used=confidence_fallback_used,
        )

    def aggregate_emotion_patterns(
        self,
        normalized_payloads: Sequence[NormalizedStyleLearningPayload],
        db_patterns: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return self._aggregate_pattern_bucket(
            "emotion_patterns", normalized_payloads, db_patterns or []
        )

    def aggregate_language_patterns(
        self,
        normalized_payloads: Sequence[NormalizedStyleLearningPayload],
        db_patterns: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return self._aggregate_pattern_bucket(
            "language_patterns", normalized_payloads, db_patterns or []
        )

    def aggregate_topic_preferences(
        self,
        normalized_payloads: Sequence[NormalizedStyleLearningPayload],
        db_patterns: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return self._aggregate_pattern_bucket(
            "topic_preferences", normalized_payloads, db_patterns or []
        )

    def normalize_pattern_payload(
        self,
        payload: Any,
        *,
        review_id: Any = None,
        group_id: Any = None,
        source_label: str = "learned_patterns",
    ) -> NormalizedStyleLearningPayload:
        raw_type = type(payload).__name__
        parse_error: Optional[str] = None
        parsed_payload = payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                parsed_payload = None
            else:
                try:
                    parsed_payload = json.loads(text)
                except (TypeError, json.JSONDecodeError) as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"
                    parsed_payload = payload
                    logger.warning(
                        "[LearningService] payload_parse_error "
                        f"source={source_label} review_id={review_id} group_id={group_id} "
                        f"payload_raw_type=str payload_parse_error={parse_error} "
                        f"raw_preview={text[:160]!r}"
                    )

        normalized = NormalizedStyleLearningPayload()
        detected_keys: set[str] = set()
        style_categories: set[str] = set()
        self._normalize_style_learning_payload(
            parsed_payload,
            normalized,
            detected_keys=detected_keys,
            style_categories=style_categories,
        )
        normalized.payload_detected_keys = sorted(detected_keys)
        normalized.style_categories = sorted(style_categories)
        logger.info(
            "[LearningService] normalize_style_learning_payload "
            f"source={source_label} review_id={review_id} group_id={group_id} "
            f"payload_raw_type={raw_type} "
            f"payload_detected_keys={normalized.payload_detected_keys} "
            f"payload_normalized_summary={normalized.summary()} "
            f"payload_parse_error={parse_error}"
        )
        return normalized

    def _normalize_style_learning_payload(
        self,
        payload: Any,
        normalized: NormalizedStyleLearningPayload,
        *,
        detected_keys: set[str],
        style_categories: set[str],
        preferred_bucket: Optional[str] = None,
    ) -> None:
        if payload is None:
            return
        if isinstance(payload, list):
            for item in payload:
                self._normalize_style_learning_payload(
                    item,
                    normalized,
                    detected_keys=detected_keys,
                    style_categories=style_categories,
                    preferred_bucket=preferred_bucket,
                )
            return
        if isinstance(payload, str) or not isinstance(payload, dict):
            return

        detected_keys.update(str(key) for key in payload.keys())
        raw_message_count = self._extract_first_numeric(payload, self._RAW_MESSAGE_KEYS)
        if raw_message_count is not None and normalized.raw_message_count is None:
            normalized.raw_message_count = int(raw_message_count)
        confidence = self._extract_first_numeric(payload, self._CONFIDENCE_KEYS)
        if confidence is not None and normalized.confidence is None:
            normalized.confidence = self._normalize_percentage(confidence)

        explicit_bucket = self._resolve_bucket_name(
            payload.get("type") or payload.get("category") or payload.get("pattern_type")
        )
        if explicit_bucket:
            style_categories.add(self._bucket_to_category(explicit_bucket))

        for key, value in payload.items():
            bucket_name = self._resolve_bucket_name(key)
            if bucket_name:
                style_categories.add(self._bucket_to_category(bucket_name))
                self._normalize_style_learning_payload(
                    value,
                    normalized,
                    detected_keys=detected_keys,
                    style_categories=style_categories,
                    preferred_bucket=bucket_name,
                )

        entry_bucket = preferred_bucket or explicit_bucket
        if entry_bucket:
            entry = self._build_pattern_entry(payload, entry_bucket)
            if entry is not None:
                self._append_pattern_entry(normalized, entry_bucket, entry)

        for nested_key in ("items", "patterns", "features", "values"):
            if nested_key in payload:
                self._normalize_style_learning_payload(
                    payload.get(nested_key),
                    normalized,
                    detected_keys=detected_keys,
                    style_categories=style_categories,
                    preferred_bucket=entry_bucket,
                )

    def _build_pattern_entry(
        self, item: Dict[str, Any], bucket_name: str
    ) -> Optional[Dict[str, Any]]:
        name = self._extract_pattern_label(item)
        if not name:
            return None
        numeric_value, value_source = self._extract_numeric_value(item)
        if numeric_value is None:
            return None
        entry = {
            "name": str(name).strip(),
            "numeric_value": numeric_value,
            "value_source": value_source,
            "bucket": bucket_name,
        }
        confidence = self._extract_first_numeric(item, self._CONFIDENCE_KEYS)
        if confidence is not None:
            entry["confidence"] = self._normalize_percentage(confidence)
        return self._apply_pattern_compatibility_fields(entry, bucket_name)

    def _append_pattern_entry(
        self,
        normalized: NormalizedStyleLearningPayload,
        bucket_name: str,
        entry: Dict[str, Any],
    ) -> None:
        if bucket_name == "emotion_patterns":
            normalized.emotion_patterns.append(entry)
        elif bucket_name == "language_patterns":
            normalized.language_patterns.append(entry)
        elif bucket_name == "topic_preferences":
            normalized.topic_preferences.append(entry)

    def _aggregate_pattern_bucket(
        self,
        bucket_name: str,
        normalized_payloads: Sequence[NormalizedStyleLearningPayload],
        db_patterns: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for payload in normalized_payloads:
            for entry in getattr(payload, bucket_name):
                self._merge_pattern_entry(merged, entry)
        for entry in db_patterns:
            if entry.get("bucket") == bucket_name:
                self._merge_pattern_entry(merged, entry)
        ranked = sorted(
            merged.values(),
            key=lambda item: (item.get("numeric_value", 0.0), item.get("name", "")),
            reverse=True,
        )
        return ranked[:20]

    def _merge_pattern_entry(
        self, merged: Dict[str, Dict[str, Any]], entry: Dict[str, Any]
    ) -> None:
        name = str(entry.get("name") or "").strip()
        if not name:
            return
        existing = merged.get(name)
        if existing is None or float(entry.get("numeric_value") or 0.0) > float(
            existing.get("numeric_value") or 0.0
        ):
            merged[name] = dict(entry)
            return
        if "confidence" in entry and "confidence" not in existing:
            existing["confidence"] = entry["confidence"]

    def _extract_pattern_label(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            for key in (
                "display_name",
                "name",
                "pattern",
                "style",
                "topic",
                "label",
                "text",
                "content",
            ):
                label = self._coerce_pattern_label(item.get(key))
                if label:
                    return label
            return None
        return self._coerce_pattern_label(item)

    def _coerce_pattern_label(self, raw_value: Any) -> Optional[str]:
        if raw_value is None:
            return None
        if isinstance(raw_value, list):
            for item in raw_value:
                label = self._coerce_pattern_label(item)
                if label:
                    return label
            return None
        if isinstance(raw_value, dict):
            return self._extract_pattern_label(raw_value)

        text = str(raw_value).strip()
        if not text:
            return None
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if parsed is not None:
                return self._coerce_pattern_label(parsed)
        return self._sanitize_text_label(text)

    def _sanitize_text_label(self, text: str) -> Optional[str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        if self._looks_like_mojibake(cleaned):
            for candidate in self._repair_text_candidates(cleaned):
                if candidate and not self._looks_like_mojibake(candidate):
                    cleaned = candidate
                    break
        if self._looks_like_mojibake(cleaned):
            return None
        if cleaned.count("?") >= max(2, len(cleaned) // 2):
            return None
        return cleaned

    def _repair_text_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        for source_encoding, target_encoding in (("gbk", "utf-8"), ("latin1", "utf-8")):
            try:
                repaired = text.encode(source_encoding).decode(target_encoding)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            repaired = repaired.strip()
            if repaired and repaired not in candidates:
                candidates.append(repaired)
        return candidates

    def _looks_like_mojibake(self, text: str) -> bool:
        if not text:
            return False
        marker_hits = sum(text.count(marker) for marker in self._MOJIBAKE_MARKERS)
        if marker_hits >= 2:
            return True
        return text.count("\ufffd") > 0

    def _apply_pattern_compatibility_fields(
        self, entry: Dict[str, Any], bucket_name: str
    ) -> Dict[str, Any]:
        normalized_entry = dict(entry)
        normalized_entry["display_name"] = normalized_entry.get("name")
        numeric_value = float(normalized_entry.get("numeric_value") or 0.0)
        if bucket_name == "emotion_patterns":
            normalized_entry.setdefault("pattern", normalized_entry["name"])
            normalized_entry.setdefault("frequency", round(numeric_value, 2))
        elif bucket_name == "language_patterns":
            normalized_entry.setdefault("style", normalized_entry["name"])
            normalized_entry.setdefault("frequency", round(numeric_value, 2))
        elif bucket_name == "topic_preferences":
            normalized_entry.setdefault("topic", normalized_entry["name"])
            normalized_entry.setdefault("interest_level", round(numeric_value, 2))
        return normalized_entry

    def _normalize_reviews(
        self, reviews: Sequence[Any]
    ) -> List[NormalizedStyleLearningPayload]:
        return [
            self.normalize_pattern_payload(
                self._extract_review_field(review, "learned_patterns"),
                review_id=self._extract_review_field(review, "id"),
                group_id=self._extract_review_field(review, "group_id"),
                source_label="learned_patterns",
            )
            for review in reviews
        ]

    # Compatibility wrappers kept for existing tests and callers.
    def _aggregate_style_learning_statistics(
        self, reviews: Sequence[Any]
    ) -> Dict[str, Any]:
        return self.aggregate_statistics_from_reviews(reviews).to_api_dict()

    def _normalize_learned_patterns_payload(
        self,
        raw_value: Any,
        *,
        review_id: Any = None,
        group_id: Any = None,
        source_label: str = "learned_patterns",
    ) -> Any:
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except (TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[LearningService] learned_patterns JSON parse failed "
                    f"source={source_label} review_id={review_id} group_id={group_id} "
                    f"raw_preview={text[:160]!r} error={type(exc).__name__}: {exc}"
                )
                return raw_value
        return raw_value

    def _ingest_learned_pattern_payload(
        self,
        payload: Any,
        patterns_data: Dict[str, List[Dict[str, Any]]],
        pattern_type_map: Dict[str, str],
        *,
        review_id: Any = None,
        group_id: Any = None,
        source_label: str = "learned_patterns",
        forced_target: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized = self.normalize_pattern_payload(
            payload,
            review_id=review_id,
            group_id=group_id,
            source_label=source_label,
        )
        bucket_pairs = (
            ("emotion_patterns", normalized.emotion_patterns),
            ("language_patterns", normalized.language_patterns),
            ("topic_preferences", normalized.topic_preferences),
        )
        for bucket_name, entries in bucket_pairs:
            target = "topic_patterns" if bucket_name == "topic_preferences" else bucket_name
            patterns_data.setdefault(target, [])
            patterns_data[target].extend(entries)
            if debug_info is not None and bucket_name == "emotion_patterns":
                debug_info.setdefault("emotion_keys", set()).update(
                    key
                    for key in normalized.payload_detected_keys
                    if key in self._EMOTION_KEYS
                )
            if debug_info is not None and bucket_name == "language_patterns":
                debug_info.setdefault("language_value_sources", []).extend(
                    entry.get("value_source") or "none" for entry in entries
                )

    async def _collect_db_patterns(self) -> List[Dict[str, Any]]:
        from sqlalchemy import desc, select

        from ...models.orm.learning import StyleLearningPattern

        patterns: List[Dict[str, Any]] = []
        async with self.db_manager.get_session() as session:
            stmt = (
                select(StyleLearningPattern)
                .order_by(desc(StyleLearningPattern.usage_count))
                .limit(100)
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                bucket_name = self._resolve_bucket_name(row.pattern_type)
                if not bucket_name:
                    continue
                pattern_name = self._coerce_pattern_label(row.pattern)
                if not pattern_name:
                    continue
                numeric_value, value_source = self._extract_numeric_value(
                    {"usage_count": row.usage_count, "confidence": row.confidence}
                )
                if numeric_value is None:
                    continue
                entry = {
                    "name": pattern_name,
                    "numeric_value": numeric_value,
                    "value_source": value_source or "style_learning_patterns",
                    "bucket": bucket_name,
                }
                if row.confidence is not None:
                    entry["confidence"] = self._normalize_percentage(row.confidence)
                patterns.append(
                    self._apply_pattern_compatibility_fields(entry, bucket_name)
                )
        return patterns

    async def _get_progress_data(self) -> List[Dict[str, Any]]:
        if not self.db_manager or not hasattr(self.db_manager, "get_style_progress_data"):
            return []
        try:
            progress_data = await self.db_manager.get_style_progress_data()
            if not isinstance(progress_data, list):
                logger.warning(
                    "[LearningService] style_progress returned non-list "
                    f"payload={self._summarize_for_log(progress_data)}"
                )
                return []

            normalized_progress = [
                self._normalize_progress_item(item)
                for item in progress_data
                if isinstance(item, dict)
            ]
            quality_sources = sorted(
                {
                    item.get("quality_value_source", "missing")
                    for item in normalized_progress
                }
            )
            sample_sources = sorted(
                {
                    item.get("sample_count_source", "unavailable")
                    for item in normalized_progress
                }
            )
            quality_missing_count = sum(
                1 for item in normalized_progress if item.get("quality_percent") is None
            )
            payload_summary = [
                {
                    "label": item.get("label") or item.get("group_id") or "unknown",
                    "quality_percent": item.get("quality_percent"),
                    "sample_count": item.get("sample_count"),
                }
                for item in normalized_progress[:5]
            ]
            logger.info(
                "[LearningService] style_progress normalized "
                f"progress_item_count={len(normalized_progress)} "
                f"progress_item_keys={sorted(normalized_progress[0].keys()) if normalized_progress else []} "
                f"progress_quality_value_source={quality_sources} "
                f"progress_quality_missing_count={quality_missing_count} "
                f"progress_sample_count_source={sample_sources} "
                f"progress_payload_summary={payload_summary}"
            )
            return normalized_progress
        except Exception as exc:
            logger.warning(
                "[LearningService] style_progress fetch failed "
                f"error={type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return []

    def _normalize_progress_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        quality_score, quality_percent, quality_source = self._extract_progress_quality(
            item
        )
        sample_count, sample_source = self._extract_progress_sample_count(item)
        label = self._build_progress_label(item)
        return {
            "group_id": item.get("group_id"),
            "label": label,
            "timestamp": self._coerce_timestamp(item.get("timestamp")) or 0.0,
            "quality_score": quality_score,
            "quality_percent": quality_percent,
            "score": quality_score,
            "quality_value_source": quality_source,
            "quality_missing_reason": item.get("quality_missing_reason") or "",
            "sample_count": sample_count,
            "sample_count_source": sample_source,
            "processed_messages": int(item.get("processed_messages") or 0),
            "filtered_count": int(item.get("filtered_count") or 0),
            "message_count": int(item.get("message_count") or 0),
            "success": bool(item.get("success")),
            "batch_name": item.get("batch_name") or "",
        }

    def _extract_progress_quality(
        self, item: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], str]:
        raw_source = str(item.get("quality_value_source") or "").strip() or "missing"
        quality_percent = self._coerce_optional_float(item.get("quality_percent"))
        if quality_percent is not None:
            normalized_percent = self._normalize_percentage(quality_percent)
            normalized_score = (
                round(normalized_percent / 100.0, 4)
                if normalized_percent > 1.0
                else round(float(quality_percent), 4)
            )
            return (
                normalized_score,
                normalized_percent,
                raw_source if raw_source != "missing" else "quality_percent",
            )

        quality_score = self._coerce_optional_float(item.get("quality_score"))
        if quality_score is not None:
            return (
                round(float(quality_score), 4),
                self._normalize_percentage(quality_score),
                raw_source if raw_source != "missing" else "quality_score",
            )

        score = self._coerce_optional_float(item.get("score"))
        if score is not None:
            return (
                round(float(score), 4),
                self._normalize_percentage(score),
                raw_source if raw_source != "missing" else "score",
            )

        return None, None, raw_source

    def _extract_progress_sample_count(self, item: Dict[str, Any]) -> Tuple[int, str]:
        for key in (
            "sample_count",
            "filtered_count",
            "message_count",
            "processed_messages",
            "total_samples",
        ):
            numeric_value = self._coerce_optional_float(item.get(key))
            if numeric_value is None:
                continue
            return max(int(numeric_value), 0), key
        return 0, "unavailable"

    def _build_progress_label(self, item: Dict[str, Any]) -> str:
        if item.get("label"):
            return str(item.get("label"))
        if item.get("batch_name"):
            return str(item.get("batch_name"))
        if item.get("group_id"):
            return f"group_{item.get('group_id')}"
        timestamp = self._coerce_timestamp(item.get("timestamp"))
        if timestamp:
            try:
                return datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")
            except (OverflowError, OSError, ValueError):
                return str(timestamp)
        return "unknown"

    async def _get_review_cache_signature(self) -> Tuple[int, float]:
        from sqlalchemy import func, select

        from ...models.orm.learning import StyleLearningReview

        async with self.db_manager.get_session() as session:
            stmt = select(
                func.count(StyleLearningReview.id),
                func.max(StyleLearningReview.timestamp),
            )
            result = await session.execute(stmt)
            total_count, latest_timestamp = result.one()
        return int(total_count or 0), float(latest_timestamp or 0.0)

    async def _get_patterns_cache_signature(self) -> Tuple[int, float, int, float]:
        from sqlalchemy import func, select

        from ...models.orm.learning import StyleLearningPattern, StyleLearningReview

        async with self.db_manager.get_session() as session:
            review_stmt = select(
                func.count(StyleLearningReview.id),
                func.max(StyleLearningReview.timestamp),
            )
            pattern_stmt = select(
                func.count(StyleLearningPattern.id),
                func.max(StyleLearningPattern.updated_at),
            )
            review_result = await session.execute(review_stmt)
            pattern_result = await session.execute(pattern_stmt)
            review_count, review_latest = review_result.one()
            pattern_count, pattern_latest = pattern_result.one()
        return (
            int(review_count or 0),
            float(review_latest or 0.0),
            int(pattern_count or 0),
            float(pattern_latest or 0.0),
        )

    def _get_cached_statistics(
        self, signature: Tuple[int, float]
    ) -> Optional[Dict[str, Any]]:
        if self._results_cache.get("signature") == signature:
            return dict(self._results_cache.get("statistics") or {})
        return None

    def _set_cached_statistics(
        self, signature: Tuple[int, float], statistics: Dict[str, Any]
    ) -> None:
        self.__class__._results_cache = {
            "signature": signature,
            "statistics": dict(statistics),
        }

    def _get_cached_patterns(
        self, signature: Tuple[int, float, int, float]
    ) -> Optional[Dict[str, Any]]:
        if self._patterns_cache.get("signature") == signature:
            return dict(self._patterns_cache.get("payload") or {})
        return None

    def _set_cached_patterns(
        self, signature: Tuple[int, float, int, float], payload: Dict[str, Any]
    ) -> None:
        self.__class__._patterns_cache = {
            "signature": signature,
            "payload": dict(payload),
        }

    def _statistics_from_payload(
        self, payload: Dict[str, Any]
    ) -> StyleLearningStatisticsDTO:
        raw_message_count = payload.get("raw_message_count")
        if raw_message_count is None:
            raw_message_count = payload.get("total_samples")
        return StyleLearningStatisticsDTO(
            style_type_count=int(
                payload.get("style_type_count") or payload.get("unique_styles") or 0
            ),
            average_confidence=self._coerce_optional_float(
                payload.get("average_confidence", payload.get("avg_confidence"))
            ),
            raw_message_count=int(raw_message_count or 0),
            last_updated_at=self._coerce_timestamp(
                payload.get("last_updated_at", payload.get("latest_update"))
            ),
            total_reviews=int(payload.get("total_reviews") or 0),
            pending_reviews=int(payload.get("pending_reviews") or 0),
            approved_reviews=int(payload.get("approved_reviews") or 0),
            rejected_reviews=int(payload.get("rejected_reviews") or 0),
            style_feature_total_count=int(payload.get("style_feature_total_count") or 0),
            raw_message_count_source=str(
                payload.get("raw_message_count_source") or "unavailable"
            ),
            raw_message_count_value=int(payload.get("raw_message_count_value") or 0),
            confidence_value_count=int(payload.get("confidence_value_count") or 0),
            confidence_fallback_used=bool(payload.get("confidence_fallback_used")),
        )

    def _resolve_bucket_name(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        if key in self._EMOTION_KEYS:
            return "emotion_patterns"
        if key in self._LANGUAGE_KEYS:
            return "language_patterns"
        if key in self._TOPIC_KEYS:
            return "topic_preferences"
        return None

    def _bucket_to_category(self, bucket_name: str) -> str:
        if bucket_name == "emotion_patterns":
            return "emotion"
        if bucket_name == "language_patterns":
            return "language"
        return "topic"

    def _extract_numeric_value(
        self, payload: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[str]]:
        for key in self._VALUE_KEYS:
            numeric = self._coerce_optional_float(payload.get(key))
            if numeric is None:
                continue
            if key in {"score", "confidence", "weight"}:
                numeric = self._normalize_percentage(numeric)
            return numeric, key
        return None, None

    def _extract_first_numeric(
        self, payload: Any, keys: Iterable[str]
    ) -> Optional[float]:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    numeric = self._coerce_optional_float(payload.get(key))
                    if numeric is not None:
                        return numeric
            for value in payload.values():
                numeric = self._extract_first_numeric(value, keys)
                if numeric is not None:
                    return numeric
        elif isinstance(payload, list):
            for item in payload:
                numeric = self._extract_first_numeric(item, keys)
                if numeric is not None:
                    return numeric
        return None

    def _get_last_updated_timestamp(self, reviews: Sequence[Any]) -> Optional[float]:
        timestamps = [
            self._coerce_timestamp(
                self._extract_review_field(review, "review_time")
                or self._extract_review_field(review, "timestamp")
            )
            for review in reviews
        ]
        timestamps = [item for item in timestamps if item is not None]
        return max(timestamps) if timestamps else None

    @staticmethod
    def _extract_review_field(review: Any, field_name: str) -> Any:
        if isinstance(review, dict):
            return review.get(field_name)
        return getattr(review, field_name, None)

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return float(value.timestamp())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_percentage(value: float) -> float:
        normalized = float(value)
        if 0.0 <= normalized <= 1.0:
            normalized *= 100.0
        if normalized < 0:
            normalized = 0.0
        if normalized > 100:
            normalized = 100.0
        return round(normalized, 1)

    @staticmethod
    def _summarize_for_log(value: Any, limit: int = 160) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            text = value.strip()
            return f"str(len={len(text)}, preview={text[:limit]!r})"
        if isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:8]}, size={len(value)})"
        if isinstance(value, list):
            return f"list(len={len(value)}, preview_types={[type(item).__name__ for item in value[:3]]})"
        return f"{type(value).__name__}({str(value)[:limit]!r})"
