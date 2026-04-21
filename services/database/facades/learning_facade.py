"""
学习 Facade — 人格学习审核、风格学习审核、学习批次/会话、统计的业务入口
"""
import time
import json
from typing import Dict, List, Optional, Any

from astrbot.api import logger

from ._base import BaseFacade
from sqlalchemy import delete as sa_delete, desc, func, or_, select
from ....models.orm.learning import (
    LearningBatch,
    LearningSession,
    PersonaLearningReview,
    StyleLearningPattern,
    StyleLearningReview,
)
from ....models.orm.message import FilteredMessage, RawMessage
from ....models.orm.performance import LearningPerformanceHistory


class LearningFacade(BaseFacade):
    _QUALITY_REPAIR_MARKER = '[quality_repair]'
    """学习管理 Facade — 包装所有学习相关的数据库方法"""

    # Persona Learning Review methods

    async def add_persona_learning_review(self, review_data: Dict[str, Any]) -> int:
        """创建人格学习审核记录

        Args:
            review_data: 审核数据字典

        Returns:
            新记录的 id，失败返回 0
        """
        try:
            async with self.get_session() as session:

                metadata = review_data.get('metadata', {})
                record = PersonaLearningReview(
                    timestamp=self._to_float_ts(
                        review_data.get('timestamp'), default=time.time()
                    ),
                    group_id=review_data.get('group_id', ''),
                    update_type=review_data.get('update_type', ''),
                    original_content=review_data.get('original_content', ''),
                    new_content=review_data.get('new_content', ''),
                    proposed_content=review_data.get('proposed_content', ''),
                    confidence_score=review_data.get('confidence_score', 0.0),
                    reason=review_data.get('reason', ''),
                    status='pending',
                    metadata_=json.dumps(metadata, ensure_ascii=False) if metadata else None,
                )
                session.add(record)
                await session.commit()
                await session.refresh(record)
                return record.id
        except Exception as e:
            self._logger.error(f"[LearningFacade] 添加人格学习审核记录失败: {e}")
            return 0

    async def get_pending_persona_update_records(self) -> List[Dict[str, Any]]:
        """获取所有待审核的人格更新记录

        Returns:
            待审核记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(PersonaLearningReview)
                    .where(PersonaLearningReview.status == 'pending')
                    .order_by(desc(PersonaLearningReview.timestamp))
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'timestamp': r.timestamp,
                        'group_id': r.group_id,
                        'update_type': r.update_type,
                        'original_content': r.original_content,
                        'new_content': r.new_content,
                        'proposed_content': r.proposed_content,
                        'confidence_score': r.confidence_score,
                        'reason': r.reason,
                        'status': r.status,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                        'metadata': json.loads(r.metadata_) if r.metadata_ else {},
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取待审核人格更新记录失败: {e}")
            return []

    async def save_persona_update_record(self, record: Dict[str, Any]) -> int:
        """保存人格更新记录（add_persona_learning_review 的别名）

        Args:
            record: 记录数据字典

        Returns:
            新记录的 id，失败返回 0
        """
        return await self.add_persona_learning_review(record)

    async def update_persona_update_record_status(
        self, record_id: int, new_status: str, reviewer_comment: str = ''
    ) -> bool:
        """更新人格更新记录的状态

        Args:
            record_id: 记录 ID
            new_status: 新状态 (approved/rejected)
            reviewer_comment: 审核评论

        Returns:
            是否更新成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(PersonaLearningReview).where(
                    PersonaLearningReview.id == record_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                record.status = new_status
                record.reviewer_comment = reviewer_comment
                record.review_time = time.time()
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 更新人格更新记录状态失败: {e}")
            return False

    async def delete_persona_update_record(self, record_id: int) -> bool:
        """删除人格更新记录

        Args:
            record_id: 记录 ID

        Returns:
            是否删除成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(PersonaLearningReview).where(
                    PersonaLearningReview.id == record_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                await session.execute(
                    sa_delete(PersonaLearningReview).where(
                        PersonaLearningReview.id == record_id
                    )
                )
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 删除人格更新记录失败: {e}")
            return False

    async def get_persona_update_record_by_id(
        self, record_id: int
    ) -> Optional[Dict[str, Any]]:
        """根据 ID 获取人格更新记录

        Args:
            record_id: 记录 ID

        Returns:
            记录字典或 None
        """
        try:
            async with self.get_session() as session:

                stmt = select(PersonaLearningReview).where(
                    PersonaLearningReview.id == record_id
                )
                result = await session.execute(stmt)
                r = result.scalar_one_or_none()
                if not r:
                    return None
                return {
                    'id': r.id,
                    'timestamp': r.timestamp,
                    'group_id': r.group_id,
                    'update_type': r.update_type,
                    'original_content': r.original_content,
                    'new_content': r.new_content,
                    'proposed_content': r.proposed_content,
                    'confidence_score': r.confidence_score,
                    'reason': r.reason,
                    'status': r.status,
                    'reviewer_comment': r.reviewer_comment,
                    'review_time': r.review_time,
                    'metadata': json.loads(r.metadata_) if r.metadata_ else {},
                }
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取人格更新记录失败: {e}")
            return None

    async def get_reviewed_persona_update_records(
        self, limit: int = 50, offset: int = 0, status_filter: str = None
    ) -> List[Dict[str, Any]]:
        """获取已审核的人格更新记录

        Args:
            limit: 返回数量限制
            offset: 偏移量
            status_filter: 状态过滤

        Returns:
            已审核记录列表
        """
        try:
            async with self.get_session() as session:

                if status_filter:
                    stmt = (
                        select(PersonaLearningReview)
                        .where(PersonaLearningReview.status == status_filter)
                        .order_by(desc(PersonaLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )
                else:
                    stmt = (
                        select(PersonaLearningReview)
                        .where(PersonaLearningReview.status.in_(['approved', 'rejected']))
                        .order_by(desc(PersonaLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'timestamp': r.timestamp,
                        'group_id': r.group_id,
                        'update_type': r.update_type,
                        'original_content': r.original_content,
                        'new_content': r.new_content,
                        'proposed_content': r.proposed_content,
                        'confidence_score': r.confidence_score,
                        'reason': r.reason,
                        'status': r.status,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                        'metadata': json.loads(r.metadata_) if r.metadata_ else {},
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取已审核人格更新记录失败: {e}")
            return []

    async def get_pending_persona_learning_reviews(
        self, limit: int = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取待审核的人格学习审核记录

        Args:
            limit: 可选的返回数量限制
            offset: 分页偏移量

        Returns:
            待审核记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(PersonaLearningReview)
                    .where(PersonaLearningReview.status == 'pending')
                    .order_by(desc(PersonaLearningReview.timestamp))
                )
                if offset > 0:
                    stmt = stmt.offset(offset)
                if limit is not None:
                    stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'timestamp': r.timestamp,
                        'group_id': r.group_id,
                        'update_type': r.update_type,
                        'original_content': r.original_content,
                        'new_content': r.new_content,
                        'proposed_content': r.proposed_content,
                        'confidence_score': r.confidence_score,
                        'reason': r.reason,
                        'status': r.status,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                        'metadata': json.loads(r.metadata_) if r.metadata_ else {},
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取待审核人格学习审核记录失败: {e}")
            return []

    async def get_reviewed_persona_learning_updates(
        self, limit=50, offset=0, status_filter=None
    ) -> List[Dict]:
        """获取已审核的人格学习更新记录

        Args:
            limit: 返回数量限制
            offset: 偏移量
            status_filter: 状态过滤

        Returns:
            已审核记录列表
        """
        try:
            async with self.get_session() as session:

                if status_filter:
                    stmt = (
                        select(PersonaLearningReview)
                        .where(PersonaLearningReview.status == status_filter)
                        .order_by(desc(PersonaLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )
                else:
                    stmt = (
                        select(PersonaLearningReview)
                        .where(PersonaLearningReview.status.in_(['approved', 'rejected']))
                        .order_by(desc(PersonaLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'timestamp': r.timestamp,
                        'group_id': r.group_id,
                        'update_type': r.update_type,
                        'original_content': r.original_content,
                        'new_content': r.new_content,
                        'proposed_content': r.proposed_content,
                        'confidence_score': r.confidence_score,
                        'reason': r.reason,
                        'status': r.status,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取已审核人格学习更新记录失败: {e}")
            return []

    async def delete_persona_learning_review_by_id(self, review_id: int) -> bool:
        """根据 ID 删除人格学习审核记录

        Args:
            review_id: 审核记录 ID

        Returns:
            是否删除成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(PersonaLearningReview).where(
                    PersonaLearningReview.id == review_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                await session.execute(
                    sa_delete(PersonaLearningReview).where(
                        PersonaLearningReview.id == review_id
                    )
                )
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 删除人格学习审核记录失败: {e}")
            return False

    async def get_persona_learning_review_by_id(
        self, review_id: int
    ) -> Optional[Dict]:
        """根据 ID 获取人格学习审核记录（get_persona_update_record_by_id 的别名）

        Args:
            review_id: 审核记录 ID

        Returns:
            记录字典或 None
        """
        return await self.get_persona_update_record_by_id(review_id)

    async def update_persona_learning_review_status(
        self, review_id, new_status, reviewer_comment='',
        modified_content=None,
    ) -> bool:
        """更新人格学习审核记录状态

        Args:
            review_id: 审核记录 ID
            new_status: 新状态
            reviewer_comment: 审核评论
            modified_content: 用户修改后的内容（可选）

        Returns:
            是否更新成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(PersonaLearningReview).where(
                    PersonaLearningReview.id == review_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                record.status = new_status
                record.reviewer_comment = reviewer_comment
                record.review_time = time.time()

                if modified_content:
                    record.proposed_content = modified_content
                    record.new_content = modified_content

                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 更新人格学习审核记录状态失败: {e}")
            return False

    # Style Learning Review methods

    async def create_style_learning_review(
        self, review_data: Dict[str, Any]
    ) -> int:
        """创建风格学习审核记录

        Args:
            review_data: 审核数据字典

        Returns:
            新记录的 id，失败返回 0
        """
        try:
            async with self.get_session() as session:

                learned_patterns = review_data.get('learned_patterns', [])
                record = StyleLearningReview(
                    type=review_data.get('type', ''),
                    group_id=review_data.get('group_id', ''),
                    timestamp=self._to_float_ts(
                        review_data.get('timestamp'), default=time.time()
                    ),
                    learned_patterns=json.dumps(learned_patterns, ensure_ascii=False)
                    if isinstance(learned_patterns, (list, dict))
                    else learned_patterns,
                    few_shots_content=review_data.get('few_shots_content', ''),
                    status='pending',
                    description=review_data.get('description', ''),
                )
                session.add(record)
                await session.commit()
                await session.refresh(record)
                return record.id
        except Exception as e:
            self._logger.error(f"[LearningFacade] 创建风格学习审核记录失败: {e}")
            return 0

    async def get_pending_style_reviews(self, limit=None, offset=0) -> List[Dict]:
        """获取待审核的风格学习记录

        Args:
            limit: 可选的返回数量限制
            offset: 分页偏移量

        Returns:
            待审核记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(StyleLearningReview)
                    .where(StyleLearningReview.status == 'pending')
                    .order_by(desc(StyleLearningReview.timestamp))
                )
                if offset > 0:
                    stmt = stmt.offset(offset)
                if limit is not None:
                    stmt = stmt.limit(limit)

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'type': r.type,
                        'group_id': r.group_id,
                        'timestamp': r.timestamp,
                        'learned_patterns': json.loads(r.learned_patterns)
                        if r.learned_patterns
                        else [],
                        'few_shots_content': r.few_shots_content,
                        'status': r.status,
                        'description': r.description,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                        'created_at': r.created_at,
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取待审核风格学习记录失败: {e}")
            return []

    async def get_approved_few_shots(
        self, group_id: str, limit: int = 3
    ) -> List[str]:
        """获取指定群组已审批的 few-shot 对话内容

        Args:
            group_id: 群组 ID
            limit: 返回条数上限

        Returns:
            few_shots_content 文本列表，按时间倒序
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(StyleLearningReview.few_shots_content)
                    .where(
                        StyleLearningReview.status == 'approved',
                        StyleLearningReview.group_id == group_id,
                        StyleLearningReview.few_shots_content.isnot(None),
                        StyleLearningReview.few_shots_content != '',
                    )
                    .order_by(desc(StyleLearningReview.timestamp))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取已审批 few-shots 失败: {e}")
            return []

    async def get_reviewed_style_learning_updates(
        self, limit=50, offset=0, status_filter=None
    ) -> List[Dict]:
        """获取已审核的风格学习更新记录

        Args:
            limit: 返回数量限制
            offset: 偏移量
            status_filter: 状态过滤

        Returns:
            已审核记录列表
        """
        try:
            async with self.get_session() as session:

                if status_filter:
                    stmt = (
                        select(StyleLearningReview)
                        .where(StyleLearningReview.status == status_filter)
                        .order_by(desc(StyleLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )
                else:
                    stmt = (
                        select(StyleLearningReview)
                        .where(StyleLearningReview.status.in_(['approved', 'rejected']))
                        .order_by(desc(StyleLearningReview.review_time))
                        .offset(offset)
                        .limit(limit)
                    )

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        'id': r.id,
                        'type': r.type,
                        'group_id': r.group_id,
                        'timestamp': r.timestamp,
                        'learned_patterns': json.loads(r.learned_patterns)
                        if r.learned_patterns
                        else [],
                        'few_shots_content': r.few_shots_content,
                        'status': r.status,
                        'description': r.description,
                        'reviewer_comment': r.reviewer_comment,
                        'review_time': r.review_time,
                    }
                    for r in rows
                ]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取已审核风格学习更新记录失败: {e}")
            return []

    async def update_style_review_status(
        self, review_id, new_status, reviewer_comment=''
    ) -> bool:
        """更新风格学习审核记录状态

        Args:
            review_id: 审核记录 ID
            new_status: 新状态 (approved/rejected)
            reviewer_comment: 审核评论

        Returns:
            是否更新成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(StyleLearningReview).where(
                    StyleLearningReview.id == review_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                record.status = new_status
                record.reviewer_comment = reviewer_comment
                record.review_time = time.time()
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 更新风格学习审核记录状态失败: {e}")
            return False

    async def delete_style_review_by_id(self, review_id: int) -> bool:
        """根据 ID 删除风格学习审核记录

        Args:
            review_id: 审核记录 ID

        Returns:
            是否删除成功
        """
        try:
            async with self.get_session() as session:

                stmt = select(StyleLearningReview).where(
                    StyleLearningReview.id == review_id
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                if not record:
                    return False

                await session.execute(
                    sa_delete(StyleLearningReview).where(
                        StyleLearningReview.id == review_id
                    )
                )
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 删除风格学习审核记录失败: {e}")
            return False

    # Learning Batch/Session methods

    async def get_learning_batch_history(
        self, group_id=None, limit=20
    ) -> List[Dict]:
        """获取学习批次历史

        Args:
            group_id: 可选的群组 ID 过滤
            limit: 返回数量限制

        Returns:
            学习批次记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(LearningBatch)
                    .order_by(desc(LearningBatch.start_time))
                    .limit(limit)
                )
                if group_id:
                    stmt = stmt.where(LearningBatch.group_id == group_id)

                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取学习批次历史失败: {e}")
            return []

    async def get_recent_learning_batches(self, limit=5) -> List[Dict]:
        """获取最近的学习批次

        Args:
            limit: 返回数量限制

        Returns:
            学习批次记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(LearningBatch)
                    .order_by(desc(LearningBatch.start_time))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取最近学习批次失败: {e}")
            return []

    async def get_learning_sessions(self, group_id, limit=5) -> List[Dict]:
        """获取指定群组的学习会话

        Args:
            group_id: 群组 ID
            limit: 返回数量限制

        Returns:
            学习会话记录列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(LearningSession)
                    .where(LearningSession.group_id == group_id)
                    .order_by(desc(LearningSession.start_time))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取学习会话失败: {e}")
            return []

    async def get_recent_learning_sessions(self, days=7) -> List[Dict]:
        """获取最近 N 天的学习会话

        Args:
            days: 天数

        Returns:
            学习会话记录列表
        """
        try:
            async with self.get_session() as session:

                cutoff = time.time() - (days * 24 * 3600)
                stmt = (
                    select(LearningSession)
                    .where(LearningSession.start_time > cutoff)
                    .order_by(desc(LearningSession.start_time))
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取最近学习会话失败: {e}")
            return []

    async def save_learning_session_record(
        self, group_id, session_data
    ) -> bool:
        """保存学习会话记录

        Args:
            group_id: 群组 ID
            session_data: 会话数据字典

        Returns:
            是否保存成功
        """
        try:
            async with self.get_session() as session:

                sid = session_data.get('session_id', '')

                # 检查是否已存在，存在则更新
                existing = (await session.execute(
                    select(LearningSession).where(LearningSession.session_id == sid)
                )).scalar_one_or_none()

                if existing:
                    existing.message_count = session_data.get('message_count', existing.message_count)
                    existing.end_time = self._to_float_ts(session_data.get('end_time')) or existing.end_time
                    existing.learning_quality = session_data.get('learning_quality') or existing.learning_quality
                    existing.status = session_data.get('status', existing.status)
                else:
                    existing = LearningSession(
                        session_id=sid,
                        group_id=group_id,
                        batch_id=session_data.get('batch_id'),
                        start_time=self._to_float_ts(
                            session_data.get('start_time'), default=time.time()
                        ),
                        end_time=self._to_float_ts(session_data.get('end_time')),
                        message_count=session_data.get('message_count', 0),
                        learning_quality=session_data.get('learning_quality'),
                        status=session_data.get('status', 'active'),
                    )
                    session.add(existing)

                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 保存学习会话记录失败: {e}")
            return False

    async def save_learning_performance_record(
        self, group_id, performance_data
    ) -> bool:
        """保存学习性能记录

        Args:
            group_id: 群组 ID
            performance_data: 性能数据字典

        Returns:
            是否保存成功
        """
        try:
            async with self.get_session() as session:

                metadata = performance_data.get('metadata', {})
                record = LearningPerformanceHistory(
                    group_id=group_id,
                    session_id=performance_data.get('session_id', ''),
                    timestamp=int(
                        self._to_float_ts(
                            performance_data.get('timestamp'),
                            default=time.time(),
                        )
                    ),
                    quality_score=performance_data.get('quality_score'),
                    learning_time=performance_data.get('learning_time'),
                    success=performance_data.get('success', True),
                    successful_pattern=json.dumps(
                        performance_data.get('successful_pattern', []),
                        ensure_ascii=False,
                    )
                    if isinstance(performance_data.get('successful_pattern'), (list, dict))
                    else performance_data.get('successful_pattern'),
                    failed_pattern=json.dumps(
                        performance_data.get('failed_pattern', []),
                        ensure_ascii=False,
                    )
                    if isinstance(performance_data.get('failed_pattern'), (list, dict))
                    else performance_data.get('failed_pattern'),
                    created_at=int(time.time()),
                )
                session.add(record)
                await session.commit()
                return True
        except Exception as e:
            self._logger.error(f"[LearningFacade] 保存学习性能记录失败: {e}")
            return False

    # Statistics methods

    async def count_pending_persona_updates(self) -> int:
        """统计待审核的人格更新记录数

        Returns:
            待审核记录数量
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(func.count())
                    .select_from(PersonaLearningReview)
                    .where(PersonaLearningReview.status == 'pending')
                )
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception as e:
            self._logger.error(f"[LearningFacade] 统计待审核人格更新数量失败: {e}")
            return 0

    async def count_style_learning_patterns(self) -> int:
        """统计风格学习模式总数

        Returns:
            风格学习模式数量
        """
        try:
            async with self.get_session() as session:

                stmt = select(func.count()).select_from(StyleLearningPattern)
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception as e:
            self._logger.error(f"[LearningFacade] 统计风格学习模式数量失败: {e}")
            return 0

    async def count_refined_messages(self) -> int:
        """统计筛选后消息总数

        Returns:
            筛选后消息数量
        """
        try:
            async with self.get_session() as session:

                stmt = select(func.count()).select_from(FilteredMessage)
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception as e:
            self._logger.error(f"[LearningFacade] 统计筛选后消息数量失败: {e}")
            return 0

    async def get_style_learning_statistics(self) -> Dict[str, Any]:
        """获取风格学习统计信息

        Returns:
            包含 total_reviews, pending_reviews, approved_reviews 的字典
        """
        try:
            async with self.get_session() as session:

                total_stmt = select(func.count()).select_from(StyleLearningReview)
                total_result = await session.execute(total_stmt)
                total = total_result.scalar() or 0

                pending_stmt = (
                    select(func.count())
                    .select_from(StyleLearningReview)
                    .where(StyleLearningReview.status == 'pending')
                )
                pending_result = await session.execute(pending_stmt)
                pending = pending_result.scalar() or 0

                approved_stmt = (
                    select(func.count())
                    .select_from(StyleLearningReview)
                    .where(StyleLearningReview.status == 'approved')
                )
                approved_result = await session.execute(approved_stmt)
                approved = approved_result.scalar() or 0

                return {
                    'total_reviews': total,
                    'pending_reviews': pending,
                    'approved_reviews': approved,
                }
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取风格学习统计失败: {e}")
            return {
                'total_reviews': 0,
                'pending_reviews': 0,
                'approved_reviews': 0,
            }

    async def get_style_learning_statistics(self) -> Dict[str, Any]:
        """鑾峰彇椋庢牸瀛︿範缁熻淇℃伅"""
        try:
            def _parse_payload(raw_value: Any) -> Any:
                if isinstance(raw_value, str):
                    text = raw_value.strip()
                    if not text:
                        return None
                    try:
                        return json.loads(text)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        return None
                return raw_value

            def _collect_confidences(payload: Any) -> List[float]:
                values: List[float] = []
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if key == "confidence" and isinstance(value, (int, float)):
                            numeric = float(value)
                            if numeric > 1.0:
                                numeric = numeric / 100.0
                            values.append(max(0.0, min(numeric, 1.0)))
                        else:
                            values.extend(_collect_confidences(value))
                elif isinstance(payload, list):
                    for item in payload:
                        values.extend(_collect_confidences(item))
                return values

            async with self.get_session() as session:
                review_stats_stmt = select(
                    func.count().label("total_reviews"),
                    func.count(func.distinct(StyleLearningReview.type)).label("unique_styles"),
                    func.count().filter(StyleLearningReview.status == 'pending').label("pending_reviews"),
                    func.count().filter(StyleLearningReview.status == 'approved').label("approved_reviews"),
                    func.count().filter(StyleLearningReview.status == 'rejected').label("rejected_reviews"),
                    func.max(StyleLearningReview.timestamp).label("latest_update"),
                )
                review_stats = (await session.execute(review_stats_stmt)).one()

                review_payload_rows = (
                    await session.execute(select(StyleLearningReview.learned_patterns))
                ).all()
                confidence_values: List[float] = []
                for row in review_payload_rows:
                    confidence_values.extend(_collect_confidences(_parse_payload(row[0])))

                batch_row = (
                    await session.execute(
                        select(
                            func.count(LearningBatch.id),
                            func.sum(LearningBatch.filtered_count),
                            func.sum(LearningBatch.message_count),
                            func.sum(LearningBatch.processed_messages),
                            func.max(
                                func.coalesce(
                                    LearningBatch.end_time, LearningBatch.start_time
                                )
                            ),
                        )
                    )
                ).one()
                batch_count = int(batch_row[0] or 0)
                raw_message_count = 0
                raw_message_count_source = "unavailable"
                if batch_count > 0:
                    for source, value in (
                        ("learning_batches.filtered_count_sum", batch_row[1]),
                        ("learning_batches.message_count_sum", batch_row[2]),
                        ("learning_batches.processed_messages_sum", batch_row[3]),
                    ):
                        if value is not None and int(value) > 0:
                            raw_message_count = int(value)
                            raw_message_count_source = source
                            break
                    if raw_message_count_source == "unavailable":
                        for source, value in (
                            ("learning_batches.filtered_count_sum", batch_row[1]),
                            ("learning_batches.message_count_sum", batch_row[2]),
                            ("learning_batches.processed_messages_sum", batch_row[3]),
                        ):
                            if value is not None:
                                raw_message_count = int(value or 0)
                                raw_message_count_source = source
                                break
                else:
                    filtered_table_count = (
                        await session.execute(select(func.count()).select_from(FilteredMessage))
                    ).scalar()
                    if filtered_table_count is not None:
                        raw_message_count = int(filtered_table_count or 0)
                        raw_message_count_source = "filtered_messages_table_count"
                    else:
                        raw_table_count = (
                            await session.execute(select(func.count()).select_from(RawMessage))
                        ).scalar()
                        if raw_table_count is not None:
                            raw_message_count = int(raw_table_count or 0)
                            raw_message_count_source = "raw_messages_table_count"

                avg_confidence = (
                    round(sum(confidence_values) / len(confidence_values), 4)
                    if confidence_values
                    else None
                )

                performance_latest_update = (
                    await session.execute(
                        select(func.max(LearningPerformanceHistory.timestamp))
                    )
                ).scalar()
                latest_update = self._pick_latest_timestamp(
                    review_stats.latest_update,
                    batch_row[4],
                    performance_latest_update,
                )

                return {
                    'unique_styles': int(review_stats.unique_styles or 0),
                    'avg_confidence': avg_confidence,
                    'total_samples': raw_message_count,
                    'latest_update': latest_update,
                    'total_reviews': int(review_stats.total_reviews or 0),
                    'pending_reviews': int(review_stats.pending_reviews or 0),
                    'approved_reviews': int(review_stats.approved_reviews or 0),
                    'rejected_reviews': int(review_stats.rejected_reviews or 0),
                    'raw_message_count_source': raw_message_count_source,
                    'confidence_value_count': len(confidence_values),
                }
        except Exception as e:
            self._logger.error(f"[LearningFacade] 鑾峰彇椋庢牸瀛︿範缁熻澶辫触: {e}")
            return {
                'unique_styles': 0,
                'avg_confidence': None,
                'total_samples': 0,
                'latest_update': None,
                'total_reviews': 0,
                'pending_reviews': 0,
                'approved_reviews': 0,
                'rejected_reviews': 0,
                'raw_message_count_source': 'unavailable',
                'confidence_value_count': 0,
            }

    def _normalize_quality_value(
        self, raw_value: Optional[float]
    ) -> tuple[Optional[float], Optional[float]]:
        if raw_value is None:
            return None, None
        numeric_value = float(raw_value)
        if numeric_value > 1.0:
            quality_percent = round(numeric_value, 1)
            quality_score = round(numeric_value / 100.0, 4)
        else:
            quality_score = round(numeric_value, 4)
            quality_percent = round(numeric_value * 100.0, 1)
        return quality_score, quality_percent

    def _pick_latest_timestamp(self, *values: Any) -> Optional[float]:
        normalized_values: List[float] = []
        for value in values:
            normalized_value = self._to_float_ts(value)
            if normalized_value is not None:
                normalized_values.append(normalized_value)
        if not normalized_values:
            return None
        return max(normalized_values)

    def _resolve_progress_quality(
        self,
        batch_row: LearningBatch,
        history_rows: List[LearningPerformanceHistory],
    ) -> tuple[Optional[float], Optional[float], str, str]:
        if batch_row.quality_score is not None:
            quality_score, quality_percent = self._normalize_quality_value(
                batch_row.quality_score
            )
            return (
                quality_score,
                quality_percent,
                'learning_batches.quality_score',
                '',
            )

        batch_timestamp = self._pick_latest_timestamp(
            batch_row.end_time, batch_row.start_time
        )
        best_history: Optional[LearningPerformanceHistory] = None
        best_delta: Optional[float] = None
        for history in history_rows:
            if history.quality_score is None:
                continue
            history_timestamp = self._to_float_ts(history.timestamp)
            if history_timestamp is None:
                continue
            if batch_timestamp is None:
                best_history = history
                best_delta = 0.0
                break
            delta = abs(history_timestamp - batch_timestamp)
            if best_delta is None or delta < best_delta:
                best_history = history
                best_delta = delta

        if best_history is not None and (best_delta is None or best_delta <= 21600):
            quality_score, quality_percent = self._normalize_quality_value(
                best_history.quality_score
            )
            return (
                quality_score,
                quality_percent,
                'learning_performance_history.quality_score_backfill',
                '',
            )

        error_message = str(batch_row.error_message or '').strip()
        if self._QUALITY_REPAIR_MARKER in error_message:
            return None, None, 'historical_missing', error_message
        return None, None, 'historical_missing', 'legacy_batch_missing_quality'

    def _resolve_progress_sample_count(
        self, batch_row: LearningBatch
    ) -> tuple[int, str]:
        if batch_row.filtered_count is not None:
            return int(batch_row.filtered_count), 'filtered_count'
        if batch_row.message_count is not None:
            return int(batch_row.message_count), 'message_count'
        if batch_row.processed_messages is not None:
            return int(batch_row.processed_messages), 'processed_messages'
        return 0, 'unavailable'

    async def get_style_progress_data(
        self, group_id=None
    ) -> List[Dict]:
        """获取风格学习进度数据（从 learning_batches 表查询）

        Args:
            group_id: 可选的群组 ID 过滤

        Returns:
            学习批次进度列表
        """
        try:
            async with self.get_session() as session:

                stmt = (
                    select(LearningBatch)
                    .where(
                        or_(
                            LearningBatch.filtered_count > 0,
                            LearningBatch.message_count > 0,
                            LearningBatch.processed_messages > 0,
                        )
                    )
                    .order_by(desc(LearningBatch.start_time))
                    .limit(30)
                )
                if group_id:
                    stmt = stmt.where(LearningBatch.group_id == group_id)

                result = await session.execute(stmt)
                rows = result.scalars().all()
                items: List[Dict[str, Any]] = []
                quality_sources = set()
                sample_sources = set()
                quality_missing_count = 0
                quality_backfilled_count = 0

                history_rows: List[LearningPerformanceHistory] = []
                if rows:
                    history_stmt = (
                        select(LearningPerformanceHistory)
                        .where(LearningPerformanceHistory.quality_score.is_not(None))
                        .order_by(desc(LearningPerformanceHistory.timestamp))
                        .limit(max(120, len(rows) * 8))
                    )
                    if group_id:
                        history_stmt = history_stmt.where(
                            LearningPerformanceHistory.group_id == group_id
                        )
                    history_result = await session.execute(history_stmt)
                    history_rows = list(history_result.scalars().all())

                history_by_group: Dict[str, List[LearningPerformanceHistory]] = {}
                for history in history_rows:
                    history_by_group.setdefault(history.group_id or '', []).append(history)

                for row in rows:
                    quality_score, quality_percent, quality_value_source, missing_reason = (
                        self._resolve_progress_quality(
                            row, history_by_group.get(row.group_id or '', [])
                        )
                    )
                    sample_count, sample_count_source = self._resolve_progress_sample_count(
                        row
                    )
                    item = {
                        'group_id': row.group_id,
                        'label': (
                            row.batch_name
                            or (f"group_{row.group_id}" if row.group_id else '')
                        ),
                        'timestamp': row.end_time or row.start_time or 0,
                        'quality_score': quality_score,
                        'quality_percent': quality_percent,
                        'score': quality_score,
                        'quality_value_source': quality_value_source,
                        'quality_missing_reason': missing_reason,
                        'success': bool(row.success),
                        'processed_messages': row.processed_messages or 0,
                        'filtered_count': row.filtered_count or 0,
                        'message_count': row.message_count or 0,
                        'sample_count': sample_count,
                        'sample_count_source': sample_count_source,
                        'batch_name': row.batch_name or '',
                    }
                    items.append(item)
                    quality_sources.add(quality_value_source)
                    sample_sources.add(sample_count_source)
                    if quality_score is None:
                        quality_missing_count += 1
                    elif quality_value_source == 'learning_performance_history.quality_score_backfill':
                        quality_backfilled_count += 1

                payload_summary = [
                    {
                        'label': item.get('label') or item.get('group_id') or 'unknown',
                        'quality_percent': item.get('quality_percent'),
                        'sample_count': item.get('sample_count'),
                        'quality_value_source': item.get('quality_value_source'),
                    }
                    for item in items[:5]
                ]
                self._logger.info(
                    "[LearningFacade] style progress summary "
                    f"progress_item_count={len(items)} "
                    f"progress_item_keys={sorted(items[0].keys()) if items else []} "
                    f"progress_quality_value_source={sorted(quality_sources)} "
                    f"progress_quality_missing_count={quality_missing_count} "
                    f"progress_quality_backfilled_count={quality_backfilled_count} "
                    f"progress_sample_count_source={sorted(sample_sources)} "
                    f"progress_payload_summary={payload_summary}"
                )
                return items
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取风格学习进度数据失败: {e}")
            return []

    async def get_learning_patterns_data(
        self, group_id=None
    ) -> Dict[str, Any]:
        """获取学习模式分布数据

        按 pattern_type 分组统计 StyleLearningPattern 记录。

        Args:
            group_id: 可选的群组 ID 过滤

        Returns:
            按模式类型分组的计数字典
        """
        try:
            async with self.get_session() as session:

                stmt = select(
                    StyleLearningPattern.pattern_type,
                    func.count().label('count'),
                ).group_by(StyleLearningPattern.pattern_type)
                if group_id:
                    stmt = stmt.where(StyleLearningPattern.group_id == group_id)

                result = await session.execute(stmt)
                rows = result.all()
                pattern_counts = {row[0]: row[1] for row in rows}
                return pattern_counts
        except Exception as e:
            self._logger.error(f"[LearningFacade] 获取学习模式分布数据失败: {e}")
            return {}
