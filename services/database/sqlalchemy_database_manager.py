"""
DomainRouter — 薄路由层，将所有数据库方法委托给领域 Facade

前身为 4308 行的单体 SQLAlchemyDatabaseManager，现已拆分为
11 个领域 Facade，本文件仅保留生命周期管理、会话/连接基础设施
以及方法路由。
"""
import os
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from astrbot.api import logger

from ...config import PluginConfig
from ...core.database.engine import DatabaseEngine


class SQLAlchemyDatabaseManager:
    """DomainRouter — 薄路由层，委托给 11 个领域 Facade。

    对外接口（方法签名、返回类型）与旧版完全一致，消费者无需任何改动。
    """

    # Lifecycle

    def __init__(self, config: PluginConfig, context=None):
        self.config = config
        self.context = context
        self.engine: Optional[DatabaseEngine] = None
        self._started = False
        self._starting = False
        self._start_lock = asyncio.Lock()

        # Facades（在 start() 中初始化）
        self._affection = None
        self._message = None
        self._learning = None
        self._jargon = None
        self._persona = None
        self._social = None
        self._expression = None
        self._psychological = None
        self._reinforcement = None
        self._metrics = None
        self._admin = None

    async def start(self) -> bool:
        """启动数据库管理器（带并发保护）"""
        async with self._start_lock:
            if self._started:
                logger.debug("[DomainRouter] 已启动，跳过")
                return True

            if self._starting:
                logger.warning("[DomainRouter] 正在启动中，等待…")
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    if self._started:
                        return True
                logger.error("[DomainRouter] 启动超时")
                return False

            try:
                self._starting = True
                logger.info("[DomainRouter] 开始启动…")

                db_url = self._get_database_url()

                if hasattr(self.config, 'db_type') and self.config.db_type.lower() == 'mysql':
                    await self._ensure_mysql_database_exists()

                self.engine = DatabaseEngine(db_url, echo=False)
                logger.info("[DomainRouter] 数据库引擎已创建")

                await self.engine.create_tables(enable_auto_migration=True)

                if await self.engine.health_check():
                    self._init_facades()
                    self._started = True
                    self._starting = False
                    logger.info("[DomainRouter] 数据库启动成功")
                    return True

                self._started = False
                self._starting = False
                logger.error("[DomainRouter] 数据库健康检查失败")
                return False

            except Exception as e:
                self._started = False
                self._starting = False
                logger.error(f"[DomainRouter] 启动失败: {e}", exc_info=True)
                return False

    async def stop(self) -> bool:
        """停止数据库管理器"""
        if not self._started:
            return True
        try:
            if self.engine:
                await self.engine.close()
            self._started = False
            logger.info("[DomainRouter] 数据库已停止")
            return True
        except Exception as e:
            logger.error(f"[DomainRouter] 停止失败: {e}")
            return False

    # Facade initialization

    def _init_facades(self):
        """初始化所有领域 Facade"""
        from .facades import (
            AffectionFacade, MessageFacade, LearningFacade,
            JargonFacade, PersonaFacade, SocialFacade,
            ExpressionFacade, PsychologicalFacade, ReinforcementFacade,
            MetricsFacade, AdminFacade,
        )
        self._affection = AffectionFacade(self.engine, self.config)
        self._message = MessageFacade(self.engine, self.config)
        self._learning = LearningFacade(self.engine, self.config)
        self._jargon = JargonFacade(self.engine, self.config)
        self._persona = PersonaFacade(self.engine, self.config)
        self._social = SocialFacade(self.engine, self.config)
        self._expression = ExpressionFacade(self.engine, self.config)
        self._psychological = PsychologicalFacade(self.engine, self.config)
        self._reinforcement = ReinforcementFacade(self.engine, self.config)
        self._metrics = MetricsFacade(self.engine, self.config)
        self._admin = AdminFacade(self.engine, self.config)
        logger.info("[DomainRouter] 11 个领域 Facade 已初始化")

    # Infrastructure: database URL

    def _get_database_url(self) -> str:
        """获取数据库连接 URL"""
        if hasattr(self.config, 'db_type') and self.config.db_type.lower() == 'mysql':
            host = getattr(self.config, 'mysql_host', 'localhost')
            port = getattr(self.config, 'mysql_port', 3306)
            user = getattr(self.config, 'mysql_user', 'root')
            password = getattr(self.config, 'mysql_password', '')
            database = getattr(self.config, 'mysql_database', 'astrbot_self_learning')
            return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"

        db_path = getattr(self.config, 'messages_db_path', None)
        if not db_path:
            db_path = os.path.join(self.config.data_dir, 'messages.db')
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        return f"sqlite:///{db_path}"

    async def _ensure_mysql_database_exists(self):
        """确保 MySQL 数据库存在

        使用 aiomysql 直连 MySQL 服务器以检查/创建目标数据库。
        显式禁用 SSL 以避免 MySQL 8 默认 TLS 握手导致的
        struct.unpack 解包异常。
        """
        import aiomysql
        host = getattr(self.config, 'mysql_host', 'localhost')
        port = getattr(self.config, 'mysql_port', 3306)
        user = getattr(self.config, 'mysql_user', 'root')
        password = getattr(self.config, 'mysql_password', '')
        database = getattr(self.config, 'mysql_database', 'astrbot_self_learning')

        try:
            conn = await asyncio.wait_for(
                aiomysql.connect(
                    host=host, port=port, user=user,
                    password=password, charset='utf8mb4',
                    ssl=False, connect_timeout=10,
                ),
                timeout=15,
            )
        except asyncio.TimeoutError:
            logger.error("[DomainRouter] 连接 MySQL 超时 (15s)")
            raise
        except Exception as e:
            logger.error(f"[DomainRouter] 连接 MySQL 失败: {e}")
            raise

        try:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s",
                    (database,),
                )
                if not await cursor.fetchone():
                    logger.info(f"[DomainRouter] 数据库 {database} 不存在，正在创建...")
                    await cursor.execute(
                        f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                    await conn.commit()
                    logger.info(f"[DomainRouter] 数据库 {database} 创建成功")
        except Exception as e:
            logger.error(f"[DomainRouter] 确保 MySQL 数据库存在失败: {e}")
            raise
        finally:
            conn.close()

    # Infrastructure: session

    @asynccontextmanager
    async def get_session(self):
        """获取 ORM 会话（async context manager）"""
        if not self.engine:
            if self._starting:
                logger.debug("[DomainRouter] 等待 engine 创建…")
                for _ in range(30):
                    await asyncio.sleep(0.1)
                    if self.engine:
                        break
                if not self.engine:
                    raise RuntimeError("数据库管理器启动超时，engine未创建")
            else:
                raise RuntimeError("数据库管理器未启动，engine不存在")

        if not self._started:
            logger.debug("[DomainRouter] get_session: _started=False 但 engine 存在，继续执行")

        session = self.engine.get_session()
        try:
            async with session:
                yield session
        except Exception:
            raise

    # Domain delegates: AffectionFacade

    async def get_user_affection(self, group_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._affection.get_user_affection(group_id, user_id)

    async def update_user_affection(
        self, group_id: str, user_id: str, new_level: int,
        change_reason: str = "", bot_mood: str = "",
    ) -> bool:
        return await self._affection.update_user_affection(
            group_id, user_id, new_level, change_reason, bot_mood,
        )

    async def get_all_user_affections(self, group_id: str) -> List[Dict[str, Any]]:
        return await self._affection.get_all_user_affections(group_id)

    async def get_total_affection(self, group_id: str) -> int:
        return await self._affection.get_total_affection(group_id)

    async def save_bot_mood(
        self, group_id: str, mood_type: str, mood_intensity: float,
        mood_description: str, duration_hours: int = 24,
    ) -> bool:
        return await self._affection.save_bot_mood(
            group_id, mood_type, mood_intensity, mood_description, duration_hours,
        )

    async def get_current_bot_mood(self, group_id: str) -> Optional[Dict[str, Any]]:
        return await self._affection.get_current_bot_mood(group_id)

    # Domain delegates: MessageFacade

    async def save_raw_message(self, message_data) -> int:
        return await self._message.save_raw_message(message_data)

    async def get_recent_raw_messages(
        self, group_id: str, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_recent_raw_messages(group_id, limit)

    async def get_unprocessed_messages(
        self, limit: Optional[int] = None, group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_unprocessed_messages(limit, group_id=group_id)

    async def mark_messages_processed(self, message_ids: List[int]) -> bool:
        return await self._message.mark_messages_processed(message_ids)

    async def get_messages_by_timerange(
        self, group_id: str, start_time: int, end_time: int, limit: int = 500,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_messages_by_timerange(
            group_id, start_time, end_time, limit,
        )

    async def get_messages_by_group_and_timerange(
        self, group_id: str, start_time: int, end_time: int, limit: int = 500,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_messages_by_group_and_timerange(
            group_id, start_time, end_time, limit,
        )

    async def get_messages_for_replay(
        self, group_id: str, days: int = 30, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_messages_for_replay(group_id, days, limit)

    async def get_recent_filtered_messages(
        self, group_id: str, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_recent_filtered_messages(group_id, limit)

    async def get_filtered_messages_for_learning(
        self, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return await self._message.get_filtered_messages_for_learning(limit)

    async def add_filtered_message(self, filtered_data: Dict[str, Any]) -> int:
        return await self._message.add_filtered_message(filtered_data)

    async def save_bot_message(
        self, group_id: str, message: str, timestamp: int = None,
    ) -> bool:
        return await self._message.save_bot_message(group_id, message, timestamp)

    async def get_recent_bot_responses(
        self, group_id: str, limit: int = 10,
    ) -> List[str]:
        return await self._message.get_recent_bot_responses(group_id, limit)

    async def get_message_statistics(
        self, group_id: str = None,
    ) -> Dict[str, Any]:
        return await self._message.get_message_statistics(group_id)

    async def get_messages_statistics(self) -> Dict[str, Any]:
        return await self._message.get_messages_statistics()

    async def get_group_messages_statistics(self, group_id: str) -> Dict[str, Any]:
        return await self._message.get_group_messages_statistics(group_id)

    async def get_group_user_statistics(
        self, group_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        return await self._message.get_group_user_statistics(group_id)

    async def get_groups_for_social_analysis(self) -> List[Dict[str, Any]]:
        return await self._message.get_groups_for_social_analysis()

    # Legacy compatibility: WebUI chat/statistics routes

    async def get_chat_history(
        self,
        group_id: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """兼容旧版 WebUI 聊天历史接口。"""
        from sqlalchemy import desc, select
        from ...models.orm.message import RawMessage

        try:
            if group_id and start_time is not None and end_time is not None:
                return await self.get_messages_by_group_and_timerange(
                    group_id, start_time, end_time, limit
                )
            if group_id:
                return await self.get_recent_raw_messages(group_id, limit=limit)

            async with self.get_session() as session:
                stmt = select(RawMessage).order_by(desc(RawMessage.timestamp)).limit(limit)
                result = await session.execute(stmt)
                return [self._message._row_to_dict(row) for row in result.scalars().all()]
        except Exception as e:
            logger.error(f"[DomainRouter] 获取聊天历史失败: {e}")
            return []

    async def get_message_by_id(self, message_id: int) -> Optional[Dict[str, Any]]:
        """兼容旧版 WebUI 单条消息查询接口。"""
        from ...models.orm.message import BotMessage, RawMessage

        try:
            async with self.get_session() as session:
                message = await session.get(RawMessage, message_id)
                if message:
                    data = self._message._row_to_dict(message)
                    data["message_type"] = "raw"
                    return data

                bot_message = await session.get(BotMessage, message_id)
                if bot_message:
                    data = self._message._row_to_dict(bot_message)
                    data["message_type"] = "bot"
                    return data
        except Exception as e:
            logger.error(f"[DomainRouter] 获取消息详情失败: {e}")
        return None

    async def delete_message(self, message_id: int) -> bool:
        """兼容旧版 WebUI 删除消息接口。"""
        from sqlalchemy import delete
        from ...models.orm.message import BotMessage, FilteredMessage, RawMessage

        try:
            async with self.get_session() as session:
                raw_message = await session.get(RawMessage, message_id)
                if raw_message:
                    await session.execute(
                        delete(FilteredMessage).where(
                            FilteredMessage.raw_message_id == message_id
                        )
                    )
                    await session.delete(raw_message)
                    await session.commit()
                    return True

                bot_message = await session.get(BotMessage, message_id)
                if bot_message:
                    await session.delete(bot_message)
                    await session.commit()
                    return True
        except Exception as e:
            logger.error(f"[DomainRouter] 删除消息失败: {e}")
        return False

    async def get_chat_statistics(self, group_id: str = None) -> Dict[str, Any]:
        """兼容旧版 WebUI 聊天统计接口。"""
        from sqlalchemy import and_, distinct, func, select
        from ...models.orm.message import RawMessage

        try:
            async with self.get_session() as session:
                total_stmt = select(func.count()).select_from(RawMessage)
                users_stmt = select(func.count(distinct(RawMessage.sender_id))).select_from(
                    RawMessage
                )
                range_stmt = select(
                    func.min(RawMessage.timestamp),
                    func.max(RawMessage.timestamp),
                ).select_from(RawMessage)

                if group_id:
                    condition = and_(RawMessage.group_id == group_id)
                    total_stmt = total_stmt.where(condition)
                    users_stmt = users_stmt.where(condition)
                    range_stmt = range_stmt.where(condition)

                total_messages = (await session.execute(total_stmt)).scalar() or 0
                total_users = (await session.execute(users_stmt)).scalar() or 0
                date_range = (await session.execute(range_stmt)).one()

                return {
                    "total_messages": int(total_messages),
                    "total_users": int(total_users),
                    "date_range": {
                        "start_time": date_range[0],
                        "end_time": date_range[1],
                    }
                    if date_range and date_range[0] is not None
                    else None,
                    "group_id": group_id,
                }
        except Exception as e:
            logger.error(f"[DomainRouter] 获取聊天统计失败: {e}")
            return {
                "total_messages": 0,
                "total_users": 0,
                "date_range": None,
                "group_id": group_id,
            }

    # Domain delegates: LearningFacade

    async def add_persona_learning_review(
        self,
        review_data: Dict[str, Any] = None,
        *,
        group_id: str = None,
        proposed_content: str = None,
        learning_source: str = '',
        confidence_score: float = 0.5,
        raw_analysis: str = '',
        metadata: Dict[str, Any] = None,
        original_content: str = '',
        new_content: str = '',
    ) -> int:
        """兼容新旧两种调用方式：单 dict 或关键字参数。"""
        if review_data is None:
            review_data = {
                'group_id': group_id or '',
                'proposed_content': proposed_content or '',
                'update_type': learning_source,
                'confidence_score': confidence_score,
                'reason': raw_analysis,
                'metadata': metadata or {},
                'original_content': original_content,
                'new_content': new_content,
            }
        return await self._learning.add_persona_learning_review(review_data)

    async def get_pending_persona_update_records(self) -> List[Dict[str, Any]]:
        return await self._learning.get_pending_persona_update_records()

    async def save_persona_update_record(self, record_data: Dict[str, Any]) -> int:
        return await self._learning.save_persona_update_record(record_data)

    async def delete_persona_update_record(self, record_id: int) -> bool:
        return await self._learning.delete_persona_update_record(record_id)

    async def get_persona_update_record_by_id(
        self, record_id: int,
    ) -> Optional[Dict[str, Any]]:
        return await self._learning.get_persona_update_record_by_id(record_id)

    async def get_reviewed_persona_update_records(
        self, limit: int = 50, offset: int = 0, status_filter: str = None,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_reviewed_persona_update_records(
            limit=limit, offset=offset, status_filter=status_filter,
        )

    async def update_persona_update_record_status(
        self, record_id: int, status: str, comment: str = None,
    ) -> bool:
        return await self._learning.update_persona_update_record_status(
            record_id, status, comment,
        )

    async def create_style_learning_review(
        self, review_data: Dict[str, Any],
    ) -> int:
        return await self._learning.create_style_learning_review(review_data)

    async def get_pending_style_reviews(
        self, limit: int = 50, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_pending_style_reviews(limit, offset)

    async def get_reviewed_style_learning_updates(
        self, limit: int = 50, offset: int = 0, status_filter: str = None,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_reviewed_style_learning_updates(
            limit=limit, offset=offset, status_filter=status_filter,
        )

    async def update_style_review_status(
        self, review_id: int, status: str, reviewer_comment: str = '',
    ) -> bool:
        return await self._learning.update_style_review_status(
            review_id, status, reviewer_comment,
        )

    async def delete_style_review_by_id(self, review_id: int) -> bool:
        return await self._learning.delete_style_review_by_id(review_id)

    async def get_approved_few_shots(
        self, group_id: str, limit: int = 3,
    ) -> List[str]:
        return await self._learning.get_approved_few_shots(group_id, limit)

    async def get_pending_persona_learning_reviews(
        self, limit: int = 50, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_pending_persona_learning_reviews(limit, offset)

    async def get_reviewed_persona_learning_updates(
        self, limit: int = 50, offset: int = 0, status_filter: str = None,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_reviewed_persona_learning_updates(
            limit=limit, offset=offset, status_filter=status_filter,
        )

    async def delete_persona_learning_review_by_id(self, review_id: int) -> bool:
        return await self._learning.delete_persona_learning_review_by_id(review_id)

    async def get_persona_learning_review_by_id(
        self, review_id: int,
    ) -> Optional[Dict[str, Any]]:
        return await self._learning.get_persona_learning_review_by_id(review_id)

    async def update_persona_learning_review_status(
        self, review_id: int, status: str, comment: str = None,
        modified_content: str = None,
    ) -> bool:
        return await self._learning.update_persona_learning_review_status(
            review_id, status, comment, modified_content,
        )

    async def get_learning_batch_history(
        self, group_id: str = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_learning_batch_history(group_id, limit)

    async def get_recent_learning_batches(
        self, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_recent_learning_batches(limit)

    async def get_learning_sessions(self, group_id: str) -> List[Dict[str, Any]]:
        return await self._learning.get_learning_sessions(group_id)

    async def get_recent_learning_sessions(
        self, days: int = 7,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_recent_learning_sessions(days)

    async def save_learning_session_record(
        self, group_id: str, session_data: Dict[str, Any],
    ) -> bool:
        return await self._learning.save_learning_session_record(group_id, session_data)

    async def save_learning_performance_record(
        self, group_id: str, performance_data: Dict[str, Any],
    ) -> bool:
        return await self._learning.save_learning_performance_record(
            group_id, performance_data,
        )

    async def count_pending_persona_updates(self) -> int:
        return await self._learning.count_pending_persona_updates()

    async def count_style_learning_patterns(self) -> int:
        return await self._learning.count_style_learning_patterns()

    async def count_refined_messages(self) -> int:
        return await self._learning.count_refined_messages()

    async def get_style_learning_statistics(self) -> Dict[str, Any]:
        return await self._learning.get_style_learning_statistics()

    async def get_style_progress_data(
        self, group_id: str = None,
    ) -> List[Dict[str, Any]]:
        return await self._learning.get_style_progress_data(group_id)

    async def get_learning_patterns_data(
        self, group_id: str = None,
    ) -> Dict[str, Any]:
        return await self._learning.get_learning_patterns_data(group_id)

    # Domain delegates: JargonFacade

    async def get_jargon(self, chat_id: str, content: str) -> Optional[Dict[str, Any]]:
        return await self._jargon.get_jargon(chat_id, content)

    async def insert_jargon(self, jargon_data: Dict[str, Any]) -> Optional[int]:
        return await self._jargon.insert_jargon(jargon_data)

    async def update_jargon(self, jargon_data: Dict[str, Any]) -> bool:
        return await self._jargon.update_jargon(jargon_data)

    async def get_jargon_statistics(self, group_id: str = None) -> Dict[str, Any]:
        return await self._jargon.get_jargon_statistics(group_id)

    async def get_recent_jargon_list(
        self, group_id: str = None, chat_id: str = None,
        limit: int = 50, offset: int = 0, only_confirmed: bool = False,
    ) -> List[Dict[str, Any]]:
        return await self._jargon.get_recent_jargon_list(
            group_id, chat_id, limit, offset, only_confirmed,
        )

    async def get_jargon_count(
        self, chat_id: str = None, only_confirmed: bool = False,
    ) -> int:
        return await self._jargon.get_jargon_count(chat_id, only_confirmed)

    async def search_jargon(
        self, keyword: str, chat_id: str = None,
        confirmed_only: bool = False, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return await self._jargon.search_jargon(
            keyword=keyword, chat_id=chat_id,
            confirmed_only=confirmed_only, limit=limit,
        )

    async def get_jargon_by_id(self, jargon_id: int) -> Optional[Dict[str, Any]]:
        return await self._jargon.get_jargon_by_id(jargon_id)

    async def delete_jargon_by_id(self, jargon_id: int) -> bool:
        return await self._jargon.delete_jargon_by_id(jargon_id)

    async def set_jargon_global(self, jargon_id: int, is_global: bool) -> bool:
        return await self._jargon.set_jargon_global(jargon_id, is_global)

    async def sync_global_jargon_to_group(self, target_chat_id: str) -> int:
        return await self._jargon.sync_global_jargon_to_group(target_chat_id)

    async def save_or_update_jargon(
        self, chat_id: str, content: str, jargon_data: Dict[str, Any],
    ) -> Optional[int]:
        return await self._jargon.save_or_update_jargon(
            chat_id, content, jargon_data,
        )

    async def get_global_jargon_list(
        self, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return await self._jargon.get_global_jargon_list(limit)

    async def get_jargon_groups(self) -> List[Dict[str, Any]]:
        return await self._jargon.get_jargon_groups()

    # Domain delegates: PersonaFacade

    async def backup_persona(self, group_id: str, backup_data: Dict[str, Any]) -> bool:
        backup_data.setdefault('group_id', group_id)
        return await self._persona.backup_persona(backup_data)

    async def get_persona_backups(self, limit: int = 10) -> List[Dict[str, Any]]:
        return await self._persona.get_persona_backups(limit)

    async def restore_persona_backup(
        self, backup_id: int,
    ) -> Optional[Dict[str, Any]]:
        return await self._persona.restore_persona_backup(backup_id)

    async def get_persona_update_history(
        self, group_id: str = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return await self._persona.get_persona_update_history(group_id, limit)

    # Domain delegates: SocialFacade

    async def load_user_profile(self, qq_id: str) -> Optional[Dict[str, Any]]:
        return await self._social.load_user_profile(qq_id)

    async def save_user_profile(
        self, qq_id: str, profile_data: Dict[str, Any],
    ) -> bool:
        return await self._social.save_user_profile(qq_id, profile_data)

    async def load_user_preferences(
        self, user_id: str, group_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._social.load_user_preferences(user_id, group_id)

    async def save_user_preferences(
        self, user_id: str, group_id: str, prefs: Dict[str, Any],
    ) -> bool:
        return await self._social.save_user_preferences(user_id, group_id, prefs)

    async def get_social_relations_by_group(
        self, group_id: str,
    ) -> List[Dict[str, Any]]:
        return await self._social.get_social_relations_by_group(group_id)

    async def get_social_relationships(
        self, group_id: str,
    ) -> List[Dict[str, Any]]:
        return await self._social.get_social_relationships(group_id)

    async def load_social_graph(self, group_id: str) -> List[Dict[str, Any]]:
        return await self._social.load_social_graph(group_id)

    async def save_social_relation(
        self, group_id: str, relation_data: Dict[str, Any],
    ) -> bool:
        return await self._social.save_social_relation(group_id, relation_data)

    async def get_user_social_relations(
        self, group_id: str, user_id: str,
    ) -> Dict[str, Any]:
        return await self._social.get_user_social_relations(group_id, user_id)

    async def clear_social_relations(self, group_id: str) -> bool:
        """兼容旧版 WebUI 清空社交关系接口。"""
        from sqlalchemy import delete
        from ...models.orm.social_relation import (
            SocialRelation,
            UserSocialProfile,
            UserSocialRelationComponent,
        )

        try:
            async with self.get_session() as session:
                await session.execute(
                    delete(UserSocialRelationComponent).where(
                        UserSocialRelationComponent.group_id == group_id
                    )
                )
                await session.execute(
                    delete(UserSocialProfile).where(
                        UserSocialProfile.group_id == group_id
                    )
                )
                await session.execute(
                    delete(SocialRelation).where(SocialRelation.group_id == group_id)
                )
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"[DomainRouter] 清空社交关系失败: {e}")
            return False

    # Domain delegates: ExpressionFacade

    async def get_all_expression_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        return await self._expression.get_all_expression_patterns()

    async def get_expression_patterns_statistics(self) -> Dict[str, Any]:
        return await self._expression.get_expression_patterns_statistics()

    async def get_group_expression_patterns(
        self, group_id: str, limit: int = None,
    ) -> List[Dict[str, Any]]:
        return await self._expression.get_group_expression_patterns(group_id, limit)

    async def get_recent_week_expression_patterns(
        self, group_id: str = None, limit: int = 50, hours: int = 168,
    ) -> List[Dict[str, Any]]:
        return await self._expression.get_recent_week_expression_patterns(
            group_id, limit, hours,
        )

    async def load_style_profile(
        self, profile_name: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._expression.load_style_profile(profile_name)

    async def save_style_profile(
        self, profile_name: str, profile_data: Dict[str, Any],
    ) -> bool:
        return await self._expression.save_style_profile(profile_name, profile_data)

    async def save_style_learning_record(
        self, record_data: Dict[str, Any],
    ) -> bool:
        return await self._expression.save_style_learning_record(record_data)

    async def save_language_style_pattern(
        self, language_style: str, pattern_data: Dict[str, Any],
    ) -> bool:
        return await self._expression.save_language_style_pattern(
            language_style, pattern_data,
        )

    async def get_diversity_metrics(self, group_id: str = None) -> Dict[str, Any]:
        """兼容旧版 WebUI 多样性指标接口。"""
        try:
            recent_messages = (
                await self.get_recent_raw_messages(group_id, limit=200) if group_id else []
            )
            texts = [msg.get("message", "") for msg in recent_messages if msg.get("message")]
            total_chars = sum(len(text) for text in texts)
            unique_chars = len(set("".join(texts))) if texts else 0
            unique_messages = len(set(texts)) if texts else 0

            if group_id:
                patterns = await self.get_group_expression_patterns(group_id, limit=50)
            else:
                grouped_patterns = await self.get_all_expression_patterns()
                patterns = [
                    pattern
                    for group_patterns in grouped_patterns.values()
                    for pattern in group_patterns
                ]

            vocabulary_diversity = (
                round(min((unique_chars / max(total_chars, 1)) * 1000, 100), 1)
                if texts
                else 0
            )
            topic_diversity = (
                round(min((unique_messages / max(len(texts), 1)) * 100, 100), 1)
                if texts
                else 0
            )
            style_diversity = round(min(len(patterns) * 5, 100), 1)
            total_score = round(
                (vocabulary_diversity + topic_diversity + style_diversity) / 3, 1
            )

            return {
                "vocabulary_diversity": vocabulary_diversity,
                "topic_diversity": topic_diversity,
                "style_diversity": style_diversity,
                "total_score": total_score,
                "sample_size": len(texts),
            }
        except Exception as e:
            logger.error(f"[DomainRouter] 获取多样性指标失败: {e}")
            return {
                "vocabulary_diversity": 0,
                "topic_diversity": 0,
                "style_diversity": 0,
                "total_score": 0,
                "sample_size": 0,
            }

    # Domain delegates: PsychologicalFacade

    async def load_emotion_profile(
        self, user_id: str, group_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._psychological.load_emotion_profile(user_id, group_id)

    async def save_emotion_profile(
        self, user_id: str, group_id: str, profile: Dict[str, Any],
    ) -> bool:
        return await self._psychological.save_emotion_profile(
            user_id, group_id, profile,
        )

    async def get_affection_metrics(self, group_id: str) -> Dict[str, Any]:
        """兼容旧版 WebUI 好感度指标接口。"""
        try:
            affections = await self.get_all_user_affections(group_id)
            levels = [int(item.get("affection_level", 0) or 0) for item in affections]
            total_users = len(levels)
            average_affection = round(sum(levels) / total_users, 2) if total_users else 0

            distribution = [
                {"range": "0-20", "count": sum(1 for level in levels if 0 <= level <= 20)},
                {"range": "21-40", "count": sum(1 for level in levels if 21 <= level <= 40)},
                {"range": "41-60", "count": sum(1 for level in levels if 41 <= level <= 60)},
                {"range": "61-80", "count": sum(1 for level in levels if 61 <= level <= 80)},
                {"range": "81-100", "count": sum(1 for level in levels if 81 <= level <= 100)},
            ]

            return {
                "average_affection": average_affection,
                "total_users": total_users,
                "high_affection_count": sum(1 for level in levels if level >= 70),
                "low_affection_count": sum(1 for level in levels if level <= 30),
                "distribution": distribution,
            }
        except Exception as e:
            logger.error(f"[DomainRouter] 获取好感度指标失败: {e}")
            return {
                "average_affection": 0,
                "total_users": 0,
                "high_affection_count": 0,
                "low_affection_count": 0,
                "distribution": [],
            }

    # Domain delegates: ReinforcementFacade

    async def get_learning_history_for_reinforcement(
        self, group_id: str, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return await self._reinforcement.get_learning_history_for_reinforcement(
            group_id, limit,
        )

    async def save_reinforcement_learning_result(
        self, group_id: str, result_data: Dict[str, Any],
    ) -> bool:
        return await self._reinforcement.save_reinforcement_learning_result(
            group_id, result_data,
        )

    async def get_persona_fusion_history(
        self, group_id: str, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._reinforcement.get_persona_fusion_history(group_id, limit)

    async def save_persona_fusion_result(
        self, group_id: str, fusion_data: Dict[str, Any],
    ) -> bool:
        return await self._reinforcement.save_persona_fusion_result(
            group_id, fusion_data,
        )

    async def get_learning_performance_history(
        self, group_id: str, limit: int = 30,
    ) -> List[Dict[str, Any]]:
        return await self._reinforcement.get_learning_performance_history(
            group_id, limit,
        )

    async def save_strategy_optimization_result(
        self, group_id: str, optimization_data: Dict[str, Any],
    ) -> bool:
        return await self._reinforcement.save_strategy_optimization_result(
            group_id, optimization_data,
        )

    # Domain delegates: MetricsFacade

    async def get_group_statistics(
        self, group_id: str = None,
    ) -> Dict[str, Any]:
        return await self._metrics.get_group_statistics(group_id)

    async def get_detailed_metrics(
        self, group_id: str = None,
    ) -> Dict[str, Any]:
        return await self._metrics.get_detailed_metrics(group_id)

    async def get_trends_data(self) -> Dict[str, Any]:
        return await self._metrics.get_trends_data()

    # Domain delegates: AdminFacade

    async def clear_all_messages_data(self) -> bool:
        return await self._admin.clear_all_messages_data()

    async def export_messages_learning_data(
        self, group_id: str = None,
    ) -> Dict[str, Any]:
        return await self._admin.export_messages_learning_data(group_id)

    async def clear_messages_data(self) -> Dict[str, Any]:
        return await self._admin.clear_messages_data()

    async def clear_persona_reviews_data(self) -> Dict[str, Any]:
        return await self._admin.clear_persona_reviews_data()

    async def clear_style_learning_data(self) -> Dict[str, Any]:
        return await self._admin.clear_style_learning_data()

    async def clear_jargon_data(self) -> Dict[str, Any]:
        return await self._admin.clear_jargon_data()

    async def clear_learning_history_data(self) -> Dict[str, Any]:
        return await self._admin.clear_learning_history_data()

    async def clear_all_plugin_data(self) -> Dict[str, Any]:
        return await self._admin.clear_all_plugin_data()

    async def get_data_statistics(self) -> Dict[str, int]:
        return await self._admin.get_data_statistics()
