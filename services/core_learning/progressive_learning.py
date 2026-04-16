"""
渐进式学习服务 - 协调各个组件实现智能自适应学习
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.star import Context

from ...config import PluginConfig
from ...constants import UPDATE_TYPE_PROGRESSIVE_PERSONA_LEARNING
from ...exceptions import LearningError

from ...utils.json_utils import safe_parse_llm_json, clean_llm_json_response

from ..database import DatabaseManager


@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    messages_processed: int = 0
    filtered_messages: int = 0
    style_updates: int = 0
    quality_score: float = 0.0
    success: bool = False


class ProgressiveLearningService:
    """渐进式学习服务"""
    
    def __init__(self, config: PluginConfig, context: Context,
                 db_manager: DatabaseManager,
                 message_collector,
                 multidimensional_analyzer,
                 style_analyzer,
                 quality_monitor,
                 persona_manager, # 添加 persona_manager 参数
                 ml_analyzer, # 添加 ml_analyzer 参数
                 prompts: Any): # 添加 prompts 参数
        self.config = config
        self.context = context
        self.db_manager = db_manager
        
        # 注入各个组件服务
        self.message_collector = message_collector
        self.multidimensional_analyzer = multidimensional_analyzer
        self.style_analyzer = style_analyzer
        self.quality_monitor = quality_monitor
        self.persona_manager = persona_manager # 注入 persona_manager
        self.ml_analyzer = ml_analyzer # 注入 ml_analyzer
        self.prompts = prompts # 保存 prompts 实例
        
        # 学习状态 - 使用字典管理每个群组的学习状态
        self.learning_active = {} # 改为字典，按群组ID管理
        
        # 增量更新回调函数，降低耦合性
        self.update_system_prompt_callback = None

        self._group_sessions: Dict[str, LearningSession] = {}
        self._latest_learning_results: Dict[str, Dict[str, Any]] = {}
        self._first_batch_result_waiters: Dict[str, asyncio.Future] = {}
        self.learning_sessions: List[LearningSession] = [] # 历史学习会话，可以从数据库加载
        self.learning_lock = asyncio.Lock() # 添加异步锁防止竞态条件

        # 学习控制参数
        self.batch_size = config.max_messages_per_batch
        self.learning_interval = config.learning_interval_hours * 3600 # 转换为秒
        self.quality_threshold = config.style_update_threshold

        logger.info("渐进式学习服务初始化完成")

    def _get_refine_timeout_seconds(self) -> float:
        timeout = getattr(self.config, "refine_timeout_seconds", None)
        if isinstance(timeout, (int, float)) and timeout > 0:
            return float(timeout)
        return 45.0

    def _get_refine_retry_count(self) -> int:
        retry_count = getattr(self.config, "refine_retry_count", 1)
        try:
            return max(0, int(retry_count))
        except (TypeError, ValueError):
            return 1

    def _get_refine_recent_message_limit(self) -> int:
        configured_limit = getattr(self.config, "refine_recent_message_limit", None)
        if isinstance(configured_limit, int) and configured_limit > 0:
            return max(5, min(configured_limit, 30))

        batch_size = getattr(self, "batch_size", 50) or 50
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = 50

        return max(20, min(30, max(20, batch_size // 2 or 20)))

    def _compact_refine_value(
        self,
        value: Any,
        *,
        max_depth: int = 2,
        max_items: int = 6,
        max_str_len: int = 500,
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return value if len(value) <= max_str_len else value[:max_str_len] + "...<truncated>"
        if isinstance(value, (int, float, bool)):
            return value
        if max_depth <= 0:
            return str(value)[:max_str_len]
        if isinstance(value, list):
            return [
                self._compact_refine_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_str_len=max_str_len,
                )
                for item in value[:max_items]
            ]
        if isinstance(value, dict):
            compacted: Dict[str, Any] = {}
            for key, item in list(value.items())[:max_items]:
                compacted[key] = self._compact_refine_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_str_len=max_str_len,
                )
            return compacted
        return str(value)[:max_str_len]

    def _build_recent_messages_refine_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        filtered_messages = [msg for msg in messages if isinstance(msg, dict)]
        ordered_messages = sorted(filtered_messages, key=lambda item: item.get("timestamp", 0))
        recent_limit = limit or self._get_refine_recent_message_limit()
        recent_messages = ordered_messages[-recent_limit:]

        return {
            "message_count": len(ordered_messages),
            "recent_message_count": len(recent_messages),
            "recent_messages": [
                {
                    "message_id": msg.get("message_id", msg.get("id", "")),
                    "sender_id": msg.get("sender_id", ""),
                    "timestamp": msg.get("timestamp", 0),
                    "message_length": len(str(msg.get("message", ""))),
                    "message_preview": self._compact_refine_value(
                        str(msg.get("message", "")).strip(),
                        max_depth=0,
                        max_str_len=240,
                    ),
                    "filter_reason": msg.get("filter_reason", ""),
                }
                for msg in recent_messages
            ],
        }

    def _build_refine_analysis_payload(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        style_analysis = analysis_data.get("style_analysis")
        style_profile = analysis_data.get("style_profile")
        learning_insights = analysis_data.get("learning_insights")

        payload: Dict[str, Any] = {
            "message_count": analysis_data.get("message_count", 0),
            "style_analysis_success": bool(analysis_data.get("style_analysis_success", False)),
            "degraded_mode": bool(analysis_data.get("degraded_mode", False)),
        }

        for key in ("degraded_reason", "filtering_reason", "filtering_detail", "analysis_timestamp"):
            if analysis_data.get(key) is not None:
                payload[key] = analysis_data.get(key)

        if isinstance(style_analysis, dict):
            payload["style_analysis"] = self._compact_refine_value(style_analysis, max_depth=2, max_items=6)
        elif style_analysis is not None:
            payload["style_analysis"] = self._compact_refine_value(style_analysis, max_depth=1, max_items=4)

        if isinstance(style_profile, dict):
            payload["style_profile"] = self._compact_refine_value(style_profile, max_depth=2, max_items=6)

        if isinstance(learning_insights, dict):
            payload["learning_insights"] = self._compact_refine_value(learning_insights, max_depth=1, max_items=6)

        for key in ("style_attributes", "style_features", "enhanced_prompt", "refine_retry_meta"):
            if analysis_data.get(key) is not None:
                payload[key] = self._compact_refine_value(analysis_data.get(key), max_depth=1, max_items=6)

        return payload

    async def _run_refine_with_timeout_retry(
        self,
        llm_adapter,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        timeout_seconds: Optional[float] = None,
        retry_count: Optional[int] = None,
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        timeout_seconds = float(timeout_seconds or self._get_refine_timeout_seconds())
        max_retry = self._get_refine_retry_count() if retry_count is None else max(0, int(retry_count))
        last_error: Optional[str] = None
        attempts = 0
        started_at = time.perf_counter()

        for attempt in range(max_retry + 1):
            attempts = attempt + 1
            attempt_start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    llm_adapter.refine_chat_completion(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    ),
                    timeout=timeout_seconds,
                )
                if not response:
                    raise ValueError("empty_refine_response")

                clean_response = clean_llm_json_response(response)
                parsed_response = safe_parse_llm_json(clean_response)
                if not isinstance(parsed_response, dict) or not parsed_response:
                    raise ValueError(f"invalid_refine_payload:{type(parsed_response)}")

                duration = time.perf_counter() - attempt_start
                total_duration = time.perf_counter() - started_at
                logger.info(
                    f"[LearningBatch] refine_call_success attempts={attempts} "
                    f"attempt_duration={duration:.2f}s total_duration={total_duration:.2f}s "
                    f"timeout_seconds={timeout_seconds}"
                )
                return parsed_response, {
                    "success": True,
                    "attempts": attempts,
                    "timeout_seconds": timeout_seconds,
                    "total_duration_seconds": total_duration,
                    "last_error": None,
                    "degraded_mode": False,
                    "needs_refine_retry": False,
                }
            except asyncio.TimeoutError:
                last_error = f"timeout_after_{timeout_seconds:.2f}s"
                logger.warning(
                    f"[LearningBatch] refine_call_timeout attempt={attempts} "
                    f"timeout_seconds={timeout_seconds}"
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"[LearningBatch] refine_call_failed attempt={attempts} error={exc}"
                )

            if attempts <= max_retry:
                await asyncio.sleep(min(1.5, 0.5 * attempts))

        total_duration = time.perf_counter() - started_at
        logger.warning(
            f"[LearningBatch] refine_call_failed_final attempts={attempts} "
            f"timeout_seconds={timeout_seconds} total_duration={total_duration:.2f}s "
            f"last_error={last_error}"
        )
        return None, {
            "success": False,
            "attempts": attempts,
            "timeout_seconds": timeout_seconds,
            "total_duration_seconds": total_duration,
            "last_error": last_error,
            "degraded_mode": True,
            "needs_refine_retry": True,
        }

    def _resolve_umo(self, group_id: str) -> str:
        """将group_id解析为unified_msg_origin以支持多配置文件"""
        if hasattr(self, 'group_id_to_unified_origin'):
            return self.group_id_to_unified_origin.get(group_id, group_id)
        return group_id

    def set_update_system_prompt_callback(self, callback):
        """
        设置增量更新回调函数
        
        Args:
            callback: 异步回调函数，接受 group_id 参数
        """
        self.update_system_prompt_callback = callback
        logger.info("增量更新回调函数已设置")

    async def start(self):
        """服务启动时加载历史学习会话"""
        # 假设每个群组有独立的学习会话，这里需要一个 group_id
        # 为了简化，暂时假设加载一个默认的或全局的学习会话
        # 实际应用中，可能需要根据当前处理的群组ID来加载
        default_group_id = "global_learning" # 或者从配置中获取
        # 这里可以加载所有历史会话，或者只加载最近的N个
        # 为了简化，我们暂时不从数据库加载历史会话列表，只在每次会话结束时保存
        # 如果需要加载历史会话，需要 DatabaseManager 提供 load_all_learning_sessions 方法
        logger.info("渐进式学习服务启动，准备开始学习。")

    def _create_batch_result(
        self,
        *,
        success: bool,
        generated_learning_content: bool = False,
        persona_applied: bool = False,
        persona_review_written: bool = False,
        session_updates_written: bool = False,
        processed_messages: int = 0,
        filtered_messages: int = 0,
        degraded_mode: bool = False,
        reason: str = "",
        style_analysis_success: bool = False,
    ) -> Dict[str, Any]:
        return {
            "success": bool(success),
            "generated_learning_content": bool(generated_learning_content),
            "persona_applied": bool(persona_applied),
            "persona_review_written": bool(persona_review_written),
            "session_updates_written": bool(session_updates_written),
            "processed_messages": int(processed_messages or 0),
            "filtered_messages": int(filtered_messages or 0),
            "degraded_mode": bool(degraded_mode),
            "reason": str(reason or ""),
            "style_analysis_success": bool(style_analysis_success),
        }

    def _reset_first_batch_waiter(self, group_id: str) -> None:
        existing_waiter = self._first_batch_result_waiters.pop(group_id, None)
        if existing_waiter and not existing_waiter.done():
            existing_waiter.cancel()

        loop = asyncio.get_running_loop()
        self._first_batch_result_waiters[group_id] = loop.create_future()
        self._latest_learning_results.pop(group_id, None)

    def _publish_learning_result(
        self,
        group_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_result = self._create_batch_result(**result)
        self._latest_learning_results[group_id] = normalized_result

        waiter = self._first_batch_result_waiters.get(group_id)
        if waiter and not waiter.done():
            waiter.set_result(normalized_result)

        return normalized_result

    async def wait_for_first_learning_result(
        self,
        group_id: str,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        existing_result = self._latest_learning_results.get(group_id)
        if existing_result:
            return existing_result

        waiter = self._first_batch_result_waiters.get(group_id)
        if not waiter:
            return self._create_batch_result(
                success=False,
                reason="first_batch_result_waiter_missing",
            )

        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
        except asyncio.TimeoutError:
            return self._create_batch_result(
                success=False,
                reason="first_batch_result_wait_timeout",
            )

    async def start_learning(self, group_id: str) -> bool:
        """启动学习流程 - 优化为后台任务执行"""
        async with self.learning_lock: # 使用锁防止竞态条件
            try:
                logger.info(
                    f"[ProgressiveLearning] group={group_id} start_requested "
                    f"active={self.learning_active.get(group_id, False)} "
                    f"batch_size={self.batch_size} "
                    f"interval_seconds={self.learning_interval} "
                    f"quality_threshold={self.quality_threshold}"
                )
                # 检查该群组是否已经在学习
                if self.learning_active.get(group_id, False):
                    logger.info(
                        f"[ProgressiveLearning] group={group_id} decision=skip "
                        "reason=already_active"
                    )
                    return True # 返回True表示学习状态正常
                
                # 设置该群组为学习状态
                self.learning_active[group_id] = True
                self._reset_first_batch_waiter(group_id)
                
                # 创建新的学习会话
                session_id = f"session_{group_id}_{int(time.time())}"
                self._group_sessions[group_id] = LearningSession(
                    session_id=session_id,
                    start_time=datetime.now().isoformat()
                )
                # 保存新的学习会话到数据库
                await self.db_manager.save_learning_session_record(group_id, self._group_sessions[group_id].__dict__)
                
                logger.info(
                    f"[ProgressiveLearning] group={group_id} session_started "
                    f"session_id={session_id}"
                )
                
                # 创建后台任务，确保不阻塞主线程
                learning_task = asyncio.create_task(self._learning_loop_safe(group_id))
                
                # 设置任务完成回调
                def on_learning_complete(task):
                    if task.exception():
                        logger.error(f"群组 {group_id} 学习任务异常完成: {task.exception()}")
                    else:
                        logger.info(f"群组 {group_id} 学习任务正常完成")
                    # 清除该群组的学习状态
                    self.learning_active[group_id] = False
                    
                learning_task.add_done_callback(on_learning_complete)

                def on_learning_complete_with_result(task):
                    if task.cancelled() or task.exception():
                        return

                    result = self._latest_learning_results.get(group_id)
                    if not result:
                        return

                    logger.info(
                        f"[LearningTask] group={group_id} success={result.get('success')} "
                        f"generated_learning_content={result.get('generated_learning_content')} "
                        f"persona_applied={result.get('persona_applied')} "
                        f"persona_review_written={result.get('persona_review_written')} "
                        f"session_updates_written={result.get('session_updates_written')} "
                        f"processed_messages={result.get('processed_messages')} "
                        f"filtered_messages={result.get('filtered_messages')} "
                        f"degraded_mode={result.get('degraded_mode')} "
                        f"reason={result.get('reason')}"
                    )

                learning_task.add_done_callback(on_learning_complete_with_result)
                
                return True
                
            except Exception as e:
                logger.error(f"启动群组 {group_id} 学习失败: {e}")
                # 确保清除学习状态
                self.learning_active[group_id] = False
                self._publish_learning_result(
                    group_id,
                    self._create_batch_result(
                        success=False,
                        reason=f"start_learning_failed:{e}",
                    ),
                )
                return False

    async def stop_learning(self, group_id: str = None):
        """停止学习流程"""
        if group_id:
            # 停止特定群组的学习
            self.learning_active[group_id] = False
            if group_id not in self._latest_learning_results:
                self._publish_learning_result(
                    group_id,
                    self._create_batch_result(
                        success=False,
                        reason="learning_stopped_before_first_batch_result",
                    ),
                )
            logger.info(f"停止群组 {group_id} 的学习任务")
        else:
            # 停止所有群组的学习
            for gid in list(self.learning_active.keys()):
                self.learning_active[gid] = False
            for gid in list(self.learning_active.keys()):
                if gid not in self._latest_learning_results:
                    self._publish_learning_result(
                        gid,
                        self._create_batch_result(
                            success=False,
                            reason="learning_stopped_before_first_batch_result",
                        ),
                    )
            logger.info("停止所有群组的学习任务")
        
        if group_id:
            session = self._group_sessions.pop(group_id, None)
            if session:
                session.end_time = datetime.now().isoformat()
                session.success = True
                await self.db_manager.save_learning_session_record(group_id, session.__dict__)
                self.learning_sessions.append(session)
                logger.info(f"学习会话结束: {session.session_id}")
        else:
            for gid, session in list(self._group_sessions.items()):
                session.end_time = datetime.now().isoformat()
                session.success = True
                await self.db_manager.save_learning_session_record(gid, session.__dict__)
                self.learning_sessions.append(session)
                logger.info(f"学习会话结束: {session.session_id}")
            self._group_sessions.clear()

    async def _learning_loop_safe(self, group_id: str):
        """安全的学习循环 - 在后台线程执行，包含完整错误处理"""
        last_result = self._latest_learning_results.get(group_id)
        try:
            while self.learning_active.get(group_id, False):
                try:
                    # 检查是否应该暂停学习
                    should_pause, reason = await self.quality_monitor.should_pause_learning()
                    if should_pause:
                        last_result = self._publish_learning_result(
                            group_id,
                            self._create_batch_result(
                                success=False,
                                degraded_mode=True,
                                reason=f"learning_paused:{reason}",
                            ),
                        )
                        logger.warning(
                            f"[ProgressiveLearning] group={group_id} decision=pause "
                            f"reason={reason}"
                        )
                        await self.stop_learning(group_id)
                        break
                    
                    # 执行一个学习批次 - 在后台执行
                    logger.info(
                        f"[ProgressiveLearning] group={group_id} batch_start"
                    )
                    batch_result = await self._execute_learning_batch_background(group_id)
                    if isinstance(batch_result, dict):
                        last_result = self._publish_learning_result(group_id, batch_result)
                        logger.info(
                            f"[ProgressiveLearning] group={group_id} batch_result "
                            f"success={batch_result.get('success')} "
                            f"generated_learning_content={batch_result.get('generated_learning_content')} "
                            f"persona_applied={batch_result.get('persona_applied')} "
                            f"session_updates_written={batch_result.get('session_updates_written')} "
                            f"degraded_mode={batch_result.get('degraded_mode')} "
                            f"reason={batch_result.get('reason')}"
                        )
                    
                    # 等待下一个学习周期
                    logger.debug(
                        f"[ProgressiveLearning] group={group_id} sleeping "
                        f"seconds={self.learning_interval}"
                    )
                    await asyncio.sleep(self.learning_interval)
                    
                except asyncio.CancelledError:
                    last_result = self._publish_learning_result(
                        group_id,
                        self._create_batch_result(
                            success=False,
                            reason="learning_task_cancelled",
                        ),
                    )
                    logger.info(f"群组 {group_id} 学习任务被取消")
                    break
                except Exception as e:
                    last_result = self._publish_learning_result(
                        group_id,
                        self._create_batch_result(
                            success=False,
                            degraded_mode=True,
                            reason=f"learning_loop_exception:{e}",
                        ),
                    )
                    logger.error(f"群组 {group_id} 学习循环异常: {e}", exc_info=True)
                    await asyncio.sleep(60) # 异常时等待1分钟
        finally:
            # 确保清理资源
            session = self._group_sessions.pop(group_id, None)
            if session:
                session.end_time = datetime.now().isoformat()
                await self.db_manager.save_learning_session_record(group_id, session.__dict__)
            logger.info(f"学习循环结束 for group {group_id}")

        return last_result or self._latest_learning_results.get(group_id) or self._create_batch_result(
            success=False,
            reason="learning_loop_finished_without_batch_result",
        )

    async def _execute_learning_batch(self, group_id: str, relearn_mode: bool = False, from_force_learning: bool = False):
        """执行一个学习批次 - 集成强化学习

        Args:
            group_id: 群组ID
            relearn_mode: 重新学习模式，如果为True则忽略"已处理"标记，获取所有历史消息
        """
        try:
            batch_start_time = datetime.now()

            # 1. 获取消息（根据模式决定是否忽略"已处理"标记）
            if relearn_mode:
                # 重新学习模式：获取所有历史消息，忽略已处理标记
                logger.info(f" 重新学习模式：获取群组 {group_id} 的所有历史消息（忽略已处理标记）")
                # 使用 get_recent_raw_messages 获取所有历史消息（不考虑已处理标记）
                unprocessed_messages = await self.db_manager.get_recent_raw_messages(
                    group_id=group_id,
                    limit=self.batch_size * 10 # 重新学习时获取更多消息
                )
                logger.info(f"获取到 {len(unprocessed_messages) if unprocessed_messages else 0} 条历史消息用于重新学习")
            else:
                # 正常模式：只获取未处理的消息
                unprocessed_messages = await self.message_collector.get_unprocessed_messages(
                    limit=self.batch_size,
                    group_id=group_id,
                )

            if not unprocessed_messages:
                if relearn_mode:
                    logger.warning(f"群组 {group_id} 没有找到历史消息")
                else:
                    logger.debug("没有未处理的消息，跳过此批次")
                return self._create_batch_result(
                    success=False,
                    processed_messages=0,
                    filtered_messages=0,
                    reason="no_unprocessed_messages_for_learning_batch",
                )

            logger.info(f"开始处理 {len(unprocessed_messages)} 条消息（relearn_mode={relearn_mode}）")
            
            # 2. 使用多维度分析器筛选消息
            filter_stage_start = time.perf_counter()
            filtered_messages = await self._filter_messages_with_context(unprocessed_messages)
            filter_stage_duration = time.perf_counter() - filter_stage_start
            logger.info(
                f"[LearningBatch] group={group_id} stage=filter "
                f"duration={filter_stage_duration:.2f}s "
                f"input_count={len(unprocessed_messages)} output_count={len(filtered_messages)}"
            )
            
            if not filtered_messages:
                logger.debug("没有通过筛选的消息")
                await self._mark_messages_processed(unprocessed_messages)
                return self._create_batch_result(
                    success=False,
                    processed_messages=len(unprocessed_messages),
                    filtered_messages=0,
                    reason="no_messages_available_after_filtering",
                )

            # 2.5 将筛选后的消息写入 FilteredMessage 表（供 WebUI 统计）
            saved_count = 0
            for msg in filtered_messages:
                try:
                    await self.message_collector.add_filtered_message({
                        "raw_message_id": msg.get("id"),
                        "message": msg.get("message", ""),
                        "sender_id": msg.get("sender_id", ""),
                        "group_id": msg.get("group_id", group_id),
                        "timestamp": msg.get("timestamp", int(time.time())),
                        "confidence": msg.get("relevance_score", 1.0),
                        "filter_reason": msg.get("filter_reason", "batch_learning"),
                    })
                    saved_count += 1
                except Exception:
                    pass  # best-effort, don't block learning
            if saved_count:
                logger.debug(f"已保存 {saved_count}/{len(filtered_messages)} 条筛选消息到 FilteredMessage 表")

            # 3. 获取当前人格设置 (针对特定群组)
            current_persona = await self._get_current_persona(group_id)
            style_analysis, updated_persona = await self._build_learning_batch_artifacts(
                group_id,
                filtered_messages,
                current_persona,
            )
            ml_tuning_info = self._extract_analysis_data(style_analysis).get("ml_tuning_info")
            
            # 8. 质量监控评估
            # 确保参数不为None，提供默认值
            if current_persona is None:
                current_persona = {"prompt": "默认人格"}
                logger.warning("current_persona为None，使用默认值")
            
            if updated_persona is None:
                updated_persona = current_persona.copy()
                logger.warning("updated_persona为None，使用current_persona的副本")
                
            quality_metrics = await self.quality_monitor.evaluate_learning_batch(
                current_persona,
                updated_persona,
                filtered_messages
            )

            # 9. 应用学习更新（对话风格学习不判断质量直接应用，人格学习加入审查）
            # 注意：对话风格（表达模式）学习总是成功，人格学习在_apply_learning_updates中会加入审查
            # 传递 relearn_mode 和 ml_tuning_info 参数
            apply_result = await self._apply_learning_updates(
                group_id,
                style_analysis,
                filtered_messages,
                current_persona,
                updated_persona,
                quality_metrics,
                relearn_mode=relearn_mode,
                ml_tuning_info=ml_tuning_info,
            )
            session_updates_written = await self._write_learning_session_updates(
                group_id,
                self._extract_analysis_data(style_analysis),
            )
            analysis_data = self._extract_analysis_data(style_analysis)
            batch_result = self._build_learning_batch_result(
                processed_count=len(unprocessed_messages),
                filtered_count=len(filtered_messages),
                analysis_data=analysis_data,
                persona_result=apply_result,
                session_updates_written=session_updates_written,
            )
            self._log_learning_batch_outcome(
                group_id=group_id,
                batch_result=batch_result,
            )
            logger.info(f"学习更新已应用（对话风格学习已完成，人格学习已加入审查），质量得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
            success = batch_result["success"]
            
            # 10. 【新增】保存学习性能记录
            # 正确处理 AnalysisResult 对象进行序列化
            style_analysis_for_db = style_analysis.data if hasattr(style_analysis, 'data') else style_analysis
            successful_pattern, failed_pattern = self._build_learning_pattern_payload(
                analysis_data,
                batch_result,
            )
            await self.db_manager.save_learning_performance_record(group_id, {
                'session_id': self._group_sessions[group_id].session_id if group_id in self._group_sessions else '',
                'timestamp': time.time(),
                'quality_score': quality_metrics.consistency_score,
                'learning_time': (datetime.now() - batch_start_time).total_seconds(),
                'success': success,
                'successful_pattern': successful_pattern,
                'failed_pattern': failed_pattern,
            })
            
            # 11. 标记消息为已处理
            await self._mark_messages_processed(unprocessed_messages)
            
            # 12. 更新学习会话统计并持久化
            group_session = self._group_sessions.get(group_id)
            if group_session:
                group_session.messages_processed += len(unprocessed_messages)
                group_session.filtered_messages += len(filtered_messages)
                group_session.quality_score = quality_metrics.consistency_score
                group_session.success = success
                await self.db_manager.save_learning_session_record(group_id, group_session.__dict__)
            
            # 13. 【新增】学习成功后更新增量内容到system_prompt
            if success:
                try:
                    # 使用回调函数进行增量更新，降低耦合性
                    if self.update_system_prompt_callback:
                        await self.update_system_prompt_callback(group_id)
                        logger.info(f"定时更新增量内容完成: {group_id}")
                    else:
                        logger.debug("未设置增量更新回调函数，跳过增量内容更新")
                except Exception as e:
                    logger.error(f"定时增量内容更新失败: {e}")
            
            # 14. 定期执行策略优化
            if success and group_session and group_session.messages_processed % 500 == 0:
                try:
                    await self.ml_analyzer.reinforcement_strategy_optimization(group_id)
                    logger.info("执行了策略优化检查")
                except Exception as e:
                    logger.error(f"策略优化失败: {e}")
            
            # 记录批次耗时
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            logger.info(f"学习批次完成，耗时: {batch_duration:.2f}秒")
            
            return self._create_batch_result(
                success=False,
                degraded_mode=True,
                reason=f"finalize_learning_batch_exception:{e}",
            )
        except Exception as e:
            logger.error(f"学习批次执行失败: {e}")
            raise LearningError(f"学习批次执行失败: {str(e)}")

    async def _execute_learning_batch_background(self, group_id: str):
        """在后台执行学习批次 - 使用线程池避免阻塞主协程"""
        try:
            batch_start_time = datetime.now()
            
            # 1. 异步获取数据
            unprocessed_messages = await self.message_collector.get_unprocessed_messages(
                limit=self.batch_size,
                group_id=group_id,
            )
            logger.info(
                f"[ProgressiveLearning] group={group_id} batch_fetch "
                f"unprocessed_messages={len(unprocessed_messages) if unprocessed_messages else 0}"
            )
            
            if not unprocessed_messages:
                logger.info(
                    f"[ProgressiveLearning] group={group_id} decision=skip "
                    "reason=no_unprocessed_messages"
                )
                return self._create_batch_result(
                    success=False,
                    processed_messages=0,
                    filtered_messages=0,
                    reason="no_unprocessed_messages_for_background_batch",
                )
            
            logger.info(
                f"[ProgressiveLearning] group={group_id} batch_processing_start "
                f"message_count={len(unprocessed_messages)}"
            )
            
            # 2. 并行执行筛选和获取人格
            filtered_messages, current_persona = await asyncio.gather(
                self._filter_messages_with_context(unprocessed_messages),
                self._get_current_persona(group_id),
                return_exceptions=True
            )
            
            # 处理异常结果
            if isinstance(filtered_messages, Exception):
                logger.error(f"消息筛选异常: {filtered_messages}")
                filtered_messages = []
            
            if isinstance(current_persona, Exception):
                logger.error(f"获取人格异常: {current_persona}")
                current_persona = {}
            
            if not filtered_messages:
                logger.info(
                    f"[ProgressiveLearning] group={group_id} decision=skip "
                    "reason=no_messages_after_filtering"
                )
                await self._mark_messages_processed(unprocessed_messages)
                return self._create_batch_result(
                    success=False,
                    processed_messages=len(unprocessed_messages),
                    filtered_messages=0,
                    reason="no_messages_available_after_background_filtering",
                )
            
            style_analysis, updated_persona = await self._build_learning_batch_artifacts(
                group_id,
                filtered_messages,
                current_persona,
            )

            # 4. 质量评估和应用更新
            return await self._finalize_learning_batch(
                group_id, current_persona, updated_persona, filtered_messages,
                unprocessed_messages, batch_start_time, style_analysis # 传递 style_analysis
            )
            
        except Exception as e:
            logger.error(f"后台学习批次执行失败: {e}", exc_info=True)

            return self._create_batch_result(
                success=False,
                degraded_mode=True,
                reason=f"background_learning_batch_exception:{e}",
            )

    async def _execute_reinforcement_learning_background(self, group_id: str, filtered_messages, current_persona):
        """在后台执行强化学习"""
        if not self.config.enable_ml_analysis:
            return {}
        
        try:
            return await self.ml_analyzer.reinforcement_memory_replay(
                group_id, filtered_messages, current_persona
            )
        except Exception as e:
            logger.error(f"后台强化学习失败: {e}")
            return {}

    async def _execute_style_analysis_background(self, group_id: str, filtered_messages):
        """在后台执行风格分析"""
        from ...core.interfaces import AnalysisResult
        try:
            return await self.style_analyzer.analyze_conversation_style(group_id, filtered_messages)
        except Exception as e:
            logger.error(f"后台风格分析失败: {e}")
            return AnalysisResult(success=False, confidence=0.0, data={}, error=str(e))

    async def _execute_incremental_tuning_background(self, group_id: str, base_persona, incremental_updates):
        """在后台执行增量微调"""
        try:
            return await self.ml_analyzer.reinforcement_incremental_tuning(
                group_id, base_persona, incremental_updates
            )
        except Exception as e:
            logger.error(f"后台增量微调失败: {e}")
            return {}

    def _extract_analysis_data(self, style_analysis: Any) -> Dict[str, Any]:
        """统一提取 style_analysis 负载。"""
        if isinstance(style_analysis, dict):
            return dict(style_analysis)
        if hasattr(style_analysis, "data") and isinstance(style_analysis.data, dict):
            return dict(style_analysis.data)
        return {}

    def _build_fallback_style_analysis(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """当 LLM 风格分析失败时，至少基于真实消息构建可消费的学习结果。"""
        texts = [msg.get("message", "").strip() for msg in messages if msg.get("message")]
        if not texts:
            return {}

        total_length = sum(len(text) for text in texts)
        avg_length = total_length / max(len(texts), 1)
        question_ratio = sum(1 for text in texts if "?" in text or "？" in text) / max(len(texts), 1)
        exclamation_ratio = sum(1 for text in texts if "!" in text or "！" in text) / max(len(texts), 1)

        common_phrases: List[str] = []
        for text in texts:
            snippet = text[:20]
            if snippet and snippet not in common_phrases:
                common_phrases.append(snippet)
            if len(common_phrases) >= 5:
                break

        expression_features: List[str] = []
        if avg_length >= 40:
            expression_features.append("long_responses")
        elif avg_length >= 15:
            expression_features.append("medium_responses")
        else:
            expression_features.append("short_responses")

        if question_ratio >= 0.3:
            expression_features.append("question_driven")
        if exclamation_ratio >= 0.2:
            expression_features.append("emotionally_expressive")

        tone = "balanced"
        if exclamation_ratio >= 0.2:
            tone = "enthusiastic"
        elif question_ratio >= 0.3:
            tone = "interactive"

        return {
            "text_style": f"style_summary_from_{len(texts)}_messages",
            "expression_features": expression_features,
            "tone": tone,
            "topics": common_phrases[:3],
            "common_phrases": common_phrases,
        }

    def _derive_style_attributes(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """从风格分析结果推导 PersonaUpdater 可直接消费的 style_attributes。"""
        style_attributes: Dict[str, Any] = {}
        style_report = analysis_data.get("style_analysis")
        if isinstance(style_report, dict):
            if style_report.get("tone"):
                style_attributes["tone"] = style_report["tone"]
            if style_report.get("emotion"):
                style_attributes["emotion"] = style_report["emotion"]

        style_profile = analysis_data.get("style_profile")
        if isinstance(style_profile, dict):
            formality_level = style_profile.get("formality_level")
            if isinstance(formality_level, (int, float)):
                if formality_level >= 0.6:
                    style_attributes["formality"] = "formal"
                elif formality_level <= 0.4:
                    style_attributes["formality"] = "casual"

            emotional_expression = style_profile.get("emotional_expression")
            if isinstance(emotional_expression, (int, float)) and "emotion" not in style_attributes:
                if emotional_expression >= 0.7:
                    style_attributes["emotion"] = "high_expression"
                elif emotional_expression <= 0.3:
                    style_attributes["emotion"] = "restrained_expression"

        return style_attributes

    def _derive_style_features(self, analysis_data: Dict[str, Any]) -> List[str]:
        """提取一组简洁的风格特征用于日志和 PersonaUpdater。"""
        features: List[str] = []
        style_report = analysis_data.get("style_analysis")
        if isinstance(style_report, dict):
            expression_features = style_report.get("expression_features")
            if isinstance(expression_features, list):
                features.extend(str(item) for item in expression_features[:5] if item)
            elif isinstance(expression_features, str):
                features.append(expression_features)

            common_phrases = style_report.get("common_phrases")
            if isinstance(common_phrases, list):
                features.extend(f"common_phrase:{item}" for item in common_phrases[:3] if item)

        deduped: List[str] = []
        for feature in features:
            if feature not in deduped:
                deduped.append(feature)
        return deduped[:8]

    def _build_learning_insights(
        self,
        group_id: str,
        messages: List[Dict[str, Any]],
        analysis_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """构造 session_updates / 临时人格更新可复用的学习洞察。"""
        style_report = analysis_data.get("style_analysis", {})
        style_profile = analysis_data.get("style_profile", {})

        interaction_patterns = "interaction_style_extracted_from_current_batch"
        if isinstance(style_report, dict):
            expression_features = style_report.get("expression_features")
            if isinstance(expression_features, list) and expression_features:
                interaction_patterns = ", ".join(str(item) for item in expression_features[:3])
            elif style_report.get("text_style"):
                interaction_patterns = str(style_report["text_style"])

        effective_strategies = (
            f"preserve_group_{group_id}_patterns_from_{len(messages)}_messages"
        )
        if isinstance(style_report, dict) and style_report.get("common_phrases"):
            phrases = style_report["common_phrases"]
            if isinstance(phrases, list) and phrases:
                effective_strategies = "reuse_common_phrases: " + " / ".join(
                    str(item) for item in phrases[:3]
                )

        improvement_suggestions = (
            "merge_learned_style_into_future_persona_and_response_generation"
        )
        if isinstance(style_profile, dict) and isinstance(style_profile.get("vocabulary_richness"), (int, float)):
            improvement_suggestions = (
                "use_quantified_style_profile_to_adjust_wording_and_rhythm"
            )

        return {
            "interaction_patterns": interaction_patterns,
            "improvement_suggestions": improvement_suggestions,
            "effective_strategies": effective_strategies,
            "learning_focus": f"processed_{len(messages)}_messages_with_reusable_style_output",
        }

    def _has_generated_learning_content(self, analysis_data: Dict[str, Any]) -> bool:
        return bool(
            analysis_data.get("enhanced_prompt")
            or analysis_data.get("learning_insights")
            or analysis_data.get("style_analysis")
        )

    def _compose_learning_reason(
        self,
        analysis_data: Dict[str, Any],
        generated_learning_content: bool,
    ) -> str:
        reasons: List[str] = []

        filtering_detail = analysis_data.get("filtering_detail")
        if filtering_detail:
            reasons.append(str(filtering_detail))

        degraded_reason = analysis_data.get("degraded_reason")
        if degraded_reason:
            reasons.append(str(degraded_reason))

        reason = analysis_data.get("reason")
        if reason:
            reasons.append(str(reason))

        if not generated_learning_content:
            reasons.append("no_learning_content_generated")

        if generated_learning_content and not reasons:
            reasons.append("learning_content_generated")

        deduped_reasons: List[str] = []
        for item in reasons:
            if item and item not in deduped_reasons:
                deduped_reasons.append(item)
        return "; ".join(deduped_reasons)

    def _build_learning_batch_result(
        self,
        *,
        processed_count: int,
        filtered_count: int,
        analysis_data: Dict[str, Any],
        persona_result: Optional[Dict[str, Any]],
        session_updates_written: bool,
    ) -> Dict[str, Any]:
        persona_result = persona_result or {}
        generated_learning_content = self._has_generated_learning_content(analysis_data)
        persona_applied = bool(persona_result.get("persona_update_applied"))
        persona_review_written = bool(persona_result.get("persona_review_written"))
        style_analysis_success = bool(
            analysis_data.get("style_analysis_success", analysis_data.get("style_analysis"))
        )
        degraded_mode = bool(analysis_data.get("degraded_mode"))
        reason = self._compose_learning_reason(analysis_data, generated_learning_content)

        return self._create_batch_result(
            success=bool(
                generated_learning_content
                or persona_applied
                or persona_review_written
                or session_updates_written
            ),
            generated_learning_content=generated_learning_content,
            persona_applied=persona_applied,
            persona_review_written=persona_review_written,
            session_updates_written=session_updates_written,
            processed_messages=processed_count,
            filtered_messages=filtered_count,
            degraded_mode=degraded_mode,
            reason=reason,
            style_analysis_success=style_analysis_success,
        )

    def _build_learning_pattern_payload(
        self,
        analysis_data: Dict[str, Any],
        batch_result: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        key_summary = {
            "style_analysis_success": batch_result.get("style_analysis_success", False),
            "generated_learning_content": batch_result.get("generated_learning_content", False),
            "persona_applied": batch_result.get("persona_applied", False),
            "persona_review_written": batch_result.get("persona_review_written", False),
            "session_updates_written": batch_result.get("session_updates_written", False),
            "processed_messages": batch_result.get("processed_messages", 0),
            "filtered_messages": batch_result.get("filtered_messages", 0),
            "degraded_mode": batch_result.get("degraded_mode", False),
            "needs_refine_retry": analysis_data.get("needs_refine_retry", False),
            "refine_attempts": analysis_data.get("refine_attempts", 0),
            "reason": batch_result.get("reason", ""),
            "style_features": analysis_data.get("style_features", []),
            "message_count": analysis_data.get("message_count", 0),
        }

        successful_pattern: Dict[str, Any] = {}
        failed_pattern: Dict[str, Any] = {}

        if batch_result.get("generated_learning_content"):
            for field in ("style_analysis", "learning_insights", "enhanced_prompt"):
                value = analysis_data.get(field)
                if value:
                    successful_pattern[field] = value
            successful_pattern["key_summary"] = key_summary
        else:
            failed_pattern = dict(key_summary)

        return successful_pattern, failed_pattern

    async def _save_learning_performance(
        self,
        group_id: str,
        quality_score: float,
        learning_time: float,
        analysis_data: Dict[str, Any],
        batch_result: Dict[str, Any],
    ) -> None:
        successful_pattern, failed_pattern = self._build_learning_pattern_payload(
            analysis_data,
            batch_result,
        )
        await self.db_manager.save_learning_performance_record(
            group_id,
            {
                "session_id": self._group_sessions[group_id].session_id if group_id in self._group_sessions else "",
                "timestamp": time.time(),
                "quality_score": quality_score,
                "learning_time": learning_time,
                "success": batch_result.get("success", False),
                "successful_pattern": successful_pattern,
                "failed_pattern": failed_pattern,
            },
        )

    def _normalize_style_analysis_result(
        self,
        group_id: str,
        style_analysis: Any,
        messages: List[Dict[str, Any]],
    ):
        """确保 style_analysis 至少包含后续链路能消费的一种结构。"""
        from ...core.interfaces import AnalysisResult

        if isinstance(style_analysis, AnalysisResult):
            success = style_analysis.success
            confidence = style_analysis.confidence
            timestamp = style_analysis.timestamp
            error = style_analysis.error
            analysis_data = self._extract_analysis_data(style_analysis)
        elif isinstance(style_analysis, dict):
            success = bool(style_analysis.get("success", True))
            confidence = float(style_analysis.get("confidence", 0.6))
            timestamp = time.time()
            error = None
            analysis_data = dict(style_analysis)
        else:
            success = False
            confidence = 0.0
            timestamp = time.time()
            error = f"unexpected_style_analysis_type:{type(style_analysis)}"
            analysis_data = {}

        raw_style_analysis_success = bool(success)
        analysis_data.setdefault("message_count", len(messages))
        analysis_data.setdefault("analysis_timestamp", datetime.now().isoformat())
        analysis_data.setdefault("style_analysis_success", raw_style_analysis_success)
        analysis_data.setdefault("degraded_mode", False)

        if error:
            analysis_data["degraded_mode"] = True
            analysis_data.setdefault("degraded_reason", f"style_analysis_error:{error}")

        if not analysis_data.get("style_analysis"):
            fallback_style_analysis = self._build_fallback_style_analysis(messages)
            if fallback_style_analysis:
                analysis_data["style_analysis"] = fallback_style_analysis
                analysis_data["degraded_mode"] = True
                analysis_data.setdefault(
                    "degraded_reason",
                    "style_analysis_missing_use_message_derived_fallback",
                )

        if not analysis_data.get("learning_insights"):
            analysis_data["learning_insights"] = self._build_learning_insights(
                group_id,
                messages,
                analysis_data,
            )

        if not analysis_data.get("style_attributes"):
            style_attributes = self._derive_style_attributes(analysis_data)
            if style_attributes:
                analysis_data["style_attributes"] = style_attributes

        if not analysis_data.get("style_features"):
            style_features = self._derive_style_features(analysis_data)
            if style_features:
                analysis_data["style_features"] = style_features

        if not success and self._has_generated_learning_content(analysis_data):
            success = True
            confidence = max(confidence, 0.55)

        return AnalysisResult(
            success=success,
            confidence=confidence,
            data=analysis_data,
            timestamp=timestamp,
            error=error,
        )

    async def _build_learning_batch_artifacts(
        self,
        group_id: str,
        filtered_messages: List[Dict[str, Any]],
        current_persona: Optional[Dict[str, Any]],
    ):
        """执行真实风格分析并生成可落地的人格学习内容。"""
        style_analysis = await self._execute_style_analysis_background(
            group_id,
            filtered_messages,
        )
        style_analysis = self._normalize_style_analysis_result(
            group_id,
            style_analysis,
            filtered_messages,
        )

        base_persona = current_persona or {"prompt": "默认人格", "name": "default"}
        refine_stage_start = time.perf_counter()
        updated_persona, refine_meta = await self._generate_updated_persona_with_refinement_v2(
            group_id,
            base_persona,
            style_analysis,
            filtered_messages,
        )
        refine_stage_duration = time.perf_counter() - refine_stage_start
        logger.info(
            f"[LearningBatch] group={group_id} stage=refine "
            f"duration={refine_stage_duration:.2f}s "
            f"attempts={refine_meta.get('attempts', 0)} "
            f"degraded_mode={refine_meta.get('degraded_mode', False)} "
            f"needs_refine_retry={refine_meta.get('needs_refine_retry', False)}"
        )
        if not isinstance(updated_persona, dict):
            logger.warning(f"更新后人格结构异常，回退为当前人格副本: {type(updated_persona)}")
            updated_persona = dict(base_persona)

        analysis_data = self._extract_analysis_data(style_analysis)
        if isinstance(refine_meta, dict) and refine_meta:
            analysis_data["refine_retry_meta"] = refine_meta
            analysis_data["refine_attempts"] = refine_meta.get("attempts", 0)
            analysis_data["needs_refine_retry"] = bool(refine_meta.get("needs_refine_retry", False))
            if refine_meta.get("degraded_mode"):
                analysis_data["degraded_mode"] = True
                refine_error = refine_meta.get("last_error") or "refine_failed"
                existing_degraded_reason = analysis_data.get("degraded_reason")
                if existing_degraded_reason:
                    analysis_data["degraded_reason"] = f"{existing_degraded_reason}; refine_failed:{refine_error}"
                else:
                    analysis_data["degraded_reason"] = f"refine_failed:{refine_error}"
        filter_reasons = [
            msg.get("filter_reason")
            for msg in filtered_messages
            if isinstance(msg, dict) and msg.get("filter_reason")
        ]
        if filter_reasons:
            analysis_data["filtering_reason"] = filter_reasons[0]
            if filter_reasons[0] == "style_learning_no_filter":
                analysis_data["filtering_detail"] = (
                    "style_learning_uses_raw_messages_without_llm_filter_by_design"
                )
        filter_details = [
            msg.get("filter_detail")
            for msg in filtered_messages
            if isinstance(msg, dict) and msg.get("filter_detail")
        ]
        if filter_details:
            analysis_data["filtering_detail"] = filter_details[0]
        current_prompt = (base_persona or {}).get("prompt", "")
        updated_prompt = updated_persona.get("prompt", "")
        if updated_prompt and updated_prompt != current_prompt:
            analysis_data["enhanced_prompt"] = updated_prompt

        if not analysis_data.get("learning_insights"):
            analysis_data["learning_insights"] = self._build_learning_insights(
                group_id,
                filtered_messages,
                analysis_data,
            )

        if self.config.enable_ml_analysis and self.ml_analyzer:
            replay_persona = dict(base_persona)
            replay_persona.setdefault("description", replay_persona.get("prompt", ""))
            logger.info(
                f"[LearningBatch] group={group_id} trigger=memory_replay "
                f"message_count={len(filtered_messages)}"
            )
            try:
                replay_stage_start = time.perf_counter()
                replay_result = await self.ml_analyzer.reinforcement_memory_replay(
                    group_id,
                    filtered_messages,
                    replay_persona,
                    from_learning_batch=True,
                )
                replay_stage_duration = time.perf_counter() - replay_stage_start
                logger.info(
                    f"[LearningBatch] group={group_id} stage=replay "
                    f"duration={replay_stage_duration:.2f}s "
                    f"message_count={len(filtered_messages)}"
                )
                if replay_result:
                    analysis_data["memory_replay_result"] = replay_result
                    logger.info(
                        f"[LearningBatch] group={group_id} memory_replay_result "
                        f"keys={list(replay_result.keys())}"
                    )
                else:
                    logger.info(
                        f"[LearningBatch] group={group_id} memory_replay_result_empty"
                    )
            except Exception as e:
                logger.error(
                    f"[LearningBatch] group={group_id} memory_replay_failed: {e}",
                    exc_info=True,
                )
        else:
            logger.info(
                f"[LearningBatch] group={group_id} memory_replay_skipped "
                f"enable_ml_analysis={self.config.enable_ml_analysis} "
                f"ml_analyzer_available={bool(self.ml_analyzer)}"
            )

        style_analysis.data = analysis_data
        logger.info(
            f"[LearningBatch] group={group_id} "
            f"style_analysis_success={analysis_data.get('style_analysis_success', style_analysis.success)} "
            f"generated_learning_content={self._has_generated_learning_content(analysis_data)} "
            f"degraded_mode={analysis_data.get('degraded_mode', False)} "
            f"reason={self._compose_learning_reason(analysis_data, self._has_generated_learning_content(analysis_data))} "
            f"analysis_fields={list(analysis_data.keys())}"
        )
        return style_analysis, updated_persona

    async def _write_learning_session_updates(
        self,
        group_id: str,
        analysis_data: Dict[str, Any],
    ) -> bool:
        """尽量将学习洞察写入 session_updates，供 LLM Hook 注入。"""
        temporary_persona_updater = getattr(self.ml_analyzer, "temporary_persona_updater", None)
        learning_insights = analysis_data.get("learning_insights")

        if not temporary_persona_updater:
            logger.info(
                f"[LearningBatch] group={group_id} session_updates_written=False "
                "reason=temporary_persona_updater_unavailable"
            )
            return False

        if not isinstance(learning_insights, dict) or not learning_insights:
            logger.info(
                f"[LearningBatch] group={group_id} session_updates_written=False "
                "reason=no_learning_insights"
            )
            return False

        try:
            updates_bucket = getattr(temporary_persona_updater, "session_updates", None)
            if not isinstance(updates_bucket, dict):
                logger.info(
                    f"[LearningBatch] group={group_id} session_updates_written=False "
                    "reason=session_updates_bucket_unavailable"
                )
                return False

            insight_text = (
                f"[Learning Insights - {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n"
                f"- interaction_patterns: {learning_insights.get('interaction_patterns', 'n/a')}\n"
                f"- improvement_suggestions: {learning_insights.get('improvement_suggestions', 'n/a')}\n"
                f"- effective_strategies: {learning_insights.get('effective_strategies', 'n/a')}\n"
                f"- learning_focus: {learning_insights.get('learning_focus', 'n/a')}"
            )

            if group_id not in updates_bucket:
                updates_bucket[group_id] = []

            if insight_text not in updates_bucket[group_id]:
                updates_bucket[group_id].append(insight_text)

            logger.info(
                f"[LearningBatch] group={group_id} session_updates_written=True "
                f"session_update_count={len(updates_bucket[group_id])}"
            )
            return True
        except Exception as e:
            logger.error(f"[LearningBatch] 写入 session_updates 失败: {e}", exc_info=True)
            return False

    def _log_learning_batch_outcome(
        self,
        group_id: str,
        batch_result: Dict[str, Any],
    ):
        """输出批次级学习结果日志，便于确认是否真的学到了。"""
        logger.info(
            f"[LearningBatch] group={group_id} "
            f"style_analysis_success={batch_result.get('style_analysis_success', False)} "
            f"generated_learning_content={batch_result.get('generated_learning_content', False)} "
            f"persona_applied={batch_result.get('persona_applied', False)} "
            f"persona_review_written={batch_result.get('persona_review_written', False)} "
            f"session_updates_written={batch_result.get('session_updates_written', False)} "
            f"processed_messages={batch_result.get('processed_messages', 0)} "
            f"filtered_messages={batch_result.get('filtered_messages', 0)} "
            f"degraded_mode={batch_result.get('degraded_mode', False)} "
            f"reason={batch_result.get('reason', '')}"
        )

    async def _finalize_learning_batch(self, group_id: str, current_persona, updated_persona,
                                     filtered_messages, unprocessed_messages, batch_start_time, style_analysis=None):
        """完成学习批次的最终处理

        Args:
            style_analysis: 风格分析结果，用于保存对话风格学习记录
        """
        try:
            # 质量监控评估
            # 确保参数不为None，提供默认值
            if current_persona is None:
                current_persona = {"prompt": "默认人格"}
                logger.warning("_finalize_learning_batch: current_persona为None，使用默认值")

            if updated_persona is None:
                updated_persona = current_persona.copy()
                logger.warning("_finalize_learning_batch: updated_persona为None，使用current_persona的副本")

            quality_metrics = await self.quality_monitor.evaluate_learning_batch(
                current_persona, updated_persona, filtered_messages
            )

            # 应用学习更新（对话风格学习不判断质量直接应用，人格学习加入审查）
            # 传递 style_analysis 用于保存对话风格学习记录
            # 如果 style_analysis 为 None，创建一个空的 AnalysisResult
            from ...core.interfaces import AnalysisResult
            if style_analysis is None:
                style_analysis, updated_persona = await self._build_learning_batch_artifacts(
                    group_id,
                    filtered_messages,
                    current_persona,
                )
            apply_result = await self._apply_learning_updates(
                group_id,
                style_analysis,
                filtered_messages,
                current_persona,
                updated_persona,
                quality_metrics,
                relearn_mode=False,
                ml_tuning_info=None,
            )
            session_updates_written = await self._write_learning_session_updates(
                group_id,
                self._extract_analysis_data(style_analysis),
            )
            analysis_data = self._extract_analysis_data(style_analysis)
            batch_result = self._build_learning_batch_result(
                processed_count=len(unprocessed_messages),
                filtered_count=len(filtered_messages),
                analysis_data=analysis_data,
                persona_result=apply_result,
                session_updates_written=session_updates_written,
            )
            self._log_learning_batch_outcome(
                group_id=group_id,
                batch_result=batch_result,
            )
            logger.info(f"学习更新已应用（对话风格学习已完成，人格学习已加入审查），质量得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
            success = batch_result["success"]

            # 记录学习批次到数据库（使用 ORM）
            try:
                batch_name = f"batch_{group_id}_{int(time.time())}"
                start_time = batch_start_time.timestamp()
                end_time = time.time()

                async with self.db_manager.get_session() as session:
                    from ...models.orm.learning import LearningBatch
                    batch_record = LearningBatch(
                        batch_id=batch_name,
                        batch_name=batch_name,
                        group_id=group_id,
                        start_time=start_time,
                        end_time=end_time,
                        quality_score=quality_metrics.consistency_score,
                        processed_messages=len(unprocessed_messages),
                        message_count=len(unprocessed_messages),
                        filtered_count=len(filtered_messages),
                        success=success,
                    )
                    session.add(batch_record)
                    await session.commit()
                    logger.debug(f"学习批次记录已保存: {batch_name}")
            except Exception as e:
                logger.debug(f"无法记录学习批次（不影响学习功能）: {e}")

            # 保存学习性能记录
            successful_pattern, failed_pattern = self._build_learning_pattern_payload(
                analysis_data,
                batch_result,
            )
            await self.db_manager.save_learning_performance_record(group_id, {
                'session_id': self._group_sessions[group_id].session_id if group_id in self._group_sessions else '',
                'timestamp': time.time(),
                'quality_score': quality_metrics.consistency_score,
                'learning_time': end_time - start_time,
                'success': success,
                'successful_pattern': successful_pattern,
                'failed_pattern': failed_pattern,
            })
            
            # 标记消息为已处理
            await self._mark_messages_processed(unprocessed_messages)
            
            # 更新会话统计
            bg_session = self._group_sessions.get(group_id)
            if bg_session:
                bg_session.messages_processed += len(unprocessed_messages)
                bg_session.filtered_messages += len(filtered_messages)
                bg_session.quality_score = quality_metrics.consistency_score
                bg_session.success = success
                await self.db_manager.save_learning_session_record(group_id, bg_session.__dict__)

            # 定期执行策略优化 - 不阻塞主流程
            if success and bg_session and bg_session.messages_processed % 500 == 0:
                asyncio.create_task(self._execute_strategy_optimization_background(group_id))
            
            batch_duration = end_time - start_time
            return batch_result
            logger.info(f"后台学习批次完成，耗时: {batch_duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"完成学习批次失败: {e}")

            return self._create_batch_result(
                success=False,
                degraded_mode=True,
                reason=f"finalize_learning_batch_exception:{e}",
            )
        except Exception as e:
            logger.error(f"瀹屾垚瀛︿範鎵規澶辫触: {e}")
            return self._create_batch_result(
                success=False,
                degraded_mode=True,
                reason=f"finalize_learning_batch_exception:{e}",
            )

    async def _execute_strategy_optimization_background(self, group_id: str):
        """在后台执行策略优化，不阻塞主流程"""
        try:
            await self.ml_analyzer.reinforcement_strategy_optimization(group_id)
            logger.info("后台策略优化完成")
        except Exception as e:
            logger.error(f"后台策略优化失败: {e}")

    async def _generate_updated_persona_with_refinement(self, group_id: str, current_persona: Dict[str, Any], style_analysis: Any) -> Dict[str, Any]:
        """使用提炼模型生成更新后的人格"""
        try:
            # 正确处理AnalysisResult对象和字典类型
            from ...core.interfaces import AnalysisResult
            
            if isinstance(style_analysis, AnalysisResult):
                # 如果是AnalysisResult对象，提取data属性
                analysis_data = style_analysis.data if style_analysis.data else {}
                logger.debug(f"从AnalysisResult提取data: success={style_analysis.success}, confidence={style_analysis.confidence}")
            elif isinstance(style_analysis, dict):
                analysis_data = style_analysis
                logger.debug("使用字典形式的style_analysis")
            elif hasattr(style_analysis, 'data'):
                # 兼容其他具有data属性的对象
                analysis_data = style_analysis.data if style_analysis.data else {}
                logger.debug(f"从对象提取data属性: {type(style_analysis)}")
            else:
                analysis_data = {}
                logger.warning(f"style_analysis类型不正确: {type(style_analysis)}, 使用空字典")
            
            # 使用多维度分析器的框架适配器生成人格更新
            if hasattr(self.multidimensional_analyzer, 'llm_adapter') and self.multidimensional_analyzer.llm_adapter:
                llm_adapter = self.multidimensional_analyzer.llm_adapter
                
                if llm_adapter.has_refine_provider() and llm_adapter.providers_configured >= 2:
                    # 准备输入数据
                    current_persona_json = json.dumps(current_persona, ensure_ascii=False, indent=2, default=self._json_serializer)
                    style_analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2, default=self._json_serializer)
                    
                    # 调用框架适配器
                    response = await llm_adapter.refine_chat_completion(
                        prompt=self.prompts.PROGRESSIVE_LEARNING_GENERATE_UPDATED_PERSONA_PROMPT.format(
                            current_persona_json=current_persona_json,
                            style_analysis_json=style_analysis_json
                        ),
                        temperature=0.6
                    )
                    
                    if response:
                        # 清理响应文本，移除markdown标识符（使用统一的json_utils工具）
                        clean_response = clean_llm_json_response(response)

                        try:
                            updated_persona = safe_parse_llm_json(clean_response)
                            logger.info("使用提炼模型成功生成更新后的人格")
                            return updated_persona
                        except json.JSONDecodeError as e:
                            logger.error(f"提炼模型返回的JSON格式不正确: {e}, 响应: {clean_response}")
                            return await self._generate_updated_persona(group_id, current_persona, style_analysis)
                else:
                    logger.warning("提炼模型Provider未配置，使用传统方法生成人格")
                    return await self._generate_updated_persona(group_id, current_persona, style_analysis)
            else:
                logger.warning("框架适配器未找到，使用传统方法生成人格")
                return await self._generate_updated_persona(group_id, current_persona, style_analysis)
            
        except Exception as e:
            logger.error(f"使用提炼模型生成人格失败: {e}")
            return await self._generate_updated_persona(group_id, current_persona, style_analysis)

    async def _generate_updated_persona_with_refinement_v2(
        self,
        group_id: str,
        current_persona: Dict[str, Any],
        style_analysis: Any,
        filtered_messages: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """使用压缩输入的 refinement 版本，带超时和一次自动重试。"""
        start_time = time.perf_counter()
        refine_meta: Dict[str, Any] = {
            "success": False,
            "attempts": 0,
            "timeout_seconds": self._get_refine_timeout_seconds(),
            "total_duration_seconds": 0.0,
            "last_error": None,
            "degraded_mode": False,
            "needs_refine_retry": False,
        }

        try:
            if not hasattr(self.context, "persona_manager") or not self.context.persona_manager:
                refine_meta["last_error"] = "persona_manager_unavailable"
                refine_meta["degraded_mode"] = True
                refine_meta["needs_refine_retry"] = True
                return dict(current_persona), refine_meta

            default_persona = await self.context.persona_manager.get_default_persona_v3(self._resolve_umo(group_id))
            if not default_persona:
                refine_meta["last_error"] = "default_persona_unavailable"
                refine_meta["degraded_mode"] = True
                refine_meta["needs_refine_retry"] = True
                return dict(current_persona), refine_meta

            if hasattr(self.multidimensional_analyzer, "llm_adapter") and self.multidimensional_analyzer.llm_adapter:
                llm_adapter = self.multidimensional_analyzer.llm_adapter
                if llm_adapter.has_refine_provider() and llm_adapter.providers_configured >= 2:
                    from ...core.interfaces import AnalysisResult

                    if isinstance(style_analysis, AnalysisResult):
                        analysis_data = style_analysis.data if style_analysis.data else {}
                    elif isinstance(style_analysis, dict):
                        analysis_data = style_analysis
                    elif hasattr(style_analysis, "data"):
                        analysis_data = style_analysis.data if style_analysis.data else {}
                    else:
                        analysis_data = {}

                    compact_current_persona = {
                        "name": current_persona.get("name", default_persona.get("name", "default")),
                        "prompt": self._compact_refine_value(current_persona.get("prompt", default_persona.get("prompt", "")), max_depth=0, max_str_len=2500),
                        "description": self._compact_refine_value(current_persona.get("description", current_persona.get("prompt", "")), max_depth=0, max_str_len=500),
                        "style_parameters": self._compact_refine_value(current_persona.get("style_parameters", {}), max_depth=1, max_items=6),
                    }
                    refine_payload = self._build_refine_analysis_payload(analysis_data)
                    recent_messages_payload = self._build_recent_messages_refine_payload(filtered_messages)

                    prompt = (
                        self.prompts.PROGRESSIVE_LEARNING_GENERATE_UPDATED_PERSONA_PROMPT.format(
                            current_persona_json=json.dumps(compact_current_persona, ensure_ascii=False, indent=2, default=self._json_serializer),
                            style_analysis_json=json.dumps(refine_payload, ensure_ascii=False, indent=2, default=self._json_serializer),
                        )
                        + "\n\n【最近消息摘要】\n"
                        + json.dumps(recent_messages_payload, ensure_ascii=False, indent=2, default=self._json_serializer)
                        + "\n\n【精炼约束】请仅依据压缩后的分析和最近消息摘要进行更新；优先保留原有人格框架，避免扩写无关内容。"
                    )
                    logger.info(
                        f"[LearningBatch] group={group_id} stage=refine_input "
                        f"message_count={refine_payload.get('message_count', 0)} "
                        f"recent_message_count={recent_messages_payload.get('recent_message_count', 0)} "
                        f"prompt_chars={len(prompt)}"
                    )

                    refined_persona, call_meta = await self._run_refine_with_timeout_retry(
                        llm_adapter,
                        prompt=prompt,
                        temperature=0.6,
                    )
                    refine_meta.update(call_meta)
                    refine_meta["total_duration_seconds"] = time.perf_counter() - start_time

                    if refined_persona:
                        logger.info(
                            f"[LearningBatch] group={group_id} filter provider bound "
                            f"refine provider bound reinforce provider bound "
                            f"stage=refine success=True attempts={refine_meta.get('attempts', 0)} "
                            f"duration={refine_meta['total_duration_seconds']:.2f}s"
                        )
                        return refined_persona, refine_meta

                    refine_meta["degraded_mode"] = True
                    refine_meta["needs_refine_retry"] = True
                    refine_meta["last_error"] = refine_meta.get("last_error") or "refine_failed_use_fallback"
                    fallback_persona = await self._generate_updated_persona(group_id, current_persona, style_analysis)
                    if not isinstance(fallback_persona, dict):
                        fallback_persona = dict(current_persona)
                    return fallback_persona, refine_meta

                refine_meta["last_error"] = "refine_provider_unavailable"
                refine_meta["degraded_mode"] = True
                refine_meta["needs_refine_retry"] = True
                fallback_persona = await self._generate_updated_persona(group_id, current_persona, style_analysis)
                if not isinstance(fallback_persona, dict):
                    fallback_persona = dict(current_persona)
                refine_meta["total_duration_seconds"] = time.perf_counter() - start_time
                return fallback_persona, refine_meta

            refine_meta["last_error"] = "llm_adapter_unavailable"
            refine_meta["degraded_mode"] = True
            refine_meta["needs_refine_retry"] = True
            fallback_persona = await self._generate_updated_persona(group_id, current_persona, style_analysis)
            if not isinstance(fallback_persona, dict):
                fallback_persona = dict(current_persona)
            refine_meta["total_duration_seconds"] = time.perf_counter() - start_time
            return fallback_persona, refine_meta
        except Exception as exc:
            logger.error(f"refine生成失败，回退到传统方法: {exc}", exc_info=True)
            refine_meta["last_error"] = str(exc)
            refine_meta["degraded_mode"] = True
            refine_meta["needs_refine_retry"] = True
            refine_meta["total_duration_seconds"] = time.perf_counter() - start_time
            fallback_persona = await self._generate_updated_persona(group_id, current_persona, style_analysis)
            if not isinstance(fallback_persona, dict):
                fallback_persona = dict(current_persona)
            return fallback_persona, refine_meta

    def _json_serializer(self, obj):
        """自定义JSON序列化器，处理不能直接序列化的对象"""
        try:
            # 检查对象的类型名称，避免循环导入
            class_name = obj.__class__.__name__
            
            if class_name == 'StyleProfile':
                # 将StyleProfile对象转换为字典
                if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    return obj.to_dict()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                # 对于有to_dict方法的对象
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                # 对于其他dataclass或对象，尝试使用__dict__
                return obj.__dict__
            else:
                # 如果都不行，转换为字符串
                return str(obj)
        except Exception as e:
            logger.warning(f"JSON序列化对象时出现错误: {e}, 对象类型: {type(obj)}, 转换为字符串")
            return str(obj)

    async def _filter_messages_with_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对话风格学习不需要筛选，直接返回所有消息"""

        # 对话风格学习不需要LLM筛选，直接学习所有原始消息
        logger.info(
            f"[LearningBatch] filter_mode=direct_raw_messages message_count={len(messages)} "
            "skip_llm_filter=True reason=style_learning_uses_raw_messages_without_llm_filter_by_design"
        )

        # 为每条消息添加默认的相关性评分
        for message in messages:
            message['relevance_score'] = 1.0 # 默认完全相关
            message['filter_reason'] = 'style_learning_no_filter'
            message['filter_detail'] = 'style_learning_uses_raw_messages_without_llm_filter_by_design'

        return messages

    async def _get_current_persona(self, group_id: str) -> Dict[str, Any]:
        """获取当前人格设置 (针对特定群组)"""
        try:
            # 通过 PersonaManagerService 获取当前人格
            persona = await self.persona_manager.get_current_persona(group_id)
            if persona:
                return persona

            # 如果没有特定群组的人格，尝试从框架获取默认人格
            if hasattr(self.context, 'persona_manager') and self.context.persona_manager:
                try:
                    default_persona = await self.context.persona_manager.get_default_persona_v3(self._resolve_umo(group_id))
                    if default_persona:
                        return {
                            'prompt': default_persona.get('prompt', '默认人格'),
                            'name': default_persona.get('name', 'default'),
                            'style_parameters': {},
                            'last_updated': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.warning(f"从框架获取默认人格失败: {e}")

            # 如果都失败，返回默认结构
            return {
                'prompt': "默认人格",
                'name': 'default',
                'style_parameters': {},
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取当前人格失败 for group {group_id}: {e}")
            return {'prompt': '默认人格', 'name': 'default', 'style_parameters': {}}

    async def _generate_updated_persona(self, group_id: str, current_persona: Dict[str, Any], style_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成更新后的人格 - 直接在原有文本后面追加增量学习内容"""
        try:
            # 使用新版框架API获取当前人格
            if not hasattr(self.context, 'persona_manager') or not self.context.persona_manager:
                logger.warning(f"无法获取PersonaManager for group {group_id}")
                return current_persona

            default_persona = await self.context.persona_manager.get_default_persona_v3(self._resolve_umo(group_id))
            if not default_persona:
                logger.warning(f"无法获取当前人格 for group {group_id}")
                return current_persona

            # 获取原有人格文本
            original_prompt = default_persona.get('prompt', '')

            # 构建增量学习内容
            learning_content = []

            # 正确处理AnalysisResult对象和字典类型
            from ...core.interfaces import AnalysisResult

            if isinstance(style_analysis, AnalysisResult):
                # 如果是AnalysisResult对象，提取data属性
                analysis_data = style_analysis.data if style_analysis.data else {}
                logger.debug(f"从AnalysisResult提取data: success={style_analysis.success}, confidence={style_analysis.confidence}")
            elif isinstance(style_analysis, dict):
                analysis_data = style_analysis
                logger.debug("使用字典形式的style_analysis")
            elif hasattr(style_analysis, 'data'):
                # 兼容其他具有data属性的对象
                analysis_data = style_analysis.data if style_analysis.data else {}
                logger.debug(f"从对象提取data属性: {type(style_analysis)}")
            else:
                analysis_data = {}
                logger.warning(f"style_analysis类型不正确: {type(style_analysis)}, 使用空字典")

            # 修复：从实际的 style_analysis 结构中提取内容
            # 优先提取 enhanced_prompt 和 learning_insights（如果有）
            if 'enhanced_prompt' in analysis_data:
                learning_content.append(analysis_data['enhanced_prompt'])
                logger.debug("找到 enhanced_prompt 字段")

            if 'learning_insights' in analysis_data:
                insights = analysis_data['learning_insights']
                if insights:
                    learning_content.append(insights)
                    logger.debug("找到 learning_insights 字段")

            # 新增：从 style_analysis 字段提取内容（StyleAnalyzer返回的结构）
            if not learning_content and 'style_analysis' in analysis_data:
                style_report = analysis_data['style_analysis']
                if isinstance(style_report, dict):
                    # 提取关键的风格分析内容
                    extracted_parts = []

                    # 提取文本风格描述
                    if 'text_style' in style_report:
                        extracted_parts.append(f"文本风格: {style_report['text_style']}")

                    # 提取表达特点
                    if 'expression_features' in style_report:
                        features = style_report['expression_features']
                        if isinstance(features, list):
                            extracted_parts.append(f"表达特点: {', '.join(features)}")
                        elif isinstance(features, str):
                            extracted_parts.append(f"表达特点: {features}")

                    # 提取语气倾向
                    if 'tone' in style_report:
                        extracted_parts.append(f"语气倾向: {style_report['tone']}")

                    # 提取话题偏好
                    if 'topics' in style_report:
                        topics = style_report['topics']
                        if isinstance(topics, list):
                            extracted_parts.append(f"话题偏好: {', '.join(topics)}")
                        elif isinstance(topics, str):
                            extracted_parts.append(f"话题偏好: {topics}")

                    if extracted_parts:
                        learning_content.append("【对话风格学习结果】\n" + "\n".join(extracted_parts))
                        logger.debug(f"从 style_analysis 提取了 {len(extracted_parts)} 个风格特征")

            # 新增：如果还是没有内容，从 style_profile 提取
            if not learning_content and 'style_profile' in analysis_data:
                style_profile = analysis_data['style_profile']
                if isinstance(style_profile, dict):
                    profile_parts = []

                    # 提取语气强度
                    if 'tone_intensity' in style_profile:
                        profile_parts.append(f"语气强度: {style_profile['tone_intensity']:.2f}")

                    # 提取情感倾向
                    if 'sentiment' in style_profile:
                        profile_parts.append(f"情感倾向: {style_profile['sentiment']:.2f}")

                    # 提取词汇丰富度
                    if 'vocabulary_richness' in style_profile:
                        profile_parts.append(f"词汇丰富度: {style_profile['vocabulary_richness']:.2f}")

                    if profile_parts:
                        learning_content.append("【风格量化指标】\n" + "\n".join(profile_parts))
                        logger.debug(f"从 style_profile 提取了 {len(profile_parts)} 个量化指标")

            # 新增：如果还是没有内容，尝试提取任何有用的信息
            if not learning_content:
                # 尝试从顶层提取任何看起来有用的字段
                useful_fields = ['summary', 'description', 'analysis', 'insights', 'findings']
                for field in useful_fields:
                    if field in analysis_data and analysis_data[field]:
                        learning_content.append(f"【{field}】\n{analysis_data[field]}")
                        logger.debug(f"从顶层字段 {field} 提取了内容")
                        break

            # 直接在原有文本后面追加新内容
            if learning_content:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                new_content = f"\n\n【学习更新 - {timestamp}】\n" + "\n".join(learning_content)

                # 创建更新后的人格 (Personality是TypedDict)
                updated_persona = dict(default_persona)
                updated_persona['prompt'] = original_prompt + new_content
                updated_persona['last_updated'] = timestamp

                logger.info(f" 成功追加 {len(learning_content)} 项学习内容到人格 for group {group_id}")
                return updated_persona
            else:
                logger.warning(f" style_analysis中没有可提取的学习内容 for group {group_id}, 数据结构: {list(analysis_data.keys())}")
                # 即使没有学习内容，也返回一个副本以确保有updated_persona用于对比
                return dict(default_persona)

        except Exception as e:
            logger.error(f"生成更新人格失败 for group {group_id}: {e}", exc_info=True)
            return current_persona

    async def _apply_learning_updates(self, group_id: str, style_analysis: Dict[str, Any], messages: List[Dict[str, Any]],
                                     current_persona: Dict[str, Any] = None, updated_persona: Dict[str, Any] = None,
                                     quality_metrics = None, relearn_mode: bool = False, ml_tuning_info: Dict[str, Any] = None):
        """应用学习更新，并创建人格学习审查记录和风格学习记录

        Args:
            group_id: 群组ID
            style_analysis: 风格分析结果
            messages: 处理的消息列表
            current_persona: 当前人格
            updated_persona: 更新后的人格
            quality_metrics: 质量指标
            relearn_mode: 重新学习模式，为True时即使内容相同也创建审查记录
            ml_tuning_info: 强化学习调优信息（包含是否使用保守融合策略等）
        """
        try:
            # 处理可能的list类型参数
            if isinstance(current_persona, list):
                logger.warning(f"current_persona为list类型(长度{len(current_persona)})，转换为空字典")
                current_persona = {}

            if isinstance(updated_persona, list):
                logger.warning(f"updated_persona为list类型(长度{len(updated_persona)})，转换为空字典")
                updated_persona = {}

            # 1. 保存对话风格学习记录（不需要审查，直接保存）
            await self._save_style_learning_record(group_id, style_analysis, messages, quality_metrics)

            # 2. 更新人格prompt（通过 PersonaManagerService）
            logger.info(f"应用人格更新 for group {group_id}")
            persona_update_applied = False
            persona_review_written = False
            persona_review_id = None

            # 正确处理 AnalysisResult 对象
            if hasattr(style_analysis, 'success'):
                # 这是一个 AnalysisResult 对象
                if not style_analysis.success:
                    logger.error(f"风格分析失败，跳过人格更新: {style_analysis.error}")
                    return {
                        "persona_update_applied": False,
                        "persona_review_written": False,
                        "persona_review_id": None,
                    }

                # 使用 AnalysisResult 的 data 属性
                style_analysis_dict = style_analysis.data
                confidence = style_analysis.confidence
                logger.debug(f"使用 AnalysisResult 对象，置信度: {confidence:.3f}")
            elif isinstance(style_analysis, dict):
                # 向后兼容：如果传入的是字典
                style_analysis_dict = style_analysis
                confidence = style_analysis.get('confidence', 0.5)
                logger.debug("使用字典形式的 style_analysis（向后兼容）")
            else:
                logger.error(f"style_analysis 类型不正确: {type(style_analysis)}")
                return {
                    "persona_update_applied": False,
                    "persona_review_written": False,
                    "persona_review_id": None,
                }

            update_success = await self.persona_manager.update_persona(group_id, style_analysis_dict, messages)
            persona_update_applied = bool(update_success)
            if not update_success:
                logger.error(f"通过 PersonaManagerService 更新人格失败 for group {group_id}")

            # 2. 创建人格学习审查记录（新增）
            # 重新学习模式：即使内容相同也创建审查记录（作为重新确认）
            # 正常模式：只在内容不同时创建审查记录
            should_create_review = False
            if relearn_mode:
                # 重新学习模式：总是创建审查记录
                should_create_review = bool(updated_persona and current_persona)
                if should_create_review:
                    # 检查是否有实质性变化
                    has_changes = updated_persona.get('prompt', '') != current_persona.get('prompt', '')
                    if has_changes:
                        logger.info(f" 重新学习模式：检测到人格变化，创建审查记录（group: {group_id}）")
                    else:
                        logger.info(f" 重新学习模式：未检测到人格变化，但仍创建审查记录供审核（group: {group_id}）")
                else:
                    logger.warning(f" 重新学习模式：无法创建审查记录 - updated_persona={bool(updated_persona)}, current_persona={bool(current_persona)}")
            elif updated_persona and current_persona and updated_persona.get('prompt') != current_persona.get('prompt'):
                # 正常模式：只在内容不同时创建
                should_create_review = True
                logger.info(f" 正常模式：检测到人格变化，创建审查记录（group: {group_id}）")
            else:
                logger.debug(f" 正常模式：人格未变化，跳过审查记录 - updated={bool(updated_persona)}, current={bool(current_persona)}, same_prompt={updated_persona.get('prompt') == current_persona.get('prompt') if updated_persona and current_persona else 'N/A'}")

            if should_create_review:
                try:
                    # 提取原人格和新人格的完整文本
                    original_prompt = current_persona.get('prompt', '')
                    new_prompt = updated_persona.get('prompt', '')

                    # 计算新增内容（用于单独标记）
                    if len(new_prompt) > len(original_prompt):
                        incremental_content = new_prompt[len(original_prompt):].strip()
                    else:
                        incremental_content = new_prompt

                    # 准备元数据（包含高亮信息）
                    metadata = {
                        "progressive_learning": True,
                        "message_count": len(messages),
                        "style_analysis_fields": list(style_analysis.data.keys()) if (hasattr(style_analysis, "data") and isinstance(style_analysis.data, dict)) else (list(style_analysis.keys()) if isinstance(style_analysis, dict) else []),
                        "original_prompt_length": len(original_prompt),
                        "new_prompt_length": len(new_prompt),
                        "incremental_content": incremental_content, # 单独记录增量内容，用于高亮
                        "incremental_start_pos": len(original_prompt), # 标记新增内容的起始位置
                        "relearn_mode": relearn_mode # 标记是否���重新学习模式
                    }

                    # 添加强化学习调优信息到元数据
                    if ml_tuning_info:
                        metadata['ml_tuning'] = ml_tuning_info

                    # 获取质量得分
                    confidence_score = quality_metrics.consistency_score if quality_metrics and hasattr(quality_metrics, 'consistency_score') else 0.5

                    # 构建 raw_analysis 说明（包含强化学习信息）
                    raw_analysis_parts = [f"基于{len(messages)}条消息的风格分析"]
                    if relearn_mode:
                        raw_analysis_parts.append("（重新学习）")
                    if ml_tuning_info and ml_tuning_info.get('applied'):
                        if ml_tuning_info.get('used_conservative_fusion'):
                            raw_analysis_parts.append(f"强化学习生成的prompt过短({ml_tuning_info['tuned_length']} vs {ml_tuning_info['original_length']})，采用保守融合策略")
                        else:
                            raw_analysis_parts.append(f"已应用强化学习优化，预期改进: {ml_tuning_info['expected_improvement']:.2%}")
                    raw_analysis = "；".join(raw_analysis_parts)

                    # 创建审查记录 - proposed_content 仅包含新增内容，审批时拼接 original_content
                    review_id = await self.db_manager.add_persona_learning_review(
                        group_id=group_id,
                        proposed_content=incremental_content, # 仅新增内容，不重复原始人格
                        learning_source=UPDATE_TYPE_PROGRESSIVE_PERSONA_LEARNING,
                        confidence_score=confidence_score,
                        raw_analysis=raw_analysis,
                        metadata=metadata,
                        original_content=original_prompt, # 原人格完整文本
                        new_content=new_prompt # 完整新人格文本（original + incremental），用于审批应用
                    )

                    persona_review_written = True
                    persona_review_id = review_id
                    logger.info(f" 已创建人格学习审查记录 (ID: {review_id})，置信度: {confidence_score:.3f}")

                except Exception as review_error:
                    logger.error(f"创建人格学习审查记录失败: {review_error}", exc_info=True)
            else:
                logger.debug(f"人格未变化或缺少必要参数，跳过审查记录创建")

            # 3. 记录学习更新
            if group_id in self._group_sessions:
                self._group_sessions[group_id].style_updates += 1

            return {
                "persona_update_applied": persona_update_applied,
                "persona_review_written": persona_review_written,
                "persona_review_id": persona_review_id,
            }

        except Exception as e:
            logger.error(f"应用学习更新失败 for group {group_id}: {e}")
            return {
                "persona_update_applied": False,
                "persona_review_written": False,
                "persona_review_id": None,
            }

    async def _mark_messages_processed(self, messages: List[Dict[str, Any]]):
        """标记消息为已处理"""
        message_ids = [msg['id'] for msg in messages if 'id' in msg]
        if message_ids:
            await self.message_collector.mark_messages_processed(message_ids)

    async def get_learning_status(self, group_id: str = None) -> Dict[str, Any]:
        """获取学习状态"""
        if group_id:
            # 获取特定群组的状态
            return {
                'learning_active': self.learning_active.get(group_id, False),
                'group_id': group_id,
                'current_session': self._group_sessions[group_id].__dict__ if group_id in self._group_sessions else None,
                'total_sessions': len(self.learning_sessions),
                'statistics': await self.message_collector.get_statistics(),
                'quality_report': await self.quality_monitor.get_quality_report(),
                'last_update': datetime.now().isoformat()
            }
        else:
            return {
                'learning_active_groups': {gid: active for gid, active in self.learning_active.items()},
                'active_groups_count': sum(1 for active in self.learning_active.values() if active),
                'group_sessions': {gid: s.__dict__ for gid, s in self._group_sessions.items()},
                'total_sessions': len(self.learning_sessions),
                'statistics': await self.message_collector.get_statistics(),
                'quality_report': await self.quality_monitor.get_quality_report(),
                'last_update': datetime.now().isoformat()
            }

    async def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        try:
            # 获取风格趋势
            style_trends = await self.style_analyzer.get_style_trends()
            
            # 获取用户分析（示例用户）
            user_insights = {}
            if self.multidimensional_analyzer.user_profiles:
                sample_user_id = list(self.multidimensional_analyzer.user_profiles.keys())
                user_insights = await self.multidimensional_analyzer.get_user_insights(sample_user_id)
            
            # 获取社交图谱
            social_graph = await self.multidimensional_analyzer.export_social_graph()
            
            return {
                'style_trends': style_trends,
                'user_insights_sample': user_insights,
                'social_graph_summary': {
                    'total_nodes': len(social_graph.get('nodes', [])),
                    'total_edges': len(social_graph.get('edges', [])),
                    'statistics': social_graph.get('statistics', {})
                },
                'learning_performance': {
                    'successful_sessions': len([s for s in self.learning_sessions if s.success]),
                    'average_quality_score': sum(s.quality_score for s in self.learning_sessions) / 
                                           max(len(self.learning_sessions), 1),
                    'total_messages_processed': sum(s.messages_processed for s in self.learning_sessions)
                }
            }
            
        except Exception as e:
            logger.error(f"获取学习洞察失败: {e}")
            return {"error": str(e)}

    async def stop(self):
        """停止服务"""
        try:
            await self.stop_learning() # 停止所有群组的学习
            logger.info("渐进式学习服务已停止")
            return True
        except Exception as e:
            logger.error(f"停止渐进式学习服务失败: {e}")
            return False

    async def _save_style_learning_record(self, group_id: str, style_analysis: Dict[str, Any],
                                         messages: List[Dict[str, Any]], quality_metrics=None):
        """
        保存对话风格学习记录（直接保存，不需要审查）

        Args:
            group_id: 群组ID
            style_analysis: 风格分析结果（可以为空，会基于消息创建简单记录）
            messages: 处理的消息列表
            quality_metrics: 质量指标
        """
        try:
            # 处理 AnalysisResult 对象，提取其 data 属性
            if style_analysis and hasattr(style_analysis, 'data'):
                style_analysis_dict = style_analysis.data
            elif isinstance(style_analysis, dict):
                style_analysis_dict = style_analysis
            else:
                style_analysis_dict = {}

            # 即使没有 style_analysis，也应该基于消息创建学习记录
            if not style_analysis_dict and not messages:
                logger.debug(f"群组 {group_id} 没有风格分析结果且没有消息，跳过风格学习记录保存")
                return

            # 1. 保存表达模式到 expression_patterns 表
            expression_patterns = style_analysis_dict.get('expression_patterns', [])

            # 在 fewshot 模式下，style_analysis 可能不包含 expression_patterns。
            # 此时从数据库获取 bot 消息与用户消息合并，提取 user->bot 对话对。
            if not expression_patterns and messages:
                try:
                    merged = await self._merge_bot_messages_for_pairs(group_id, messages)
                    if merged:
                        expression_patterns = self._extract_fewshot_pairs_from_merged(merged, group_id)
                except Exception as pair_err:
                    logger.debug(f"提取 fewshot 对话对失败: {pair_err}")

            if expression_patterns:
                await self._save_expression_patterns(group_id, expression_patterns)

            # 2. 构建 few_shots 内容（仅使用原始 A/B 对话对，不做 LLM 总结）
            few_shots_content = ''
            if expression_patterns:
                few_shots_content = self._build_few_shots_from_patterns(expression_patterns)

            # 如果没有 few_shots_content，从消息中构建简单的学习内容
            if not few_shots_content and messages:
                few_shots_content = f"基于 {len(messages)} 条对话消息的风格学习"

            # 3. 构建学习模式列表
            learned_patterns = []
            for pattern in expression_patterns[:10]: # 取前10个模式
                learned_patterns.append({
                    'situation': pattern.get('situation', ''),
                    'expression': pattern.get('expression', ''),
                    'weight': pattern.get('weight', 1.0),
                    'confidence': pattern.get('confidence', 0.8)
                })

            # 4. 获取质量得分
            confidence_score = quality_metrics.consistency_score if quality_metrics and hasattr(quality_metrics, 'consistency_score') else 0.75

            # 5. 构建描述
            pattern_count = len(learned_patterns) if learned_patterns else 0
            message_count = len(messages) if messages else 0
            description = f"群组 {group_id} 的对话风格学习结果（处理 {message_count} 条消息，提取 {pattern_count} 个表达模式）"

            # 6. 保存风格学习记录（使用 ORM）
            # 提取到有效对话对时设为 pending 等待审查，否则自动批准
            try:
                async with self.db_manager.get_session() as session:
                    from ...models.orm.learning import StyleLearningReview
                    from datetime import datetime

                    current_timestamp = time.time()
                    has_patterns = bool(learned_patterns)

                    review = StyleLearningReview(
                        type='对话风格学习',
                        group_id=group_id,
                        timestamp=current_timestamp,
                        learned_patterns=json.dumps(learned_patterns, ensure_ascii=False),
                        few_shots_content=few_shots_content,
                        status='pending' if has_patterns else 'approved',
                        description=description,
                        reviewer_comment=None if has_patterns else '自动批准（无有效对话对）',
                        review_time=None if has_patterns else current_timestamp,
                        created_at=datetime.fromtimestamp(current_timestamp),
                        updated_at=datetime.fromtimestamp(current_timestamp),
                    )

                    session.add(review)
                    await session.commit()
                    await session.refresh(review)

                    logger.info(f" 对话风格学习记录已保存 (ID: {review.id})，处理 {message_count} 条消息，提取 {pattern_count} 个模式")

            except Exception as e:
                logger.error(f"保存对话风格学习记录失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"保存风格学习记录失败: {e}", exc_info=True)

    def _build_few_shots_from_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """从表达模式构建 few-shots 内容"""
        few_shots = "*Here are few shots of dialogs, you need to imitate the tone of 'B' in the following dialogs to respond:\n"

        for i, pattern in enumerate(patterns[:5], 1): # 只取前5个
            situation = pattern.get('situation', '')
            expression = pattern.get('expression', '')
            if situation and expression:
                few_shots += f"A: {situation}\nB: {expression}\n\n"

        return few_shots.strip()

    async def _merge_bot_messages_for_pairs(
        self, group_id: str, user_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge user messages with bot messages from DB to form a timeline.

        Fetches recent bot messages for the group and interleaves them with
        user messages sorted by timestamp, producing a unified stream that
        allows user->bot pair extraction.
        """
        bot_texts = await self.db_manager.get_recent_bot_responses(group_id, limit=50)
        if not bot_texts:
            return []

        # bot_texts is List[str]; build dicts with bot sender_id
        bot_msgs = []
        # Retrieve full BotMessage records to get timestamps
        async with self.db_manager.get_session() as session:
            from ...models.orm.message import BotMessage
            from sqlalchemy import select, desc
            stmt = (
                select(BotMessage)
                .where(BotMessage.group_id == group_id)
                .order_by(desc(BotMessage.timestamp))
                .limit(50)
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                bot_msgs.append({
                    'sender_id': 'bot',
                    'message': row.message,
                    'timestamp': row.timestamp,
                })

        if not bot_msgs:
            return []

        merged = list(user_messages) + bot_msgs
        merged.sort(key=lambda m: m.get('timestamp', 0))
        return merged

    @staticmethod
    def _extract_fewshot_pairs_from_merged(
        merged: List[Dict[str, Any]], group_id: str
    ) -> List[Dict[str, Any]]:
        """Extract user->bot conversation pairs from a merged message timeline.

        Mirrors the logic of ExpressionPatternLearner._extract_few_shot_pairs
        but operates on plain dicts and returns expression pattern dicts.
        """
        pairs = []
        current_time = time.time()

        for i in range(len(merged) - 1):
            msg = merged[i]
            nxt = merged[i + 1]

            msg_is_bot = msg.get('sender_id') == 'bot'
            nxt_is_bot = nxt.get('sender_id') == 'bot'
            msg_text = msg.get('message', '').strip()
            nxt_text = nxt.get('message', '').strip()

            if not msg_is_bot and nxt_is_bot and msg_text and nxt_text:
                if len(msg_text) < 3 or len(nxt_text) < 3:
                    continue
                if msg_text.startswith(('[', 'http', '@')):
                    continue
                if nxt_text.startswith(('[', 'http', '@')):
                    continue
                if '@' in msg_text or '@' in nxt_text:
                    continue

                pairs.append({
                    'situation': msg_text[:50],
                    'expression': nxt_text[:100],
                    'weight': 1.0,
                    'confidence': 0.8,
                    'group_id': group_id,
                    'last_active_time': current_time,
                    'create_time': current_time,
                })

        return pairs

    async def _save_expression_patterns(self, group_id: str, patterns: List[Dict[str, Any]]):
        """
        保存表达模式到 expression_patterns 表

        Args:
            group_id: 群组ID
            patterns: 表达模式列表
        """
        try:
            if not patterns:
                return

            # 使用 ORM 批量保存表达模式
            async with self.db_manager.get_session() as session:
                from ...models.orm.expression import ExpressionPattern
                import time

                current_time = time.time()
                objects = []

                for pattern in patterns:
                    situation = pattern.get('situation', '').strip()
                    expression = pattern.get('expression', '').strip()

                    if not situation or not expression:
                        continue

                    objects.append(ExpressionPattern(
                        group_id=group_id,
                        situation=situation,
                        expression=expression,
                        weight=float(pattern.get('weight', 1.0)),
                        last_active_time=current_time,
                        create_time=current_time
                    ))

                if objects:
                    session.add_all(objects)
                    await session.commit()
                    logger.info(f"已保存 {len(objects)} 个表达模式到数据库 (群组: {group_id})")

        except Exception as e:
            logger.error(f"保存表达模式失败: {e}", exc_info=True)
