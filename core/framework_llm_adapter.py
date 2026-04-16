"""
AstrBot 框架 LLM 适配器
用于替换自定义 LLMClient，直接使用 AstrBot 框架的 Provider 系统
"""
import asyncio
import time
from typing import Optional, List, Dict, Any
from astrbot.api import logger
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.entities import LLMResponse

class FrameworkLLMAdapter:
    """AstrBot框架LLM适配器，用于替换自定义LLMClient"""
    
    def __init__(self, context):
        self.context = context
        self.filter_provider: Optional[Provider] = None
        self.refine_provider: Optional[Provider] = None
        self.reinforce_provider: Optional[Provider] = None
        self.providers_configured = 0
        self._needs_lazy_init = False
        self._config = None
        # 延迟初始化冷却: 避免在providers尚未就绪时高频重试
        self._last_lazy_init_attempt: float = 0
        self._lazy_init_cooldown: float = 30.0

        # 添加调用统计
        self.call_stats = {
            'filter': {'total_calls': 0, 'total_time': 0, 'errors': 0},
            'refine': {'total_calls': 0, 'total_time': 0, 'errors': 0},
            'reinforce': {'total_calls': 0, 'total_time': 0, 'errors': 0},
            'general': {'total_calls': 0, 'total_time': 0, 'errors': 0}
        }
        
    def _collect_available_chat_completion_providers(self, emit_logs: bool = True) -> List[Provider]:
        """Collect CHAT_COMPLETION providers currently visible in the registry."""
        from astrbot.core.provider.entities import ProviderType

        available_providers = []
        try:
            all_providers = self.context.get_all_providers()
            if emit_logs:
                logger.info(f" - discovered {len(all_providers)} providers")

            for provider in all_providers:
                try:
                    provider_meta = provider.meta()
                except Exception as meta_error:
                    if emit_logs:
                        logger.debug(f"skip provider with unreadable meta: {meta_error}")
                    continue

                if provider_meta.provider_type == ProviderType.CHAT_COMPLETION:
                    available_providers.append(provider)
                    if emit_logs:
                        logger.debug(
                            f" provider {provider_meta.id} available (type: {provider_meta.provider_type.value})"
                        )

            if emit_logs:
                logger.info(
                    f" discovered {len(available_providers)} CHAT_COMPLETION providers"
                )
        except Exception as e:
            logger.warning(f"failed to collect available providers: {e}")

        return available_providers

    def _select_provider_for_role(
        self,
        role_name: str,
        configured_provider_id: Optional[str],
        current_provider: Optional[Provider],
        available_providers: List[Provider],
        available_provider_map: Dict[str, Provider],
        used_provider_ids,
    ):
        """Resolve a single role, keeping existing bindings if the target is not yet visible."""

        current_provider_id = None
        if current_provider:
            try:
                current_provider_id = current_provider.meta().id
            except Exception:
                current_provider_id = None

        if configured_provider_id:
            if current_provider_id == configured_provider_id:
                if current_provider_id:
                    used_provider_ids.add(current_provider_id)
                return current_provider, True

            provider = available_provider_map.get(configured_provider_id)
            if provider is not None:
                provider_id = provider.meta().id
                used_provider_ids.add(provider_id)
                logger.info(f"{role_name} provider bound: {provider_id}")
                return provider, True

            if current_provider is not None:
                if current_provider_id:
                    used_provider_ids.add(current_provider_id)
                logger.warning(
                    f"{role_name} provider pending: {configured_provider_id} "
                    f"(current bound: {current_provider_id})"
                )
                return current_provider, False

            logger.warning(f"{role_name} provider pending: {configured_provider_id}")
            return None, False

        if current_provider is not None:
            if current_provider_id:
                used_provider_ids.add(current_provider_id)
            return current_provider, True

        for provider in available_providers:
            try:
                provider_id = provider.meta().id
            except Exception:
                continue
            if provider_id in used_provider_ids:
                continue
            used_provider_ids.add(provider_id)
            logger.info(f"{role_name} provider bound: {provider_id}")
            return provider, True

        if available_providers:
            provider = available_providers[0]
            provider_id = provider.meta().id
            used_provider_ids.add(provider_id)
            logger.info(f"{role_name} provider bound: {provider_id}")
            return provider, True

        return None, False

    def initialize_providers(self, config):
        """Initialize provider bindings from the AstrBot registry."""
        self._config = config

        logger.info("[LLM adapter] initialize provider bindings")
        logger.info(f" - filter_provider_id: {config.filter_provider_id}")
        logger.info(f" - refine_provider_id: {config.refine_provider_id}")
        logger.info(f" - reinforce_provider_id: {config.reinforce_provider_id}")

        has_configured_provider_ids = bool(
            config.filter_provider_id or config.refine_provider_id or config.reinforce_provider_id
        )

        available_providers = self._collect_available_chat_completion_providers(
            emit_logs=True
        )
        provider_registry_ready = len(available_providers) > 0

        if not provider_registry_ready:
            self._needs_lazy_init = True
            self.providers_configured = sum(
                1
                for provider in (self.filter_provider, self.refine_provider, self.reinforce_provider)
                if provider is not None
            )
            if has_configured_provider_ids:
                logger.warning(
                    "[LLM adapter] provider registry is not ready (0 visible providers); "
                    "keep pending and retry later"
                )
            else:
                logger.warning(
                    "[LLM adapter] no provider is currently available and no provider_id "
                    "is configured; keep pending and retry later"
                )
            return False

        available_provider_map = {}
        for provider in available_providers:
            try:
                available_provider_map[provider.meta().id] = provider
            except Exception:
                continue

        used_provider_ids = set()
        new_filter_provider, filter_resolved = self._select_provider_for_role(
            "filter",
            config.filter_provider_id,
            self.filter_provider,
            available_providers,
            available_provider_map,
            used_provider_ids,
        )
        new_refine_provider, refine_resolved = self._select_provider_for_role(
            "refine",
            config.refine_provider_id,
            self.refine_provider,
            available_providers,
            available_provider_map,
            used_provider_ids,
        )
        new_reinforce_provider, reinforce_resolved = self._select_provider_for_role(
            "reinforce",
            config.reinforce_provider_id,
            self.reinforce_provider,
            available_providers,
            available_provider_map,
            used_provider_ids,
        )

        self.filter_provider = new_filter_provider
        self.refine_provider = new_refine_provider
        self.reinforce_provider = new_reinforce_provider
        self.providers_configured = sum(
            1
            for provider in (self.filter_provider, self.refine_provider, self.reinforce_provider)
            if provider is not None
        )

        all_roles_ready = filter_resolved and refine_resolved and reinforce_resolved
        if has_configured_provider_ids and not all_roles_ready:
            self._needs_lazy_init = True
            logger.warning(
                "[LLM adapter] configured provider ids are still not fully visible; "
                "keep pending and retry binding"
            )
        else:
            self._needs_lazy_init = False if self.providers_configured > 0 else True

        if self.filter_provider:
            logger.info(f"filter provider bound: {self.filter_provider.meta().id}")
        if self.refine_provider:
            logger.info(f"refine provider bound: {self.refine_provider.meta().id}")
        if self.reinforce_provider:
            logger.info(f"reinforce provider bound: {self.reinforce_provider.meta().id}")

        if self.providers_configured == 0:
            logger.error(
                "No usable AI provider is currently bound. "
                "Please wait until AstrBot finishes loading the provider registry."
            )
        elif self.providers_configured < 3:
            logger.info(
                f"Bound {self.providers_configured}/3 AI providers; some advanced "
                "paths may stay in fallback mode until the rest become visible."
            )
        else:
            logger.info("All 3 AI providers are bound successfully.")

        config_summary = []
        if self.filter_provider:
            config_summary.append(f"filter: {self.filter_provider.meta().id}")
        if self.refine_provider:
            config_summary.append(f"refine: {self.refine_provider.meta().id}")
        if self.reinforce_provider:
            config_summary.append(f"reinforce: {self.reinforce_provider.meta().id}")

        if config_summary:
            logger.info(f" Provider binding summary: {' | '.join(config_summary)}")
        else:
            logger.warning(" All providers are still unbound; plugin features will be limited")

        return all_roles_ready

    def _try_lazy_init(self):
        """Attempt to lazily bind providers again when the registry becomes ready."""
        if not self._needs_lazy_init or not self._config:
            return

        now = time.time()
        registry_has_visible_providers = bool(
            self._collect_available_chat_completion_providers(emit_logs=False)
        )

        if not registry_has_visible_providers:
            if now - self._last_lazy_init_attempt < self._lazy_init_cooldown:
                return

        self._last_lazy_init_attempt = now
        logger.info("[LLM adapter] trying lazy provider rebinding...")
        try:
            if self.initialize_providers(self._config):
                self._needs_lazy_init = False
                logger.info(
                    f"[LLM adapter] lazy provider rebinding succeeded; "
                    f"{self.providers_configured} providers are bound"
                )
            elif self.providers_configured > 0:
                logger.warning(
                    "[LLM adapter] lazy rebinding finished with partial bindings; "
                    "keep retrying until all configured provider ids are visible"
                )
            else:
                logger.warning(
                    "[LLM adapter] lazy rebinding still found no bindable provider"
                )
        except Exception as e:
            logger.warning(f"[LLM adapter] lazy provider rebinding failed: {e}")

    async def wait_for_provider_binding(self, config, timeout: float = 120.0, initial_delay: float = 1.0, poll_interval: float = 1.0, max_poll_interval: float = 10.0) -> bool:
        """Poll until the requested providers become visible and binding succeeds."""
        self._config = config
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)

        start_time = time.time()
        interval = poll_interval
        while True:
            if self.initialize_providers(config):
                return True

            if time.time() - start_time >= timeout:
                logger.warning(
                    "[LLM adapter] provider-ready polling timed out; "
                    "keep current bindings and rely on later retries"
                )
                return False

            await asyncio.sleep(interval)
            interval = min(interval * 2, max_poll_interval)

    async def filter_chat_completion(

        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用筛选模型进行对话补全"""
        # 尝试延迟初始化
        self._try_lazy_init()

        # 确保 contexts 不为 None，避免 Provider 内部调用 len(None)
        if contexts is None:
            contexts = []

        if not self.filter_provider:
            logger.warning("筛选Provider未配置，尝试使用备选Provider或降级处理")
            # 尝试使用其他可用的Provider作为备选
            fallback_provider = self.refine_provider or self.reinforce_provider
            if fallback_provider:
                logger.info(f"使用备选Provider: {fallback_provider.meta().id}")
                try:
                    response = await fallback_provider.text_chat(
                        prompt=prompt,
                        contexts=contexts,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                    return response.completion_text if response else None
                except Exception as e:
                    logger.error(f"备选Provider调用失败: {e}")
                    return None
            else:
                logger.error("没有可用的Provider，无法执行筛选任务")
                return None
            
        try:
            start_time = time.time()
            self.call_stats['filter']['total_calls'] += 1
            
            logger.debug(f"调用筛选Provider: {self.filter_provider.meta().id}")
            response = await self.filter_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # 统计调用时间
            elapsed_time = time.time() - start_time
            self.call_stats['filter']['total_time'] += elapsed_time
            
            return response.completion_text if response else None
        except Exception as e:
            # 统计错误
            elapsed_time = time.time() - start_time
            self.call_stats['filter']['total_time'] += elapsed_time
            self.call_stats['filter']['errors'] += 1
            
            logger.error(f"筛选模型调用失败: {e}")
            return None
    
    async def refine_chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用提炼模型进行对话补全"""
        # 尝试延迟初始化
        self._try_lazy_init()

        # 确保 contexts 不为 None，避免 Provider 内部调用 len(None)
        if contexts is None:
            contexts = []

        if not self.refine_provider:
            logger.warning("提炼Provider未配置，尝试使用备选Provider或降级处理")
            # 尝试使用其他可用的Provider作为备选
            fallback_provider = self.filter_provider or self.reinforce_provider
            if fallback_provider:
                logger.info(f"使用备选Provider: {fallback_provider.meta().id}")
                try:
                    response = await fallback_provider.text_chat(
                        prompt=prompt,
                        contexts=contexts,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                    return response.completion_text if response else None
                except Exception as e:
                    logger.error(f"备选Provider调用失败: {e}")
                    return None
            else:
                logger.error("没有可用的Provider，无法执行提炼任务")
                return None
            
        try:
            start_time = time.time()
            self.call_stats['refine']['total_calls'] += 1
            
            logger.debug(f"调用提炼Provider: {self.refine_provider.meta().id}")
            response = await self.refine_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # 统计调用时间
            elapsed_time = time.time() - start_time
            self.call_stats['refine']['total_time'] += elapsed_time
            
            return response.completion_text if response else None
        except Exception as e:
            # 统计错误
            elapsed_time = time.time() - start_time
            self.call_stats['refine']['total_time'] += elapsed_time
            self.call_stats['refine']['errors'] += 1
            
            logger.error(f"提炼模型调用失败: {e}")
            return None
    
    async def reinforce_chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用强化模型进行对话补全"""
        # 尝试延迟初始化
        self._try_lazy_init()

        # 确保 contexts 不为 None，避免 Provider 内部调用 len(None)
        if contexts is None:
            contexts = []

        if not self.reinforce_provider:
            logger.warning("强化Provider未配置，尝试使用备选Provider或降级处理")
            # 尝试使用其他可用的Provider作为备选
            fallback_provider = self.refine_provider or self.filter_provider
            if fallback_provider:
                logger.info(f"使用备选Provider: {fallback_provider.meta().id}")
                try:
                    response = await fallback_provider.text_chat(
                        prompt=prompt,
                        contexts=contexts,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                    return response.completion_text if response else None
                except Exception as e:
                    logger.error(f"备选Provider调用失败: {e}")
                    return None
            else:
                logger.error("没有可用的Provider，无法执行强化任务")
                return None
            
        try:
            start_time = time.time()
            self.call_stats['reinforce']['total_calls'] += 1
            
            logger.debug(f"调用强化Provider: {self.reinforce_provider.meta().id}")
            response = await self.reinforce_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # 统计调用时间
            elapsed_time = time.time() - start_time
            self.call_stats['reinforce']['total_time'] += elapsed_time
            
            return response.completion_text if response else None
        except Exception as e:
            # 统计错误
            elapsed_time = time.time() - start_time
            self.call_stats['reinforce']['total_time'] += elapsed_time
            self.call_stats['reinforce']['errors'] += 1
            
            logger.error(f"强化模型调用失败: {e}")
            return None

    def get_call_statistics(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        stats = {}
        total_calls = 0
        total_time = 0
        total_errors = 0
        
        for provider_type, data in self.call_stats.items():
            calls = data['total_calls']
            time_spent = data['total_time']
            errors = data['errors']
            
            total_calls += calls
            total_time += time_spent
            total_errors += errors
            
            avg_time = (time_spent / calls * 1000) if calls > 0 else 0
            success_rate = ((calls - errors) / calls) if calls > 0 else 1.0
            
            stats[provider_type] = {
                'total_calls': calls,
                'avg_response_time_ms': round(avg_time, 2),
                'success_rate': success_rate,
                'error_count': errors
            }
        
        # 添加总体统计
        overall_avg_time = (total_time / total_calls * 1000) if total_calls > 0 else 0
        overall_success_rate = ((total_calls - total_errors) / total_calls) if total_calls > 0 else 1.0
        
        stats['overall'] = {
            'total_calls': total_calls,
            'avg_response_time_ms': round(overall_avg_time, 2),
            'success_rate': overall_success_rate,
            'error_count': total_errors
        }
        
        return stats

    def has_filter_provider(self) -> bool:
        """检查是否有筛选Provider"""
        return self.filter_provider is not None
    
    def has_refine_provider(self) -> bool:
        """检查是否有提炼Provider"""
        return self.refine_provider is not None
    
    def has_reinforce_provider(self) -> bool:
        """检查是否有强化Provider"""
        return self.reinforce_provider is not None

    def get_provider_info(self) -> Dict[str, str]:
        """获取Provider信息"""
        info = {}
        if self.filter_provider:
            info['filter'] = f"{self.filter_provider.meta().id} ({self.filter_provider.meta().model})"
        if self.refine_provider:
            info['refine'] = f"{self.refine_provider.meta().id} ({self.refine_provider.meta().model})"
        if self.reinforce_provider:
            info['reinforce'] = f"{self.reinforce_provider.meta().id} ({self.reinforce_provider.meta().model})"
        return info

    async def generate_response(self, prompt: str, temperature: float = 0.7, model_type: str = "filter") -> Optional[str]:
        """
        通用的生成响应方法，根据model_type调用对应的Provider

        Args:
            prompt: 提示词
            temperature: 温度参数
            model_type: 模型类型 ("filter", "refine", "reinforce", "general")

        Returns:
            LLM响应文本，如果失败返回None
        """
        try:
            if model_type == "filter":
                return await self.filter_chat_completion(prompt=prompt, temperature=temperature)
            elif model_type == "refine":
                return await self.refine_chat_completion(prompt=prompt, temperature=temperature)
            elif model_type == "reinforce":
                return await self.reinforce_chat_completion(prompt=prompt, temperature=temperature)
            elif model_type == "general":
                return await self.filter_chat_completion(prompt=prompt, temperature=temperature)
            else:
                logger.error(f"不支持的模型类型: {model_type}")
                return None
        except Exception as e:
            logger.error(f"generate_response调用失败: {e}")
            return None
