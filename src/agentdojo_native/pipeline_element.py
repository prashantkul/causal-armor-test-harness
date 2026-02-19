"""CausalArmor pipeline element for native AgentDojo integration.

Implements :class:`BasePipelineElement` to intercept assistant tool calls,
run the CausalArmor defense pipeline, and replace/block defended calls
before ``ToolsExecutor`` executes them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from causal_armor import (
    CausalArmorConfig,
    CausalArmorMiddleware,
    DefenseResult,
)
from causal_armor.providers.gemini import GeminiActionProvider, GeminiSanitizerProvider
from causal_armor.providers.vllm import VLLMProxyProvider

from .adapters import agentdojo_messages_to_ca, ca_toolcall_to_functioncall, functioncall_to_ca_toolcall
from .metrics import GuardMetrics

logger = logging.getLogger(__name__)


class CausalArmorPipelineElement(BasePipelineElement):
    """AgentDojo pipeline element that runs CausalArmor on proposed tool calls.

    Placed inside a ``ToolsExecutionLoop`` *before* ``ToolsExecutor`` so
    that defended actions are rewritten before execution.

    Parameters
    ----------
    gemini_tool_declarations:
        Gemini ``function_declarations`` for the action regenerator.
    untrusted_tool_names:
        Tools whose results may contain prompt injections.
    guard_enabled:
        When *False*, all tool calls pass through unchanged.
    config:
        Optional :class:`CausalArmorConfig` override.
    """

    def __init__(
        self,
        gemini_tool_declarations: list[dict[str, Any]],
        untrusted_tool_names: frozenset[str],
        guard_enabled: bool = True,
        config: CausalArmorConfig | None = None,
    ) -> None:
        self._tool_declarations = gemini_tool_declarations
        self._untrusted_tool_names = untrusted_tool_names
        self._guard_enabled = guard_enabled
        self._config = config
        self.metrics: list[GuardMetrics] = []

    def _build_middleware(self) -> CausalArmorMiddleware:
        cfg = self._config or CausalArmorConfig.from_env()
        return CausalArmorMiddleware(
            action_provider=GeminiActionProvider(tools=self._tool_declarations),
            proxy_provider=VLLMProxyProvider(),
            sanitizer_provider=GeminiSanitizerProvider(),
            config=cfg,
        )

    async def _run_guard(
        self,
        ca_messages: list,
        tool_calls: list[FunctionCall],
    ) -> dict:
        """Run guard on all tool calls within a single event loop."""
        middleware = self._build_middleware()

        defended_tool_calls: list[FunctionCall] = []
        blocked_results: list[ChatToolResultMessage] = []
        results: list[DefenseResult] = []

        try:
            for tc in tool_calls:
                ca_tc = functioncall_to_ca_toolcall(tc)

                t0 = time.monotonic()
                result = await middleware.guard(
                    ca_messages,
                    ca_tc,
                    untrusted_tool_names=self._untrusted_tool_names,
                )
                elapsed = time.monotonic() - t0

                # Collect metrics
                metric = GuardMetrics(
                    tool_name=tc.function,
                    was_defended=result.was_defended,
                    is_attack_detected=(
                        result.detection.is_attack_detected
                        if result.detection
                        else False
                    ),
                    latency_seconds=elapsed,
                )
                if result.detection:
                    attr = result.detection.attribution
                    metric.delta_user_normalized = attr.delta_user_normalized
                    metric.span_attributions = dict(
                        attr.span_attributions_normalized
                    )
                self.metrics.append(metric)

                results.append(result)

                if not result.was_defended:
                    defended_tool_calls.append(tc)
                elif result.regenerated:
                    defended_tool_calls.append(
                        ca_toolcall_to_functioncall(result.final_action)
                    )
                else:
                    blocked_results.append(
                        ChatToolResultMessage(
                            role="tool",
                            content=[
                                text_content_block_from_string(
                                    f"Tool call to '{tc.function}' was blocked "
                                    "by the security guard. The requested action "
                                    "was deemed unsafe."
                                )
                            ],
                            tool_call_id=tc.id,
                            tool_call=tc,
                            error="Blocked by CausalArmor security guard.",
                        )
                    )
        finally:
            await middleware.close()

        return {
            "defended": defended_tool_calls,
            "blocked": blocked_results,
            "results": results,
        }

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args

        last_msg = messages[-1]

        # Only intercept assistant messages with tool calls.
        if last_msg["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if last_msg["tool_calls"] is None or len(last_msg["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        if not self._guard_enabled:
            logger.info("[GUARD] CausalArmor DISABLED — passing through")
            return query, runtime, env, messages, extra_args

        logger.info("[GUARD] CausalArmor ENABLED — running LOO attribution")

        ca_messages = agentdojo_messages_to_ca(messages)

        tool_msgs = [m for m in ca_messages if m.role.value == "tool"]
        logger.info(
            "[GUARD] Context has %d messages, %d tool results",
            len(ca_messages),
            len(tool_msgs),
        )

        # Run all guard() calls + close() in a single event loop to avoid
        # httpx connection issues across separate asyncio.run() calls.
        guard_output = asyncio.run(
            self._run_guard(ca_messages, last_msg["tool_calls"])
        )
        defended_tool_calls = guard_output["defended"]
        blocked_results = guard_output["blocked"]
        results = guard_output["results"]

        # Build the replacement assistant message with only non-blocked calls.
        if defended_tool_calls:
            new_msg: ChatAssistantMessage = ChatAssistantMessage(
                role="assistant",
                content=last_msg["content"],
                tool_calls=defended_tool_calls,
            )
            new_messages: list[ChatMessage] = [*messages[:-1], new_msg, *blocked_results]
        elif blocked_results:
            # ALL calls were blocked — keep the assistant message but with
            # empty tool_calls, and add the blocked results.
            new_msg = ChatAssistantMessage(
                role="assistant",
                content=last_msg["content"],
                tool_calls=[],
            )
            new_messages = [*messages[:-1], new_msg, *blocked_results]
        else:
            new_messages = list(messages)

        # Log defense summary
        for result in results:
            orig = result.original_action
            final = result.final_action
            if result.was_defended:
                logger.info("[GUARD] BLOCKED: %s(%s)", orig.name, orig.arguments)
                if result.regenerated:
                    logger.info(
                        "[GUARD] REPLACED WITH: %s(%s)",
                        final.name,
                        final.arguments,
                    )
            else:
                logger.info("[GUARD] PASSED: %s(%s)", orig.name, orig.arguments)

        return query, runtime, env, new_messages, extra_args
