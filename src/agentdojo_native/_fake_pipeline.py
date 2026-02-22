"""Stub pipeline for AgentDojo attacks that need a pipeline name.

AgentDojo attacks like ``important_instructions`` extract the model name
from ``pipeline.name`` to personalize injection strings.  Since we build
our own ``AgentPipeline`` instead of using ``AgentPipeline.from_config``,
we provide this minimal stub for ``load_attack()``.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


class FakePipeline(BasePipelineElement):
    """Minimal pipeline stub that only provides a ``name``."""

    _MODEL_ALIASES: dict[str, str] = {
        # Gemini
        "gemini-2.0-flash": "gemini-2.0-flash-001",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
        "gemini-1.5-flash": "gemini-1.5-flash-002",
        "gemini-1.5-pro": "gemini-1.5-pro-002",
        # OpenAI
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4.1-2025-04-14": "gpt-4o-2024-05-13",
        # Anthropic
        "claude-sonnet-4-20250514": "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-5-20250929": "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-latest": "claude-3-7-sonnet-20250219",
    }

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.name = self._MODEL_ALIASES.get(model_name, model_name)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        raise NotImplementedError("FakePipeline is a stub for attack loading only")
