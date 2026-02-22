"""Stub pipeline for AgentDojo attacks that need a pipeline name.

AgentDojo attacks like ``important_instructions`` extract the model name
from ``pipeline.name`` to personalize injection strings.  Since we use
our own LangGraph agent instead of AgentDojo's pipeline, we provide
this minimal stub.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


class FakePipeline(BasePipelineElement):
    """Minimal pipeline stub that only provides a ``name``."""

    # AgentDojo MODEL_NAMES maps specific model IDs to prose names.
    # Map common short names to a recognized full ID.
    _MODEL_ALIASES: dict[str, str] = {
        "gemini-2.0-flash": "gemini-2.0-flash-001",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
        "gemini-1.5-flash": "gemini-1.5-flash-002",
        "gemini-1.5-pro": "gemini-1.5-pro-002",
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
