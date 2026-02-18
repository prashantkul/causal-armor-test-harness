"""Bridge AgentDojo tools to LangChain StructuredTool instances.

AgentDojo tools use ``Depends``-annotated parameters to receive the task
environment.  This module wraps them as LangChain ``StructuredTool`` objects
that delegate to an :class:`EnvironmentHandle`, ensuring environment mutations
propagate to the utility / security checkers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import Function, FunctionCall, FunctionsRuntime, TaskEnvironment
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentHandle:
    """Mutable wrapper shared by all tool wrappers in a scenario.

    Holds the ``FunctionsRuntime``, the live ``TaskEnvironment``, and a
    call trace so environment mutations propagate to AgentDojo's
    utility / security checkers.
    """

    runtime: FunctionsRuntime
    env: TaskEnvironment
    call_trace: list[FunctionCall] = field(default_factory=list)


def _build_args_schema(func: Function) -> type[BaseModel]:
    """Return the Pydantic model describing user-facing parameters.

    AgentDojo ``Function.parameters`` already is a Pydantic model.  We
    re-export it directly since LangChain's ``StructuredTool`` expects a
    ``type[BaseModel]`` as ``args_schema``.
    """
    return func.parameters


def agentdojo_to_langchain_tool(
    func: Function,
    handle: EnvironmentHandle,
) -> StructuredTool:
    """Wrap a single AgentDojo ``Function`` as a LangChain ``StructuredTool``.

    The returned tool delegates to ``handle.runtime.run_function`` so that
    environment mutations are visible to subsequent AgentDojo checkers.
    """

    def _invoke(**kwargs: Any) -> str:
        result, error = handle.runtime.run_function(
            handle.env, func.name, kwargs
        )
        # Record the call for trace-based utility/security checks
        handle.call_trace.append(
            FunctionCall(function=func.name, args=kwargs)
        )
        if error:
            return f"Error: {error}"
        return tool_result_to_str(result)

    return StructuredTool(
        name=func.name,
        description=func.description,
        args_schema=_build_args_schema(func),
        func=_invoke,
    )


def bridge_suite_tools(
    functions: list[Function],
    handle: EnvironmentHandle,
) -> list[StructuredTool]:
    """Bridge all AgentDojo tools in a suite to LangChain ``StructuredTool``."""
    return [agentdojo_to_langchain_tool(f, handle) for f in functions]
