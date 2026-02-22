"""Convert AgentDojo Function schemas to provider-specific tool declarations.

Reuses AgentDojo's own ``resolve_refs()`` and
``remove_additional_properties_recursively()`` for schema normalization.
"""

from __future__ import annotations

from typing import Any

from agentdojo.agent_pipeline.llms.google_llm import (
    remove_additional_properties_recursively,
    resolve_refs,
)
from agentdojo.functions_runtime import Function


def functions_to_gemini_declarations(
    functions: list[Function],
) -> list[dict[str, Any]]:
    """Convert AgentDojo Functions to Gemini ``function_declarations`` format.

    Returns ``[{"function_declarations": [...]}]`` — the structure expected
    by :class:`causal_armor.providers.gemini.GeminiActionProvider`.
    """
    declarations: list[dict[str, Any]] = []

    for func in functions:
        schema = resolve_refs(func.parameters)
        remove_additional_properties_recursively(schema)

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        declaration: dict[str, Any] = {
            "name": func.name,
            "description": func.description or func.name,
        }

        if properties:
            declaration["parameters"] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                declaration["parameters"]["required"] = required

        declarations.append(declaration)

    return [{"function_declarations": declarations}]


def functions_to_openai_tools(
    functions: list[Function],
) -> list[dict[str, Any]]:
    """Convert AgentDojo Functions to OpenAI ``tools`` format.

    Returns ``[{"type": "function", "function": {"name", "description",
    "parameters"}}]`` — the structure expected by
    :class:`causal_armor.providers.openai.OpenAIActionProvider`.
    """
    tools: list[dict[str, Any]] = []

    for func in functions:
        schema = resolve_refs(func.parameters)
        remove_additional_properties_recursively(schema)

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        tools.append({
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description or func.name,
                "parameters": parameters,
            },
        })

    return tools


def functions_to_anthropic_tools(
    functions: list[Function],
) -> list[dict[str, Any]]:
    """Convert AgentDojo Functions to Anthropic ``tools`` format.

    Returns ``[{"name", "description", "input_schema"}]`` — the structure
    expected by
    :class:`causal_armor.providers.anthropic.AnthropicActionProvider`.
    """
    tools: list[dict[str, Any]] = []

    for func in functions:
        schema = resolve_refs(func.parameters)
        remove_additional_properties_recursively(schema)

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            input_schema["required"] = required

        tools.append({
            "name": func.name,
            "description": func.description or func.name,
            "input_schema": input_schema,
        })

    return tools
