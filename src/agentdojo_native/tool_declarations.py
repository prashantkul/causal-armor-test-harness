"""Convert AgentDojo Function schemas to Gemini function declarations.

Reuses AgentDojo's own ``resolve_refs()`` and
``remove_additional_properties_recursively()`` — these are the correct
schema transforms for the Gemini API.
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
