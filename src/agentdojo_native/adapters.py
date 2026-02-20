"""Converters between AgentDojo message types and CausalArmor types."""

from __future__ import annotations

import json
from collections.abc import Sequence

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import (
    ChatMessage,
    get_text_content_as_str,
)

from causal_armor import Message, MessageRole, ToolCall


def agentdojo_messages_to_ca(messages: Sequence[ChatMessage]) -> list[Message]:
    """Convert a sequence of AgentDojo messages to CausalArmor Messages."""
    result: list[Message] = []
    for msg in messages:
        role_str = msg["role"]

        if role_str == "system":
            content = get_text_content_as_str(msg["content"])
            result.append(Message(role=MessageRole.SYSTEM, content=content))

        elif role_str == "user":
            content = get_text_content_as_str(msg["content"])
            result.append(Message(role=MessageRole.USER, content=content))

        elif role_str == "assistant":
            content_blocks = msg["content"]
            content = (
                get_text_content_as_str(content_blocks)
                if content_blocks is not None
                else ""
            )
            result.append(Message(role=MessageRole.ASSISTANT, content=content))

        elif role_str == "tool":
            content = get_text_content_as_str(msg["content"])
            tool_call = msg["tool_call"]
            tool_name = tool_call.function if tool_call is not None else None
            tool_call_id = msg.get("tool_call_id")
            result.append(
                Message(
                    role=MessageRole.TOOL,
                    content=content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )

    return result


def functioncall_to_ca_toolcall(fc: FunctionCall) -> ToolCall:
    """Convert an AgentDojo FunctionCall to a CausalArmor ToolCall."""
    args = dict(fc.args)
    return ToolCall(
        name=fc.function,
        arguments=args,
        raw_text=json.dumps({"name": fc.function, "arguments": args}),
    )


def ca_toolcall_to_functioncall(tc: ToolCall) -> FunctionCall:
    """Convert a CausalArmor ToolCall back to an AgentDojo FunctionCall."""
    return FunctionCall(
        function=tc.name,
        args=dict(tc.arguments),
        id=f"call_{tc.name}",
    )
