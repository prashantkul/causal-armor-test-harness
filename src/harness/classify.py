"""Auto-classify untrusted tools via canary injection.

Runs each user task's ground-truth pipeline with canary strings in
injection placeholders.  Any tool whose output contains a canary is
untrusted (its results flow through an injection vector).

This is fast (no LLM calls), deterministic, and auto-adapts to new
AgentDojo versions without manual maintenance.
"""

from __future__ import annotations

import logging

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import get_text_content_as_str

logger = logging.getLogger(__name__)

_CANARY_PREFIX = "___CAUSAL_ARMOR_CANARY___"


def classify_untrusted_tools(suite: TaskSuite) -> frozenset[str]:
    """Identify which tools in *suite* produce untrusted output.

    For each injection vector, inserts a unique canary string.  Then
    runs every user task's ground-truth pipeline and checks which
    tool-result messages contain a canary.  Tools that surface a canary
    are classified as untrusted.

    Returns
    -------
    frozenset[str]
        Tool names whose output may contain injected content.
    """
    injection_defaults = suite.get_injection_vector_defaults()
    canary_map = {
        vec_id: f"{_CANARY_PREFIX}{vec_id}___"
        for vec_id in injection_defaults
    }

    environment = suite.load_and_inject_default_environment(canary_map)
    canary_values = set(canary_map.values())

    untrusted: set[str] = set()

    for user_task in suite.user_tasks.values():
        gt_pipeline = GroundTruthPipeline(user_task)
        runtime = FunctionsRuntime(suite.tools)
        env_copy = environment.model_copy(deep=True)

        try:
            _, _, _, messages, _ = gt_pipeline.query(
                user_task.PROMPT, runtime, env_copy
            )
        except Exception:
            logger.warning(
                "Ground-truth pipeline failed for %s — skipping",
                user_task.ID,
            )
            continue

        for msg in messages:
            if msg["role"] != "tool":
                continue
            content = get_text_content_as_str(msg["content"])
            for canary in canary_values:
                if canary in content:
                    tool_call = msg.get("tool_call")
                    if tool_call is not None:
                        untrusted.add(tool_call.function)
                        logger.debug(
                            "Tool %s classified as untrusted "
                            "(canary found via %s)",
                            tool_call.function,
                            user_task.ID,
                        )

    logger.info(
        "Suite %s: %d/%d tools classified as untrusted: %s",
        suite.name,
        len(untrusted),
        len(suite.tools),
        sorted(untrusted),
    )
    return frozenset(untrusted)
