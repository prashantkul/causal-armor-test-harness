"""Per-call guard metrics for benchmark result tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardMetrics:
    """Per-call metrics collected by the CausalArmor pipeline element."""

    tool_name: str
    was_defended: bool
    is_attack_detected: bool
    latency_seconds: float
    delta_user_normalized: float | None = None
    span_attributions: dict[str, float] = field(default_factory=dict)
