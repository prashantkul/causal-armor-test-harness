"""Metrics aggregation for benchmark results.

Computes per-suite and overall metrics from :class:`SuiteRunResult` objects:
- **ASR** (Attack Success Rate): fraction of scenarios where injection succeeded
- **Detection rate**: fraction of attack scenarios where the guard detected an attack
- **FPR** (False Positive Rate): fraction of benign tool calls the guard wrongly blocked
- **Utility rate**: fraction of scenarios where the user task was completed
- **Guard latency**: average / max time spent in the guard node
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.runner import ScenarioResult, SuiteRunResult


@dataclass
class SuiteMetrics:
    """Aggregated metrics for one suite run."""

    suite_name: str
    total_scenarios: int = 0
    utility_count: int = 0
    attack_success_count: int = 0
    detection_count: int = 0
    total_guard_calls: int = 0
    false_positive_count: int = 0
    guard_latencies: list[float] = field(default_factory=list)
    error_count: int = 0

    @property
    def utility_rate(self) -> float:
        return self.utility_count / self.total_scenarios if self.total_scenarios else 0.0

    @property
    def asr(self) -> float:
        """Attack Success Rate (lower is better for defense)."""
        return (
            self.attack_success_count / self.total_scenarios
            if self.total_scenarios
            else 0.0
        )

    @property
    def detection_rate(self) -> float:
        return (
            self.detection_count / self.total_scenarios
            if self.total_scenarios
            else 0.0
        )

    @property
    def fpr(self) -> float:
        """False positive rate: benign calls wrongly defended."""
        if self.total_guard_calls == 0:
            return 0.0
        return self.false_positive_count / self.total_guard_calls

    @property
    def avg_guard_latency(self) -> float:
        return (
            sum(self.guard_latencies) / len(self.guard_latencies)
            if self.guard_latencies
            else 0.0
        )

    @property
    def max_guard_latency(self) -> float:
        return max(self.guard_latencies) if self.guard_latencies else 0.0


def compute_suite_metrics(suite_result: SuiteRunResult) -> SuiteMetrics:
    """Compute aggregated metrics from a suite run."""
    m = SuiteMetrics(suite_name=suite_result.suite_name)

    for s in suite_result.scenarios:
        m.total_scenarios += 1

        if s.error:
            m.error_count += 1
            continue

        if s.utility:
            m.utility_count += 1

        if not s.security:
            m.attack_success_count += 1

        # Guard-level metrics
        for gm in s.guard_metrics:
            m.total_guard_calls += 1
            m.guard_latencies.append(gm.latency_seconds)

            if gm.is_attack_detected:
                m.detection_count += 1

            # False positive: guard defended but no attack was actually detected
            # as a proper attack (heuristic: defended on a call where
            # delta_user_normalized is very low, suggesting benign)
            if gm.was_defended and not gm.is_attack_detected:
                m.false_positive_count += 1

    return m


def compute_all_metrics(
    suite_results: list[SuiteRunResult],
) -> list[SuiteMetrics]:
    """Compute metrics for each suite."""
    return [compute_suite_metrics(sr) for sr in suite_results]
