"""Reporting utilities — Rich console tables and JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness.metrics import SuiteMetrics, compute_all_metrics
from harness.runner import SuiteRunResult

console = Console()


def print_suite_table(metrics_list: list[SuiteMetrics]) -> None:
    """Print a Rich table summarising per-suite metrics."""
    table = Table(
        title="CausalArmor Benchmark Results",
        show_lines=True,
    )
    table.add_column("Suite", style="bold cyan")
    table.add_column("Scenarios", justify="right")
    table.add_column("Utility", justify="right")
    table.add_column("ASR", justify="right")
    table.add_column("Detection", justify="right")
    table.add_column("FPR", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Max Latency", justify="right")
    table.add_column("Errors", justify="right")

    for m in metrics_list:
        asr_style = "green" if m.asr < 0.1 else "red"
        table.add_row(
            m.suite_name,
            str(m.total_scenarios),
            f"{m.utility_rate:.1%}",
            f"[{asr_style}]{m.asr:.1%}[/{asr_style}]",
            f"{m.detection_rate:.1%}",
            f"{m.fpr:.1%}",
            f"{m.avg_guard_latency:.1f}s",
            f"{m.max_guard_latency:.1f}s",
            str(m.error_count),
        )

    # Totals row
    if len(metrics_list) > 1:
        total_scenarios = sum(m.total_scenarios for m in metrics_list)
        total_utility = sum(m.utility_count for m in metrics_list)
        total_attacks = sum(m.attack_success_count for m in metrics_list)
        all_latencies = [
            lat for m in metrics_list for lat in m.guard_latencies
        ]
        total_errors = sum(m.error_count for m in metrics_list)

        overall_utility = total_utility / total_scenarios if total_scenarios else 0
        overall_asr = total_attacks / total_scenarios if total_scenarios else 0
        avg_lat = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        max_lat = max(all_latencies) if all_latencies else 0
        asr_style = "green" if overall_asr < 0.1 else "red"

        table.add_row(
            "[bold]TOTAL[/bold]",
            str(total_scenarios),
            f"{overall_utility:.1%}",
            f"[{asr_style}]{overall_asr:.1%}[/{asr_style}]",
            "—",
            "—",
            f"{avg_lat:.1f}s",
            f"{max_lat:.1f}s",
            str(total_errors),
            style="bold",
        )

    console.print(table)


def export_json(
    suite_results: list[SuiteRunResult],
    output_path: Path,
) -> None:
    """Write full scenario-level results to a JSON file."""
    data = []
    for sr in suite_results:
        suite_data = {
            "suite_name": sr.suite_name,
            "attack_name": sr.attack_name,
            "guard_enabled": sr.guard_enabled,
            "scenarios": [],
        }
        for s in sr.scenarios:
            scenario_data = {
                "user_task_id": s.user_task_id,
                "injection_task_id": s.injection_task_id,
                "attack_name": s.attack_name,
                "utility": s.utility,
                "security": s.security,
                "total_seconds": round(s.total_seconds, 2),
                "error": s.error,
                "guard_metrics": [
                    {
                        "tool_name": gm.tool_name,
                        "was_defended": gm.was_defended,
                        "is_attack_detected": gm.is_attack_detected,
                        "latency_seconds": round(gm.latency_seconds, 3),
                        "delta_user_normalized": (
                            round(gm.delta_user_normalized, 4)
                            if gm.delta_user_normalized is not None
                            else None
                        ),
                        "span_attributions": {
                            k: round(v, 4)
                            for k, v in gm.span_attributions.items()
                        },
                    }
                    for gm in s.guard_metrics
                ],
            }
            suite_data["scenarios"].append(scenario_data)
        data.append(suite_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"\nResults written to [bold]{output_path}[/bold]")


def report(
    suite_results: list[SuiteRunResult],
    output_path: Path | None = None,
) -> None:
    """Print console table and optionally export JSON."""
    metrics_list = compute_all_metrics(suite_results)
    print_suite_table(metrics_list)

    if output_path:
        export_json(suite_results, output_path)
