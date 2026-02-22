"""Generate Pareto frontier and distribution plots from benchmark results.

Usage
-----
::

    python scripts/plot_results.py \\
        --baseline results/all_baseline.json \\
        --guarded  results/all_guarded.json \\
        --output   results/plots/

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def extract_suite_metrics(data: list[dict]) -> dict[str, dict]:
    """Extract per-suite utility/ASR/latency from raw JSON results."""
    metrics = {}
    for suite in data:
        name = suite["suite_name"]
        scenarios = suite["scenarios"]
        total = len(scenarios)
        errors = sum(1 for s in scenarios if s["error"])
        valid = total - errors

        utility_count = sum(1 for s in scenarios if s["utility"] and not s["error"])
        attack_count = sum(1 for s in scenarios if not s["security"] and not s["error"])

        latencies = []
        detections = 0
        for s in scenarios:
            if s["error"]:
                continue
            scenario_detected = False
            for gm in s["guard_metrics"]:
                latencies.append(gm["latency_seconds"])
                if gm["is_attack_detected"]:
                    scenario_detected = True
            if scenario_detected:
                detections += 1

        metrics[name] = {
            "total": total,
            "valid": valid,
            "errors": errors,
            "utility_rate": utility_count / valid if valid else 0,
            "asr": attack_count / valid if valid else 0,
            "detection_rate": detections / valid if valid else 0,
            "latencies": latencies,
            "avg_latency": np.mean(latencies) if latencies else 0,
        }
    return metrics


def plot_pareto_frontier(
    baseline_metrics: dict[str, dict],
    guarded_metrics: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot Utility vs ASR Pareto frontier per suite."""
    fig, ax = plt.subplots(figsize=(10, 7))

    suites = sorted(set(baseline_metrics.keys()) | set(guarded_metrics.keys()))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(suites), 3)))
    markers_baseline = "o"
    markers_guarded = "s"

    for i, suite in enumerate(suites):
        color = colors[i]

        if suite in baseline_metrics:
            bm = baseline_metrics[suite]
            ax.scatter(
                bm["utility_rate"] * 100,
                bm["asr"] * 100,
                color=color,
                marker=markers_baseline,
                s=150,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"  {suite}\n  (baseline)",
                (bm["utility_rate"] * 100, bm["asr"] * 100),
                fontsize=8,
                color=color,
            )

        if suite in guarded_metrics:
            gm = guarded_metrics[suite]
            ax.scatter(
                gm["utility_rate"] * 100,
                gm["asr"] * 100,
                color=color,
                marker=markers_guarded,
                s=150,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"  {suite}\n  (guarded)",
                (gm["utility_rate"] * 100, gm["asr"] * 100),
                fontsize=8,
                color=color,
            )

        # Draw arrow from baseline to guarded
        if suite in baseline_metrics and suite in guarded_metrics:
            bm = baseline_metrics[suite]
            gm = guarded_metrics[suite]
            ax.annotate(
                "",
                xy=(gm["utility_rate"] * 100, gm["asr"] * 100),
                xytext=(bm["utility_rate"] * 100, bm["asr"] * 100),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    linestyle="--",
                ),
            )

    # Aggregate totals
    for label, metrics, marker in [
        ("TOTAL (baseline)", baseline_metrics, markers_baseline),
        ("TOTAL (guarded)", guarded_metrics, markers_guarded),
    ]:
        total_valid = sum(m["valid"] for m in metrics.values())
        if total_valid == 0:
            continue
        total_utility = sum(
            m["utility_rate"] * m["valid"] for m in metrics.values()
        ) / total_valid
        total_asr = sum(
            m["asr"] * m["valid"] for m in metrics.values()
        ) / total_valid

        ax.scatter(
            total_utility * 100,
            total_asr * 100,
            color="black",
            marker=marker,
            s=250,
            edgecolors="red",
            linewidths=2,
            zorder=10,
        )
        ax.annotate(
            f"  {label}",
            (total_utility * 100, total_asr * 100),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Utility Rate (%)", fontsize=13)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title(
        "CausalArmor Pareto Frontier: Utility vs ASR\n"
        "(circles = baseline, squares = guarded)",
        fontsize=14,
    )
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

    # Ideal region annotation
    ax.fill_between([50, 105], 0, 15, alpha=0.05, color="green")
    ax.text(
        75, 5, "Ideal region\n(high utility, low ASR)",
        ha="center", va="center", fontsize=9, color="green", alpha=0.6,
    )

    ax.grid(True, alpha=0.3)
    ax.legend(
        ["Baseline (no guard)", "Guarded (CausalArmor)"],
        loc="upper right",
        fontsize=10,
    )

    plt.tight_layout()
    out = output_dir / "pareto_frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_distributions(
    baseline_metrics: dict[str, dict],
    guarded_metrics: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot per-suite bar charts and guard latency distributions."""
    suites = sorted(set(baseline_metrics.keys()) | set(guarded_metrics.keys()))

    # --- Bar chart: Utility & ASR side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(suites))
    width = 0.35

    # Utility bars
    ax = axes[0]
    baseline_util = [baseline_metrics.get(s, {}).get("utility_rate", 0) * 100 for s in suites]
    guarded_util = [guarded_metrics.get(s, {}).get("utility_rate", 0) * 100 for s in suites]

    bars1 = ax.bar(x - width / 2, baseline_util, width, label="Baseline", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, guarded_util, width, label="Guarded", color="#55A868", alpha=0.85)

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Utility Rate by Suite", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # ASR bars
    ax = axes[1]
    baseline_asr = [baseline_metrics.get(s, {}).get("asr", 0) * 100 for s in suites]
    guarded_asr = [guarded_metrics.get(s, {}).get("asr", 0) * 100 for s in suites]

    bars1 = ax.bar(x - width / 2, baseline_asr, width, label="Baseline", color="#C44E52", alpha=0.85)
    bars2 = ax.bar(x + width / 2, guarded_asr, width, label="Guarded", color="#8172B2", alpha=0.85)

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Attack Success Rate by Suite", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(baseline_asr, default=0), max(guarded_asr, default=0)) * 1.3 + 5)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = output_dir / "utility_asr_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # --- Latency distribution (guarded only) ---
    all_latencies = []
    suite_latencies = {}
    for s in suites:
        lats = guarded_metrics.get(s, {}).get("latencies", [])
        if lats:
            suite_latencies[s] = lats
            all_latencies.extend(lats)

    if all_latencies:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(all_latencies, bins=30, color="#55A868", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(all_latencies), color="red", linestyle="--", label=f"Mean: {np.mean(all_latencies):.1f}s")
        ax.axvline(np.median(all_latencies), color="blue", linestyle="--", label=f"Median: {np.median(all_latencies):.1f}s")
        ax.set_xlabel("Guard Latency (seconds)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Guard Latency Distribution (all suites)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Box plot per suite
        ax = axes[1]
        if suite_latencies:
            labels = list(suite_latencies.keys())
            data = [suite_latencies[l] for l in labels]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors_box = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp["boxes"], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Guard Latency (seconds)", fontsize=12)
            ax.set_title("Guard Latency by Suite", fontsize=13)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = output_dir / "latency_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # --- Detection rate + FPR bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    detection_rates = [guarded_metrics.get(s, {}).get("detection_rate", 0) * 100 for s in suites]
    bars = ax.bar(suites, detection_rates, color="#DD8452", alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("CausalArmor Attack Detection Rate by Suite", fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "detection_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline JSON results")
    parser.add_argument("--guarded", type=Path, required=True, help="Guarded JSON results")
    parser.add_argument("--output", type=Path, default=Path("results/plots"), help="Output directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    baseline_data = load_results(args.baseline)
    guarded_data = load_results(args.guarded)

    baseline_metrics = extract_suite_metrics(baseline_data)
    guarded_metrics = extract_suite_metrics(guarded_data)

    print("\n=== Baseline Metrics ===")
    for name, m in baseline_metrics.items():
        print(f"  {name}: utility={m['utility_rate']:.1%}  ASR={m['asr']:.1%}  errors={m['errors']}")

    print("\n=== Guarded Metrics ===")
    for name, m in guarded_metrics.items():
        print(f"  {name}: utility={m['utility_rate']:.1%}  ASR={m['asr']:.1%}  detection={m['detection_rate']:.1%}  avg_latency={m['avg_latency']:.1f}s  errors={m['errors']}")

    plot_pareto_frontier(baseline_metrics, guarded_metrics, args.output)
    plot_distributions(baseline_metrics, guarded_metrics, args.output)

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
