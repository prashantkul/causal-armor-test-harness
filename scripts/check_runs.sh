#!/usr/bin/env bash
# Quick status check for running benchmark processes.
set -euo pipefail

echo "=== Running Benchmarks ==="
procs=$(ps aux | grep "run_benchmark" | grep -v grep | grep -v check_runs || true)
if [ -z "$procs" ]; then
    echo "  No benchmark processes running."
else
    echo "$procs" | awk '{
        # Extract provider, suite, guard mode from command line
        cmd = ""
        for (i=11; i<=NF; i++) cmd = cmd " " $i
        provider = ""; suite = ""; guard = ""
        for (i=11; i<=NF; i++) {
            if ($i == "--provider" || $i == "-p") provider = $(i+1)
            if ($i == "--suite" || $i == "-s") suite = $(i+1)
            if ($i == "--guard") guard = "guarded"
            if ($i == "--no-guard") guard = "baseline"
        }
        if (provider == "") next
        elapsed = $10
        printf "  %-10s %-12s %-10s  (CPU time: %s)\n", provider, suite, guard, elapsed
    }'
fi

echo ""
echo "=== vLLM Proxy ==="
container=$(docker ps --filter "ancestor=nvcr.io/nvidia/vllm:26.01-py3" --format '{{.Names}}' 2>/dev/null || true)
if [ -z "$container" ]; then
    container=$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | grep -i vllm | awk '{print $1}' || true)
fi
if [ -z "$container" ]; then
    echo "  No vLLM container found."
else
    echo "  Container: $container"
    # Show last few request logs (POST /v1/completions lines)
    recent=$(docker logs --tail 20 "$container" 2>&1 | grep -E "POST /v1/completions|ValueError|Error" | tail -5 || true)
    if [ -n "$recent" ]; then
        echo "$recent" | sed 's/^/  /'
    else
        echo "  (no recent completions activity)"
    fi
    # Request count in last 60s
    count=$(docker logs --since 60s "$container" 2>&1 | grep -c "POST /v1/completions" || echo "0")
    echo "  Requests in last 60s: $count"
fi

echo ""
results_dir="$(cd "$(dirname "$0")/.." && pwd)/results"
python3 - "$results_dir" << 'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

results = Path(sys.argv[1])
suites = ["banking", "workspace", "travel", "slack"]
providers = ["gemini", "openai", "anthropic"]
modes = ["baseline", "guarded"]
reps = 3

console = Console()

# ── Run Matrix ──

# Detect running benchmark processes
running = set()
try:
    out = subprocess.check_output(
        ["ps", "aux"], text=True, stderr=subprocess.DEVNULL,
    )
    for line in out.splitlines():
        if "run_benchmark" not in line or "grep" in line or "check_runs" in line:
            continue
        parts = line.split()
        provider = suite = mode = None
        for i, tok in enumerate(parts):
            if tok in ("--provider", "-p") and i + 1 < len(parts):
                provider = parts[i + 1]
            if tok in ("--suite", "-s") and i + 1 < len(parts):
                suite = parts[i + 1]
            if tok == "--guard":
                mode = "guarded"
            if tok == "--no-guard":
                mode = "baseline"
        if provider and suite and mode:
            running.add((suite, mode, provider))
except Exception:
    pass

# Check which result files exist
done = {}
for suite in suites:
    for provider in providers:
        for mode in modes:
            n = sum(
                1 for run in range(1, reps + 1)
                if (results / f"{suite}_{provider}_{mode}_run{run}.json").exists()
            )
            done[(suite, mode, provider)] = n

table = Table(title="Run Matrix", title_style="bold")
table.add_column("Suite", style="cyan")
table.add_column("Mode", style="cyan")
for p in providers:
    table.add_column(p.capitalize(), justify="center")

totals = {p: 0 for p in providers}
for suite in suites:
    for mode in modes:
        cells = []
        for provider in providers:
            n = done[(suite, mode, provider)]
            totals[provider] += n
            is_running = (suite, mode, provider) in running
            if n == reps:
                cell = Text(f"{n}/{reps}", style="bold green")
            elif n > 0 and is_running:
                cell = Text(f"{n}/{reps}", style="bold blue")
            elif is_running:
                cell = Text(f"{n}/{reps}", style="blue")
            elif n > 0:
                cell = Text(f"{n}/{reps}", style="yellow")
            else:
                cell = Text(f"{n}/{reps}", style="dim")
            cells.append(cell)
        table.add_row(suite, mode, *cells)

# Totals row
table.add_section()
total_cells = []
for p in providers:
    t = totals[p]
    style = "bold green" if t == len(suites) * len(modes) * reps else "yellow"
    total_cells.append(Text(f"{t}/{len(suites) * len(modes) * reps}", style=style))
table.add_row(
    Text("Total", style="bold"), "", *total_cells,
)

console.print()
console.print(table)
total_done = sum(totals.values())
total_all = len(suites) * len(providers) * len(modes) * reps
if total_done == total_all:
    console.print(f"\n  [bold green]{total_done}/{total_all} completed[/bold green]")
else:
    console.print(
        f"\n  [green]{total_done}[/green]/{total_all} completed, "
        f"[yellow]{total_all - total_done}[/yellow] remaining"
    )
console.print(
    "  Legend: [bold green]done[/bold green]  "
    "[blue]running[/blue]  "
    "[yellow]partial[/yellow]  "
    "[dim]pending[/dim]"
)

# ── Completed Results ──

result_files = sorted(results.glob("*.json"))
rows = []
for f in result_files:
    if f.name.startswith("benchmark_") or "verify" in f.name:
        continue
    if f.is_symlink():
        continue
    try:
        with open(f) as fh:
            data = json.load(fh)
        d = data[0] if isinstance(data, list) else data
        scenarios = d.get("scenarios", [])
        n = len(scenarios)
        if n == 0:
            continue
        util = sum(1 for s in scenarios if s.get("utility")) / n * 100
        asr = sum(1 for s in scenarios if not s.get("security")) / n * 100
        errs = sum(1 for s in scenarios if s.get("error"))
        rows.append((f.name, n, util, asr, errs))
    except Exception:
        continue

if rows:
    detail = Table(title="Completed Results", title_style="bold")
    detail.add_column("File", style="cyan", no_wrap=True)
    detail.add_column("N", justify="right")
    detail.add_column("Utility", justify="right")
    detail.add_column("ASR", justify="right")
    detail.add_column("Errors", justify="right")
    for fname, n, util, asr, errs in rows:
        asr_style = "green" if asr < 5 else "yellow" if asr < 20 else "red"
        err_style = "green" if errs == 0 else "yellow" if errs < 10 else "red"
        detail.add_row(
            fname,
            str(n),
            f"{util:.1f}%",
            Text(f"{asr:.1f}%", style=asr_style),
            Text(str(errs), style=err_style),
        )
    console.print()
    console.print(detail)
PYEOF
