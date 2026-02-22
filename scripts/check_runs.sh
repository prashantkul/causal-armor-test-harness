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
echo "=== Completed Results ==="
results_dir="$(dirname "$0")/../results"
for f in "$results_dir"/*.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    # Skip verification and old benchmark files
    [[ "$fname" == benchmark_* ]] && continue
    [[ "$fname" == *verify* ]] && continue
    python3 -c "
import json, sys
with open('$f') as fh:
    data = json.load(fh)
d = data[0] if isinstance(data, list) else data
scenarios = d.get('scenarios', [])
n = len(scenarios)
if n == 0:
    sys.exit(0)
util = sum(1 for s in scenarios if s.get('utility')) / n * 100
asr = sum(1 for s in scenarios if not s.get('security')) / n * 100
errs = sum(1 for s in scenarios if s.get('error'))
print(f'  {\"$fname\":<50s}  n={n:>4d}  util={util:5.1f}%  asr={asr:5.1f}%  err={errs}')
" 2>/dev/null || true
done
