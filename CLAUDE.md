# CLAUDE.md — CausalArmor Test Harness

## Important: Open source library

This is a published open source benchmark harness. Exercise extra care:

- **Do not introduce errors** — test changes locally before committing
- **Always verify benchmark results** are reasonable before publishing
- **Do not commit credentials** — `.env` files are git-ignored for a reason

## Project overview

Benchmark harness for evaluating [CausalArmor](https://github.com/prashantkul/causal-armor) prompt-injection defense on [AgentDojo](https://github.com/ethz-spylab/agentdojo) task suites (banking, travel, workspace, slack).

Two execution paths exist:
- **`src/harness/`** — LangGraph-based (original, uses `ca-bench` CLI)
- **`src/agentdojo_native/`** — Native AgentDojo pipeline (no LangGraph dependency, preferred)

## Repository layout

```
src/
  harness/                    # LangGraph-based harness (legacy)
    cli.py                    # Typer CLI entry point (ca-bench)
    runner.py                 # Scenario/suite/full benchmark runner
    bridge.py                 # AgentDojo tools ↔ LangChain adapter
    classify.py               # Canary-injection tool classifier
    metrics.py                # SuiteMetrics aggregation
    report.py                 # JSON + Rich table output
  agentdojo_native/           # Native integration (preferred)
    run_benchmark.py          # CLI + runner (python -m agentdojo_native.run_benchmark)
    pipeline_element.py       # CausalArmorPipelineElement (BasePipelineElement)
    adapters.py               # AgentDojo ↔ CausalArmor message converters
    tool_declarations.py      # AgentDojo Function → Gemini declarations
    classify.py               # Canary-injection tool classifier
    metrics.py                # GuardMetrics dataclass
scripts/
  start_vllm.sh               # Start vLLM Docker container (Gemma 3 12B)
  plot_results.py              # Generate plots from results
notebooks/
  benchmark_analysis.ipynb     # Full analysis: metrics, Pareto frontier, CI, plots
results/                       # Benchmark output JSON files + plots/
causal_armor.toml              # Defense configuration
```

## Running benchmarks

### Prerequisites
- vLLM server running (`scripts/start_vllm.sh`) serving `google/gemma-3-12b-it`
- `.env` file with `GOOGLE_API_KEY`, proxy/model settings
- Python 3.11+ with venv at `.venv/`

### Native integration (preferred)

```bash
# Single suite, guarded
.venv/bin/python -m agentdojo_native.run_benchmark \
    --suite banking --attack important_instructions --guard \
    -o results/banking_guarded.json

# Single suite, baseline (no guard)
.venv/bin/python -m agentdojo_native.run_benchmark \
    --suite banking --attack important_instructions --no-guard \
    -o results/banking_baseline.json

# All suites (omit --suite)
.venv/bin/python -m agentdojo_native.run_benchmark \
    --attack important_instructions --guard
```

### LangGraph harness (legacy)

```bash
uv run ca-bench run --suite banking --attack important_instructions --verbose
```

## Key architectural decisions

- **CausalArmorPipelineElement** intercepts tool calls *after* the LLM proposes them but *before* execution — post-LLM, pre-execution
- **Retry logic**: Transient errors (429/503) automatically retry 3x with exponential backoff (10s/20s/40s)
- **Blocked call handling**: When the guard blocks ALL tool calls, the assistant message is modified to remove function_call parts (Gemini rejects orphaned function_response parts with 400)
- **Canary-injection classification**: Automatically identifies untrusted tools by injecting canary strings into attack vectors and checking which tool outputs contain them

## Common issues

- **400 INVALID_ARGUMENT from Gemini**: Usually means orphaned function_response parts without matching function_call — check `pipeline_element.py` blocked-call handling
- **429 RESOURCE_EXHAUSTED**: Gemini rate limits — retry logic handles this automatically
- **vLLM OOM**: Reduce `GPU_MEM_UTIL` in `start_vllm.sh` or use `--max-model-len 4096`

## Git workflow

- **Main branch**: `main`
- **PRs target**: `main`
- Result JSON files in `results/` and notebooks are tracked for reproducibility
