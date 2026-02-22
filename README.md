# CausalArmor Test Harness

AgentDojo benchmark harness for evaluating the [CausalArmor](https://github.com/prashantkul/causal-armor) prompt-injection defense.

## Overview

This harness runs AgentDojo benchmark scenarios (banking, travel, workspace, slack) through a LangGraph agent guarded by CausalArmor. It measures:

- **Utility** -- does the agent still complete the legitimate task?
- **Security (ASR)** -- does the prompt injection attack succeed?
- **Detection rate** -- does CausalArmor detect the injection?
- **False positive rate** -- does the guard incorrectly block benign tool calls?
- **Guard latency** -- how long does each CausalArmor evaluation take?

## Architecture

```
User Task (AgentDojo)
  |
  v
LangGraph Agent (Gemini)
  |
  v
LLM proposes tool call --> CausalArmor Guard (LOO attribution via vLLM proxy)
  |                              |
  | (passed)                     | (blocked -> regenerate action)
  v                              v
Tool Execution            Safe replacement action
  |
  v
LLM continues...
```

Key design: CausalArmor operates **post-LLM, pre-execution**. It lets the LLM see all tool outputs (including injected content), then evaluates whether the LLM's proposed action was anomalously influenced by any tool result using leave-one-out (LOO) causal attribution.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker with NVIDIA GPU support (for vLLM proxy model)
- Google API key (for Gemini agent + CausalArmor action/sanitizer models)
- Hugging Face token (for vLLM model downloads)

## Setup

### 1. Clone and install

```bash
git clone <this-repo>
cd causal-armor-test-harness
uv sync
```

### 2. Configure environment

Create a `.env` file:

```bash
GOOGLE_API_KEY=<your-google-api-key>

# CausalArmor model configuration
CAUSAL_ARMOR_PROXY_MODEL=google/gemma-3-12b-it
CAUSAL_ARMOR_PROXY_BASE_URL=http://localhost:8000
CAUSAL_ARMOR_SANITIZER_MODEL=gemini-2.5-flash
CAUSAL_ARMOR_ACTION_MODEL=gemini-2.5-flash

# CausalArmor detection tuning
CAUSAL_ARMOR_CONFIG_PATH=causal_armor.toml
CAUSAL_ARMOR_MARGIN_TAU=0.0

# Hugging Face (for vLLM model downloads)
HF_TOKEN=<your-hf-token>
```

### 3. Start vLLM proxy server

```bash
bash scripts/start_vllm.sh
```

This starts a Docker container serving `google/gemma-3-12b-it` on port 8000. Tunable environment variables:

| Variable | Default | Description |
|---|---|---|
| `GPU_MEM_UTIL` | `0.5` | vLLM GPU memory utilization fraction |
| `CONTAINER_MEM` | `32g` | Docker container memory limit |
| `SHM_SIZE` | `4g` | Shared memory for NCCL/PyTorch |

Wait until vLLM reports "ready" before running benchmarks.

### 4. CausalArmor tuning (optional)

Edit `causal_armor.toml`:

```toml
[causal_armor]
margin_tau = 0.0            # Detection threshold (0.0 = most sensitive)
privileged_tools = []       # Tools that skip guard evaluation
enable_cot_masking = true   # Mask chain-of-thought for scoring
mask_cot_for_scoring = true
enable_sanitization = true  # Sanitize flagged content before regeneration
log_attributions = true     # Log per-span attribution scores
```

## Usage

### CLI Commands

The CLI is available as `ca-bench` (or `uv run python -m harness.cli`).

#### Run benchmarks

```bash
# Single scenario (quick test)
uv run ca-bench run --suite banking --attack important_instructions \
  --user-task user_task_0 --injection-task injection_task_0 --verbose

# Full suite with attack
uv run ca-bench run --suite banking --attack important_instructions --verbose

# Baseline (no guard)
uv run ca-bench run --suite banking --attack important_instructions --no-guard

# All suites
uv run ca-bench run --attack important_instructions

# Different agent model
uv run ca-bench run --suite banking --attack important_instructions \
  --agent-model gemini-2.5-pro
```

**Options:**

| Flag | Description |
|---|---|
| `--suite`, `-s` | Suite to run: `banking`, `travel`, `workspace`, `slack` |
| `--user-task`, `-u` | Filter to a single user task ID |
| `--injection-task`, `-i` | Filter to a single injection task ID |
| `--attack`, `-a` | Attack name (e.g. `important_instructions`) |
| `--no-guard` | Run without CausalArmor (baseline) |
| `--agent-model` | Agent LLM model (default: `gemini-2.0-flash`) |
| `--version`, `-v` | AgentDojo benchmark version (default: `v1.1`) |
| `--output`, `-o` | JSON output path (auto-generated if omitted) |
| `--verbose` | Enable debug logging |

#### List tasks in a suite

```bash
uv run ca-bench list-tasks banking
```

#### Show tool classification

```bash
uv run ca-bench classify banking
```

Shows which tools are classified as **untrusted** (their outputs may contain injections) vs **trusted**.

### Output

Results are written as JSON to `results/` with a table printed to the console:

```
                     CausalArmor Benchmark Results
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Suite   ┃ Scenarios ┃ Utility ┃  ASR ┃ Detection ┃  FPR ┃ Latency ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ banking │        10 │  60.0%  │ 5.0% │    95.0%  │ 2.0% │   3.1s  │
└─────────┴───────────┴─────────┴──────┴───────────┴──────┴─────────┘
```

## Project Structure

```
src/harness/
  cli.py              # Typer CLI entry point
  runner.py           # Scenario/suite/full benchmark runner
  bridge.py           # AgentDojo tools <-> LangChain adapter
  classify.py         # Automatic untrusted tool classification (canary injection)
  metrics.py          # Aggregate guard metrics (detection rate, FPR, latency)
  report.py           # JSON + Rich table output
  _fake_pipeline.py   # Stub pipeline for AgentDojo attack loader

scripts/
  start_vllm.sh       # Start vLLM Docker container

causal_armor.toml     # CausalArmor configuration
```

## Key Dependencies

- [AgentDojo](https://github.com/ethz-spylab/agentdojo) -- benchmark framework (task suites, attacks, evaluation)
- [CausalArmor](https://github.com/prashantkul/causal-armor) -- LOO-based prompt injection defense
- [causal-armor-langgraph-demo](https://github.com/prashantkul/causal-armor-langgraph-demo) -- LangGraph agent with pluggable guard node
- [vLLM](https://github.com/vllm-project/vllm) -- fast LLM serving for proxy model
