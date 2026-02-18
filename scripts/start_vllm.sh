#!/usr/bin/env bash
# Start vLLM serving Gemma 3 12B for CausalArmor LOO proxy scoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

docker run -it --gpus all -p 8000:8000 \
  --env-file "$PROJECT_DIR/.env" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve google/gemma-3-12b-it --dtype auto --max-model-len 8192 \
  --gpu-memory-utilization 0.8
