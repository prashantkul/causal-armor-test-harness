#!/usr/bin/env bash
# Start vLLM serving Gemma 3 12B for CausalArmor LOO proxy scoring.
#
# Docker-level memory constraints prevent vLLM from consuming all GPU/system
# RAM on unified-memory systems (e.g. DGX Spark GB10).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# GPU memory utilisation (vLLM-level, fraction of visible GPU memory)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

# Docker container memory limit (host RAM ceiling)
CONTAINER_MEM="${CONTAINER_MEM:-32g}"

# Shared memory for NCCL / PyTorch
SHM_SIZE="${SHM_SIZE:-4g}"

docker run -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --memory "$CONTAINER_MEM" \
  --shm-size "$SHM_SIZE" \
  -p 8000:8000 \
  --env-file "$PROJECT_DIR/.env" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve google/gemma-3-12b-it \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_MEM_UTIL"
