#!/usr/bin/env bash
# Run TwoTower on CPU. Usage: ./run.sh "prompt" [mode] [max_new_tokens]
# mode: ar | mask_diffusion (default) | mock_ar
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROMPT="${1:?usage: run.sh \"prompt\" [mode] [max_new_tokens]}"
MODE="${2:-mask_diffusion}"; N="${3:-32}"
MODEL="${TT_MODEL:-$HERE/twotower}"
export PYTHONPATH="$HERE/mamba_shim"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 8)}"
python "$MODEL/inference_cpu.py" --model "$MODEL" --mode "$MODE" \
    --max-new-tokens "$N" --block-size 16 --steps-per-block 16 "$PROMPT"
