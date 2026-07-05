#!/usr/bin/env bash
# One-shot: download the model + apply CPU patches. Governed by NVIDIA's license.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${TT_MODEL:-$HERE/twotower}"
echo ">> downloading model + code to $MODEL (NVIDIA Nemotron Open Model License)"
huggingface-cli download nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 --local-dir "$MODEL"
echo ">> applying CPU patches"
( cd "$MODEL"
  patch -p0 --forward < "$HERE/patches/modeling_nemotron_h.cpu.patch" || true
  # inference patch just maps .cuda() -> .to("cpu"); produce inference_cpu.py
  sed 's/\.cuda()/.to("cpu")/g' inference.py > inference_cpu.py )
echo ">> done. run:  ./run.sh \"Explain why the sky is blue.\""
