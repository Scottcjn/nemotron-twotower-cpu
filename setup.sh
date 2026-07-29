#!/usr/bin/env bash
# One-shot: download the model + apply CPU patches. Governed by NVIDIA's license.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${TT_MODEL:-$HERE/twotower}"

# Apply a patch idempotently, and stop if it neither applies nor is already
# applied. The old `|| true` swallowed a genuinely failed patch, which then
# showed up much later as "'list' object has no attribute 'device'" mid-run.
apply_patch() {
    local p="$1" name
    name="$(basename "$p")"
    if patch -p0 --forward --silent --dry-run <"$p" >/dev/null 2>&1; then
        patch -p0 --forward --silent <"$p"
        echo "   applied $name"
    elif patch -p0 --reverse --silent --dry-run <"$p" >/dev/null 2>&1; then
        echo "   already applied, skipping $name"
    else
        echo "!! $name does not apply to $MODEL -- upstream files may have changed" >&2
        exit 1
    fi
}

echo ">> downloading model + code to $MODEL (NVIDIA Nemotron Open Model License)"
huggingface-cli download nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 --local-dir "$MODEL"

echo ">> applying CPU patches"
(
    cd "$MODEL"
    apply_patch "$HERE/patches/modeling_nemotron_h.cpu.patch"
    # inference_cpu.py is a patched COPY, so NVIDIA's inference.py stays pristine
    # and re-running setup.sh always starts from a clean copy.
    cp -f inference.py inference_cpu.py
    apply_patch "$HERE/patches/inference.cpu.patch"
)

echo ">> done. run:  ./run.sh \"Explain why the sky is blue.\""
