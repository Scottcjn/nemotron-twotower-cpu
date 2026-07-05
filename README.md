# nemotron-twotower-cpu

Run NVIDIA's [Nemotron-Labs-TwoTower](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16) diffusion language model on **any CPU**, `ppc64le`, `arm64`, `x86_64`, with **no NVIDIA GPU and no CUDA**.

TwoTower ships with a hard dependency on `mamba_ssm` and `causal_conv1d`, which are Triton/CUDA-only. That makes the released model unrunnable on non-NVIDIA hardware, including Apple Silicon, IBM POWER, and any GPU-less machine. This repo removes that dependency with a small pure-PyTorch kernel shim plus a handful of one-line fixes to an untested CPU code path, so the model runs anywhere PyTorch does. It is slow (no fused kernels), but correct, which is exactly what you want for an offline distillation teacher.

## Proven on

| Machine | Arch | What ran |
|---|---|---|
| IBM POWER8 S824 | `ppc64le` | full 60B model, **AR + two-tower mask-diffusion generation** (CPU, no GPU) |
| Apple M2 Mac mini | `arm64` | kernel shim **bit-exact** (`0.00e+00` error), MPS available |

Real transcripts in [`transcripts/`](transcripts/). Sample (POWER8, mask-diffusion, CPU):

```
Prompt: Explain in one sentence why the sky is blue.
Generated (31 NFE, 32 tokens):
 The sky is blue because the atmosphere scatters shorter wavelengths of light
 more than longer ones, with blue light being scattered the most.
```

## How it works

Three pieces, all small:

1. **`mamba_shim/`**, a drop-in `mamba_ssm` + `causal_conv1d` package in pure PyTorch. Put it on `PYTHONPATH` and the model's CUDA-kernel imports resolve to portable ATen code instead. Only one function needs to be numerically exact (`rmsnorm_fn`, the gated group-RMSNorm used in every forward); the rest are real reference implementations of the Mamba-2 SSD scan and causal depthwise conv, verified against independent references. On CPU the model's own `is_fast_path_available` gate routes to its native `torch_forward` path, so the fused-kernel functions are only exercised by the diffusion denoiser, for which this repo provides correct scans.

2. **`patches/modeling_nemotron_h.cpu.patch`**, six one-line fixes to a code path NVIDIA never ran. Their `HybridMambaAttentionDynamicCache` stores `conv_states` / `ssm_states` as lists of per-layer tensors (as documented), but `torch_forward` and a few cache methods call `.device` / `.zero_()` on the *list* instead of `[layer_idx]`. These never fire on CUDA (the fast path skips `torch_forward` entirely), so they were latent until run on CPU. See [`UPSTREAM_BUGS.md`](UPSTREAM_BUGS.md), worth reporting upstream.

3. **`patches/inference.cpu.patch`**, the demo `inference.py` hardcodes `.cuda()`; this maps it to CPU. The model itself is device-agnostic.

## Quick start

```bash
# 1. get NVIDIA's model + code (governed by their license, see NOTICE.md)
huggingface-cli download nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 --local-dir twotower

# 2. apply the CPU patches
cd twotower
patch -p0 < ../patches/modeling_nemotron_h.cpu.patch
patch -p0 < ../inference.cpu.patch   # writes inference_cpu.py

# 3. run, with the shim on PYTHONPATH
export PYTHONPATH=$PWD/../mamba_shim
export CUDA_VISIBLE_DEVICES=""
python inference_cpu.py --mode mask_diffusion --max-new-tokens 32 \
    "Explain in one sentence why the sky is blue."
```

Or just `bash run.sh "your prompt"`. Verify the kernels on your arch first with `python verify_kernels.py`.

## Requirements

- `torch` (any build, CPU is fine), `transformers >= 4.48`, `einops`, `safetensors`, `accelerate`
- ~120 GB RAM for the full 60B in bf16 (both towers). The model does not fit smaller machines; a 24 GB box is a *student* host, not a teacher host.

## Speed & the roadmap

This is a **correctness** port, not a fast one, generation takes tens of minutes for a short completion. That is acceptable for an offline teacher that generates a distillation corpus in the background, but do not expect interactive speed.

The SSD scan in `mamba_shim` is fully vectorized (segsum form, verified bit-exact against a naive reference on ppc64le/arm64/x86_64), so the scan itself is not the bottleneck. Profiling on POWER8 shows the end-to-end cost is dominated by the broader CPU forward: many small MoE and per-layer operations that do not saturate available cores (the process uses ~3 of 24 threads), plus the base `transformers` `torch_forward` selective-scan path in the frozen context tower, which this repo does not modify. Real speedups from here are a deeper effort, a threaded/fused CPU forward, or simply running on a machine with an NVIDIA GPU where the original kernels apply. This port's job is reach (it runs where nothing else does), not throughput.

The point of running the teacher on a CPU at all: **distillation for hardware people actually own.** Generate a corpus from the slow CPU teacher, distill a small (1-3B) student, and that student runs fast on the same `arm64`/NEON machines this shim is already proven on, laptops, Macs, phones, Raspberry Pis.

## License

Our code (the shim, patches, scripts) is MIT, see [`LICENSE`](LICENSE). It contains no NVIDIA weights and no NVIDIA source; the patches are diffs you apply to files you download yourself. Use of NVIDIA's model is governed by the NVIDIA Nemotron Open Model License, see [`NOTICE.md`](NOTICE.md). The Mamba-2 and causal-conv algorithms reimplemented here are from [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache-2.0).
