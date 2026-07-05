# Attribution & third-party terms

This repository contains **only** original code (a kernel shim, patch files, and
scripts) under the MIT License. It does **not** redistribute any NVIDIA model
weights or NVIDIA source code.

- **The model**, `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`, you download
  yourself from NVIDIA. Its use is governed by the **NVIDIA Nemotron Open Model
  License Agreement**:
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
  The `patches/` here are unified diffs applied to files *you* obtain under that
  license; we do not distribute the underlying files.

- **Algorithms**, the pure-PyTorch reimplementations in `mamba_shim/` (Mamba-2
  SSD selective scan, gated RMSNorm, causal depthwise conv1d) are clean-room
  reimplementations of algorithms from **state-spaces/mamba** (Apache-2.0) and
  the Mamba-2 paper. No code was copied from `mamba_ssm` or `causal_conv1d`.
