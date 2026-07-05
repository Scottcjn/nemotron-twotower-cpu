# SPDX-License-Identifier: AGPL-3.0-or-later
def selective_state_update(*a, **k):
    raise RuntimeError("selective_state_update shim: CUDA-only path; CPU must use torch_forward")
