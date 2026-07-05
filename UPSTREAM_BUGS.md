# Upstream bugs found running the CPU path

While porting Nemotron-Labs-TwoTower to CPU we hit six latent bugs in
`modeling_nemotron_h.py`. All are in the `torch_forward` slow path and the
`HybridMambaAttentionDynamicCache` update methods, code that is **never
executed on CUDA**, because `NemotronHMamba2Mixer.forward` routes to
`cuda_kernels_forward` whenever `is_fast_path_available and "cuda" in
device.type`. On any CPU (or a build without `mamba_ssm`/`causal_conv1d`), the
fast path is unavailable, `torch_forward` runs, and these fire.

The cache stores `conv_states` and `ssm_states` as **lists of per-layer
tensors**, the class docstring says so explicitly ("Each of these lists has
`num_layers` tensors"). Most of `torch_forward` correctly indexes
`ssm_states[self.layer_idx]`, but a few sites call tensor methods on the list
object itself:

| Location | Buggy | Fix |
|---|---|---|
| `update_conv_state` (cache_init) | `new_conv_state.to(self.conv_states.device)` | `...to(self.conv_states[layer_idx].device)` |
| `update_conv_state` (roll) | `new_conv_state[:, 0, :].to(self.conv_states.device)` | `...[layer_idx].device` |
| `update_ssm_state` | `new_ssm_state.to(self.ssm_states.device)` | `...[layer_idx].device` |
| `reset` | `self.conv_states.zero_()` / `self.ssm_states.zero_()` | `[t.zero_() for t in ...]` |
| `torch_forward` (single-step) | `cache_device = cache_params.ssm_states.device` | `...ssm_states[self.layer_idx].device` |

Symptom before the fix:

```
AttributeError: 'list' object has no attribute 'device'
  File ".../modeling_nemotron_h.py", line 566, in torch_forward
    cache_device = cache_params.ssm_states.device
```

The full diff is in [`patches/modeling_nemotron_h.cpu.patch`](patches/modeling_nemotron_h.cpu.patch).
These are the minimal changes to make the documented list-based cache work on
the torch path; they do not affect the CUDA fast path.
