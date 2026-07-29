# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cover the shim behaviour the denoiser actually depends on.

verify_kernels.py checks three things on a fresh arch: rmsnorm_fn exactness, a
plain SSD scan, and that the conv is causal. But every call the two-tower
denoiser makes goes through `initial_states` (seeded from the context state)
and, on the context-extend path, `return_final_states=True` -- neither of which
verify_kernels.py exercises, and `_ssd_naive` (kept "for the self-test") was
never called by anything.

The property that matters is stitching: scanning a prefix, then scanning the
next block seeded from the prefix's final state, must equal scanning the whole
thing in one pass. That is exactly what `_denoiser_block_mamba` assumes.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mamba_shim"))

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: E402
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn  # noqa: E402
from mamba_ssm.ops.triton.ssd_combined import (  # noqa: E402
    _ssd_naive,
    mamba_chunk_scan_combined,
)

ATOL = 1e-4


def ssm_inputs(seed=0, B=2, L=7, H=4, P=3, G=2, N=5):
    torch.manual_seed(seed)
    return dict(
        x=torch.randn(B, L, H, P),
        dt=torch.rand(B, L, H),
        A=-torch.rand(H),
        Bm=torch.randn(B, L, G, N),
        Cm=torch.randn(B, L, G, N),
        D=torch.randn(H),
        dt_bias=torch.randn(H),
        s0=torch.randn(B, H, P, N),
    )


def scan(t, chunk_size=16, **kw):
    return mamba_chunk_scan_combined(
        t["x"], t["dt"], t["A"], t["Bm"], t["Cm"], chunk_size,
        D=t["D"], dt_bias=t["dt_bias"], dt_softplus=True, **kw)


def naive(t, chunk_size=16, **kw):
    return _ssd_naive(
        t["x"], t["dt"], t["A"], t["Bm"], t["Cm"], chunk_size,
        D=t["D"], dt_bias=t["dt_bias"], dt_softplus=True, **kw)


def slice_steps(t, sl):
    out = dict(t)
    for k in ("x", "dt", "Bm", "Cm"):
        out[k] = t[k][:, sl]
    return out


# --------------------------------------------------------------------------
# SSD scan: the paths the denoiser uses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seeded", [False, True])
def test_scan_matches_sequential_reference(seeded):
    t = ssm_inputs()
    kw = {"initial_states": t["s0"]} if seeded else {}
    assert (scan(t, **kw) - naive(t, **kw)).abs().max() < ATOL


@pytest.mark.parametrize("seeded", [False, True])
def test_final_state_matches_sequential_reference(seeded):
    t = ssm_inputs(seed=1)
    kw = {"initial_states": t["s0"]} if seeded else {}
    y, s = scan(t, return_final_states=True, **kw)
    ry, rs = naive(t, return_final_states=True, **kw)
    assert (y - ry).abs().max() < ATOL
    assert s.shape == t["s0"].shape
    assert (s - rs).abs().max() < ATOL


def test_initial_state_actually_changes_the_output():
    """Guard against a shim that accepts initial_states and ignores it."""
    t = ssm_inputs(seed=2)
    assert (scan(t) - scan(t, initial_states=t["s0"])).abs().max() > 1e-3


def test_zero_initial_state_is_the_same_as_none():
    t = ssm_inputs(seed=3)
    zero = torch.zeros_like(t["s0"])
    assert (scan(t) - scan(t, initial_states=zero)).abs().max() < ATOL


@pytest.mark.parametrize("split", [1, 3, 6])
def test_seeded_block_scan_stitches_onto_a_prefix(split):
    """The denoiser's core assumption: prefix, then block seeded from the
    prefix's final state == one pass over the whole sequence."""
    t = ssm_inputs(seed=4, L=8)
    whole_y, whole_s = scan(t, return_final_states=True)

    head_y, head_s = scan(slice_steps(t, slice(0, split)), return_final_states=True)
    tail_y, tail_s = scan(slice_steps(t, slice(split, None)),
                          initial_states=head_s, return_final_states=True)

    assert (head_y - whole_y[:, :split]).abs().max() < ATOL
    assert (tail_y - whole_y[:, split:]).abs().max() < ATOL
    assert (tail_s - whole_s).abs().max() < ATOL


@pytest.mark.parametrize("L", [1, 2, 15, 16, 17, 33])
def test_scan_lengths_around_the_chunk_boundary(L):
    t = ssm_inputs(seed=5, L=L)
    y, s = scan(t, chunk_size=16, initial_states=t["s0"], return_final_states=True)
    ry, rs = naive(t, chunk_size=16, initial_states=t["s0"], return_final_states=True)
    assert y.shape == t["x"].shape
    assert (y - ry).abs().max() < ATOL
    assert (s - rs).abs().max() < ATOL


def test_scan_preserves_input_dtype():
    t = ssm_inputs(seed=6)
    t["x"] = t["x"].to(torch.bfloat16)
    y, s = scan(t, initial_states=t["s0"], return_final_states=True)
    assert y.dtype is torch.bfloat16 and s.dtype is torch.bfloat16


def test_single_head_group_broadcast():
    """H // G repeat_interleave: one B/C group feeding several heads."""
    t = ssm_inputs(seed=7, H=6, G=3)
    assert (scan(t, initial_states=t["s0"]) - naive(t, initial_states=t["s0"])).abs().max() < ATOL


# --------------------------------------------------------------------------
# causal conv: the denoiser passes a context history in
# --------------------------------------------------------------------------

def conv_inputs(seed=0, B=2, D=5, L=9, K=4):
    torch.manual_seed(seed)
    return (torch.randn(B, D, L), torch.randn(D, K), torch.randn(D),
            torch.randn(B, D, K - 1))


def test_conv_with_history_matches_plain_convolution():
    x, w, b, hist = conv_inputs()
    got = causal_conv1d_fn(x, w, b, activation="silu", initial_states=hist)
    ref = F.silu(F.conv1d(torch.cat([hist, x], -1), w.unsqueeze(1), b, groups=w.shape[0]))
    assert got.shape == x.shape
    assert (got - ref).abs().max() < ATOL


def test_conv_history_stitches_onto_a_prefix():
    """Same stitching property, for the conv half of the mixer."""
    x, w, b, _ = conv_inputs(seed=1, L=12)
    K = w.shape[-1]
    split = 7
    whole = causal_conv1d_fn(x, w, b, activation="silu")
    hist = x[..., split - (K - 1):split]
    tail = causal_conv1d_fn(x[..., split:], w, b, activation="silu", initial_states=hist)
    assert (tail - whole[..., split:]).abs().max() < ATOL


def test_conv_history_is_not_ignored():
    x, w, b, hist = conv_inputs(seed=2)
    with_hist = causal_conv1d_fn(x, w, b, initial_states=hist)
    without = causal_conv1d_fn(x, w, b)
    assert (with_hist - without).abs().max() > 1e-3


def test_conv_left_pad_equals_zero_history():
    x, w, b, _ = conv_inputs(seed=3)
    zeros = torch.zeros(x.shape[0], x.shape[1], w.shape[-1] - 1)
    assert (causal_conv1d_fn(x, w, b) -
            causal_conv1d_fn(x, w, b, initial_states=zeros)).abs().max() < ATOL


def test_conv_is_causal():
    x, w, b, _ = conv_inputs(seed=4)
    y = causal_conv1d_fn(x, w, b, activation="silu")
    x2 = x.clone()
    x2[..., 5:] += 99.0
    y2 = causal_conv1d_fn(x2, w, b, activation="silu")
    assert torch.allclose(y[..., :5], y2[..., :5], atol=ATOL)


def test_conv_update_single_step_matches_full_conv():
    """Decode step: rolling the state and dotting must equal position L-1 of a
    full causal conv over the same window."""
    x, w, b, _ = conv_inputs(seed=5, L=8)
    K = w.shape[-1]
    full = causal_conv1d_fn(x, w, b, activation="silu")
    # state holds the K inputs ending at the previous position; update() rolls
    # the oldest out and writes the new token at index -1.
    state = x[..., -K - 1:-1].contiguous()
    y = causal_conv1d_update(x[..., -1], state, w, b, activation="silu")
    assert (y - full[..., -1]).abs().max() < ATOL
    assert (state[..., -1] - x[..., -1]).abs().max() == 0, "state must be updated in place"


# --------------------------------------------------------------------------
# gated group RMSNorm
# --------------------------------------------------------------------------

@pytest.mark.parametrize("group_size", [None, 8])
def test_rmsnorm_gate_before_norm(group_size):
    torch.manual_seed(8)
    B, L, D = 2, 4, 32
    x, z, w = torch.randn(B, L, D), torch.randn(B, L, D), torch.randn(D)
    got = rmsnorm_fn(x=x, weight=w, z=z, eps=1e-6, group_size=group_size,
                     norm_before_gate=False)
    g = x.float() * F.silu(z.float())
    gs = D if group_size is None else group_size
    gg = g.reshape(B, L, D // gs, gs)
    ref = (gg * torch.rsqrt(gg.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(B, L, D) * w
    assert (got - ref).abs().max() < ATOL


def test_rmsnorm_gate_after_norm_differs():
    """norm_before_gate is honoured, not silently fixed to one branch."""
    torch.manual_seed(9)
    x, z, w = torch.randn(2, 4, 32), torch.randn(2, 4, 32), torch.randn(32)
    before = rmsnorm_fn(x=x, weight=w, z=z, eps=1e-6, group_size=8, norm_before_gate=True)
    after = rmsnorm_fn(x=x, weight=w, z=z, eps=1e-6, group_size=8, norm_before_gate=False)
    assert (before - after).abs().max() > 1e-3
    ref = (x.float() * torch.rsqrt(
        x.float().reshape(2, 4, 4, 8).pow(2).mean(-1, keepdim=True) + 1e-6
    ).expand(2, 4, 4, 8).reshape(2, 4, 32)) * w * F.silu(z.float())
    assert (before - ref).abs().max() < ATOL


def test_rmsnorm_without_gate_preserves_dtype():
    torch.manual_seed(10)
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    out = rmsnorm_fn(x=x, weight=torch.randn(32), eps=1e-6, group_size=8)
    assert out.dtype is torch.bfloat16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
