# SPDX-License-Identifier: AGPL-3.0-or-later
import torch, torch.nn.functional as F

def causal_conv1d_fn(x, weight, bias=None, activation=None, initial_states=None, **kw):
    """x:(B,D,L) depthwise causal conv. weight:(D,K). initial_states:(B,D,K-1) or None.
    Returns (B,D,L). activation in {None,'silu','swish'}."""
    B, D, L = x.shape
    K = weight.shape[-1]
    if initial_states is not None:
        xp = torch.cat([initial_states.to(x.dtype), x], dim=-1)      # (B,D,K-1+L)
    else:
        xp = F.pad(x, (K - 1, 0))                                    # left causal pad
    y = F.conv1d(xp, weight.unsqueeze(1), bias=bias, groups=D)[..., :L]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y

def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, **kw):
    """Single-step decode: roll conv_state, insert x, depthwise dot with weight."""
    D, K = weight.shape
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    y = (conv_state * weight).sum(-1)
    if bias is not None: y = y + bias
    if activation in ("silu", "swish"): y = F.silu(y)
    return y
