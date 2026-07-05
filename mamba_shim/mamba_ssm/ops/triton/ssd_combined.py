# SPDX-License-Identifier: AGPL-3.0-or-later
# Vectorized Mamba-2 SSD scan (segsum form), no Python sequence loop.
# For block length L this is the quadratic single-chunk form: O(L^2) but fully
# vectorized (einsum), so it uses all cores instead of one. Matches the naive
# sequential recurrence bit-for-bit (see _ssd_naive, kept for the self-test).
import torch, torch.nn.functional as F


def _segsum(x):
    """x:(...,L) -> (...,L,L): seg[...,i,j] = sum_{j<k<=i} x[...,k] for i>=j (log-decay
    from just-after-j to i), -inf above the diagonal. Mamba-2 reference form."""
    L = x.shape[-1]
    xr = x[..., None].expand(*x.shape, L)                  # (...,L(i),L(j)): value = x[...,i]
    mask0 = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device), -1)
    xr = xr.masked_fill(~mask0, 0)
    seg = xr.cumsum(dim=-2)                                 # cumulative over i
    mask1 = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device), 0)
    return seg.masked_fill(~mask1, float("-inf"))


def _prep(x, dt, A, B, C, D, dt_bias, dt_softplus):
    Bsz, L, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    x = x.float(); B = B.float(); C = C.float(); A = A.float(); dt = dt.float()
    if dt_bias is not None: dt = dt + dt_bias.float()
    if dt_softplus: dt = F.softplus(dt)
    Bh = B.repeat_interleave(H // G, dim=2)                 # (B,L,H,N)
    Ch = C.repeat_interleave(H // G, dim=2)
    return x, dt, A, Bh, Ch, (D.float() if D is not None else None), Bsz, L, H, P, N


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None,
                              dt_bias=None, dt_softplus=False, initial_states=None,
                              return_final_states=False, **kw):
    dtype = x.dtype
    x, dt, A, Bh, Ch, D, Bsz, L, H, P, N = _prep(x, dt, A, B, C, D, dt_bias, dt_softplus)
    dlog = dt * A                                           # (B,L,H) per-step log-decay
    dcs = dlog.permute(0, 2, 1)                             # (B,H,L)
    Lmat = torch.exp(_segsum(dcs))                          # (B,H,L,L) decay j->i, 0 above diag
    CB = torch.einsum("blhn,bshn->bhls", Ch, Bh)           # (B,H,L(i),L(j))  C_i . B_j
    dtj = dt.permute(0, 2, 1)[:, :, None, :]               # (B,H,1,L) dt at step j
    scores = CB * Lmat * dtj                               # lower-tri already (Lmat 0 above)
    y = torch.einsum("bhls,bshp->blhp", scores, x)         # (B,L,H,P)

    Ai = torch.exp(dlog.cumsum(dim=1))                     # (B,L,H) decay from step0->i
    if initial_states is not None:
        s0 = initial_states.float()                        # (B,H,P,N)
        yinit = torch.einsum("blhn,bhpn->blhp", Ch, s0)    # C_i . s0
        y = y + yinit * Ai[..., None]
    if D is not None:
        y = y + x * D[None, None, :, None]
    y = y.to(dtype)
    if not return_final_states:
        return y
    # final state after L steps: decayed s0 + sum_j (decay j->L) dt_j B_j x_j
    total = torch.exp(dlog.sum(dim=1))                     # (B,H) full-seq decay
    decay_jL = torch.exp((dlog.sum(dim=1, keepdim=True) - dlog.cumsum(dim=1)))  # (B,L,H) decay j->L (exclusive)
    dBx = (dt[..., None, None] * Bh[:, :, :, None, :] * x[:, :, :, :, None])    # (B,L,H,P,N)
    fs = (dBx * decay_jL[..., None, None]).sum(dim=1)      # (B,H,P,N)
    if initial_states is not None:
        fs = fs + initial_states.float() * total[..., None, None]
    return y, fs.to(dtype)


def _ssd_naive(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
               dt_softplus=False, initial_states=None, return_final_states=False, **kw):
    dtype = x.dtype
    x, dt, A, Bh, Ch, D, Bsz, L, H, P, N = _prep(x, dt, A, B, C, D, dt_bias, dt_softplus)
    dA = torch.exp(dt * A)
    st = initial_states.float() if initial_states is not None else x.new_zeros(Bsz, H, P, N)
    out = []
    for t in range(L):
        st = st * dA[:, t][..., None, None] + dt[:, t][..., None, None] * Bh[:, t][:, :, None, :] * x[:, t][:, :, :, None]
        out.append((st * Ch[:, t][:, :, None, :]).sum(-1))
    y = torch.stack(out, 1)
    if D is not None: y = y + x * D[None, None, :, None]
    y = y.to(dtype)
    return (y, st.to(dtype)) if return_final_states else y


def mamba_split_conv1d_scan_combined(*a, **k):
    raise RuntimeError("mamba_split_conv1d_scan_combined: fused CUDA path; CPU decomposes elsewhere")
