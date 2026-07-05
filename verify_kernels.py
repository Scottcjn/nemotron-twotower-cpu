# SPDX-License-Identifier: AGPL-3.0-or-later
# Proves the mamba_shim kernels are numerically correct on THIS architecture.
# Run before trusting the port on a new arch (ppc64le / arm64 / x86_64):
#   python verify_kernels.py
import os, sys, platform, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba_shim"))
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from causal_conv1d import causal_conv1d_fn

print(f"arch={platform.machine()} torch={torch.__version__} "
      f"mps={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
torch.manual_seed(0)
ok = True

# 1. gated group-RMSNorm (the one function that MUST be exact, used every forward)
B, L, D, gs = 2, 4, 32, 8
x, z, w = torch.randn(B, L, D), torch.randn(B, L, D), torch.randn(D)
xg = (x.float() * F.silu(z.float())).reshape(B, L, D // gs, gs)
ref = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(B, L, D) * w.float()
e = (rmsnorm_fn(x=x, weight=w, z=z, eps=1e-6, group_size=gs, norm_before_gate=False) - ref).abs().max().item()
print(f"rmsnorm_fn        err={e:.2e}  {'OK' if e < 1e-5 else 'FAIL'}"); ok &= e < 1e-5

# 2. Mamba-2 SSD scan vs an independent sequential reference
Bs, Ll, H, P, G, N = 2, 6, 4, 3, 2, 5
x = torch.randn(Bs, Ll, H, P); dt = torch.rand(Bs, Ll, H); A = -torch.rand(H)
Bm = torch.randn(Bs, Ll, G, N); Cm = torch.randn(Bs, Ll, G, N); Dm = torch.randn(H); dtb = torch.randn(H)
y = mamba_chunk_scan_combined(x, dt, A, Bm, Cm, 16, D=Dm, dt_bias=dtb, dt_softplus=True)
dtr = F.softplus(dt + dtb); Bh = Bm.repeat_interleave(H // G, 2); Ch = Cm.repeat_interleave(H // G, 2)
dA = torch.exp(dtr * A); st = torch.zeros(Bs, H, P, N); out = []
for t in range(Ll):
    st = st * dA[:, t][..., None, None] + dtr[:, t][..., None, None] * Bh[:, t][:, :, None, :] * x[:, t][:, :, :, None]
    out.append((st * Ch[:, t][:, :, None, :]).sum(-1))
ref = torch.stack(out, 1) + x * Dm[None, None, :, None]
e = (y - ref).abs().max().item()
print(f"ssd_scan          err={e:.2e}  {'OK' if e < 1e-4 else 'FAIL'}"); ok &= e < 1e-4

# 3. causal depthwise conv, output[t] must not depend on x[>t]
xc, wc = torch.randn(1, 4, 8), torch.randn(4, 3)
y1 = causal_conv1d_fn(xc, wc); xc2 = xc.clone(); xc2[:, :, 5:] += 99.0
c = torch.allclose(y1[:, :, :5], causal_conv1d_fn(xc2, wc)[:, :, :5], atol=1e-5)
print(f"causal_conv1d     causal={c}  {'OK' if c else 'FAIL'}"); ok &= c

print("\n=== KERNEL SHIM:", "ALL PASS" if ok else "FAIL", "on", platform.machine(), "===")
sys.exit(0 if ok else 1)
