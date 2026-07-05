# Pure-torch shim for mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn
# Matches Mamba-2 gated (group) RMSNorm semantics so NemotronH runs on CPU/ppc64le.
import torch
import torch.nn.functional as F

def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
               norm_before_gate=False, **kwargs):
    dt = x.dtype
    x = x.float()
    if z is not None:
        z = z.float()
        if not norm_before_gate:            # gate BEFORE norm (NemotronH uses this)
            x = x * F.silu(z)
    if group_size is None:
        var = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(var + eps)
    else:                                   # group RMSNorm: normalize within each group
        shp = x.shape
        xg = x.reshape(*shp[:-1], shp[-1] // group_size, group_size)
        var = xg.pow(2).mean(-1, keepdim=True)
        out = (xg * torch.rsqrt(var + eps)).reshape(shp)
    out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:  # gate AFTER norm
        out = out * F.silu(z)
    return out.to(dt)

class RMSNorm(torch.nn.Module):            # some paths import the class form
    def __init__(self, hidden_size, eps=1e-6, group_size=None, **kw):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps; self.group_size = group_size
    def forward(self, x, z=None):
        return rmsnorm_fn(x, self.weight, None, z, self.eps, self.group_size, False)
RMSNormGated = RMSNorm
