import torch.nn.functional as F
import torch


def swiglu_custom(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    h1 = F.silu(x1) * x2
    h2 = F.silu(x2) * x1
    out = torch.cat([h1, h2], dim=-1)
    return out
