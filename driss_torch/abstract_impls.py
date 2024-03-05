import torch
from torch.library import impl_abstract


@impl_abstract("DrissTorch::saturated_cast")
def saturated_cast_meta(
    x: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
):
    return torch.empty_like(x, dtype=out_dtype), torch.empty(
        (), device=x.device, dtype=torch.float32
    )
