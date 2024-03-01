import torch
from torch.library import impl_abstract

print(__name__)


@impl_abstract("DrissTorch::saturated_cast")
def saturated_cast_meta(
    x: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
):
    return torch.empty_like(x, dtype=out_dtype)
