from pathlib import Path
from typing import Tuple

import torch

lib_path = Path(__file__).parent / ".." / "build" / "libdriss_torch.so"
torch.ops.load_library(str(lib_path.resolve()))
torch.ops.load_library(lib_path)
torch.ops.import_module("driss_torch.abstract_impls")


ops = torch.ops.DrissTorch


def list_ops():
    raise NotImplementedError("This function does not do what I think it should.")
    return ops.__dir__()


def saturated_cast(
    x: torch.Tensor,
    amax: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This op takes in a tensor and returns the fp8 saturated casted version of it.
    Args;
        x: The input tensor.
        out_dtype: The output data type, must be a float8 dtype.
        scale: An on device tensor, this is expected to be a singleton tensor whose value is
            the max(abs(x) before casting, we will use this to calculate the scale
            using the formula `scale = amax / max(max_abs(x), 1e-12)`
        transpose: If true will transpose the input tensor during casting
    Returns:
        The output tensor. And the on device scale tensor.
    """
    assert not transpose, "Transpose is not supported yet"
    return ops.saturated_cast(x, amax, out_dtype, transpose)
