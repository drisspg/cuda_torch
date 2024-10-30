from pathlib import Path
from typing import Optional

import torch

lib_path = Path(__file__).parent / "lib" / "libdriss_torch.so"
torch.ops.load_library(str(lib_path.resolve()))
torch.ops.load_library(lib_path)
torch.ops.import_module("driss_torch.abstract_impls")


ops = torch.ops.DrissTorch
Tensor = torch.Tensor


def list_ops():
    raise NotImplementedError("This function does not do what I think it should.")
    return ops.__dir__()


def saturated_cast(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    """This op takes in a tensor and returns the fp8 saturated casted version of it.
    Args;
        x: The input tensor.
        out_dtype: The output data type, must be a float8 dtype.
        scale: An on device tensor, this is expected to be a singleton tensor whose value is
            the max(abs(x) before casting, we will use this to calculate the scale
            using the formula `scale = amax / max(max_abs(x), 1e-12)`
        transpose: If true will transpose the input tensor during casting
    Returns:
        The output tensor.
    """
    assert not transpose, "Transpose is not supported yet"
    return ops.saturated_cast(x, scale, out_dtype, transpose)


def amax(x: torch.Tensor) -> float:
    """This op takes in a tensor and returns the max absolute value of it."""
    return ops.amax(x)


def sweep_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    use_fast_accum: bool,
    cluster_shape_x: int,
    cluster_shape_y: int,
    cluster_shape_z: int,
    transposed: bool,
    swizzle: int,
) -> torch.Tensor:
    return ops.sweep_mm(
        x,
        w,
        x_scale,
        w_scale,
        bias,
        out_dtype,
        use_fast_accum,
        cluster_shape_x,
        cluster_shape_y,
        cluster_shape_z,
        transposed,
        swizzle,
    )
