import pytest
import torch
from driss_torch import saturated_cast
from float8_experimental.float8_utils import tensor_to_amax, tensor_to_scale


def eager_scaled_quant(
    a: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Quantize tensor to fp8 using a delayed scaled and calculate abs_max

    Args:
        a: Input tensor to quantize
        scale: Scale to apply to input tensor, calculated from previous abs_max
        fp8_dtype: FP8 datatype to quantize to
    """
    out = a * scale
    out = torch.where(out > torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max, out)
    out = torch.where(out < -1 * torch.finfo(fp8_dtype).max, -1 * torch.finfo(fp8_dtype).max, out)
    return out.to(fp8_dtype)


@pytest.mark.parametrize("num_rows", [3, 64, 512, 4096])
@pytest.mark.parametrize("num_cols", [7, 17, 127, 512, 3212, 4097])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
def test_cast_eager(num_rows: int, num_cols: int, in_dtype: torch.dtype, fp8_dtype: torch.dtype):
    torch.manual_seed(0)
    a = torch.rand(num_rows, num_cols, dtype=in_dtype, device="cuda")
    amax = tensor_to_amax(a).to(torch.float32)
    scale = tensor_to_scale(a, fp8_dtype)

    cast_pytorch = eager_scaled_quant(a, scale, fp8_dtype)
    cast_custom, scale_custom = saturated_cast(a, amax, fp8_dtype)

    custom_fp32 = cast_custom.to(torch.float32)
    pytorch_fp32 = cast_pytorch.to(torch.float32)
    torch.testing.assert_close(custom_fp32, pytorch_fp32, atol=1e-5, rtol=0.20)
    torch.testing.assert_close(scale, scale_custom)
    # I worked through examples and I am pretty convinced that the fused kernel is more accurate than
    # eager pytorch
    # The fused kernel says that scaler: 57344.066406 is an example of a scale when the amax is 0.9999988675117493
    # while eager pytorch says 57344.0703125
    # The actual value is 57344.0 / 0.9999988675117493 = 57344.064941479795
    # The difference for the fused kernel is 0.00146 and for pytorch it is: 0.00537
    # The unfortunate thing is that since we then take this scale and multiply it by the input tensor, and
    # convert to fp8e4 or fp8e5 there will be values that get braodcasted near the end of the of range where small
    # epsilon in scale can cause a large difference in the fp8 tensor since the dynamic range is so small at the
    # end of the range.


@pytest.mark.parametrize("num_rows", [3, 64, 512, 4096])
@pytest.mark.parametrize("num_cols", [7, 17, 127, 512, 3212, 4097])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float32])
def test_cast_compile(num_rows: int, num_cols: int, in_dtype: torch.dtype, fp8_dtype: torch.dtype):
    torch._dynamo.reset()
    a = torch.rand(num_rows, num_cols, dtype=in_dtype, device="cuda")
    amax = tensor_to_amax(a).to(torch.float32)
    scale = tensor_to_scale(a, fp8_dtype)

    cast_custom_compile_func = torch.compile(saturated_cast, fullgraph=True)
    cast_custom, scale_custom = saturated_cast(a, amax, fp8_dtype)
    cast_custom_compile, scale_custom_compile = cast_custom_compile_func(a, amax, fp8_dtype)

    custom_fp32 = cast_custom.to(torch.float32)
    custom_compile_fp32 = cast_custom_compile.to(torch.float32)
    torch.testing.assert_close(custom_fp32, custom_compile_fp32)
    torch.testing.assert_close(scale, scale_custom)
    torch.testing.assert_close(scale, scale_custom_compile)


@pytest.mark.xfail(reason="This test is failing, we need to investigate", strict=True)
def test_cast_edge_bug():
    a = torch.Tensor([0.3223, 0.3223, 0.3223]).to(device="cuda", dtype=torch.bfloat16).view(2, 1)
    scale = torch.Tensor([57344.0]).to("cuda")
    cast_pytorch = eager_scaled_quant(a, scale, torch.float8_e5m2)
    cast_custom = saturated_cast(a, scale, torch.float8_e5m2)

    custom_fp32, scale_custom = cast_custom.to(torch.float32)
    pytorch_fp32 = cast_pytorch.to(torch.float32)
    MAX_P_output = a.to(torch.float64) * scale.to(torch.float64)
    print("Custom diff is ", torch.max(torch.abs(MAX_P_output - custom_fp32)))
    print("PyTorch diff is ", torch.max(torch.abs(MAX_P_output - pytorch_fp32)))
    # The closest bit pattern is 0|11101|01 = 20480.0
    # Custom is producing 0|11101|00 = 16384.0 which is wrong unless rounding toward zero is set
    torch.testing.assert_close(custom_fp32, pytorch_fp32)
