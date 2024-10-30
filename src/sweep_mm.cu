#include "dispatch_fp8_rowwise_kernel.h"
#include "sweep_mm.h"
#include "torch/torch.h"

namespace driss_torch {
using namespace at;

namespace {

void check_inputs(const at::Tensor &a, const at::Tensor &b,
                  const at::Tensor &scale_a, const at::Tensor &scale_b,
                  const std::optional<at::Tensor> &bias,
                  const at::Tensor &out) {
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.device() == b.device());
  TORCH_CHECK(scale_a.device() == a.device());
  TORCH_CHECK(scale_b.device() == b.device());

  TORCH_CHECK(a.dtype() == at::kFloat8_e4m3fn || a.dtype() == at::kFloat8_e5m2);
  TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(scale_a.dtype() == at::kFloat);
  TORCH_CHECK(scale_b.dtype() == at::kFloat);

  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(0));
  TORCH_CHECK(scale_a.dim() == 2);
  TORCH_CHECK(scale_b.dim() == 2);
  TORCH_CHECK(scale_a.size(0) == a.size(0));
  TORCH_CHECK(scale_a.size(1) == 1);
  TORCH_CHECK(scale_b.size(0) == 1);
  TORCH_CHECK(scale_b.size(1) == b.size(1));

  TORCH_CHECK(a.stride(1) == 1);
  TORCH_CHECK(a.stride(0) >= a.size(1));
  TORCH_CHECK(b.stride(0) == 1);
  TORCH_CHECK(b.stride(1) >= b.size(0));
  TORCH_CHECK(scale_a.stride(0) == 1);
  TORCH_CHECK(scale_b.stride(1) == 1);

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == b.device());
    TORCH_CHECK(bias->dtype() == at::kFloat || bias->dtype() == at::kBFloat16);
    TORCH_CHECK(bias->dim() == 1);
    TORCH_CHECK(bias->size(0) == b.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

  TORCH_CHECK(out.device() == a.device());
  TORCH_CHECK(out.dtype() == at::kBFloat16);
  TORCH_CHECK(out.dim() == 2);
  TORCH_CHECK(out.size(0) == a.size(0));
  TORCH_CHECK(out.size(1) == b.size(1));
  TORCH_CHECK(out.stride(1) == 1);
  TORCH_CHECK(out.stride(0) >= out.size(1));
}

} // namespace

at::Tensor sweep_mm(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                    at::Tensor w_scale, std::optional<at::Tensor> bias,
                    at::ScalarType out_dtype, bool use_fast_accum,
                    const int64_t cluster_shape_x, int64_t cluster_shape_y,
                    const int64_t cluster_shape_z, bool transposed,
                    const int64_t swizzle) {

  const auto M = XQ.size(0);
  const auto N = WQ.size(1);

  Tensor out = at::empty({M, N}, XQ.options().dtype(out_dtype));

  check_inputs(XQ, WQ, x_scale, w_scale, bias, out);
  dispatch_fp8_rowwise_kernel(XQ, WQ, x_scale, w_scale, bias, use_fast_accum,
                              out, cluster_shape_x,
                              cluster_shape_y, cluster_shape_z, transposed,
                              swizzle);
  return out;
}

} // namespace driss_torch
