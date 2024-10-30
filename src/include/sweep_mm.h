#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor sweep_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::ScalarType out_dtype,
    bool use_fast_accum,
    const int64_t cluster_shape_x,
    const int64_t cluster_shape_y,
    const int64_t cluster_shape_z,
    bool transposed,
    const int64_t swizzle);

} // namespace driss_torch
