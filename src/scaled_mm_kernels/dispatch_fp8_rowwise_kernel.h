
#pragma once

#include <torch/torch.h>
#include <cute/layout.hpp>

namespace driss_torch {

void dispatch_fp8_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor out,
    const int64_t cluster_shape_x,
    const int64_t cluster_shape_y,
    const int64_t cluster_shape_z,
    bool transposed,
    const int64_t swizzle);
}  // driss_torch
