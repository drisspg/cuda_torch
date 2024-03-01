#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor saturated_cast(const at::Tensor &input, const at::Tensor &amax,
                          at::ScalarType dtype, bool transpose);
} // namespace driss_torch
