#include <c10/core/DispatchKey.h>
#include <torch/library.h>

// Custom up headers
#include "saturated_cast.h"
#include "amax.h"

TORCH_LIBRARY(DrissTorch, m) {
  m.impl_abstract_pystub("driss_torch.abstract_impls");
  //   Saturated cast func from bf16 to fp8 types
  m.def("saturated_cast(Tensor input, Tensor scale, ScalarType dtype, bool transpose) -> Tensor", {at::Tag::needs_fixed_stride_order});
  m.impl("saturated_cast", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::saturated_cast));
  //   Amax func
  m.def("amax(Tensor input) -> Tensor");
  m.impl("amax", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::amax));
}
