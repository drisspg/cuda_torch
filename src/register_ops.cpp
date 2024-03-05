#include <c10/core/DispatchKey.h>
#include <torch/library.h>

// Custom up headers
#include "saturated_cast.h"

TORCH_LIBRARY(DrissTorch, m) {
  m.impl_abstract_pystub("driss_torch.abstract_impls");
  //   Saturated cast func from bf16 to fp8 types
  m.def("saturated_cast(Tensor input, Tensor amax, ScalarType dtype, bool transpose) -> (Tensor, Tensor)");
  m.impl("saturated_cast", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::saturated_cast));
}
