#include <c10/core/DispatchKey.h>
#include <torch/library.h>

// Custom up headers
#include "saturated_cast.h"
#include "amax.h"
#include "sweep_mm.h"

TORCH_LIBRARY(DrissTorch, m) {
  m.impl_abstract_pystub("driss_torch.abstract_impls");
  //   Saturated cast func from bf16 to fp8 types
  m.def("saturated_cast(Tensor input, Tensor scale, ScalarType dtype, bool transpose) -> Tensor");
  m.impl("saturated_cast", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::saturated_cast));
  //   Amax func
  m.def("amax(Tensor input) -> Tensor");
  m.impl("amax", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::amax));
  // sweep_mm
  m.def("sweep_mm(Tensor x, Tensor w, Tensor x_scale, Tensor w_scale, Tensor? bias , ScalarType out_dtype, bool use_fast_accum, int cluster_shape_x, int cluster_shape_y, int cluster_shape_z, bool transposed, int swizzle) -> Tensor");
  m.impl("sweep_mm", c10::DispatchKey::CUDA, TORCH_FN(driss_torch::sweep_mm));
}
