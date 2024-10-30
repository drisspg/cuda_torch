
// Auto-generated kernel instantiation
#include "f8f8bf16_rowwise_kernel.h"

namespace driss_torch {
using namespace at;

template void handle_transposition<
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    std::true_type,
    std::true_type,
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    float>(
        at::Tensor XQ,
        at::Tensor WQ,
        at::Tensor x_scale,
        at::Tensor w_scale,
        std::optional<at::Tensor> bias,
        at::Tensor out,
        const int swizzle);

} // namespace driss_torch
