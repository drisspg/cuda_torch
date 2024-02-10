#include "include/saturated_cast.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace driss_torch {
using namespace at;

namespace {
__global__ void saturated_cast_kernel_single(
    nv_bfloat16 *input, __nv_fp8_storage_t *output, int n_rows, int n_cols,
    __nv_fp8_interpretation_t out_dtype, nv_bfloat16 *scaler) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
    const nv_bfloat16 scaled_input = __hmul(input[global_index], (*scaler));
    output[global_index] = __nv_cvt_bfloat16raw_to_fp8(
        scaled_input, __nv_saturation_t::__NV_SATFINITE, out_dtype);
  }
}

template <int coarse_factor>
__global__ void saturated_cast_kernel_double_coalesced_flat(
    nv_bfloat162 const *__restrict input,
    __nv_fp8x2_storage_t *__restrict output, const int numels,
    __nv_fp8_interpretation_t out_dtype, nv_bfloat16 const *scaler) {
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_factor;
  const int stride = 1;
  const nv_bfloat162 scale_2 = {(*scaler), (*scaler)};

  nv_bfloat162 scaled_inputs[coarse_factor];
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_idx = idx + i;
    if (temp_idx < numels) {
      scaled_inputs[i] = input[temp_idx * stride];
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_idx = idx + i;
    if (temp_idx < numels) {
      scaled_inputs[i] = __hmul2(scaled_inputs[i], scale_2);
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_idx = idx + i;
    if (temp_idx < numels) {
      output[temp_idx * stride] = __nv_cvt_bfloat16raw2_to_fp8x2(
          scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
    }
  }
}

template <int coarse_factor>
__global__ void saturated_cast_kernel_double_coalesced(
    nv_bfloat162 const *__restrict input,
    __nv_fp8x2_storage_t *__restrict output, int n_rows, int n_cols,
    __nv_fp8_interpretation_t out_dtype, nv_bfloat16 const *scaler) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_factor;
  const int row_stride = n_cols;
  const int col_stride = 1;
  const nv_bfloat162 scale_2 = {(*scaler), (*scaler)};

  nv_bfloat162 scaled_inputs[coarse_factor];
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_col = col + i;
    if (row < n_rows && temp_col < n_cols) {
      scaled_inputs[i] = input[row * row_stride + temp_col * col_stride];
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_col = col + i;
    if (row < n_rows && temp_col < n_cols) {
      scaled_inputs[i] = __hmul2(scaled_inputs[i], scale_2);
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_col = col + i;
    if (row < n_rows && temp_col < n_cols) {
      output[row * row_stride + temp_col * col_stride] =
          __nv_cvt_bfloat16raw2_to_fp8x2(
              scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
    }
  }
}

__nv_fp8_interpretation_t dtype_map(const ScalarType dtype) {
  switch (dtype) {
  case at::kFloat8_e4m3fn:
    return __nv_fp8_interpretation_t::__NV_E4M3;
  case at::kFloat8_e5m2:
    return __nv_fp8_interpretation_t::__NV_E5M2;
  default:
    TORCH_CHECK(false, "Invalid dtype");
  }
}

void dispatch_best_kernel(const Tensor &input, const Tensor &output,
                          __nv_fp8_interpretation_t out_dtype,
                          const Tensor &scale, bool transpose) {
  const int n_rows = input.size(0);
  const int n_cols = input.size(1);
  const int block_size_x = 32;
  const int block_size_y = 32;
  const auto numel = input.numel();
  int kernel_choice = 0;
  if (numel % 2 == 0 && !transpose) {
    kernel_choice = 2;
  } else if (n_cols % 2 == 0) {
    kernel_choice = 1;
  }
  switch (kernel_choice) {
  case 0: {
    const dim3 block(block_size_x, block_size_y);
    const dim3 grid(ceil_div(n_cols, block_size_x),
                    ceil_div(n_rows, block_size_y));
    saturated_cast_kernel_single<<<grid, block>>>(
        static_cast<nv_bfloat16 *>(input.data_ptr()),
        static_cast<__nv_fp8_storage_t *>(output.data_ptr()), n_rows, n_cols,
        out_dtype, static_cast<nv_bfloat16 *>(scale.data_ptr()));
    break;
  }
  case 1: {
    // / We cast to a 16x2 type, so we need to divide the number of columns by 2
    const auto packed_col_size = n_cols / 2;
    // Found 4 to be the best factor for the coalesced kernel
    const int coarse_factor = 4;
    const dim3 block(block_size_x, block_size_y);
    const dim3 grid(ceil_div(packed_col_size, block_size_x * coarse_factor),
                    ceil_div(n_rows, block_size_y));
    saturated_cast_kernel_double_coalesced<coarse_factor><<<grid, block>>>(
        static_cast<nv_bfloat162 *>(input.data_ptr()),
        static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()), n_rows,
        packed_col_size, out_dtype,
        static_cast<nv_bfloat16 *>(scale.data_ptr()));
    break;
  }
  case 2: {
    const int coarse_factor = 4;
    const dim3 block(256);
    const int packed_numel = numel / 2;
    // We divide numel by 2 because we are casting to a 16x2 type
    const dim3 grid(ceil_div(packed_numel, block.x * coarse_factor));
    saturated_cast_kernel_double_coalesced_flat<coarse_factor><<<grid, block>>>(
        static_cast<nv_bfloat162 *>(input.data_ptr()),
        static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()), packed_numel,
        out_dtype, static_cast<nv_bfloat16 *>(scale.data_ptr()));
    break;
  }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

Tensor saturated_cast(const Tensor &input, ScalarType dtype,
                      const Tensor &scale, bool transpose) {
  TORCH_CHECK(dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2,
              "Output tensor must be of type Float8_e4m3fn or Float8_e5m2")
  auto output = torch::empty(input.sizes(), input.options().dtype(dtype));

  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "Input tensor must be of type BFloat16");
  TORCH_CHECK(scale.scalar_type() == at::kBFloat16,
              "Scale must be of type BFloat16");
  dispatch_best_kernel(input, output, dtype_map(dtype), scale, transpose);
  return output;
}

} // namespace driss_torch
