#include "include/saturated_cast.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <tuple>
#include <type_traits>

namespace driss_torch {
using namespace at;

namespace {
__forceinline__ __device__ void set_scale(float *scale, float scaler) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.y == 0) {
    *scale = scaler;
  }
}
#define DISPATCH_KERNEL_SINGLE(T)                                              \
  saturated_cast_kernel_single<T><<<grid, block>>>(                            \
      static_cast<T *>(input.data_ptr()),                                      \
      static_cast<__nv_fp8_storage_t *>(output.data_ptr()), n_rows, n_cols,    \
      out_dtype, static_cast<float *>(amax.data_ptr()),                        \
      static_cast<float *>(scale.data_ptr()))

#define DISPATCH_KERNEL_DOUBLE_COALESCED(T)                                    \
  saturated_cast_kernel_double_coalesced<coarse_factor, T><<<grid, block>>>(   \
      static_cast<T *>(input.data_ptr()),                                      \
      static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()), n_rows, n_cols,  \
      out_dtype, static_cast<float *>(amax.data_ptr()),                        \
      static_cast<float *>(scale.data_ptr()))

#define DISPATCH_KERNEL_DOUBLE_COALESCED_FLAT(T)                               \
  saturated_cast_kernel_double_coalesced_flat<coarse_factor, T>                \
      <<<grid, block>>>(                                                       \
          static_cast<T *>(input.data_ptr()),                                  \
          static_cast<__nv_fp8x2_storage_t *>(output.data_ptr()),              \
          packed_numel, out_dtype, static_cast<float *>(amax.data_ptr()),      \
          static_cast<float *>(scale.data_ptr()))

float __forceinline__ __device__
get_dtype_max(__nv_fp8_interpretation_t out_dtype) {
  return out_dtype == __nv_fp8_interpretation_t::__NV_E4M3 ? 448.0f : 57344.0f;
}

template <typename HPType>
__global__ void
saturated_cast_kernel_single(HPType const *input, __nv_fp8_storage_t *output,
                             const int n_rows, const int n_cols,
                             __nv_fp8_interpretation_t out_dtype,
                             const float *amax, float *scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const float dtype_max = get_dtype_max(out_dtype);
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
    const float scaler = dtype_max / std::max((*amax), 1e-12f);
    set_scale(scale, scaler);
    if constexpr (std::is_same_v<HPType, nv_bfloat16>) {
      const HPType scaled_input = __hmul(input[global_index], scaler);
      output[global_index] = __nv_cvt_bfloat16raw_to_fp8(
          scaled_input, __nv_saturation_t::__NV_SATFINITE, out_dtype);
    } else {
      const HPType scaled_input = input[global_index] * scaler;
      output[global_index] = __nv_cvt_float_to_fp8(
          scaled_input, __nv_saturation_t::__NV_SATFINITE, out_dtype);
    }
  }
}

template <int coarse_factor, typename PackedHPType>
__global__ void saturated_cast_kernel_double_coalesced_flat(
    PackedHPType const *__restrict input,
    __nv_fp8x2_storage_t *__restrict output, const int numels,
    __nv_fp8_interpretation_t out_dtype, float const *amax, float *scale) {
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_factor;
  const int stride = 1;
  const float dtype_max = get_dtype_max(out_dtype);
  const float scaler = dtype_max / std::max((*amax), 1e-12f);
  set_scale(scale, scaler);
  const PackedHPType scale_2 = {scaler, scaler};

  PackedHPType scaled_inputs[coarse_factor];
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
      if constexpr (std::is_same_v<PackedHPType, nv_bfloat162>) {
        scaled_inputs[i] = __hmul2(scaled_inputs[i], scale_2);
      } else {
        // I can't find the right fmul2 fo this??
        scaled_inputs[i] = {scaled_inputs[i].x * scaler,
                            scaled_inputs[i].y * scaler};
      }
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_idx = idx + i;
    if (temp_idx < numels) {
      __nv_fp8x2_storage_t out;
      if constexpr (std::is_same_v<PackedHPType, nv_bfloat162>) {
        out = __nv_cvt_bfloat16raw2_to_fp8x2(
            scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
      } else {
        out = __nv_cvt_float2_to_fp8x2(
            scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
      }
      output[temp_idx * stride] = out;
    }
  }
}

template <int coarse_factor, typename PackedHPType>
__global__ void saturated_cast_kernel_double_coalesced(
    PackedHPType const *__restrict input,
    __nv_fp8x2_storage_t *__restrict output, int n_rows, int n_cols,
    __nv_fp8_interpretation_t out_dtype, float const *amax, float *scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_factor;
  const int row_stride = n_cols;
  const int col_stride = 1;
  const float dtype_max = get_dtype_max(out_dtype);
  const float scaler = dtype_max / std::max((*amax), 1e-12f);
  set_scale(scale, scaler);
  const PackedHPType scale_2 = {scaler, scaler};

  PackedHPType scaled_inputs[coarse_factor];
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
      if constexpr (std::is_same_v<PackedHPType, nv_bfloat162>) {
        scaled_inputs[i] = __hmul2(scaled_inputs[i], scale_2);
      } else {
        // I can't find the right fmul2 fo this??
        scaled_inputs[i] = {scaled_inputs[i].x * scaler,
                            scaled_inputs[i].y * scaler};
      }
    }
  }
#pragma unroll
  for (int i{0}; i < coarse_factor; ++i) {
    const int temp_col = col + i;
    if (row < n_rows && temp_col < n_cols) {
      __nv_fp8x2_storage_t out;
      if constexpr (std::is_same_v<PackedHPType, nv_bfloat162>) {
        out = __nv_cvt_bfloat16raw2_to_fp8x2(
            scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
      } else {
        out = __nv_cvt_float2_to_fp8x2(
            scaled_inputs[i], __nv_saturation_t::__NV_SATFINITE, out_dtype);
      }
      output[row * row_stride + temp_col * col_stride] = out;
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

enum KernelChoice { single, coalesced, coalesced_flat };

void dispatch_best_kernel(const Tensor &input, const Tensor &output,
                          __nv_fp8_interpretation_t out_dtype,
                          const Tensor &amax, Tensor &scale, bool transpose) {
  const int n_rows = input.size(0);
  const int n_cols = input.size(1);
  const int block_size_x = 32;
  const int block_size_y = 32;
  const auto numel = input.numel();
  int kernel_choice = KernelChoice::single;
  if (numel % 2 == 0 && !transpose) {
    kernel_choice = KernelChoice::coalesced_flat;
  } else if (n_cols % 2 == 0) {
    kernel_choice = KernelChoice::coalesced;
  }
  switch (kernel_choice) {
  case KernelChoice::single: {
    const dim3 block(block_size_x, block_size_y);
    const dim3 grid(ceil_div(n_cols, block_size_x),
                    ceil_div(n_rows, block_size_y));
    if (input.scalar_type() == at::kBFloat16) {
      DISPATCH_KERNEL_SINGLE(nv_bfloat16);
    } else if (input.scalar_type() == at::kFloat) {
      DISPATCH_KERNEL_SINGLE(float);
    }
    break;
  }
  case KernelChoice::coalesced: {
    // / We cast to a 16x2 type, so we need to divide the number of columns by 2
    const auto packed_col_size = n_cols / 2;
    // Found 4 to be the best factor for the coalesced kernel
    const int coarse_factor = 4;
    const dim3 block(block_size_x, block_size_y);
    const dim3 grid(ceil_div(packed_col_size, block_size_x * coarse_factor),
                    ceil_div(n_rows, block_size_y));
    if (input.scalar_type() == at::kBFloat16) {
      DISPATCH_KERNEL_DOUBLE_COALESCED(nv_bfloat162);
    } else if (input.scalar_type() == at::kFloat) {
      DISPATCH_KERNEL_DOUBLE_COALESCED(float2);
    }
    break;
  }
  case KernelChoice::coalesced_flat: {
    const int coarse_factor = 4;
    const dim3 block(256);
    const int packed_numel = numel / 2;
    // We divide numel by 2 because we are casting to a 16x2 type
    const dim3 grid(ceil_div(packed_numel, block.x * coarse_factor));
    if (input.scalar_type() == at::kBFloat16) {
      DISPATCH_KERNEL_DOUBLE_COALESCED_FLAT(nv_bfloat162);
    } else if (input.scalar_type() == at::kFloat) {
      DISPATCH_KERNEL_DOUBLE_COALESCED_FLAT(float2);
    }
    break;
  }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

std::tuple<Tensor, Tensor> saturated_cast(const Tensor &input,
                                          const Tensor &amax, ScalarType dtype,
                                          bool transpose) {
  TORCH_CHECK(dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2,
              "Output tensor must be of type Float8_e4m3fn or Float8_e5m2")

  TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Input tensor must be of type BFloat16 or Float, but got ",
              input.dtype());
  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D, but got ",
              input.dim());

  TORCH_CHECK(amax.scalar_type() == at::kFloat,
              "Amax tensor must be of type Float, but got ", amax.dtype())
  TORCH_CHECK(amax.numel() == 1, "Amax tensor must be a scalar, but got ",
              amax.numel());

  // Input must either be transposed or contiguous
  auto strides = input.strides();
  bool is_contiguous = input.is_contiguous();
  bool is_transposed = strides[0] == 1 && strides[1] == input.size(0);
  bool check_allowed_strides =
      (is_contiguous || is_transposed) && input.storage_offset() == 0;
  auto contig_input = check_allowed_strides ? input : input.contiguous();

  auto output =
      torch::empty_like(contig_input, contig_input.options().dtype(dtype));
  auto scale = torch::empty({}, amax.options().dtype(at::kFloat));
  dispatch_best_kernel(contig_input, output, dtype_map(dtype), amax, scale,
                       transpose);
  return {output, scale};
}

} // namespace driss_torch
