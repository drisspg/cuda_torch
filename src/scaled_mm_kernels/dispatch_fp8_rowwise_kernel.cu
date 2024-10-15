
#include "dispatch_fp8_rowwise_kernel.h"
#include "f8f8bf16_rowwise_kernel.h"

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
    const int64_t swizzle) {

    // Static dispatch based on the input parameters
    const auto dtype_a = XQ.dtype();
    const auto dtype_b = WQ.dtype();
    const auto bias_dtype = bias.has_value() ? bias.value().scalar_type() : at::kFloat;

    if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 1 &&
           cluster_shape_y == 2 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_1, cute::_2, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == false &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::false_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else if (dtype_a == at::kFloat8_e4m3fn &&
           dtype_b == at::kFloat8_e4m3fn &&
           cluster_shape_x == 2 &&
           cluster_shape_y == 1 &&
           cluster_shape_z == 1 &&
           transposed == true &&
           use_fast_accum == true &&
           bias_dtype == at::kFloat) {
        handle_transposition<
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            std::true_type,
            std::true_type,
            cutlass::float_e4m3_t,
            cutlass::float_e4m3_t,
            float>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }
    else 
    { 
        TORCH_CHECK(false, 
        "No kernel found for the given parameters: ",
        "dtype_a=", dtype_a,
        ", dtype_b=", dtype_b,
        ", cluster_shape=(", cluster_shape_x, ",", cluster_shape_y, ",", cluster_shape_z, ")",
        ", transposed=", transposed,
        ", use_fast_accum=", use_fast_accum,
        ", bias_dtype=", bias_dtype);
    }

}

}  // namespace driss_torch
