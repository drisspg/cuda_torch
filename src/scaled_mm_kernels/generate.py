import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import itertools
from typing import List, Optional, Tuple, Dict
import subprocess
import os

class FastAccum(Enum):
    TRUE = True
    FALSE = False

    def __str__(self) -> str:
        return f"std::{str(self.value).lower()}_type"

class Transposed(Enum):
    TRUE = True
    FALSE = False

    def __str__(self) -> str:
        return f"std::{str(self.value).lower()}_type"

@dataclass
class ClusterShape:
    x: int
    y: int
    z: int

    def __str__(self) -> str:
        return f"cute::Shape<cute::_{self.x}, cute::_{self.y}, cute::_{self.z}>"


class ScalarType(Enum):
    f8e4 = "f8e4"
    f8e5 = "f8e5"
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"

    @classmethod
    def _at_map(cls):
        return {
            cls.f8e4: "at::kFloat8_e4m3fn",
            cls.f8e5: "at::kFloat8_e5m2",
            cls.bf16: "at::kBFloat16",
            cls.fp16: "at::kHalf",
            cls.fp32: "at::kFloat",
        }

    @classmethod
    def _cutlass_map(cls):
        return {
            cls.f8e4: "cutlass::float_e4m3_t",
            cls.f8e5: "cutlass::float_e5m2_t",
            cls.bf16: "cutlass::bfloat16_t",
            cls.fp16: "cutlass::half_t",
            cls.fp32: "float",
        }

    def to_at(self) -> str:
        return self._at_map()[self]

    def to_cutlass(self) -> str:
        return self._cutlass_map()[self]


# Define the combinations we want to generate
DTYPES = [ScalarType.f8e4, ScalarType.f8e4]
CLUSTER_SHAPES = [
    ClusterShape(1, 2, 1),
    ClusterShape(2, 1, 1),
]
TRANSPOSED = [Transposed.FALSE, Transposed.TRUE]
# FAST_ACCUM = [FastAccum.TRUE, FastAccum.FALSE]
FAST_ACCUM = [FastAccum.TRUE]
BIAS_DTYPES = [ScalarType.fp32]
# BIAS_DTYPES = ["kBFloat16", "kFloat"]

KERNEL_TEMPLATE = """
// Auto-generated kernel instantiation
#include "f8f8bf16_rowwise_kernel.h"

namespace driss_torch {{
using namespace at;

template void handle_transposition<
    {cluster_shape},
    {transposed},
    {fast_accum},
    {dtype_a},
    {dtype_b},
    {bias_dtype}>(
        at::Tensor XQ,
        at::Tensor WQ,
        at::Tensor x_scale,
        at::Tensor w_scale,
        std::optional<at::Tensor> bias,
        at::Tensor out,
        const int swizzle);

}} // namespace driss_torch
"""

DISPATCH_HEADER_TEMPLATE = """
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
"""

DISPATCH_IMPL_TEMPLATE = """
#include "dispatch_fp8_rowwise_kernel.h"
#include "f8f8bf16_rowwise_kernel.h"

namespace driss_torch {{

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
    const int64_t swizzle) {{

    // Static dispatch based on the input parameters
    const auto dtype_a = XQ.dtype();
    const auto dtype_b = WQ.dtype();
    const auto bias_dtype = bias.has_value() ? bias.value().scalar_type() : at::kFloat;

    {dispatch_cases}
}}

}}  // namespace driss_torch
"""

def cluster_shape_abbr(shape: ClusterShape) -> str:
    return f"{shape.x}{shape.y}{shape.z}"

def bool_abbr(value: bool) -> str:
    return "T" if value else "F"


def generate_kernel_file(
    output_dir: Path,
    dtype_a: ScalarType,
    dtype_b: ScalarType,
    cluster_shape: ClusterShape,
    transposed: Transposed,
    fast_accum: FastAccum,
    bias_dtype: ScalarType,
) -> str:
    kernel_name = f"kernel_{dtype_a.name}_{dtype_b.name}_{cluster_shape_abbr(cluster_shape)}_T{bool_abbr(transposed.value)}_FA{bool_abbr(fast_accum.value)}_{bias_dtype.name}"
    content = KERNEL_TEMPLATE.format(
        dtype_a=dtype_a.to_cutlass(),
        dtype_b=dtype_b.to_cutlass(),
        cluster_shape=str(cluster_shape),
        transposed=str(transposed),
        fast_accum=str(fast_accum),
        bias_dtype=bias_dtype.to_cutlass(),
    )
    filename = f"{kernel_name}.cu"
    (output_dir / filename).write_text(content)
    return kernel_name


def generate_kernels(output_dir: Path) -> List[Dict]:
    kernel_configs = []
    for dtype_a, dtype_b, cluster_shape, transposed, fast_accum, bias_dtype in itertools.product(
        DTYPES, DTYPES, CLUSTER_SHAPES, TRANSPOSED, FAST_ACCUM, BIAS_DTYPES
    ):
        kernel_name = generate_kernel_file(
            output_dir,
            dtype_a,
            dtype_b,
            cluster_shape,
            transposed,
            fast_accum,
            bias_dtype,
        )
        kernel_configs.append(
            {
                "name": kernel_name,
                "dtype_a": dtype_a,
                "dtype_b": dtype_b,
                "cluster_shape": cluster_shape,
                "transposed": transposed,
                "fast_accum": fast_accum,
                "bias_dtype": bias_dtype,
            }
        )
    return kernel_configs


def generate_dispatch_files(output_dir: Path, kernel_configs: List[Dict]):
    dispatch_header = DISPATCH_HEADER_TEMPLATE
    (output_dir / "dispatch_fp8_rowwise_kernel.h").write_text(dispatch_header)

    dispatch_cases = []
    for config in kernel_configs:
        condition = f"""dtype_a == {config['dtype_a'].to_at()} &&
           dtype_b == {config['dtype_b'].to_at()} &&
           cluster_shape_x == {config['cluster_shape'].x} &&
           cluster_shape_y == {config['cluster_shape'].y} &&
           cluster_shape_z == {config['cluster_shape'].z} &&
           transposed == {str(config['transposed'].value).lower()} &&
           use_fast_accum == {str(config['fast_accum'].value).lower()} &&
           bias_dtype == {config['bias_dtype'].to_at()}"""

        dispatch_case = f"""if ({condition}) {{
        handle_transposition<
            {str(config['cluster_shape'])},
            {str(config['transposed'])},
            {str(config['fast_accum'])},
            {config['dtype_a'].to_cutlass()},
            {config['dtype_b'].to_cutlass()},
            {config['bias_dtype'].to_cutlass()}>(
                XQ, WQ, x_scale, w_scale, bias, out, swizzle);
        return;
    }}"""
        dispatch_cases.append(dispatch_case)

    catch_all = """
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
"""
    dispatch_cases.append(catch_all)
    dispatch_impl = DISPATCH_IMPL_TEMPLATE.format(
        dispatch_cases="\n    else ".join(dispatch_cases)
    )
    (output_dir / "dispatch_fp8_rowwise_kernel.cu").write_text(dispatch_impl)

def main(output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_configs = generate_kernels(output_dir)
    generate_dispatch_files(output_dir, kernel_configs)

    print(f"Generated and compiled kernel files, and created dispatch mechanism in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and compile CUDA kernel instantiations for static dispatch"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to output generated kernel files (default: ./generated_kernels)",
    )
    args = parser.parse_args()
    main(args.output_dir)
