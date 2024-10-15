#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

// Determine if the architecture supports rowwise scaled mm
// Currenlty failing on windows with: https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>

// Rename the global function symbol
// #define cuTensorMapEncodeTiled nvrtc_cuTensorMapEncodeTiled
#include <cute/tensor.hpp>
// #undef cuTensorMapEncodeTiled
// // Set everything back to normal

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

namespace driss_torch {

namespace{

constexpr int kNumSMsForH100 = 132;

using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;

using Multiply = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies,
    DtypeEpilogue,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

using Add = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus,
    DtypeEpilogue,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

using Cast = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::Identity,
    DtypeOutput,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

template <bool PingPong, bool FastAccum>
struct Schedule;

template <>
struct Schedule</*PingPong=*/false, /*FastAccum=*/false> {
  using type = cutlass::gemm::KernelTmaWarpSpecialized;
};

template <>
struct Schedule</*PingPong=*/true, /*FastAccum=*/false> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
};

template <>
struct Schedule</*PingPong=*/false, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
};

template <>
struct Schedule</*PingPong=*/true, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
};

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

// Cutlass rowwise kernel
template <
    typename TileShape,
    typename ClusterShape,
    typename PingPong,
    typename Transposed,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias>
void f8f8bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  // Workaround for https://github.com/pytorch/pytorch/issues/133334.
  if (M % 256 > 0) {
    int padded_M = ((M - 1) / 256 + 1) * 256;
    at::Tensor padded_x_scale = x_scale.new_empty({padded_M, 1});
    padded_x_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/M)
        .copy_(std::move(x_scale));
    x_scale = std::move(padded_x_scale);
  }

  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);

  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

  using LayoutOutput = std::conditional_t<
      Transposed::value,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // Implement rowwise scaling epilogue.
  constexpr int ColBroadcastStages = 0;
  constexpr int RowBroadcastStages = 0;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeScale>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeScale>;

  using Bias = std::conditional_t<
      Transposed::value,
      cutlass::epilogue::fusion::
          Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeBias>,
      cutlass::epilogue::fusion::
          Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeBias>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Add,
          Bias,
          cutlass::epilogue::fusion::Sm90EVT<
              Multiply,
              XScale,
              cutlass::epilogue::fusion::Sm90EVT<Multiply, WScale, Accum>>>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          DtypeAccum,
          DtypeEpilogue,
          DtypeOutput,
          LayoutOutput,
          AlignmentOutput,
          DtypeOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecialized,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutInputA,
          AlignmentInputA,
          DtypeB,
          LayoutInputB,
          AlignmentInputB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Schedule<PingPong::value, FastAccum::value>::type>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, static_cast<int>(XQ.stride(0)), 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, static_cast<int>(WQ.stride(1)), 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, static_cast<int>(out.stride(0)), 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<DtypeA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<DtypeB*>(WQ.data_ptr()),
       stride_b},
      {{{{bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr())
                           : nullptr},
         {{reinterpret_cast<DtypeScale*>(x_scale.data_ptr())},
          {{reinterpret_cast<DtypeScale*>(w_scale.data_ptr())}}}}},
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output,
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output}};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Set the swizzle size
  arguments.scheduler.max_swizzle_size = swizzle;

  // Allocate workspace memory
  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ClusterShape, typename... Types>
void dispatch_fp8_rowwise_kernel_on_tile_size(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  int M = XQ.size(0);
  int N = WQ.size(1);

  // We prefer to use smaller tiles (less wasted compute in case of padding),
  // but if this causes us to have more CUDA blocks than there are SMs on the
  // GPU then we'll hit wave quantization, hence we'll switch to larger tiles.
  if (ceildiv(M, 64 * cute::get<0>(ClusterShape{})) *
          ceildiv(N, 128 * cute::get<1>(ClusterShape{})) <=
      kNumSMsForH100 / cute::size(ClusterShape{})) {
    return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_64, cute::_128, cute::_128>,
        ClusterShape,
        /*PingPong=*/std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
  } else {
    return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_128, cute::_128, cute::_128>,
        ClusterShape,
        /*PingPong=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
  }
}

} // namespace
template <
    typename ClusterShape,
    typename Transposed,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias>
void handle_transposition(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  if constexpr (!Transposed::value) {
    dispatch_fp8_rowwise_kernel_on_tile_size<
        ClusterShape,
        Transposed,
        FastAccum,
        DtypeA,
        DtypeB,
        DtypeBias>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
  } else {
    dispatch_fp8_rowwise_kernel_on_tile_size<
        ClusterShape,
        Transposed,
        FastAccum,
        DtypeB,
        DtypeA,
        DtypeBias>(WQ.t(), XQ.t(), w_scale.t(), x_scale.t(), bias, out.t(), swizzle);
  }
}

#endif // !defined(USE_ROCM)

} // driss_torch
