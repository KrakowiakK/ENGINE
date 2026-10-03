// Copyright © 2023-2024 Apple Inc.

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

// ENGINE_QMM_TRACE=1 (tools/profile/qmm_paths.sh, gate G2): one line per quantized-matmul DISPATCH of 2..64 rows naming the
// kernel that actually runs (tile and split-K included) -- a decision-level trace once said "qmm" while MLX's split-K
// kernel, not the tile under test, ran those shapes.
// ENGINE_QMM_TRACE=1: one line per quantized matmul dispatch of 2..64 rows; ENGINE_QMM_TRACE=unique: each distinct
// (path, kernel, M, N, K, split) once (microbenchmarks dispatch the same shape thousands of times).
static void engine_qmm_trace(const char* fn, const std::string& kname, int M, int N, int K, int split) {
  static const char* mode = std::getenv("ENGINE_QMM_TRACE");
  if (!mode || M < 2 || M > 64) return;
  if (std::strcmp(mode, "unique") == 0) {
    static std::mutex mu;
    static std::set<std::string> seen;
    std::string key = std::string(fn) + "|" + kname + "|" + std::to_string(M) + "|" + std::to_string(N) + "|" +
        std::to_string(K) + "|" + std::to_string(split);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) return;
  }
  std::fprintf(stderr, "qmm trace: M=%d N=%d K=%d fn=%s kernel=%s split=%d\n", M, N, K, fn, kname.c_str(), split);
}

// ENGINE quantized-matmul policy (measured on the M3 Ultra, KB C-0064..C-0069): transposed 8-bit products of 5..16 rows
// with WIDE outputs (N >= 8192) take a 16x32x32 qmm tile, NARROW outputs take split-K with a 16x32x32 tile from 9 rows;
// 1..4 rows and 17+ rows keep MLX's choice (xr8 qmv beats every tile at 2..4 rows; a 16-row tile loses from 17 rows).
// ENGINE_QMM_POLICY=mlx restores MLX's choices everywhere; each ENGINE_Q* knob still overrides its own default.
static bool engine_qmm_policy() {
  static const bool on = [] { const char* e = std::getenv("ENGINE_QMM_POLICY"); return !(e && std::strcmp(e, "mlx") == 0); }();
  return on;
}
static int engine_qmm_int(const char* name, int policy_value, int mlx_value) {
  const char* e = std::getenv(name);
  if (e && e[0]) return std::atoi(e);
  return engine_qmm_policy() ? policy_value : mlx_value;
}
static std::array<int, 3> engine_qmm_tile(const char* name, std::array<int, 3> policy_value, std::array<int, 3> mlx_value) {
  std::array<int, 3> t = engine_qmm_policy() ? policy_value : mlx_value;
  if (const char* e = std::getenv(name); e && e[0]) { std::sscanf(e, "%d,%d,%d", &t[0], &t[1], &t[2]); }
  return t;
}


namespace {

template <typename... Args>
auto get_quantized_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_quantized_kernel(d, name, template_def, mode);
}

template <typename... Args>
auto get_qmm_nax_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_qmm_nax_kernel(d, name, template_def, mode);
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    int offset) {
  if (skip) {
    return 0;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }

  return offset;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(strides[0], offset++);
  compute_encoder.set_vector_bytes(strides[1], offset++);

  return offset;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());

  concatenate(
      kname,
      mode + "_qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_quad", mode, type_string, group_size, bits, K, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  engine_qmm_trace("qmv_quad", kname, M, N, K, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;

  // ENGINE P095: the cross-row twin for an unbatched 8-bit g64 bfloat16 qmv_fast at 2..9 rows --
  // the weight tile read once for the group instead of once per row, the per-row arithmetic the
  // stock kernel's (bit-identical outputs). ENGINE_QMV_CROSSROW8=0 restores the stock dispatch.
  static const bool crossrow8 = [] {
    const char* e = std::getenv("ENGINE_QMV_CROSSROW8");
    return !(e && e[0] == '0' && e[1] == '\0');
  }();
  static const int crossrow8_min_n = [] {
    // P095 (OBS-ENG-182) took the head alone: with a runtime input count the group kept its
    // accumulators in memory and every narrower projection LOST. P096 (OBS-ENG-185) made the count
    // a template constant; the kernel then beats the stock dispatch at M = 8 on every projection
    // (qkvi 140 vs 162 us, GDN in_proj 159 vs 181, o_proj 69 vs 90, out 72 vs 88) and at M = 4 on
    // the wide ones, so the default takes every 8-bit projection (N >= 2560). Was 65536.
    const char* e = std::getenv("ENGINE_QMV_CROSSROW8_MIN_N");
    return (e && e[0]) ? std::atoi(e) : 2560;
  }();
  const bool xr8 = crossrow8 && fast && B == 1 && M >= 2 && M <= 9 && bits == 8 &&
      group_size == 64 && mode == "affine" && x.dtype() == bfloat16 && N >= crossrow8_min_n;
  // P098: a fixed rows-per-group table for the kernelbench arms (2 = pairs everywhere, 8 = one group)
  static const int crossrow8_ipg = [] {
    const char* e = std::getenv("ENGINE_QMV_CROSSROW8_IPG");
    return (e && e[0]) ? std::atoi(e) : 0;
  }();
  const std::string xr_name = crossrow8_ipg == 2 ? "qmv_fast_xr8g2" : (crossrow8_ipg == 8 ? "qmv_fast_xr8g8" : "qmv_fast_xr8");
  concatenate(
      kname,
      mode + (fast ? (xr8 ? "_" + xr_name + "_" : "_qmv_fast_") : "_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? (xr8 ? xr_name : "qmv_fast") : "qmv"),
      mode,
      type_string,
      group_size,
      bits,
      B > 1);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  engine_qmm_trace("qmv", kname, M, N, K, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  auto& compute_encoder = metal::get_command_encoder(s);

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);
  MTL::Size grid_dims = MTL::Size(M, N / bn, B);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm_split_k", mode, type_string, group_size, bits, split_k);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (biases) {
    auto b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm", mode, type_string, group_size, bits, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_nax_" : "_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  // The gather qmm NAX kernels are instantiated with BK = 64 only; any
  // other value here makes the kernel name lookup fail.
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        bm,
        bk,
        bn,
        wm,
        wn);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  int bk = 32;
  // ENGINE: small-M tiles for the transposed qmm (verify blocks and batched rows of 5..16 rows ran in a 32-row tile, half
  // of its MMA work on padding). ENGINE_QMM_SMALL_M = largest M routed here (policy 16, MLX 0 = off), ENGINE_QMM_TILE =
  // "BM,BK,BN" (BM, BN multiples of 16; BK a multiple of 32). The kernel template already takes BM/BK/BN.
  static const int small_m = engine_qmm_int("ENGINE_QMM_SMALL_M", 16, 0);
  static const std::array<int, 3> small_tile = engine_qmm_tile("ENGINE_QMM_TILE", {16, 32, 32}, {16, 32, 32});
  // ENGINE_QMM_SMALL_MIN_N: only outputs this wide take the small tile (default 8192). Without split-K a tile's time has a
  // floor set by its serial K loop (~0.73..0.88 us per 32-wide step at any N up to 256 threadgroups), while qmv scales with
  // the bytes, so the tile wins only on wide outputs (C-0065); narrow ones take split-K below.
  static const int small_min_n = engine_qmm_int("ENGINE_QMM_SMALL_MIN_N", 8192, 8192);
  const bool small = transpose && small_m > 0 && M <= small_m && N >= small_min_n && K % small_tile[1] == 0;
  if (small) { bm = small_tile[0]; bk = small_tile[1]; bn = small_tile[2]; }
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_" : "_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0",
      small ? "_tile_" + std::to_string(bm) + "_" + std::to_string(bk) + "_" + std::to_string(bn) : "");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose && small) {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_t", mode, type_string, group_size, bits, aligned, batched, bm, bk, bn);
  } else if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_n", mode, type_string, group_size, bits, batched);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  engine_qmm_trace("qmm", kname, M, N, K, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // Choose split_k to target ~512 threadgroups
  static const int nosplit_small = engine_qmm_int("ENGINE_QMM_SMALL_NOSPLIT", 0, 0);
  if (nosplit_small > 0 && M <= nosplit_small) {
    return qmm(x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }
  // ENGINE split-K knobs: ENGINE_QMM_SPLIT_TILE = "BM,BK,BN" of the split kernel (policy 16,32,32: 12..16-row narrow
  // outputs 74 -> 52 us on 2560x6144, C-0066), ENGINE_QMM_SPLIT_K = a forced split factor, ENGINE_QMM_SPLIT_TGS = the
  // threadgroup target of the automatic factor (MLX: 512), ENGINE_QMM_SPLIT_TILE_MAX_M = the custom tile only up to this
  // many rows (policy 16: at 17..32 rows a 16-row tile makes two row tiles read every weight tile, 320x10240 M=20
  // 24.9 -> 29.9 us, C-0068).
  static const std::array<int, 3> split_tile = engine_qmm_tile("ENGINE_QMM_SPLIT_TILE", {16, 32, 32}, {32, 32, 32});
  static const int split_forced = engine_qmm_int("ENGINE_QMM_SPLIT_K", 0, 0);
  static const int split_tgs = engine_qmm_int("ENGINE_QMM_SPLIT_TGS", 512, 512);
  static const int split_tile_max_m = engine_qmm_int("ENGINE_QMM_SPLIT_TILE_MAX_M", 16, 0);
  const bool tile_applies = split_tile_max_m <= 0 || M <= split_tile_max_m;
  int bm = tile_applies ? split_tile[0] : 32, bk = tile_applies ? split_tile[1] : 32, bn = tile_applies ? split_tile[2] : 32;
  const bool split_custom = bm != 32 || bk != 32 || bn != 32;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = split_forced > 0 ? split_forced : std::max(1, split_tgs / current_tgs);

  // Each K partition must be a whole number of BK-wide (32) K-tiles as well as
  // whole quantization groups. The qmm_t_splitk kernels tile K by BK=32 and do
  // not bound the K dimension, so a partition smaller than BK (e.g. nvfp4's
  // group_size=16) would over-read into the next group's weights/scales.
  int k_align = std::max(group_size, bk);
  split_k = std::min(split_k, K / k_align);

  // Ensure K divides evenly by split_k * k_align
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(
        x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }

  int k_partition_size = K / split_k;
  int split_k_partition_stride = M * N;

  // Allocate intermediate buffer: insert split_k at the front so that
  // partition_stride = M * N matches the leading stride of the buffer.
  auto& compute_encoder = metal::get_command_encoder(s);
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  // Grid: (N_tiles, M_tiles, split_k)
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);

  bool aligned = N % bn == 0;
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false",
      split_custom ? "_tile_" + std::to_string(bm) + "_" + std::to_string(bk) + "_" + std::to_string(bn) : "");
  auto kernel = split_custom
      ? get_quantized_kernel_wrapped(d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned, bm, bk, bn)
      : get_quantized_kernel_wrapped(d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);

  engine_qmm_trace("qmm_splitk", kname, M, N, K, split_k);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Sum across split_k dimension (axis 0)
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(0)},
      {intermediate.strides(0)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {0}, compute_encoder, d, s);
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return gather_qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* const array& lhs_indices = */ lhs_indices,
        /* const array& rhs_indices = */ rhs_indices,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_t", mode, type_string, group_size, bits, aligned);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_n", mode, type_string, group_size, bits);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "gather_qmv_fast" : "gather_qmv"),
      mode,
      type_string,
      group_size,
      bits);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_gather_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "gather_qvm", mode, type_string, group_size, bits);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c++);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs_nax(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 64, bn = 64, bk = 64;
  int wm = 2, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode +
          (transpose ? "_gather_qmm_rhs_nax_nt_" : "_gather_qmm_rhs_nax_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_nax_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  if (metal::is_nax_available() && transpose &&
      (env::enable_tf32() || x_.dtype() != float32)) {
    return gather_qmm_rhs_nax(
        /* const array& x_ = */ x_,
        /* const array& w_ = */ w_,
        /* const array& scales_ = */ scales_,
        /* const std::optional<array>& biases_ = */ biases_,
        /* const array& indices_ = */ indices_,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string mode = */ mode);
  }

  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;
  // ENGINE (P019 unit 3): the 16x32 / 2-warp tile re-reads the x block once per
  // BN column block, so at prefill width the kernel runs at the memory roof and
  // ~1.4x below the dense qmm_t tile (32x32 / 4 warps). A larger BM raises the
  // arithmetic intensity but wastes more of a partial tile at every expert
  // boundary: with sorted rows the pass count is about M/BM + E, so the slots
  // touched are M + E*BM and a bigger tile only pays once M/E is large enough.
  // Break-even against the measured 1.39x is M/E ~ 25. Chosen by rows-per-expert,
  // overridable with MLX_GATHER_QMM_BM for a runtime sweep.
  if (transpose) {
    // MEASURED (P019 unit 3, REFUT-ENG-005): 16 stays the default FOR THE 4-BIT g32
    // ARM. The intensity gain of the wider tile is only 1.12x (bm 32) / 1.25x (bm
    // 64), not the 1.39x of the dense qmm_t gap, while the slots touched go as
    // M + E*BM: at 512 experts and 40-160 rows per expert every wider tile LOSES
    // (8k prefill 1126 -> 1060 -> 933 tok/s at chunk 2048). Break-even for bm 32
    // is M/E ~ 117.
    // ENGINE P052: the 8-bit g64 arm does NOT re-rank the same way -- 8/16/32/64
    // screened at 8192 (E9, the current champion's actual expert format) give
    // 1177 / 1295 / 1313 / 1149 tok/s: 32 wins, both narrower AND the next-wider
    // tile lose. Re-measure before assuming either screen transfers to a third
    // tuple; this is why the default is set PER TUPLE, not globally.
    if (group_size == 64 && bits == 8) {
      bm = 32;
      wm = 2;
    }
    if (const char* s = std::getenv("MLX_GATHER_QMM_BM")) {
      int v = atoi(s);
      if (v == 8) {
        bm = 8;
        wm = 1;
      } else if (v == 16) {
        bm = 16;
        wm = 1;
      } else if (v == 32 || v == 64) {
        bm = v;
        wm = 2;
      }
    }
  }

  // ENGINE P019 unit 5: WORK-TABLE DISPATCH. The stock kernel above charges
  // M + E*BM row slots because a BM-row block that straddles an expert boundary
  // runs a whole second K-loop pass. The work table gives every threadgroup exactly
  // one (expert, row_start, rows) item -- sum ceil(c_e / BM) items, about
  // M + E*BM/2 slots -- and is built on the device from the sorted indices, so
  // nothing here needs a device -> host sync: the grid is the host-known worst case
  // M/BM + E and the unused tail carries expert = -1 and exits at once.
  // Instantiated for (bfloat16 x, affine 4-bit g32 w) (P019) and, as of P052, also
  // (bfloat16 x, affine 8-bit g64 w) -- the format the current champion's routed experts
  // actually ship in. Both kernels are the same generic <group_size, bits> template; only
  // this gate and the .metal instantiation list decide which tuples pay for them.
  // MLX_GATHER_QMM_WT=0 restores the stock kernel.
  {
    int n_experts =
        static_cast<int>(w.size() / w.shape(-1) / w.shape(-2));
    bool is_8g64 = group_size == 64 && bits == 8;
    bool wt = transpose && mode == "affine" && x.dtype() == bfloat16 &&
        ((group_size == 32 && bits == 4) || is_8g64) &&
        (bm == 8 || bm == 16 || bm == 32 || (bm == 64 && is_8g64)) &&
        n_experts > 1 && n_experts <= 1024;
    if (const char* sw = std::getenv("MLX_GATHER_QMM_WT")) {
      if (sw[0] == '0') {
        wt = false;
      }
    }
    if (std::getenv("ENGINE_WT_DEBUG")) {
      fprintf(stderr, "ENGINE_WT_DEBUG gather_qmm: transpose=%d mode=%s dtype_bf16=%d gs=%d bits=%d bm=%d n_experts=%d wt=%d M=%d N=%d K=%d\n",
              transpose, mode.c_str(), x.dtype() == bfloat16, group_size, bits, bm, n_experts, wt, M, N, K);
    }
    if (wt) {
      constexpr int TG = 1024;
      const int NW = M / bm + n_experts + 1;
      auto& compute_encoder = metal::get_command_encoder(s);
      array work(Shape{NW * 3}, int32, nullptr, {});
      work.set_data(allocator::malloc(work.nbytes()));
      compute_encoder.add_temporary(work);

      std::string type_string = get_type_string(x.dtype());
      std::string wkname;
      wkname.reserve(64);
      concatenate(
          wkname,
          "affine_gather_qmm_work_",
          type_string,
          "_gs_",
          group_size,
          "_b_",
          bits,
          "_bm_",
          bm,
          "_tg_",
          TG);
      auto wkernel = get_quantized_kernel_wrapped(
          d, wkname, "gather_qmm_work", mode, type_string, group_size, bits, bm, TG);
      compute_encoder.set_compute_pipeline_state(wkernel);
      compute_encoder.set_input_array(indices, 0);
      compute_encoder.set_output_array(work, 1);
      compute_encoder.set_bytes(M, 2);
      compute_encoder.set_bytes(n_experts, 3);
      compute_encoder.set_bytes(NW, 4);
      compute_encoder.dispatch_threadgroups(
          MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));

      const bool align_N_wt = (N % bn) == 0;
      const bool align_K_wt = (K % bk) == 0;
      const bool align_M_wt = true;  // not read by the work-table kernel
      metal::MTLFCList wt_consts = {
          {&align_M_wt, MTL::DataType::DataTypeBool, 200},
          {&align_N_wt, MTL::DataType::DataTypeBool, 201},
          {&align_K_wt, MTL::DataType::DataTypeBool, 202},
      };
      std::string gname;
      gname.reserve(96);
      concatenate(
          gname,
          "affine_gather_qmm_rhs_wt_nt_",
          type_string,
          "_gs_",
          group_size,
          "_b_",
          bits,
          "_bm_",
          bm,
          "_bn_",
          bn,
          "_bk_",
          bk,
          "_wm_",
          wm,
          "_wn_",
          wn);
      std::string ghash;
      ghash.reserve(128);
      concatenate(
          ghash,
          gname,
          "_align_M_t_align_N_",
          align_N_wt ? 't' : 'n',
          "_align_K_",
          align_K_wt ? 't' : 'n');
      auto gkernel = get_gather_qmm_wt_kernel(
          d, gname, ghash, wt_consts, x, group_size, bits, mode, bm, bn, bk,
          wm, wn, transpose);
      compute_encoder.set_compute_pipeline_state(gkernel);
      int c = 0;
      compute_encoder.set_input_array(x, c++);
      compute_encoder.set_input_array(w, c++);
      compute_encoder.set_input_array(scales, c++);
      if (biases_) {
        array biases = ensure_row_contiguous(*biases_, d, s);
        compute_encoder.set_input_array(biases, c++);
      }
      compute_encoder.set_input_array(work, c++);
      compute_encoder.set_output_array(out, c++);
      compute_encoder.set_bytes(M, c++);
      compute_encoder.set_bytes(N, c++);
      compute_encoder.set_bytes(K, c++);
      compute_encoder.dispatch_threadgroups(
          MTL::Size((N + bn - 1) / bn, NW, 1), MTL::Size(32, wn, wm));
      return;
    }
  }

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }
  qmv(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  // ENGINE_QMV_LIMIT: rows from which a transposed quantized matmul with an output at least ENGINE_QMM_SMALL_MIN_N wide
  // leaves qmv (policy 5, MLX 0 = its table); ENGINE_QMV_LIMIT_NARROW: the same for narrower outputs (policy 9: qmv stays
  // best through 8 rows and ties at 9, split-K with a 16-row tile wins from 9..10 rows, C-0068)
  static const int qmv_limit = engine_qmm_int("ENGINE_QMV_LIMIT", 5, 0);
  static const int qmv_limit_min_n = engine_qmm_int("ENGINE_QMM_SMALL_MIN_N", 8192, 8192);
  if (transpose_ && qmv_limit > 0 && N >= qmv_limit_min_n) vector_limit = qmv_limit;
  static const int qmv_limit_narrow = engine_qmm_int("ENGINE_QMV_LIMIT_NARROW", 9, 0);
  if (transpose_ && qmv_limit_narrow > 0 && N < qmv_limit_min_n) vector_limit = qmv_limit_narrow;
  auto mode = quantization_mode_to_string(mode_);
  // It is a matrix matrix product.
  if (M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    int B = out.size() / M / N;
    if (transpose_ && B == 1) {
      qmm_splitk(
          x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
      return;
    }
    qmm(x,
        w,
        scales,
        biases,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    dispatch_qmv(
        x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    qvm(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(
      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 6) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);

  // We are walking x in order and w is also in order so we can batch up the
  // matmuls and reuse reading x and w.
  //
  // TODO: Tune 16 and 4 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void quantize_dequantize(
    const array& in,
    array& out,
    std::string mode,
    int group_size,
    int bits,
    metal::Device& d,
    const Stream& s) {
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(in, d, s);
  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_output_array(out, 1);
  auto type_string = get_type_string(in.dtype());
  std::string kname;
  concatenate(
      kname,
      mode + "_quantize_dequantize_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "quantize_dequantize", mode, type_string, group_size, bits);

  compute_encoder.set_compute_pipeline_state(kernel);

  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;
  int per_thread = std::max(group_size / simd_size, 1);
  size_t nthreads = w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() /= per_thread;
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto mode = quantization_mode_to_string(mode_);
  bool w_quantized = (inputs[1].dtype() == uint32);
  // Tensor-scale nvfp4 (global_scale_x / global_scale_w) is packed into
  // inputs by ops.cpp but no Metal qqmm kernel currently consumes the
  // global scales. Reject the request rather than silently dropping them
  // in the gemv path below.
  int base_size = w_quantized ? 3 : 2;
  if (mode_ == QuantizationMode::Nvfp4 &&
      static_cast<int>(inputs.size()) > base_size) {
    throw std::runtime_error(
        "[QQMatmul] Global scale (tensor-scale nvfp4) is not supported "
        "on the Metal backend.");
  }
  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(allocator::malloc(out.nbytes()));

    bool donate_x = inputs[0].is_donatable();
    array x = ensure_row_contiguous(inputs[0], d, s);
    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
    quantize_dequantize(x, xhat, mode, group_size_, bits_, d, s);

    // Make sure the last two dims of w and s are contiguous
    array w = ensure_row_contiguous_matrix(inputs[1], d, s);
    array scales = ensure_row_contiguous_matrix(inputs[2], d, s);

    bool non_batched = w.ndim() == 2;
    int K = x.shape(-1);
    int M = non_batched ? x.size() / K : x.shape(-2);
    int N = out.shape(-1);
    dispatch_qmv(
        xhat,
        w,
        scales,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  } else {
    throw std::runtime_error("[QQMatmul] NYI for the general case");
  }
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(w_pre, d, s);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], d, s);
      compute_encoder.set_input_array(biases, 2);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    scales.set_data(allocator::malloc(scales.nbytes()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(allocator::malloc(biases.nbytes()));
      compute_encoder.set_output_array(biases, 3);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
  }

  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto mode = quantization_mode_to_string(mode_);
  std::string kname;
  concatenate(
      kname,
      mode + (dequantize_ ? "_dequantize" : "_quantize"),
      "_",
      type_string,
      "_gs_",
      group_size_,
      "_b_",
      bits_);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      dequantize_ ? "dequantize" : "quantize",
      mode,
      type_string,
      group_size_,
      bits_);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
      : bits_ == 6                               ? 4
                                                 : 8 / bits_;
  int per_thread =
      dequantize_ ? packs_per_int : std::max(group_size_ / simd_size, 1);
  size_t nthreads =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  unary_op_gpu(inputs, out, name(), stream());
}

} // namespace mlx::core
