#include <unordered_set>
#include <chrono>
#include <cstdio>
#include <cstdlib>
// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mlx::core::fast {

struct CustomKernelCache {
  std::unordered_map<std::string, std::string> libraries;
};

static CustomKernelCache& cache() {
  static CustomKernelCache cache_;
  return cache_;
};

// ENGINE fork (T-0151): how often a "__donate<k>" kernel took its input's buffer / had to copy it (process-wide)
std::atomic<long>& donated_kernels_counter() {
  static std::atomic<long> c{0};
  return c;
}
std::atomic<long>& copied_kernels_counter() {
  static std::atomic<long> c{0};
  return c;
}
// ENGINE_DONATE_STATS=1: print both counters to stderr when the process exits
static void register_donate_stats() {
  static bool done = [] {
    if (std::getenv("ENGINE_DONATE_STATS") != nullptr) {
      std::atexit([] {
        std::fprintf(stderr, "engine donate stats: donated %ld copied %ld\n",
                     donated_kernels_counter().load(), copied_kernels_counter().load());
      });
    }
    return true;
  }();
  (void)done;
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // silence some warnings
  (void)is_precompiled_;
  (void)shared_memory_;

  auto& s = stream();

  std::vector<array> copies;

  // ENGINE fork (T-0151): a kernel whose name carries "__donate<k>" writes output 0 IN PLACE over input k. When input k is
  // donatable (no other reference) output 0 takes its buffer; otherwise input k is copied into a fresh output 0 first.
  // Either way output 0 starts as input k's values, so the kernel need only write the elements it changes.
  int donate = -1;
  if (auto p = name_.find("__donate"); p != std::string::npos) {
    register_donate_stats();
    size_t q = p + 8;
    int v = 0;
    bool any = false;
    while (q < name_.size() && name_[q] >= '0' && name_[q] <= '9') {
      v = v * 10 + (name_[q] - '0');
      q++;
      any = true;
    }
    if (any && v < static_cast<int>(inputs.size())) {
      donate = v;
    }
  }

  for (size_t oi = 0; oi < outputs.size(); oi++) {
    auto& out = outputs[oi];
    if (oi == 0 && donate >= 0) {
      const array& in = inputs[donate];
      if (in.dtype() == out.dtype() && in.size() == out.size() && in.flags().row_contiguous && in.is_donatable()) {
        out.copy_shared_buffer(in);
        long nd = ++donated_kernels_counter();
        if ((nd & 4095) == 0 && std::getenv("ENGINE_DONATE_STATS") != nullptr) {   // a server never exits normally
          std::fprintf(stderr, "engine donate stats: donated %ld copied %ld\n", nd, copied_kernels_counter().load());
        }
      } else {
        copy_gpu(in, out, CopyType::General, s);   // allocates out (contiguous) and copies
        long nc = ++copied_kernels_counter();
        if ((nc & 1023) == 1 && std::getenv("ENGINE_DONATE_STATS") != nullptr) {
          std::fprintf(stderr, "engine donate stats: COPIED %ld (donated %ld)\n", nc, donated_kernels_counter().load());
        }
      }
      continue;
    }
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  std::vector<array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);

  {
    // Clear kernels from the device library cache if needed
    auto& kernel_cache = cache();
    if (auto it = kernel_cache.libraries.find(name_);
        it != kernel_cache.libraries.end()) {
      if (it->second != source_) {
        auto& d = metal::device(s.device);
        d.clear_library(name_);
        it->second = source_;
      }
    } else {
      kernel_cache.libraries.emplace(name_, source_);
    }
  }

  // ENGINE T-0164: ENGINE_KERNEL_JIT_LOG=1 prints every custom-kernel variant on its first use in the process (the
  // library compile and pipeline creation it triggers) with the wall time they took
  static const bool jit_log = std::getenv("ENGINE_KERNEL_JIT_LOG") != nullptr;
  static std::unordered_set<std::string> jit_seen;
  const bool jit_first = jit_log && jit_seen.insert(name_).second;
  const auto jit_t0 = std::chrono::steady_clock::now();
  auto lib = d.get_library(name_, [this] { return metal::utils() + source_; });
  auto kernel = d.get_kernel(name_, lib);
  if (jit_first) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - jit_t0).count();
    fprintf(stderr, "engine jit: %.1f ms %s\n", ms, name_.c_str());
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (std::get<0>(shape_info)) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (std::get<1>(shape_info)) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (std::get<2>(shape_info)) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  auto tg_size = tx * ty * tz;
  auto max_tg_size = kernel->maxTotalThreadsPerThreadgroup();
  if (tg_size > max_tg_size) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup (" << max_tg_size
        << ").";
    throw std::invalid_argument(msg.str());
  }

  const auto [gx, gy, gz] = grid_;
  MTL::Size group_dims =
      MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  compute_encoder.add_temporaries(std::move(copies));
}

} // namespace mlx::core::fast
