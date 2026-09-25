// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(affine_qmv_fast, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmv, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qvm, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 32)  \

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                     \
      type,                                                                     \
      group_size,                                                               \
      bits,                                                                     \
      aligned)

#define instantiate_quantized_all_splitk_qmm(type, group_size, bits)                    \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 32, 32, 32, 2, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 64, 32, 32, 2, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nn, type, group_size, bits, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)  \
  instantiate_quantized_all_splitk_qmm(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

#define instantiate_gather_qmm_work(type, group_size, bits, bm, tg)                        \
  instantiate_kernel(                                                                      \
      "affine_gather_qmm_work_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_tg_" #tg, \
      affine_gather_qmm_work, type, group_size, bits, bm, tg)

// ENGINE P019 unit 5: the work-table pair, instantiated ONLY for the arm's tuple
// (bfloat16 activations, 4-bit affine weights, group 32). The host falls back to the
// plain gather_qmm_rhs for anything else, so nothing else pays for these kernels.
// ENGINE P020 unit 21: BM=8 for the work-table kernel. Any per-expert tiling computes (-rows) mod BM
// wasted rows per expert -- 9.6% of the work at BM=16 with real top-10 routing (BOUND-ENG-005 (5)) --
// so a SMALLER tile halves the waste at half the arithmetic intensity. REFUT-ENG-005 only tried bigger.
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 32, 4, 8, 32, 32, 1, 2, true)
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 32, 4, 16, 32, 32, 1, 2, true)
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 32, 4, 32, 32, 32, 2, 2, true)
instantiate_gather_qmm_work(bfloat16_t, 32, 4, 8, 1024)
instantiate_gather_qmm_work(bfloat16_t, 32, 4, 16, 1024)
instantiate_gather_qmm_work(bfloat16_t, 32, 4, 32, 1024)

// ENGINE P052: the SAME pair, for 8-bit affine weights at group 64 -- the format the CURRENT
// champion's routed experts actually ship in (`fused experts LIVE 8-bit g64`, expertBanks()).
// affine_gather_qmm_work/affine_gather_qmm_rhs_wt are templated on <group_size, bits> with no
// 4-bit-specific assumption in either kernel body (get_pack_factor<bits,8>, QuantizedBlockLoader<...,
// group_size, bits> are already the same generic machinery every other quantized kernel in this file
// uses for 8-bit); only the C++ dispatch gate in quantized.cpp restricted this pair to (32, 4), so the
// 8-bit MoE gather_qmm at prefill has been running the STOCK, un-worked-table kernel this whole time.
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 64, 8, 8, 32, 32, 1, 2, true)
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 64, 8, 16, 32, 32, 1, 2, true)
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 64, 8, 32, 32, 32, 2, 2, true)
// ENGINE P052: BM=64 too -- the 8-bit screen (unlike the 4-bit one, REFUT-ENG-005) ranks 8 < 16 < 32
// monotonically, the opposite ranking from 4-bit's, so 64 is a real candidate here, not a repeat of a
// closed question.
instantiate_gather_qmm_rhs(affine_gather_qmm_rhs_wt, affine_gather_qmm_rhs_wt_nt, bfloat16_t, 64, 8, 64, 32, 32, 2, 2, true)
instantiate_gather_qmm_work(bfloat16_t, 64, 8, 8, 1024)
instantiate_gather_qmm_work(bfloat16_t, 64, 8, 16, 1024)
instantiate_gather_qmm_work(bfloat16_t, 64, 8, 32, 1024)
instantiate_gather_qmm_work(bfloat16_t, 64, 8, 64, 1024)

instantiate_quantized_all() // clang-format on

// ENGINE P095: the cross-row 8-bit g64 qmv_fast twin, one instantiation (the checkpoint's dense format), selected by name from quantized.cpp qmv()
instantiate_kernel("affine_qmv_fast_xr8_bfloat16_t_gs_64_b_8_batch_0", affine_qmv_fast_xr8, bfloat16_t, 64, 8, false)
instantiate_kernel("affine_qmv_fast_xr8g2_bfloat16_t_gs_64_b_8_batch_0", affine_qmv_fast_xr8g2, bfloat16_t, 64, 8, false)
instantiate_kernel("affine_qmv_fast_xr8g8_bfloat16_t_gs_64_b_8_batch_0", affine_qmv_fast_xr8g8, bfloat16_t, 64, 8, false)
