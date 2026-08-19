// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/csrc/cpu/shared_kernels/internal/library.h>
#include <torchao/csrc/cpu/shared_kernels/internal/parallel.h>
#include <cassert>

namespace torchao::ops::internal {

// Result of computing tile sizes for a 2D (m, n) problem.
struct TileSizes {
  int mc;
  int nc;
};

// Chooses tile sizes (mc, nc) so that there are approximately
// target_tiles_per_thread tiles per thread. Guarantees:
//   1. mc == m or mc % m_step == 0, and
//   2. nc == n or nc % n_step == 0.
//
// This is the shared implementation behind the various
// *TilingParams::from_target_tiles_per_thread factory functions.
inline TileSizes compute_tile_sizes_per_thread(
    int m,
    int m_step,
    int n,
    int n_step,
    int target_tiles_per_thread) {
  TORCHAO_CHECK(m >= 1, "m must be >= 1");
  TORCHAO_CHECK(m_step >= 1, "m_step must be >= 1");

  TORCHAO_CHECK(n >= 1, "n must be >= 1");
  TORCHAO_CHECK(n_step >= 1, "n_step must be >= 1");
  TORCHAO_CHECK(
      target_tiles_per_thread >= 1, "target_tiles_per_thread must be >= 1");
  auto num_threads = torchao::get_num_threads();
  TORCHAO_CHECK(num_threads >= 1, "num_threads must be >= 1");

  int mc = m_step;
  int num_mc_panels = (m + mc - 1) / mc;

  int numerator = n * num_mc_panels;
  int denominator = num_threads * target_tiles_per_thread;

  // Set nc = ceil(numerator / denominator)
  int nc = (numerator + denominator - 1) / denominator;
  assert(nc >= 1);

  // Replace nc with next number n_step divides
  nc = ((nc + n_step - 1) / n_step) * n_step;

  // Clamp mc, nc to be no larger than m, n
  mc = std::min(m, mc);
  nc = std::min(n, nc);

  assert((mc == m) || (mc % m_step == 0));
  assert((nc == n) || (nc % n_step == 0));

  return TileSizes{mc, nc};
}

// Builds a *TilingParams struct (any type exposing writable `int mc` and
// `int nc` members) from a target number of tiles per thread. This is the
// shared implementation behind the various
// *TilingParams::from_target_tiles_per_thread factory functions, which differ
// only in the concrete struct type they return.
template <typename TilingParams>
inline TilingParams make_tiling_params_from_target_tiles_per_thread(
    int m,
    int m_step,
    int n,
    int n_step,
    int target_tiles_per_thread) {
  auto tile_sizes = compute_tile_sizes_per_thread(
      m, m_step, n, n_step, target_tiles_per_thread);

  TilingParams tiling_params;
  tiling_params.mc = tile_sizes.mc;
  tiling_params.nc = tile_sizes.nc;
  return tiling_params;
}

// Selects the index of the appropriate config in a sorted (ascending m_step)
// array of configs based on m. Each element of `configs` must expose an
// `int m_step` member. The first config must be set (m_step >= 1), and set
// configs must be strictly increasing in m_step. Returns the index of the
// config with the largest m_step that is <= m (or 0 if none apply).
template <typename ConfigArray>
inline int select_config_idx(const ConfigArray& configs, int m) {
  assert(m >= 1);
  assert(configs[0].m_step >= 1);

  size_t i = 0;
  while (i + 1 < configs.size() && configs[i + 1].m_step >= 1 &&
         configs[i + 1].m_step <= m) {
    assert(configs[i].m_step < configs[i + 1].m_step);
    i++;
  }

  assert(i < configs.size());
  assert(configs[i].m_step >= 1);
  assert(i == 0 || configs[i].m_step <= m);
  return static_cast<int>(i);
}

} // namespace torchao::ops::internal
