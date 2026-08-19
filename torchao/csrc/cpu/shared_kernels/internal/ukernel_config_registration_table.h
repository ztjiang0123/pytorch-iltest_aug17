// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
#include <torchao/csrc/cpu/shared_kernels/internal/packed_weights_header.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace torchao::ops {

/**
 * @brief A thread-unsafe registration table for kernel configurations.
 *
 * This table maps a combination of a weight format (header) and a CPU
 * microarchitecture to a specific UKernelConfig.
 *
 * The table is templated over the concrete UKernelConfig and
 * PackedWeightsFormat types so that each op namespace can reuse the same
 * registration logic. The only requirements on the template parameters are:
 *   - PackedWeightsFormat provides `to_packed_weights_header()`.
 *   - UKernelConfig provides `validate()`.
 */
template <typename UKernelConfig, typename PackedWeightsFormat>
struct UKernelConfigRegistrationTable {
 private:
  using Key = std::pair<torchao::ops::PackedWeightsHeader, cpuinfo_uarch>;
  struct KeyHasher {
    std::size_t operator()(const Key& k) const {
      return std::hash<torchao::ops::PackedWeightsHeader>()(k.first) ^
          std::hash<int>()(static_cast<int>(k.second));
    }
  };
  std::unordered_map<Key, UKernelConfig, KeyHasher> registration_table_;
  inline Key make_key(
      torchao::ops::PackedWeightsHeader header,
      cpuinfo_uarch uarch) const {
    return std::make_pair(header, uarch);
  }

 public:
  // Register a kernel config for a given format and uarch.
  void register_ukernel_config(
      PackedWeightsFormat format,
      cpuinfo_uarch uarch,
      UKernelConfig config) {
    auto header = format.to_packed_weights_header();
    auto key = make_key(header, uarch);
    if (registration_table_.find(key) != registration_table_.end()) {
      throw std::runtime_error(
          "UKernelConfig is already registered for this format");
    }
    config.validate();
    registration_table_[key] = config;
  }
  // Get the kernel config for a given format and uarch.
  std::optional<UKernelConfig> get_ukernel_config(
      torchao::ops::PackedWeightsHeader header,
      cpuinfo_uarch uarch) const {
    auto key = make_key(header, uarch);
    auto it = registration_table_.find(key);
    if (it == registration_table_.end()) {
      return std::nullopt;
    }
    return it->second;
  }
};

} // namespace torchao::ops
