// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once
#include <fstream>
#include <memory>
#include <string>

namespace tianjic_util {

void saveDataBlock(std::shared_ptr<uint32_t> data, int data_length,
                   std::string file_name);

template <class T> T align(T x, T n) { return ((x + n - 1) / n) * n; }
}  // namespace tianjic_util
