// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once
#include <string>
#include <sys/types.h>

namespace tianjic_util {

void makeDir(const std::string &path);

// 纯C++版本的make dir，留作备用
int makeDirBak(std::string path_name, mode_t mode = 0777);

std::string basename(std::string file_name);
} // namespace tianjic_util


