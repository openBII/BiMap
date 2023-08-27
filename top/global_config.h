// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once
#include "src/compiler/ir/basic.pb.h"
#include "toml11/toml.hpp"

class GlobalConfig
{
 public:
    static const int CHIP_X_MAX = 1;
    static const int CHIP_Y_MAX = 1;
    static const int CORE_X_MAX = 16;
    static const int CORE_Y_MAX = 10;
    static const int CORE_MAX = CORE_X_MAX * CORE_Y_MAX;
    static const int STEP_GROUP_MAX =
        4;  // 一个芯片可以包含的step group最大个数
    static const int PHASE_GROUP_MAX =
        32;  // 一个芯片可以包含的phase group最大个数
    static const bool PHASE_ADAPT =
        true;  // 所有的phase group都配置为自适应执行模式

    static toml::value CONFIG;

    static std::string OUTPUT_DIR;
    static bool OUTPUT_READABLE;
    static tianjic_ir::TestMode TEST_MODE;
};
