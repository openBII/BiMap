// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "top/global_config.h"
toml::value GlobalConfig::CONFIG = toml::parse("top/config.toml");
std::string GlobalConfig::OUTPUT_DIR = "";
bool GlobalConfig::OUTPUT_READABLE = true;
tianjic_ir::TestMode GlobalConfig::TEST_MODE = tianjic_ir::PRIM_OUTPUT;