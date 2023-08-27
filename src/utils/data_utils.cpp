// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "src/utils/data_utils.h"
        #include "spdlog/spdlog.h"
#include "src/utils/file_utils.h"
#include <cstdio>
#include <cstdlib>

namespace tianjic_util {

void saveDataBlock(std::shared_ptr<uint32_t> data, int data_length, std::string file_name) {
    std::string file_path = file_name.substr(0, file_name.find_last_of('/'));
    makeDir(file_path);

    FILE *file_ptr;
    if ((file_ptr = fopen(file_name.c_str(), "wb")) == NULL) {
        spdlog::get("console")->error("Fail to print data block to file");
        return;
    }

    fwrite(data.get(), sizeof(uint32_t), data_length, file_ptr);
    fclose(file_ptr);
}

} // namespace tianjic_util
