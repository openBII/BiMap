// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/


#include "src/utils/file_utils.h"
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace tianjic_util {

void makeDir(const std::string &path) {
    std::string cmd = "mkdir -p " + path;
    if (!opendir(path.c_str()))
        system(cmd.c_str());
}

// 纯C++版本的make dir，留作备用
int makeDirBak(std::string path_name, mode_t mode) {
    size_t pre = 0, pos;
    std::string dir;
    int mkdir_state;

    if (path_name[path_name.size() != '/'])
        path_name += '/';

    while ((pos = path_name.find_first_of('/', pre)) != std::string::npos) {
        dir = path_name.substr(0, pos++);
        pre = pos;
        if (dir.size() == 0)
            continue;

        if ((mkdir_state = ::mkdir(dir.c_str(), mode)) && errno != EEXIST) {
            throw "Directory path is made in failure";
        }
    }
    return mkdir_state;
}

std::string basename(std::string file_name){

    int pos = file_name.find_first_of('.');
    if (pos != std::string::npos){
        file_name = file_name.substr(0, pos);
    }
    return file_name;
}
} // namespace tianjic_util
