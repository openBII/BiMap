// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/


#if 1
#include <string.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include "cmdline/cmdline.h"
#include "src/compiler/ir/code.pb.h"
#include "src/compiler/ir/json_to_proto/jsonconfig.h"
#include "src/utils/file_utils.h"

using namespace std;

string OUTPUT_DIR;


int main(int argc, char **argv)
{
    cmdline::parser parser;
    parser.add<string>("input_path", 'i', "input file path", true, "");
    parser.add<string>("case_name", 'c', "case name", true, "");
    parser.add<string>("output_dir", 'o', "output file dir", false, "");
    parser.parse_check(argc, argv);

    string input_json_name(parser.get<string>("input_path"));
    string case_name(parser.get<string>("case_name"));
    string output_dir = "temp/" + case_name + "/";
    if (parser.exist("output_dir"))
        output_dir = parser.get<string>("output_dir");

    if (output_dir.back() != '/')
        output_dir += "/";
    string output_name(output_dir + case_name + ".code");
    string output_name_readable(output_dir + case_name + ".code.txt");

    tianjic_util::makeDir(output_dir);

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    tianjic_ir::code::Config config;

    JsonConfig trans(&config);
    trans.ParseConfigFile(input_json_name);
    trans.OutputProto(output_name, output_name_readable);

    return 0;
}

#endif
