// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include <dirent.h>
#include <iostream>
#include <signal.h>

#include "cmdline/cmdline.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"  // or "../stdout_sinks.h" if no colors needed
#include "spdlog/spdlog.h"
#include "src/simulator/behavior_simulator/simulator_asm.h"
#include "src/simulator/behavior_simulator/util.h"
#include "src/utils/file_utils.h"
#include "top/global_config.h"

using namespace std;

int32_t main(int32_t argc, char **argv) {
    // console->set_level(spdlog::level::off);
    cmdline::parser parser;
    parser.add<string>("device", 'd', "cpu/gpu", false, "cpu");
    parser.add<string>("config_path", 'i', "input config file", true);
    parser.add<string>("case_name", 'c', "case_name", true, "");
    parser.add<string>("output_dir", 'o', "output_dir", false, "");
    parser.add<bool>("readable", 'r', "output readble format", false, true);
    parser.parse_check(argc, argv);

    GlobalConfig::OUTPUT_READABLE = parser.get<bool>("readable");

    string case_name = tianjic_util::basename(parser.get<string>("case_name"));

    string config_path = parser.get<string>("config_path");

    auto logger = spdlog::basic_logger_mt(
        "logger", string("temp/logs/") + parser.get<string>("case_name"));
    logger->flush_on(spdlog::level::info);
    auto console = spdlog::stdout_color_mt("console");

    auto config = toml::parse("top/config.toml");

    string output_dir = "temp/" + parser.get<string>("case_name") + "/" +
                        toml::get<std::string>(config["path"]["behavior_out"]);

    if (parser.exist("output_dir"))
        output_dir = parser.get<string>("output_dir");

    GlobalConfig::OUTPUT_DIR = output_dir;

    tianjic_util::makeDir(output_dir);

    Simulator sim(case_name);
    sim.mapConfig(config_path);

    auto start = getTimeNs();
    spdlog::get("logger")->info("start behavior simulation {}", case_name);
    sim.simulate();
    auto end = getTimeNs();

    console->info("c++ case {}  running time : {} ns", case_name, end - start);
    return 0;
}
