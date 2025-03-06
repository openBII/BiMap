#include <iostream>

#include <argparse/argparse.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "base/base.h"
#include "base/config.h"
#include "base/request.h"
#include "example/example_ifce.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

std::map<std::string, Ramulator::AiMISR> Ramulator::AiMISRInfo::opcode_str_to_aim_ISR;
std::map<Ramulator::Opcode, std::string> Ramulator::AiMISRInfo::aim_opcode_to_str;

std::map<std::string, Ramulator::Type> Ramulator::AiMISRInfo::str_to_type;
std::map<Ramulator::Type, std::string> Ramulator::AiMISRInfo::type_to_str;

std::map<std::string, Ramulator::MemAccessRegion> Ramulator::AiMISRInfo::str_to_mem_access_region;
std::map<Ramulator::MemAccessRegion, std::string> Ramulator::AiMISRInfo::mem_access_region_to_str;

std::string Ramulator::trace_file_path;
bool Ramulator::use_trace_file_path;