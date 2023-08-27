// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef JSONCONFIG_H
#define JSONCONFIG_H
#include "json/json.h"
#include "src/compiler/ir/code.pb.h"

using namespace std;
class JsonConfig
{
public:
    JsonConfig(tianjic_ir::code::Config *config):_config(config),SimClock(0){}

    bool ParseConfigFile(const string &Name);
    void OutputProto(std::string output_name, std::string output_name_readable);
private:
    void ConfigAxonPI(const Json::Value &PIValue,::tianjic_ir::code::AxonParameter *axon);
    void ConfigSomaPI(const Json::Value &PIValue, ::tianjic_ir::code::SomaParameter *soma);
    void ConfigRouterPI(const Json::Value &PIValue,::tianjic_ir::code::RouterParameter *router);


    void ChipConfig(void);
    void CoreConfig(const Json::Value &CoreValue, tianjic_ir::code::Core *core);
    void PIConfig(const Json::Value &PIArray, tianjic_ir::code::Core *core);
    void PIParameterConfig(const Json::Value &PIParameterArray, tianjic_ir::code::PIParameter *pi_parameter);

    void WriteData(const Json::Value &ArgInput, uint32_t ArgAddr, int ArgInType, bool ArgDebugPrint, tianjic_ir::code::InitData *init_data);

    Json::Value Root;
    tianjic_ir::code::Config * _config;


    int32_t cx;
    int32_t cy;
    int32_t x;
    int32_t y;
    uint32_t CoreGroup;
    int32_t static_PI_Addr;
    int32_t instant_PI_Addr;
    int32_t static_PI_num;
    int32_t instant_PI_num;
    int32_t tmp_static_PI_num;
    int32_t tmp_instant_PI_num;
    int32_t total_PI_num;
    int32_t PI_store_addr;

    bool A_valid;
    bool S1_valid;
    bool R_valid;
    bool S2_valid;

    unsigned int CurrentPIGroupNum;
    Json::Value PIGroupValue;

    int SimClock;
    int CalcPhaseNum(const Json::Value &ArgCoreInfo, uint32_t ArgGroupID);
    void InitCoreMemory(const Json::Value &DataList, tianjic_ir::code::Core *core);
    void CoreMemPrintConfig(const Json::Value &CoreMemPrintValue, tianjic_ir::code::Core *core);
    void SwitchConfig(const Json::Value &config);
};

#endif // JSONCONFIG_H
