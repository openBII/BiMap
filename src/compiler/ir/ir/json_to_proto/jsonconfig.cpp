// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "jsonconfig.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#define DEBUG_OUTPUT 0
#define PRINT_TRIGGER_MATRIX 0

#if DEBUG_OUTPUT
#define OUTPUT(ARG) ARG
static bool DataDebug = true;
#else
#define OUTPUT(ARG)
static bool DataDebug = false;
#endif

bool JsonConfig::ParseConfigFile(const string &Name) {
    Json::CharReaderBuilder ReaderBuilder;
    ifstream InputStream(Name);
    if (InputStream.is_open() == true) {
        string err;
        if (Json::parseFromStream(ReaderBuilder, InputStream, &Root, &err) == true) {
            // cout<<Root.toStyledString()<<endl;
            InputStream.close();
            {
                ChipConfig();
                _config->set_sim_clock(SimClock);
            }
            return true;
        }else{
            std::cout << err << endl;
        }
    }
    throw Name + " open error";
}

int JsonConfig::CalcPhaseNum(const Json::Value &ArgCoreInfo, uint32_t ArgGroupID) {
    int _MaxPhaseNum = 0;
    for (Json::ArrayIndex _i = 0; _i < ArgCoreInfo.size(); _i++) {
        Json::Value _Core = ArgCoreInfo[_i];
        if (_Core["CoreInfo"]["CoreGroup"].asUInt() != ArgGroupID) {
            continue;
        }
        int _PhaseNum = int(_Core["PI"].size());
        _MaxPhaseNum = _MaxPhaseNum > _PhaseNum ? _MaxPhaseNum : _PhaseNum;
    }
    return _MaxPhaseNum;
}

void JsonConfig::SwitchConfig(const Json::Value &config) {
    ::tianjic_ir::code::DebugFileSwitch *debug_switch = new ::tianjic_ir::code::DebugFileSwitch();

    debug_switch->set_mchip_case(bool(config.get("MCHIP_CASE", 0).asInt()));
    debug_switch->set_dump_a_ad_ina(bool(config.get("DUMP_A_AD_INA", 0).asInt()));
    debug_switch->set_dump_a_di_ina(bool(config.get("DUMP_A_DI_INA", 0).asInt()));
    debug_switch->set_dump_a_do_ina(bool(config.get("DUMP_A_DO_INA", 0).asInt()));
    debug_switch->set_dump_a_ad_inb(bool(config.get("DUMP_A_AD_INB", 0).asInt()));
    debug_switch->set_dump_a_di_inb(bool(config.get("DUMP_A_DI_INB", 0).asInt()));
    debug_switch->set_dump_a_do_inb(bool(config.get("DUMP_A_DO_INB", 0).asInt()));
    debug_switch->set_dump_a_ad_bias(bool(config.get("DUMP_A_AD_BIAS", 0).asInt()));
    debug_switch->set_dump_a_di_bias(bool(config.get("DUMP_A_DI_BIAS", 0).asInt()));
    debug_switch->set_dump_a_do_bias(bool(config.get("DUMP_A_DO_BIAS", 0).asInt()));
    debug_switch->set_dump_a_ad_vou(bool(config.get("DUMP_A_AD_VOU", 0).asInt()));
    debug_switch->set_dump_a_do_vou(bool(config.get("DUMP_A_DO_VOU", 0).asInt()));
    debug_switch->set_dump_mac(bool(config.get("DUMP_MAC", 0).asInt()));
    debug_switch->set_dump_d_soma(bool(config.get("DUMP_D_SOMA", 0).asInt()));
    debug_switch->set_dump_a_rhead(bool(config.get("DUMP_A_RHEAD", 0).asInt()));
    debug_switch->set_dump_d_rhead(bool(config.get("DUMP_D_RHEAD", 0).asInt()));
    debug_switch->set_dump_a_dout(bool(config.get("DUMP_A_DOUT", 0).asInt()));
    debug_switch->set_dump_d_dout(bool(config.get("DUMP_D_DOUT", 0).asInt()));
    debug_switch->set_dump_d_send(bool(config.get("DUMP_D_SEND", 0).asInt()));
    debug_switch->set_dump_a_din(bool(config.get("DUMP_A_DIN", 0).asInt()));
    debug_switch->set_dump_d_din(bool(config.get("DUMP_D_DIN", 0).asInt()));
    debug_switch->set_dump_dbgmsg_t(bool(config.get("DUMP_DBGMSG_T", 0).asInt()));
    debug_switch->set_dump_dbgmsg_c(bool(config.get("DUMP_DBGMSG_C", 0).asInt()));
    debug_switch->set_dump_dbgmsg_a(bool(config.get("DUMP_DBGMSG_A", 0).asInt()));
    debug_switch->set_dump_dbgmsg_s(bool(config.get("DUMP_DBGMSG_S", 0).asInt()));
    debug_switch->set_dump_dbgmsg_r(bool(config.get("DUMP_DBGMSG_R", 0).asInt()));
    debug_switch->set_dump_step_num(config.get("DUMP_STEP_NUM", 2).asInt());
    debug_switch->set_dump_phase_num(config.get("DUMP_PHASE_NUM", 4).asInt());

    _config->set_allocated_debug_file_switch(debug_switch);
#if DEBUG_OUTPUT
    cout << "MCHIP_CASE     = " << MCHIP_CASE << endl;
    cout << "DUMP_A_AD_INA  = " << DUMP_A_AD_INA << endl;
    cout << "DUMP_A_DI_INA  = " << DUMP_A_DI_INA << endl;
    cout << "DUMP_A_DO_INA  = " << DUMP_A_DO_INA << endl;
    cout << "DUMP_A_AD_INB  = " << DUMP_A_AD_INB << endl;
    cout << "DUMP_A_DI_INB  = " << DUMP_A_DI_INB << endl;
    cout << "DUMP_A_DO_INB  = " << DUMP_A_DO_INB << endl;
    cout << "DUMP_A_AD_BIAS = " << DUMP_A_AD_BIAS << endl;
    cout << "DUMP_A_DI_BIAS = " << DUMP_A_DI_BIAS << endl;
    cout << "DUMP_A_DO_BIAS = " << DUMP_A_DO_BIAS << endl;
    cout << "DUMP_A_AD_VOU  = " << DUMP_A_AD_VOU << endl;
    cout << "DUMP_A_DO_VOU  = " << DUMP_A_DO_VOU << endl;
    cout << "DUMP_MAC       = " << DUMP_MAC << endl;
    cout << "DUMP_D_SOMA    = " << DUMP_D_SOMA << endl;
    cout << "DUMP_A_RHEAD   = " << DUMP_A_RHEAD << endl;
    cout << "DUMP_D_RHEAD   = " << DUMP_D_RHEAD << endl;
    cout << "DUMP_A_DOUT    = " << DUMP_A_DOUT << endl;
    cout << "DUMP_D_DOUT    = " << DUMP_D_DOUT << endl;
    cout << "DUMP_D_SEND    = " << DUMP_D_SEND << endl;
    cout << "DUMP_A_DIN     = " << DUMP_A_DIN << endl;
    cout << "DUMP_D_DIN     = " << DUMP_D_DIN << endl;
    cout << "DUMP_DBGMSG_T  = " << DUMP_DBGMSG_T << endl;
    cout << "DUMP_DBGMSG_C  = " << DUMP_DBGMSG_C << endl;
    cout << "DUMP_DBGMSG_A  = " << DUMP_DBGMSG_A << endl;
    cout << "DUMP_DBGMSG_S  = " << DUMP_DBGMSG_S << endl;
    cout << "DUMP_DBGMSG_R  = " << DUMP_DBGMSG_R << endl;
    cout << "DUMP_STEP_NUM  = " << DUMP_STEP_NUM << endl;
    cout << "DUMP_PHASE_NUM = " << DUMP_PHASE_NUM << endl;
    system("pause");
#endif
}

void JsonConfig::ChipConfig() {
    if (Root.isMember("debug_file_switch")) {
        SwitchConfig(Root["debug_file_switch"]);
    }
    ::tianjic_ir::code::ChipArray *chip_array = _config->add_chip_arrays();
    for (Json::ArrayIndex i = 0; i < Root["ChipArray"].size(); i++) {
        ::tianjic_ir::code::Chip *chip = chip_array->add_chips();
        Json::Value ChipValue = Root["ChipArray"][i];
        cx = ChipValue["ChipID"]["cx"].asInt();
        cy = ChipValue["ChipID"]["cy"].asInt();

        ::tianjic_ir::code::ChipID *chip_id = new ::tianjic_ir::code::ChipID();
        chip_id->set_cx(cx);
        chip_id->set_cy(cy);
        chip->set_allocated_chip_id(chip_id);

        OUTPUT(cout << i + 1 << ":++++++++++++++++ chip : (" << cx << "," << cy << ")++++++++++++++++" << endl);
        for (Json::ArrayIndex _i = 0; _i < ChipValue["ChipConfig"].size(); _i++) {
            auto phase_group_config = chip->add_phase_group_configs();
            OUTPUT(cout << _i + 1 << ":++++++++++++++++ ChipConfig set : (" << cx << "," << cy << ")++++++++++++++++" << endl);
            Json::Value PhaseGroup = ChipValue["ChipConfig"][_i];

            phase_group_config->set_phasegroup(PhaseGroup["PhaseGroup"].asUInt());
            phase_group_config->set_sim_clock(PhaseGroup["Sim_clock"].asInt());
            phase_group_config->set_trigger(PhaseGroup["trigger"].asInt());

            /*......*/
            int PhaseNum = CalcPhaseNum(ChipValue["CoreConfig"], phase_group_config->phasegroup());
            /*......*/

            if (PhaseGroup.isMember("P_adpt")) {
                phase_group_config->set_p_adpt(PhaseGroup["P_adpt"].asInt());
            }

            SimClock = phase_group_config->sim_clock() * PhaseNum > SimClock ? phase_group_config->sim_clock() * PhaseNum : SimClock;
        }
        if (ChipValue.isMember("step_clock")) {
            for (int i = 0; i < 4; i++) {
                ::tianjic_ir::code::StepClock *step_clock = chip->add_step_clock();
                if (!ChipValue["step_clock"][i]["clock0_in_step"].isNull()) {
                    step_clock->set_clock0_in_step(ChipValue["step_clock"][i]["clock0_in_step"].asInt());
                }
                if (!ChipValue["step_clock"][i]["clock1_in_step"].isNull()) {
                    step_clock->set_clock1_in_step(ChipValue["step_clock"][i]["clock1_in_step"].asInt());
                }
            }
        }
        for (Json::ArrayIndex _i = 0; _i < ChipValue["CoreConfig"].size(); _i++) {
            OUTPUT(cout << _i + 1 << ":++++++++++++++++ CoreConfig set ++++++++++++++++" << endl);
            ::tianjic_ir::code::Core *core = chip->add_cores();
            CoreConfig(ChipValue["CoreConfig"][_i], core);
            OUTPUT(cout << _i + 1 << ":---------------- CoreConfig set ----------------" << endl);
        }
        if (ChipValue.isMember("trigger") && ChipValue["trigger"].isObject()) {
            ::tianjic_ir::code::Trigger *trigger = new ::tianjic_ir::code::Trigger();
            Json::Value triggerList = ChipValue["trigger"];
            for (Json::ArrayIndex _i = 0; _i < triggerList["start"].size(); _i++) {
                trigger->add_start(triggerList["start"][_i].asInt());
            }
            for (Json::ArrayIndex _i = 0; _i < triggerList["high"].size(); _i++) {
                trigger->add_high(triggerList["high"][_i].asInt());
            }
            for (Json::ArrayIndex _i = 0; _i < triggerList["low"].size(); _i++) {
                trigger->add_low(triggerList["low"][_i].asInt());
            }
            chip->set_allocated_trigger(trigger);
        }
        OUTPUT(cout << i + 1 << ":---------------- chip : (" << cx << "," << cy << ")----------------" << endl);
    }
    if (Root.isMember("sim_clock")) {
        int clock = Root["sim_clock"].asInt();
        SimClock = clock > 0 ? clock : SimClock;
    }
}

void JsonConfig::CoreMemPrintConfig(const Json::Value &CoreMemPrintValue, ::tianjic_ir::code::Core *core) {
    for (uint32_t curPhaseNum = 0; curPhaseNum < CoreMemPrintValue.size(); curPhaseNum++) {
        ::tianjic_ir::code::MemoryOutput *mem_output = core->add_memory_output();
        if (tmp_static_PI_num != 0) {
            OUTPUT(cout << "static Phase Num:" << curPhaseNum << endl);
            uint32_t curPhaseSectionSum = 0;
            for (uint32_t curSection = 0; curSection < CoreMemPrintValue[curPhaseNum].size(); curSection++) {
                Json::Value curSectionInfo = CoreMemPrintValue[curPhaseNum][curSection];
                uint32_t start = curSectionInfo["start"].asUInt();
                uint32_t length = curSectionInfo["length"].asUInt();
                ::tianjic_ir::code::MemorySegment *seg = mem_output->add_memory_segs();
                seg->set_start(start);
                seg->set_length(length);
                curPhaseSectionSum++;
            }

            tmp_static_PI_num--;
        } else if (tmp_instant_PI_num != 0) {
            OUTPUT(cout << "instant Phase Num:" << curPhaseNum - uint32_t(static_PI_num) << endl);
            uint32_t curPhaseSectionSum = 0;
            for (uint32_t curSection = 0; curSection < CoreMemPrintValue[curPhaseNum].size(); curSection++) {
                Json::Value curSectionInfo = CoreMemPrintValue[curPhaseNum][curSection];
                uint32_t start = curSectionInfo["start"].asUInt();
                uint32_t length = curSectionInfo["length"].asUInt();
                ::tianjic_ir::code::MemorySegment *seg = mem_output->add_memory_segs();
                seg->set_start(start);
                seg->set_length(length);

                curPhaseSectionSum++;
            }
            tmp_instant_PI_num--;
        }
    }
}

void JsonConfig::CoreConfig(const Json::Value &CoreValue, ::tianjic_ir::code::Core *core) {
    ::tianjic_ir::code::CoreInfo *core_info = new ::tianjic_ir::code::CoreInfo();
    x = CoreValue["CoreInfo"]["x"].asInt();
    y = CoreValue["CoreInfo"]["y"].asInt();
    OUTPUT(cout << ":++++++++++++++++ core : (" << x << "," << y << ")++++++++++++++++" << endl);
    core_info->set_x(x);
    core_info->set_y(y);

    CoreGroup = CoreValue["CoreInfo"]["CoreGroup"].asUInt();
    static_PI_Addr = CoreValue["CoreInfo"]["static_PI_base_Addr"].asInt();
    core_info->set_phase_group(CoreGroup);
    core_info->set_static_pi_base_addr(static_PI_Addr);

    total_PI_num = static_PI_num = int(CoreValue["PI"].size());

    PI_store_addr = static_PI_Addr;

    // instant pi config
    ::tianjic_ir::code::CoreRegisters *core_registers = new ::tianjic_ir::code::CoreRegisters();
    core_registers->set_instant_pi_en(CoreValue["CoreInfo"]["registers"]["instant_PI_en"].asInt());
    core_registers->set_fixed_instant_pi(CoreValue["CoreInfo"]["registers"]["fixed_instant_PI"].asInt());

    if (core_registers->instant_pi_en()) {
        core_registers->set_instant_pi_number(CoreValue["CoreInfo"]["registers"]["instant_PI_number"].asInt());
    }
    core_registers->set_pi_loop_en(CoreValue["CoreInfo"]["registers"]["PI_loop_en"].asInt());
    core_registers->set_start_instant_pi_num(CoreValue["CoreInfo"]["registers"]["start_instant_PI_num"].asInt());
    core_registers->set_pi_sign_cxy(CoreValue["CoreInfo"]["registers"]["PI_sign_CXY"].asInt());
    core_registers->set_pi_sign_nx(CoreValue["CoreInfo"]["registers"]["PI_sign_Nx"].asInt());
    core_registers->set_pi_sign_ny(CoreValue["CoreInfo"]["registers"]["PI_sign_Ny"].asInt());
    core_registers->set_pi_cxy(CoreValue["CoreInfo"]["registers"]["PI_CXY"].asInt());
    core_registers->set_pi_nx(CoreValue["CoreInfo"]["registers"]["PI_Nx"].asInt());
    core_registers->set_pi_ny(CoreValue["CoreInfo"]["registers"]["PI_Ny"].asInt());
    core_registers->set_receive_pi_addr_base(CoreValue["CoreInfo"]["registers"]["Receive_PI_addr_base"].asInt());

    core_info->set_allocated_registers(core_registers);
    core->set_allocated_core_info(core_info);

    instant_PI_Addr = CoreValue["CoreInfo"]["registers"]["Addr_instant_PI_base"].asInt();
    //    instant_PI_Addr=0;

    if (core_registers->instant_pi_en()) {
        tmp_instant_PI_num = instant_PI_num = CoreValue["CoreInfo"]["registers"]["instant_PI_number"].asInt() + 1;
        tmp_static_PI_num = static_PI_num = total_PI_num - instant_PI_num;
    } else {
        tmp_static_PI_num = static_PI_num = total_PI_num;
        tmp_instant_PI_num = instant_PI_num = 0;
    }

    PIConfig(CoreValue["PI"], core);
    InitCoreMemory(CoreValue["initData"], core);
    tmp_static_PI_num = static_PI_num;
    tmp_instant_PI_num = instant_PI_num;
    CoreMemPrintConfig(CoreValue["MemoryOutput"], core);
#if DEBUG_OUTPUT
    cout << "x=" << x << " y=" << y << " CoreGroup=" << CoreGroup << " static_PI_Addr=" << static_PI_Addr << " static_PI_num=" << static_PI_num << " instant_PI_num=" << instant_PI_num << endl;
    cout << ":---------------- core : (" << x << "," << y << ")----------------" << endl;
    system("pause");
#endif
}

void JsonConfig::InitCoreMemory(const Json::Value &DataList, tianjic_ir::code::Core *core) {
    for (Json::ArrayIndex i = 0; i < DataList.size(); i++) {
        tianjic_ir::code::InitData *init_data = core->add_initdata();
        uint32_t _address = DataList[i]["start"].asUInt();
        uint32_t _length = DataList[i]["length"].asUInt();
        Json::Value _Data = DataList[i]["data"];

        init_data->set_start(_address);
        init_data->set_length(_length);

        uint32_t type = _Data[0].size();
        if (type == 1) {
            type = 0;
        } else if (type == 4) {
            type = 1;
        } else if (type == 16) {
            type = 3;
        }
        WriteData(_Data, _address, int32_t(type), DataDebug, init_data);
        OUTPUT(system("pause"));
    }
}

void JsonConfig::PIConfig(const Json::Value &PIArray, tianjic_ir::code::Core *core) {
    for (Json::ArrayIndex i = 0; i < PIArray.size(); i++) {

        PIGroupValue = PIArray[i];

        A_valid = PIGroupValue["A_valid"].asBool();
        S1_valid = PIGroupValue["S1_valid"].asBool();
        R_valid = PIGroupValue["R_valid"].asBool();
        S2_valid = PIGroupValue["S2_valid"].asBool();

        ::tianjic_ir::code::PIGroup *pi_group = core->add_pi_list();
        pi_group->set_a_valid(A_valid);
        pi_group->set_s1_valid(S1_valid);
        pi_group->set_r_valid(R_valid);
        pi_group->set_s2_valid(S2_valid);
        ::tianjic_ir::code::PIParameter *pi_parameter = new ::tianjic_ir::code::PIParameter();
        PIParameterConfig(PIGroupValue["PI_parameter"], pi_parameter);
        pi_group->set_allocated_pi_parameter(pi_parameter);
    }
}

void JsonConfig::PIParameterConfig(const Json::Value &PIParameterArray, ::tianjic_ir::code::PIParameter *pi_parameter) {
    if (A_valid == true) {
        ::tianjic_ir::code::AxonParameter *axon = new ::tianjic_ir::code::AxonParameter();
        ConfigAxonPI(PIParameterArray[0], axon);
        pi_parameter->set_allocated_axon(axon);
    }
    if (S1_valid == true) {
        ::tianjic_ir::code::SomaParameter *soma1 = new ::tianjic_ir::code::SomaParameter();
        ConfigSomaPI(PIParameterArray[1], soma1);
        pi_parameter->set_allocated_soma1(soma1);
    }
    if (R_valid == true) {
        ::tianjic_ir::code::RouterParameter *router = new ::tianjic_ir::code::RouterParameter();
        ConfigRouterPI(PIParameterArray[2], router);
        pi_parameter->set_allocated_router(router);
    }
    if (S2_valid == true) {
        ::tianjic_ir::code::SomaParameter *soma2 = new ::tianjic_ir::code::SomaParameter();
        ConfigSomaPI(PIParameterArray[3], soma2);
        pi_parameter->set_allocated_soma2(soma2);
    }
}

void JsonConfig::ConfigAxonPI(const Json::Value &PIValue, ::tianjic_ir::code::AxonParameter *axon) {
    uint32_t PIC = PIValue["PIC"].asUInt();

    OUTPUT(cout << "********* Axon PI *********" << endl);
    if (PIC == 0x41) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_mac_group_number_last(PIValue["mac_group_number_last"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_x2_base_addr(PIValue["x2_base_addr"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_x2_precision(PIValue["x2_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop0_extent(PIValue["loop0_extent"].asInt());
        axon->set_loop1_extent(PIValue["loop1_extent"].asInt());
        axon->set_loop2_extent(PIValue["loop2_extent"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_loop0_num_in_last_row(PIValue["loop0_num_in_last_row"].asInt());
        axon->set_x1_addr_loop1_step(PIValue["x1_addr_loop1_step"].asInt());
        axon->set_x1_addr_loop2_step(PIValue["x1_addr_loop2_step"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_x1_addr_mac_step(PIValue["x1_addr_mac_step"].asInt());
        axon->set_stride_x(PIValue["stride_x"].asInt());
        axon->set_stride_y(PIValue["stride_y"].asInt());
        axon->set_dilate_x(PIValue["dilate_x"].asInt());
        axon->set_dilate_y(PIValue["dilate_y"].asInt());
        axon->set_pad_top(PIValue["pad_top"].asInt());
        axon->set_pad_down(PIValue["pad_down"].asInt());
        axon->set_pad_left(PIValue["pad_left"].asInt());
        axon->set_pad_right(PIValue["pad_right"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x81) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_mac_group_number_last(PIValue["mac_group_number_last"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_x2_base_addr(PIValue["x2_base_addr"].asInt());
        axon->set_x2_precision(PIValue["x2_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop0_extent(PIValue["loop0_extent"].asInt());
        axon->set_loop1_extent(PIValue["loop1_extent"].asInt());
        axon->set_loop2_extent(PIValue["loop2_extent"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_loop0_num_in_last_row(PIValue["loop0_num_in_last_row"].asInt());
        axon->set_x1_addr_loop1_step(PIValue["x1_addr_loop1_step"].asInt());
        axon->set_x1_addr_loop2_step(PIValue["x1_addr_loop2_step"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_x1_addr_mac_step(PIValue["x1_addr_mac_step"].asInt());
        axon->set_stride_x(PIValue["stride_x"].asInt());
        axon->set_stride_y(PIValue["stride_y"].asInt());
        axon->set_dilate_x(PIValue["dilate_x"].asInt());
        axon->set_dilate_y(PIValue["dilate_y"].asInt());
        axon->set_pad_top(PIValue["pad_top"].asInt());
        axon->set_pad_down(PIValue["pad_down"].asInt());
        axon->set_pad_left(PIValue["pad_left"].asInt());
        axon->set_pad_right(PIValue["pad_right"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x02) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop0_extent(PIValue["loop0_extent"].asInt());
        axon->set_loop1_extent(PIValue["loop1_extent"].asInt());
        axon->set_loop2_extent(PIValue["loop2_extent"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_x1_addr_loop1_step(PIValue["x1_addr_loop1_step"].asInt());
        axon->set_x1_addr_loop2_step(PIValue["x1_addr_loop2_step"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_stride_x(PIValue["stride_x"].asInt());
        axon->set_stride_y(PIValue["stride_y"].asInt());
        axon->set_pad_top(PIValue["pad_top"].asInt());
        axon->set_pad_down(PIValue["pad_down"].asInt());
        axon->set_pad_left(PIValue["pad_left"].asInt());
        axon->set_pad_right(PIValue["pad_right"].asInt());
        axon->set_constant_b(PIValue["constant_b"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x03) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_x2_base_addr(PIValue["x2_base_addr"].asInt());
        axon->set_x2_precision(PIValue["x2_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_constant_b(PIValue["constant_b"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x43) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_x2_base_addr(PIValue["x2_base_addr"].asInt());
        axon->set_x2_precision(PIValue["x2_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_constant_b(PIValue["constant_b"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x83) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_loop4_extent(PIValue["loop4_extent"].asInt());
        axon->set_loop5_extent(PIValue["loop5_extent"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_x1_addr_loop4_step(PIValue["x1_addr_loop4_step"].asInt());
        axon->set_x1_addr_loop5_step(PIValue["x1_addr_loop5_step"].asInt());
        axon->set_constant_a(PIValue["constant_a"].asInt());
        axon->set_constant_b(PIValue["constant_b"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    } else if (PIC == 0x04) {
        axon->set_pic(PIC & 0xff);
        axon->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
        axon->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
        axon->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        axon->set_x1_precision(PIValue["x1_precision"].asInt());
        axon->set_x2_base_addr(PIValue["x2_base_addr"].asInt());
        axon->set_x2_precision(PIValue["x2_precision"].asInt());
        axon->set_bias_type(PIValue["bias_type"].asInt());
        axon->set_bias_base_addr(PIValue["bias_base_addr"].asInt());
        axon->set_o_base_addr(PIValue["o_base_addr"].asInt());
        axon->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        axon->set_o_end_addr(PIValue["o_end_addr"].asInt());
        axon->set_loop0_extent(PIValue["loop0_extent"].asInt());
        axon->set_loop3_extent(PIValue["loop3_extent"].asInt());
        axon->set_x1_addr_loop3_step(PIValue["x1_addr_loop3_step"].asInt());
        axon->set_loop0_num_in_last_row(PIValue["loop0_num_in_last_row"].asInt());
        axon->set_constant_b(PIValue["constant_b"].asInt());
        axon->set_a2s2_mode(PIValue["a2s2_mode"].asInt());
    }
    OUTPUT(cout << "********* Axon PI *********" << endl);
}

void JsonConfig::WriteData(const Json::Value &ArgInput, uint32_t ArgAddr, int ArgInType, bool ArgDebugPrint, tianjic_ir::code::InitData *init_data) {
    int k = 0;
    for (uint32_t i = 0; i < ArgInput.size(); i++, ArgAddr++) {
        uint32_t tmp = 0;
        if (ArgInType == 0) {
            tmp = ArgInput[i][0].asInt64() & 0xffffffff;
        } else if (ArgInType == 1 || ArgInType == 2) {
            for (int j = 3; j >= 0; j--) {
                tmp |= uint32_t((ArgInput[i][j].asInt64() & 0xff) << (8 * j));
            }
        } else {
            for (int j = 15; j >= 0; j--) {
                tmp |= uint32_t((ArgInput[i][j].asInt64() & 3) << (2 * j));
            }
        }
        if (ArgDebugPrint) {
            cout << "Chip(" << cx << "," << cy << ") core(" << x << "," << y << ")" << std::hex << std::setw(4) << " addr:0x" << ArgAddr + k++ << "\t: " << std::setw(8) << std::setfill('0') << tmp
                 << std::dec << endl;
        }
        init_data->add_data(int(tmp));
    }
}

void JsonConfig::ConfigSomaPI(const Json::Value &PIValue, ::tianjic_ir::code::SomaParameter *soma) {
    uint32_t PIC = PIValue["PIC"].asUInt();
    OUTPUT(cout << "********* Soma1 PI *********" << endl);
    if (PIC == 0x08) {
        cout << "PIC==0x08" << endl;
        soma->set_pic(PIC & 0xff);
        soma->set_pic_mode(PIValue["PI_S.pic_mode"].asInt());
        soma->set_reset_uin_addr(PIValue["PI_S.reset_uin_addr"].asInt());
        soma->set_reset_o_addr(PIValue["PI_S.reset_o_addr"].asInt());
        soma->set_reset_s_addr(PIValue["PI_S.reset_s_addr"].asInt());
        soma->set_reset_vm_addr(PIValue["PI_S.reset_vm_addr"].asInt());
        soma->set_reset_vtheta_addr(PIValue["PI_S.reset_vtheta_addr"].asInt());
        soma->set_tw_en(PIValue["PI_S.Tw_en"].asInt());
        soma->set_uin_base_addr(PIValue["PI_S.uin_base_addr"].asInt());
        soma->set_uin_end_addr(PIValue["PI_S.uin_end_addr"].asInt());
        soma->set_s_base_addr(PIValue["PI_S.s_base_addr"].asInt());
        soma->set_s_end_addr(PIValue["PI_S.s_end_addr"].asInt());
        soma->set_v_base_addr(PIValue["PI_S.v_base_addr"].asInt());
        soma->set_o_end_addr(PIValue["PI_S.o_end_addr"].asInt());
        soma->set_neuron_num(PIValue["PI_S.neuron_num"].asInt());
        soma->set_tw_len(PIValue["PI_S.Tw_len"].asInt());
        soma->set_y_num(PIValue["PI_S.Y_num"].asInt());
        soma->set_vm_base_addr(PIValue["PI_S.vm_base_addr"].asInt());
        soma->set_vm_end_addr(PIValue["PI_S.vm_end_addr"].asInt());
        soma->set_vtheta_base_addr(PIValue["PI_S.vtheta_base_addr"].asInt());
        soma->set_vtheta_end_addr(PIValue["PI_S.vtheta_end_addr"].asInt());
        soma->set_vinit(PIValue["PI_S.Vinit"].asInt());
        soma->set_reset_mode(PIValue["PI_S.reset_mode"].asInt());
        soma->set_fire_type(PIValue["PI_S.fire_type"].asInt());
        soma->set_vth_adpt_en(PIValue["PI_S.Vth_adpt_en"].asInt());
        soma->set_vleaky_adpt_en(PIValue["PI_S.Vleaky_adpt_en"].asInt());
        soma->set_para_base_addr(PIValue["PI_S.para_base_addr"].asInt());
        soma->set_bit_shift_num(PIValue["PI_S.bit_shift_num"].asInt());
        soma->set_row_pipeline_num(PIValue["PI_S.row_pipeline_num"].asInt());
        soma->set_row_pipeline_en(PIValue["row_pipeline_en"].asInt());
        soma->set_memory_select(PIValue["PI_S.memory_select"].asInt());
        soma->set_seed(int(PIValue["P_LIF.seed"].asInt()));
        soma->set_vth0(int(PIValue["P_LIF.Vth0"].asInt()));
        soma->set_vth_alpha(int(PIValue["P_LIF.Vth_alpha"].asInt()));
        soma->set_vth_beta(int(PIValue["P_LIF.Vth_beta"].asInt()));
        soma->set_vth_incre(int(PIValue["P_LIF.Vth_Incre"].asInt()));
        soma->set_vr(int(PIValue["P_LIF.VR"].asInt()));
        soma->set_vl(int(PIValue["P_LIF.VL"].asInt()));
        soma->set_vleaky_alpha(int(PIValue["P_LIF.Vleaky_alpha"].asInt()));
        soma->set_vleaky_beta(int(PIValue["P_LIF.Vleaky_beta"].asInt()));
        soma->set_dv(int(PIValue["P_LIF.dV"].asInt()));
        soma->set_ref_len(int(PIValue["P_LIF.Ref_len"].asInt()));
        soma->set_tw_cnt(int(PIValue["P_LIF.Tw_cnt"].asInt()));
    } else if (PIC == 0x05) {
        cout << "PIC==0x05" << endl;

        soma->set_pic(PIC & 0xff);
        soma->set_pic_mode(PIValue["pic_mode"].asInt());
        soma->set_reset_x1_addr(PIValue["reset_x1_addr"].asBool());
        soma->set_reset_o_addr(PIValue["reset_o_addr"].asBool());
        soma->set_row_pipeline_en(PIValue["row_pipeline_en"].asBool());
        soma->set_x1_base_addr(int(PIValue["x1_base_addr"].asInt64() & 0xffffffff));
        soma->set_x1_precision(int(PIValue["x1_precision"].asInt64() & 0xffffffff));
        soma->set_x1_end_addr(int(PIValue["x1_end_addr"].asInt64() & 0xffffffff));
        soma->set_o_base_addr(int(PIValue["o_base_addr"].asInt64() & 0xffffffff));
        soma->set_out_precision(int(PIValue["x2_precision"].asInt64() & 0xffffffff));
        soma->set_o_end_addr(int(PIValue["o_end_addr"].asInt64() & 0xffffffff));
        soma->set_km_num_in(int(PIValue["nif"].asInt64() & 0xffffffff));
        soma->set_nkx(int(PIValue["nkx"].asInt64() & 0xffffffff));
        soma->set_nky(int(PIValue["nky"].asInt64() & 0xffffffff));
        soma->set_km_num_out(int(PIValue["nof"].asInt64() & 0xffffffff));
        soma->set_nx(int(PIValue["nx"].asInt64() & 0xffffffff));
        soma->set_ny(int(PIValue["ny"].asInt64() & 0xffffffff));
        soma->set_kx_step(int(PIValue["kx_step"].asInt64() & 0xffffffff));
        soma->set_ky_step(int(PIValue["ky_step"].asInt64() & 0xffffffff));
        soma->set_f_step(int(PIValue["f_step"].asInt64() & 0xffffffff));
        soma->set_x_step(int(PIValue["x_step"].asInt64() & 0xffffffff));
        soma->set_y_step(int(PIValue["y_step"].asInt64() & 0xffffffff));
        soma->set_stride_x(int(PIValue["stride_x"].asInt64() & 0xffffffff));
        soma->set_stride_y(int(PIValue["stride_y"].asInt64() & 0xffffffff));
        soma->set_pad_top(int(PIValue["pad_top"].asInt64() & 0xffffffff));
        soma->set_pad_down(int(PIValue["pad_down"].asInt64() & 0xffffffff));
        soma->set_pad_left(int(PIValue["pad_left"].asInt64() & 0xffffffff));
        soma->set_pad_right(int(PIValue["pad_right"].asInt64() & 0xffffffff));
        soma->set_compare_init(uint32_t(PIValue["compare_init"].asInt64() & 0xffffffff));
        soma->set_bit_shift_num(int(PIValue["bit_shift_num"].asInt64() & 0xffffffff));
        soma->set_row_pipeline_num(int(PIValue["row_pipeline_num"].asInt64() & 0xffffffff));
        soma->set_memory_select(PIValue["memory_select"].asInt());
    } else if (PIC == 0x06) {
        int _pic_mode = PIValue["pic_mode"].asInt();
        if (_pic_mode == 0) {
            cout << "PIC==0x06" << endl;

            soma->set_pic(PIC & 0xff);
            soma->set_pic_mode(PIValue["pic_mode"].asInt());
            soma->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
            soma->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
            soma->set_reset_ciso_addr(PIValue["reset_ciso_addr"].asInt());
            soma->set_row_pipeline_en(PIValue["row_pipeline_en"].asInt());
            soma->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
            soma->set_x1_precision(PIValue["x1_precision"].asInt());
            soma->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
            soma->set_o_base_addr(PIValue["o_base_addr"].asInt());
            soma->set_out_precision(PIValue["out_precision"].asInt());
            soma->set_o_end_addr(PIValue["o_end_addr"].asInt());
            soma->set_ciso_base_addr(PIValue["ciso_base_addr"].asInt());
            soma->set_ciso_end_addr(PIValue["ciso_end_addr"].asInt());
            soma->set_km_num_in(PIValue["Km_num_in"].asInt());
            soma->set_km_num_out(PIValue["Km_num_out"].asInt());
            soma->set_km_num_ciso(PIValue["Km_num_ciso"].asInt());
            soma->set_num_in(PIValue["num_in"].asInt());
            soma->set_num_out(PIValue["num_out"].asInt());
            soma->set_num_ciso(PIValue["num_ciso"].asInt());
            soma->set_bit_shift_num(PIValue["bit_shift_num"].asInt());
            soma->set_row_pipeline_num(PIValue["row_pipeline_num"].asInt());
            soma->set_memory_select(PIValue["memory_select"].asInt());
            soma->set_in_ciso_pipe_sel(PIValue["in_ciso_pipe_sel"].asInt());

        } else if (_pic_mode == 1) {
            cout << "PIC==0x26" << endl;

            soma->set_pic(PIC & 0xff);
            soma->set_pic_mode(PIValue["pic_mode"].asInt());
            soma->set_reset_x1_addr(PIValue["reset_x1_addr"].asInt());
            soma->set_reset_o_addr(PIValue["reset_o_addr"].asInt());
            soma->set_reset_ciso_addr(PIValue["reset_ciso_addr"].asInt());
            soma->set_row_pipeline_en(PIValue["row_pipeline_en"].asInt());
            soma->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
            soma->set_x1_precision(PIValue["x1_precision"].asInt());
            soma->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
            soma->set_o_base_addr(PIValue["o_base_addr"].asInt());
            soma->set_out_precision(PIValue["out_precision"].asInt());
            soma->set_o_end_addr(PIValue["o_end_addr"].asInt());
            soma->set_ciso_base_addr(PIValue["ciso_base_addr"].asInt());
            soma->set_ciso_end_addr(PIValue["ciso_end_addr"].asInt());
            soma->set_km_num_in(PIValue["Km_num_in"].asInt());
            soma->set_km_num_out(PIValue["Km_num_out"].asInt());
            soma->set_km_num_ciso(PIValue["Km_num_ciso"].asInt());
            soma->set_num_in(PIValue["num_in"].asInt());
            soma->set_num_out(PIValue["num_out"].asInt());
            soma->set_num_ciso(PIValue["num_ciso"].asInt());
            soma->set_bit_shift_num(PIValue["bit_shift_num"].asInt());
            soma->set_row_pipeline_num(PIValue["row_pipeline_num"].asInt());
            soma->set_memory_select(PIValue["memory_select"].asInt());
            soma->set_out_ciso_sel(PIValue["out_ciso_sel"].asInt());
        }
    } else if (PIC == 0x07) {
        cout << "PIC==0x07" << endl;

        soma->set_pic(PIC & 0xff);
        soma->set_reset_x1_addr(PIValue["reset_x1_addr"].asBool());
        soma->set_reset_o_addr(PIValue["reset_o_addr"].asBool());
        soma->set_row_pipeline_en(PIValue["row_pipeline_en"].asInt());
        soma->set_x1_base_addr(PIValue["x1_base_addr"].asInt());
        soma->set_x1_precision(PIValue["x1_precision"].asInt());
        soma->set_x1_end_addr(PIValue["x1_end_addr"].asInt());
        soma->set_o_base_addr(PIValue["o_base_addr"].asInt());
        soma->set_x2_precision(PIValue["x2_precision"].asInt());
        soma->set_o_end_addr(PIValue["o_end_addr"].asInt());
        soma->set_neuron_num(PIValue["neuron_num"].asInt());
        soma->set_y_num(PIValue["Y_num"].asInt());
        soma->set_lut_base_addr(PIValue["lut_base_addr"].asInt());
        soma->set_lut_data_width(PIValue["lut_data_width"].asInt());
        soma->set_bit_shift_num(PIValue["bit_shift_num"].asInt());
        soma->set_row_pipeline_num(PIValue["row_pipeline_num"].asInt());
        soma->set_memory_select(PIValue["memory_select"].asInt());
    }
    OUTPUT(cout << "********* Soma1 PI *********" << endl);
}

void JsonConfig::ConfigRouterPI(const Json::Value &PIValue, ::tianjic_ir::code::RouterParameter *router) {
    uint32_t PIC = PIValue["PIC"].asUInt();
    OUTPUT(cout << "********* Router PI *********" << endl);
    if (PIC == 0x09) {
        cout << "PIC==0x09" << endl;

        router->set_pic(PIC & 0xff);
        router->set_rhead_mode(PIValue["Rhead_mode"].asInt());
        router->set_cxy(PIValue["CXY"].asInt());
        router->set_send_en(PIValue["Send_en"].asInt());
        router->set_receive_en(PIValue["Receive_en"].asInt());
        router->set_dout_memory_select(PIValue["Dout_memory_select"].asInt());
        router->set_addr_dout_base(PIValue["Addr_Dout_base"].asInt());
        router->set_addr_dout_length(PIValue["Addr_Dout_length"].asInt());
        router->set_addr_rhead_base(PIValue["Addr_Rhead_base"].asInt());
        router->set_addr_rhead_length(PIValue["Addr_Rhead_length"].asInt());
        router->set_addr_din_base(PIValue["Addr_Din_base"].asInt());
        router->set_addr_din_length(PIValue["Addr_Din_length"].asInt());
        router->set_send_number(PIValue["Send_number"].asInt());
        router->set_receive_number(PIValue["Receive_number"].asInt());
        router->set_nx(PIValue["Nx"].asInt());
        router->set_ny(PIValue["Ny"].asInt());
        router->set_back_sign_en(PIValue["Back_Sign_en"].asInt());
        router->set_send_pi_en(PIValue["Send_PI_en"].asInt());
        router->set_send_pi_num(PIValue["Send_PI_num"].asInt());
        router->set_receive_sign_en(PIValue["Receive_sign_en"].asInt());
        router->set_receive_sign_num(PIValue["Receive_sign_num"].asInt());
        router->set_send_pi_addr_base(PIValue["Send_PI_addr_base"].asInt());
        router->set_relay_number(PIValue["Relay_number"].asInt());
        router->set_q(PIValue["Q"].asInt());
        router->set_t_mode(PIValue["T_mode"].asInt());
        router->set_soma_in_en(PIValue["Soma_in_en"].asInt());
    }
    OUTPUT(cout << "********* Router PI *********" << endl);
}

void JsonConfig::OutputProto(std::string output_name, std::string output_name_readable) {
    fstream output(output_name, ios::out | ios::binary);
    if (output.is_open()) {
        _config->SerializeToOstream(&output);
        output.close();
    }

    fstream IRDebug(output_name_readable, ios::out);
    IRDebug << _config->DebugString();
    IRDebug.close();
}