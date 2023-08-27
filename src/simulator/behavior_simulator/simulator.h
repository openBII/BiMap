// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "src/simulator/behavior_simulator/chip_array.h"
#include "src/simulator/behavior_simulator/memory_visitor.h"
#include "src/simulator/behavior_simulator/patch.h"
#include "src/simulator/behavior_simulator/primitive/prim_02.h"
#include "src/simulator/behavior_simulator/primitive/prim_03.h"
#include "src/simulator/behavior_simulator/primitive/prim_04.h"
#include "src/simulator/behavior_simulator/primitive/prim_05.h"
#include "src/simulator/behavior_simulator/primitive/prim_06.h"
#include "src/simulator/behavior_simulator/primitive/prim_07.h"
#include "src/simulator/behavior_simulator/primitive/prim_08.h"
#include "src/simulator/behavior_simulator/primitive/prim_25.h"
#include "src/simulator/behavior_simulator/primitive/prim_26.h"
#include "src/simulator/behavior_simulator/primitive/prim_41.h"
#include "src/simulator/behavior_simulator/primitive/prim_43.h"
#include "src/simulator/behavior_simulator/primitive/prim_81.h"
#include "src/simulator/behavior_simulator/primitive/prim_83.h"
#include "src/simulator/behavior_simulator/util.h"
#include "json/json.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace Json;

static void config_data_block(shared_ptr<Core> p_core,
                              shared_ptr<Primitive> prim, const Value &blocks,
                              shared_ptr<MemoryVisitor> visitor) {
    int32_t output_cnt = 0;
    for (uint32_t i = 0; i < blocks.size(); ++i) {
        Value block = blocks[i];
        string id = block["id"].asString();
        size_t start = block["start"].asUInt();

        if (block.isMember("direction_out") ? block["direction_out"].asBool()
                                            : false) {
            // 输出块
            if (block.isMember("length")) {
                size_t block_start = start * 4;
                size_t block_length = block["length"].asUInt() * 4;
                if (size_t(prim->get_output_length()[output_cnt]) >
                    block_length) {
                    size_t block_size =
                        size_t(prim->get_output_length()[output_cnt]);
                    DataBlock output(p_core->get_id(), id, nullptr, block_start,
                                     block_length, block_size);

                    if (block.isMember("print_length")) {
                        visitor->add_output_segment(
                            {block_start, block["print_length"].asInt() * 4});
                    } else {
                        visitor->add_output_segment(
                            {block_start, block_length});
                    }
                    p_core->init_data_block(output);
                    prim->add_output_id(output.get_id());
                } else {
                    DataBlock output(p_core->get_id(), id, nullptr, block_start,
                                     block_length);
                    if (block.isMember("print_length")) {
                        visitor->add_output_segment(
                            {block_start, block["print_length"].asInt() * 4});
                    } else {
                        visitor->add_output_segment(
                            {block_start, block_length});
                    }
                    p_core->init_data_block(output);
                    prim->add_output_id(output.get_id());
                }
            } else {
                size_t block_start = start * 4;
                size_t block_length =
                    size_t(prim->get_output_length()[output_cnt]);
                DataBlock output(p_core->get_id(), id, nullptr, block_start,
                                 block_length);
                if (block.isMember("print_length")) {
                    visitor->add_output_segment(
                        {block_start, block["print_length"].asInt() * 4});
                } else {
                    visitor->add_output_segment({block_start, block_length});
                }
                p_core->init_data_block(output);
                prim->add_output_id(output.get_id());
            }
            ++output_cnt;
        } else {
            // 输入块
            if (block.isMember("input_source_id")) {
                // 例如：A-S1流水，S1的输入块,有key:input_source_id的，必有key：length
                ID input_source_id = ID::make_resource_id(
                    p_core->get_id(), block["input_source_id"].asString());
                if (block.isMember("length")) {
                    size_t block_start = start * 4;
                    size_t block_length = block["length"].asUInt() * 4;
                    DataBlock input(p_core->get_id(), id, nullptr, block_start,
                                    block_length, input_source_id);
                    p_core->init_data_block(input);
                    prim->add_input_id(input.get_id());
                } else {
                    throw runtime_error(id + " (block.isMember(length)) else");
                }
            } else {
                // 所有输入块
                if (block["init"]) {
                    string id = block["id"].asString();
                    size_t start = block["start"].asUInt();
                    Value data = block["data"];
                    shared_ptr<int32_t> pdata = new_array<int32_t>(data.size());
                    auto p = pdata.get();
                    for (uint32_t n = 0; n < data.size(); ++n) {
                        *p++ = data[n][0].asInt();
                    }
                    size_t block_start = start * 4;
                    size_t block_length = data.size() * 4;
                    DataBlock input(p_core->get_id(), id,
                                    reinterpret_pointer_cast<uint8_t>(pdata),
                                    block_start, block_length);
                    p_core->init_data_block(input);
                    continue;
                }

                Value data = block["data"];
                if (data.empty()) {
                    if (block.isMember("length")) {
                        size_t block_start = start * 4;
                        size_t block_length = block["length"].asUInt() * 4;
                        DataBlock input(p_core->get_id(), id, nullptr,
                                        block_start, block_length);
                        p_core->init_data_block(input);
                        prim->add_input_id(input.get_id());
                    } else {
                        size_t block_start = start * 4;
                        DataBlock input(p_core->get_id(), id, nullptr,
                                        block_start, 0);
                        p_core->init_data_block(input);
                        prim->add_input_id(input.get_id());
                    }
                } else {
                    shared_ptr<int32_t> pdata = new_array<int32_t>(data.size());
                    auto p = pdata.get();
                    for (uint32_t n = 0; n < data.size(); ++n) {
                        *p++ = data[n][0].asInt();
                    }
                    if (block.isMember("length")) {
                        size_t block_start = start * 4;
                        size_t block_length = block["length"].asUInt() * 4;
                        DataBlock input(
                            p_core->get_id(), id,
                            reinterpret_pointer_cast<uint8_t>(pdata),
                            block_start, block_length);
                        p_core->init_data_block(input);
                        prim->add_input_id(input.get_id());
                    } else {
                        size_t block_start = start * 4;
                        size_t block_length = data.size() * 4;
                        DataBlock input(
                            p_core->get_id(), id,
                            reinterpret_pointer_cast<uint8_t>(pdata),
                            block_start, block_length);
                        p_core->init_data_block(input);
                        prim->add_input_id(input.get_id());
                    }
                }
            }
        }
    }
}

static void config_09_data_block(shared_ptr<Core> p_core,
                                 shared_ptr<Primitive> prim,
                                 const Value &blocks,
                                 shared_ptr<MemoryVisitor> visitor) {
    for (uint32_t i = 0; i < blocks.size(); ++i) {
        Value block = blocks[i];

        if (block["init"]) {
            string id = block["id"].asString();
            size_t start = block["start"].asUInt();
            Value data = block["data"];
            shared_ptr<int32_t> pdata = new_array<int32_t>(data.size());
            auto p = pdata.get();
            for (uint32_t n = 0; n < data.size(); ++n) {
                *p++ = data[n][0].asInt();
            }
            size_t block_start = start * 4;
            size_t block_length = data.size() * 4;
            DataBlock input(p_core->get_id(), id,
                            reinterpret_pointer_cast<uint8_t>(pdata),
                            block_start, block_length);
            p_core->init_data_block(input);
        } else {
            if (block.isMember("A")) {
                // header

                size_t start = block["start"].asUInt();
                if (block.isMember("EN")) {
                    // RHead_mode=1
                    HeadAdvanced *pheader = new HeadAdvanced();
                    pheader->base.A = block["A"].asUInt();
                    pheader->base.X = block["X"].asInt();
                    pheader->base.Y = block["Y"].asInt();
                    pheader->base.S = block["S"].asUInt();
                    pheader->base.P = block["P"].asUInt();
                    pheader->base.Q = block["Q"].asUInt();
                    pheader->base.T = block["T"].asUInt();
                    pheader->A_offset = block["A_offset"].asUInt();
                    pheader->Const = block["Const"].asUInt();
                    pheader->EN = block["EN"].asUInt();
                    pheader->pack_per_Rhead = block["pack_per_Rhead"].asUInt();

                    DataBlock header_block(
                        p_core->get_id(),
                        string("packet_header_") + to_string(start),
                        reinterpret_cast<uint8_t *>(pheader), start, 8);
                    p_core->init_data_block(header_block);
                    prim->add_input_id(header_block.get_id());
                } else {
                    // Rhead_mode=0
                    HeadBase *pheader = new HeadBase();
                    pheader->A = block["A"].asUInt();
                    pheader->X = block["X"].asInt();
                    pheader->Y = block["Y"].asInt();
                    pheader->S = block["S"].asUInt();
                    pheader->P = block["P"].asUInt();
                    pheader->Q = block["Q"].asUInt();
                    pheader->T = block["T"].asUInt();

                    DataBlock header_block(
                        p_core->get_id(),
                        string("packet_header_") + to_string(start),
                        reinterpret_cast<uint8_t *>(pheader), start, 4);
                    p_core->init_data_block(header_block);
                    prim->add_input_id(header_block.get_id());
                }
            } else {
                // data

                string id = block["id"].asString();
                size_t block_start = block["start"].asUInt();
                size_t block_length = block["length"].asUInt();

                if (block.isMember("input_source_id")) {
                    // 例如：A-S1流水，S1的输入块,有key:input_source_id的，必有key：length
                    ID input_source_id = ID::make_resource_id(
                        p_core->get_id(), block["input_source_id"].asString());
                    if (block.isMember("length")) {
                        DataBlock input(p_core->get_id(), id, nullptr,
                                        block_start, block_length,
                                        input_source_id);
                        p_core->init_data_block(input);
                        prim->add_input_id(input.get_id());
                    } else {
                        throw runtime_error(id +
                                            " (block.isMember(length)) else");
                    }
                } else {
                    DataBlock input(p_core->get_id(), id, nullptr, block_start,
                                    block_length);
                    p_core->init_data_block(input);
                    prim->add_input_id(input.get_id());
                }
            }
        }
    }
}

static shared_ptr<Primitive>
single_pi_config(shared_ptr<Core> p_core, Value Root,
                 shared_ptr<MemoryVisitor> visitor) {
    if (Root.empty()) {
        return nullptr;
    }
    if (Root.isMember("axon_delay")) {
        if (Root["axon_delay"].asBool()) {
            return nullptr;
        } else {
            return nullptr;
        }
    } else {
        shared_ptr<Primitive> prim;

        if (Root["PIC"] == 0x41) {
            shared_ptr<Prim41_Parameter> para = make_shared<Prim41_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->x2_precision = Root["x2_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->niy = Root["niy"].asInt();
            para->nix = Root["nix"].asInt();
            para->nif = Root["nif"].asInt();
            para->nof = Root["nof"].asInt();
            para->nkx = Root["nkx"].asInt();
            para->nky = Root["nky"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->pad_top = Root["pad_top"].asInt();
            para->pad_down = Root["pad_down"].asInt();
            para->pad_left = Root["pad_left"].asInt();
            para->pad_right = Root["pad_right"].asInt();
            para->dilate_x = Root["dilate_x"].asInt();
            para->dilate_y = Root["dilate_y"].asInt();

            prim = make_shared<Prim41>(para);
        } else if (Root["PIC"] == 0x81) {
            shared_ptr<Prim81_Parameter> para = make_shared<Prim81_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->x2_precision = Root["x2_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->niy = Root["niy"].asInt();
            para->nix = Root["nix"].asInt();
            para->nif = Root["nif"].asInt();
            para->nof = Root["nof"].asInt();
            para->nkx = Root["nkx"].asInt();
            para->nky = Root["nky"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->pad_top = Root["pad_top"].asInt();
            para->pad_down = Root["pad_down"].asInt();
            para->pad_left = Root["pad_left"].asInt();
            para->pad_right = Root["pad_right"].asInt();
            para->dilate_x = Root["dilate_x"].asInt();
            para->dilate_y = Root["dilate_y"].asInt();

            prim = make_shared<Prim81>(para);
        } else if (Root["PIC"] == 0x02) {
            shared_ptr<Prim02_Parameter> para = make_shared<Prim02_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->bias_length = Root["bias_length"].asInt();
            para->niy = Root["niy"].asInt();
            para->nix = Root["nix"].asInt();
            para->nif = Root["nif"].asInt();
            para->nkx = Root["nkx"].asInt();
            para->nky = Root["nky"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->pad_top = Root["pad_top"].asInt();
            para->pad_down = Root["pad_down"].asInt();
            para->pad_left = Root["pad_left"].asInt();
            para->pad_right = Root["pad_right"].asInt();
            para->constant_b = Root["constant_b"].asInt();
            para->avg_pooling_en = Root["avg_pooling_en"].asBool();

            prim = make_shared<Prim02>(para);
        } else if (Root["PIC"] == 0x03) {
            shared_ptr<Prim03_Parameter> para = make_shared<Prim03_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->bias_length = Root["bias_length"].asInt();
            para->ny = Root["ny"].asInt();
            para->nx = Root["nx"].asInt();
            para->nif = Root["nif"].asInt();
            para->n_branch = Root["n_branch"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->constant_b = Root["constant_b"].asInt();
            para->tensor_en = Root["tensor_en"].asBool();

            prim = make_shared<Prim03>(para);
        } else if (Root["PIC"] == 0x43) {
            shared_ptr<Prim43_Parameter> para = make_shared<Prim43_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->x2_precision = Root["x2_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->bias_length = Root["bias_length"].asInt();
            para->ny = Root["ny"].asInt();
            para->nx = Root["nx"].asInt();
            para->nif = Root["nif"].asInt();
            para->x2_length = Root["x2_length"].asInt();
            para->n_branch = Root["n_branch"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->constant_b = Root["constant_b"].asInt();
            para->tensor_en = Root["tensor_en"].asBool();

            prim = make_shared<Prim43>(para);
        } else if (Root["PIC"] == 0x83) {
            shared_ptr<Prim83_Parameter> para = make_shared<Prim83_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->bias_length = Root["bias_length"].asInt();
            para->ny = Root["ny"].asInt();
            para->nx = Root["nx"].asInt();
            para->nif = Root["nif"].asInt();
            para->n_branch = Root["n_branch"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->constant_b = Root["constant_b"].asInt();
            para->constant_a = Root["constant_a"].asInt();
            para->tensor_en = Root["tensor_en"].asBool();

            prim = make_shared<Prim83>(para);
        } else if (Root["PIC"] == 0x04) {
            shared_ptr<Prim04_Parameter> para = make_shared<Prim04_Parameter>();
            para->x1_precision = Root["x1_precision"].asInt();
            para->x2_precision = Root["x2_precision"].asInt();
            para->bias_type = Root["bias_type"].asInt();
            para->nif = Root["nif"].asInt();
            para->nof = Root["nof"].asInt();
            para->constant_b = Root["constant_b"].asInt();

            prim = make_shared<Prim04>(para);
        } else if (Root["PIC"] == 0x06) {
            shared_ptr<Prim06_Parameter> para = make_shared<Prim06_Parameter>();
            para->length_in = Root["length_in"].asInt();
            para->length_out = Root["length_out"].asInt();
            para->length_ciso = Root["length_ciso"].asInt();
            para->num_in = Root["num_in"].asInt();
            para->num_out = Root["num_out"].asInt();
            para->num_ciso = Root["num_ciso"].asInt();
            para->x1_precision = Root["x1_precision"].asInt();
            para->out_precision = Root["out_precision"].asInt();
            para->bit_shift_num = Root["bit_shift_num"].asInt();
            para->real_length_in_en = Root["real_length_in_en"].asBool();
            para->real_num_in = Root["real_num_in"].asInt();

            prim = make_shared<Prim06>(para);
        } else if (Root["PIC"] == 0x26) {
            shared_ptr<Prim26_Parameter> para = make_shared<Prim26_Parameter>();
            para->length_in = Root["length_in"].asInt();
            para->length_out = Root["length_out"].asInt();
            para->length_ciso = Root["length_ciso"].asInt();
            para->num_in = Root["num_in"].asInt();
            para->num_out = Root["num_out"].asInt();
            para->num_ciso = Root["num_ciso"].asInt();
            para->x1_precision = Root["x1_precision"].asInt();
            para->out_precision = Root["out_precision"].asInt();
            para->bit_shift_num = Root["bit_shift_num"].asInt();

            prim = make_shared<Prim26>(para);
        } else if (Root["PIC"] == 0x05) {
            shared_ptr<Prim05_Parameter> para = make_shared<Prim05_Parameter>();

            para->x1_precision = Root["x1_precision"].asInt();
            para->out_precision = Root["out_precision"].asInt();
            para->niy = Root["niy"].asInt();
            para->nix = Root["nix"].asInt();
            para->nif = Root["nif"].asInt();
            para->nof = Root["nof"].asInt();
            para->nkx = Root["nkx"].asInt();
            para->nky = Root["nky"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->pad_top = Root["pad_top"].asInt();
            para->pad_down = Root["pad_down"].asInt();
            para->pad_left = Root["pad_left"].asInt();
            para->pad_right = Root["pad_right"].asInt();
            para->compare_init = Root["compare_init"].asInt64() & 0xffffffff;
            para->bit_shift_num = Root["bit_shift_num"].asInt();

            prim = make_shared<Prim05>(para);
        } else if (Root["PIC"] == 0x25) {
            shared_ptr<Prim25_Parameter> para = make_shared<Prim25_Parameter>();

            para->x1_precision = Root["x1_precision"].asInt();
            para->out_precision = Root["out_precision"].asInt();
            para->niy = Root["niy"].asInt();
            para->nix = Root["nix"].asInt();
            para->nif = Root["nif"].asInt();
            para->nof = Root["nof"].asInt();
            para->nkx = Root["nkx"].asInt();
            para->nky = Root["nky"].asInt();
            para->stride_x = Root["stride_x"].asInt();
            para->stride_y = Root["stride_y"].asInt();
            para->pad_top = Root["pad_top"].asInt();
            para->pad_down = Root["pad_down"].asInt();
            para->pad_left = Root["pad_left"].asInt();
            para->pad_right = Root["pad_right"].asInt();
            para->compare_init = Root["compare_init"].asInt64() & 0xffffffff;
            para->bit_shift_num = Root["bit_shift_num"].asInt();

            prim = make_shared<Prim25>(para);
        } else if (Root["PIC"] == 0x07) {
            shared_ptr<Prim07_Parameter> para = make_shared<Prim07_Parameter>();

            para->group_num = Root["group_num"].asInt();
            para->neuron_real_num = Root["neuron_real_num"].asInt();
            para->lut_data_width = Root["lut_data_width"].asInt();
            para->x1_precision = Root["x1_precision"].asInt();
            para->x2_precision = Root["x2_precision"].asInt();
            para->bit_shift_num = Root["bit_shift_num"].asInt();

            prim = make_shared<Prim07>(para);
        } else if (Root["PIC"] == 0x08) {
            shared_ptr<Prim08_Parameter> para = make_shared<Prim08_Parameter>();

            para->neuron_num = Root["neuron_num"].asInt();
            para->group_num = Root["group_num"].asInt();
            para->seed = Root["seed"].asInt();
            para->Vth0 = Root["Vth0"].asInt();
            para->Vth_adpt_en = Root["Vth_adpt_en"].asInt();
            para->Vth_alpha = Root["Vth_alpha"].asInt();
            para->Vth_beta = Root["Vth_beta"].asInt();
            para->Vth_Incre = Root["Vth_Incre"].asInt();
            para->VR = Root["VR"].asInt();
            para->VL = Root["VL"].asInt();
            para->Vleaky_adpt_en = Root["Vleaky_adpt_en"].asInt();
            para->Vleaky_alpha = Root["Vleaky_alpha"].asInt();
            para->Vleaky_beta = Root["Vleaky_beta"].asInt();
            para->dV = Root["dV"].asInt();
            para->Ref_len = Root["Ref_len"].asInt();
            para->Tw_cnt = Root["Tw_cnt"].asInt();
            para->Vinit = Root["Vinit"].asInt();
            para->Tw_len = Root["Tw_len"].asInt();
            para->Tw_en = Root["Tw_en"].asInt();
            para->VM_const_en = Root["VM_const_en"].asInt();
            para->VM_const = Root["VM_const"].asInt64() & 0xffffffff;
            para->VM_len = Root["VM_len"].asInt();
            para->Vtheta_const_en = Root["Vtheta_const_en"].asInt();
            para->Vtheta_const = Root["Vtheta_const"].asInt();
            para->Vtheta_len = Root["Vtheta_len"].asInt();
            para->ref_cnt_const_en = Root["ref_cnt_const_en"].asInt();
            para->ref_cnt_const = Root["ref_cnt_const"].asInt();
            para->reset_mode = Root["reset_mode"].asInt();
            para->fire_type = Root["fire_type"].asInt();

            prim = make_shared<Prim08>(para);
        }

        else if (Root["PIC"] == 0x09) {
            shared_ptr<Prim09_Parameter> para = make_shared<Prim09_Parameter>(
                ROUTER::BROADCAST_TYPE(Root["multicast_relay_or_not"].asUInt()),
                Root["dx"].asInt(), Root["dy"].asInt(),
                Root["multicast_relay_or_not"].asUInt() == 0
                    ? 0
                    : Root["relay_packets_num"].asInt() + 1,
                Root["received_stop_num"].asInt() + 1,
                ROUTER::PACKET_TYPE(Root["hearder_multipack"].asInt()),
                Root["send_en"].asBool(), Root["recv_en"].asBool(),
                Root["recv_address"].asInt() * 4,
                Root["din_length"].asInt() * 4,
                Root["recv_end_phase"].asUInt());

            prim = make_shared<Prim09>(para);
            config_09_data_block(p_core, prim, Root["DataBlock"], visitor);
            if (Root["recv_en"].asBool()) {
                visitor->add_output_segment(
                    {Root["output_seg"]["start"].asUInt() * 4,
                     Root["output_seg"]["length"].asUInt() * 4});
            }

            return prim;
        }
        config_data_block(p_core, prim, Root["DataBlock"], visitor);
        return prim;
    }
}

class Simulator
{
 public:
    Simulator() : _visitor_master(make_shared<MemoryVisitorMaster>()) {}

    void simulate(void) { _chip_array->execute(); }

    //    void output(void){
    //        _visitor->serialize_fstream(*_ctx);
    //    }

    void map(const string &path) {
        Value Root;

        Json::CharReaderBuilder ReaderBuilder;
        ifstream InputStream(path);
        if (InputStream.is_open() == true) {
            string err;
            if (Json::parseFromStream(ReaderBuilder, InputStream, &Root,
                                      &err) != true) {
                throw runtime_error(path + " parse error");
            }
        } else {
            throw runtime_error(path + " open error");
        }

        _ctx = make_shared<Context>();
        shared_ptr<ChipArray> chiparray1 =
            make_shared<ChipArray>("ChipArray1", _ctx);
        Value chip_list = Root["ChipArray"];

        for (ArrayIndex i = 0; i < chip_list.size(); ++i) {
            chiparray1->add_chip(config_chip(chiparray1, chip_list[i]));
        }
        _chip_array = chiparray1;
    }

 private:
    shared_ptr<Chip> config_chip(shared_ptr<ChipArray> p_chip_array,
                                 Value chip_config) {
        shared_ptr<Chip> chip =
            make_shared<Chip>(p_chip_array->get_id(), chip_config["x"].asUInt(),
                              chip_config["y"].asUInt(),
                              chip_config["step_group_id"].asUInt(), _ctx);
        Value core_list = chip_config["cores"];
        for (ArrayIndex i = 0; i < core_list.size(); ++i) {
            chip->add_core(config_core(chip, core_list[i]));
        }
        return chip;
    }

    shared_ptr<Core> config_core(shared_ptr<Chip> p_chip, Value core_config) {
        shared_ptr<Core> core =
            make_shared<Core>(p_chip->get_id(), core_config["x"].asUInt(),
                              core_config["y"].asUInt(),
                              core_config["phase_group_id"].asUInt(), _ctx);

        Value pi_list = core_config["pi_groups"];
        for (ArrayIndex i = 0; i < pi_list.size(); ++i) {
            auto visitor = make_shared<MemoryVisitor>();
            core->add_pi_group(config_pi_group(core, pi_list[i], visitor));
            _visitor_master->set_visitor(core->get_id(), i, *visitor);
        }
        core->set_memory_visitor(
            _visitor_master->get_visitor_map(core->get_id()));
        return core;
    }

    PIGroup config_pi_group(shared_ptr<Core> p_core, Value pi_group_config,
                            shared_ptr<MemoryVisitor> visitor) {
        auto config = bind(single_pi_config, p_core, placeholders::_1, visitor);
        PIGroup t;
        t.add_axon(config(pi_group_config["axon"]));
        t.add_soma1(config(pi_group_config["soma1"]));
        t.add_router(config(pi_group_config["router"]));
        t.add_soma2(config(pi_group_config["soma2"]));
        return t;
        //        return PIGroup(
        //                config(pi_group_config["axon"]),
        //                config(pi_group_config["soma1"]),
        //                config(pi_group_config["router"]),
        //                config(pi_group_config["soma2"]));
    }

    shared_ptr<ChipArray> _chip_array;
    shared_ptr<Context> _ctx;

    shared_ptr<MemoryVisitorMaster> _visitor_master;
};

#endif  // SIMULATOR_H
