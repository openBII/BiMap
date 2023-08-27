// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "spdlog/spdlog.h"
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <json/json.h>
#include <map>
#include <string>

#include "src/compiler/ir/asm.pb.h"
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

using namespace std;

// #define DEBUG
class Simulator
{
 public:
    Simulator(string case_name)
        : _case_name(case_name),
          _visitor_master(make_shared<MemoryVisitorMaster>()),
          precision_to_bits{32, 8, 8, 2} {
        precision_map[tianjic_ir::INT_32] = 0;
        precision_map[tianjic_ir::INT_8] = 1;
        precision_map[tianjic_ir::UINT_8] = 2;
        precision_map[tianjic_ir::TERNARY] = 3;
    }

    void simulate(void) { _chip_array->execute(); }

    void mapConfig(const string &path) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        int fd = open(path.c_str(), O_RDONLY);
        tianjic_ir::assembly::Config config;
        google::protobuf::io::ZeroCopyInputStream *is =
            new google::protobuf::io::FileInputStream(fd);
        if (!google::protobuf::TextFormat::Parse(is, &config)) {
            spdlog::get("console")->error("Parse asm config error");
            exit(-1);
        }
        delete is;

        auto &behavior_config = config.behavior_config();
        auto &test_config = config.test_config();
        auto &io_config = config.data_config();

        GlobalConfig::TEST_MODE = test_config.test_mode();
        _ctx = make_shared<Context>();
        _ctx->set_seed(test_config.random_seed());

        auto &step_configs = behavior_config.step_config();
        _chip_array = make_shared<ChipArray>("ChipArray1", _ctx);

        // 当前只支持一个stepgroup的写法
        for (auto step_config : step_configs) {
            _ctx->n_step = step_config.has_step_exe_number()
                               ? step_config.step_exe_number()
                               : 1;
            _chip_array->add_chip(config_chip(_chip_array, step_config));
        }
        config_ioblock(io_config);
    }

 private:
    int out_base = 0x8000 << 2;
    const int precision_to_bits[4];
    map<tianjic_ir::Precision, int> precision_map;
    int phase_id;

    bool as_pipeline_en = false;
    bool sr_pipeline_en = false;
    int pipeline_num;
    int pipeline_ny;
    int pipeline_length;

    ID soma_o_id = ID();
    ID axon_o_id = ID();

    int axon_o_base_addr;
    int soma_o_base_addr;

    shared_ptr<Prim02_Parameter> paraConfig(tianjic_ir::assembly::P02 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim03_Parameter> paraConfig(tianjic_ir::assembly::P03 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim04_Parameter> paraConfig(tianjic_ir::assembly::P04 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim41_Parameter> paraConfig(tianjic_ir::assembly::P41 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim43_Parameter> paraConfig(tianjic_ir::assembly::P43 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim81_Parameter> paraConfig(tianjic_ir::assembly::P81 p,
                                            tianjic_ir::Shape shape);
    shared_ptr<Prim83_Parameter> paraConfig(tianjic_ir::assembly::P83 p,
                                            tianjic_ir::Shape shape);

    shared_ptr<Prim05_Parameter> paraConfig(tianjic_ir::assembly::PX5 p);
    shared_ptr<Prim25_Parameter> paraConfig25(tianjic_ir::assembly::PX5 p);
    shared_ptr<Prim06_Parameter> paraConfig(tianjic_ir::assembly::P06 p);
    shared_ptr<Prim26_Parameter> paraConfig(tianjic_ir::assembly::P26 p);
    shared_ptr<Prim07_Parameter> paraConfig(tianjic_ir::assembly::P07 p);
    shared_ptr<Prim08_Parameter> paraConfig(tianjic_ir::assembly::P08 p);

    shared_ptr<Primitive> primConfig(shared_ptr<Core> p_core,
                                     tianjic_ir::assembly::AxonPrim axon,
                                     shared_ptr<MemoryVisitor> visitor);
    shared_ptr<Primitive> primConfig(shared_ptr<Core> p_core,
                                     tianjic_ir::assembly::SomaPrim soma,
                                     shared_ptr<MemoryVisitor> visitor);
    shared_ptr<Primitive> primConfig(shared_ptr<Core> p_core,
                                     tianjic_ir::assembly::RouterPrim router,
                                     shared_ptr<MemoryVisitor> visitor);

    shared_ptr<Chip> config_chip(shared_ptr<ChipArray> p_chip_array,
                                 tianjic_ir::assembly::StepConfig step_config) {
        shared_ptr<Chip> chip = make_shared<Chip>(
            p_chip_array->get_id(), step_config.chip_x(), step_config.chip_y(),
            step_config.step_group_id(), _ctx);

        auto &phase_group_configs = step_config.phase_group_config();

        for (auto &phase_group_config : phase_group_configs) {
            auto &core_configs = phase_group_config.core_config();
            int phase_group_id = phase_group_config.phase_group_id();
            for (auto &core_config : core_configs) {
                chip->add_core(config_core(chip, core_config, phase_group_id));
            }
        }
        return chip;
    }

    void config_ioblock(const tianjic_ir::DataConfig &io_config) {
        // config io block
        auto &dynamic_blocks = io_config.dynamic_blocks();
        auto &static_blocks = io_config.static_blocks();

        auto get_core = [&](auto blk) {
            return _ctx->getCoreById(ID::make_core_id(
                ID::make_chip_id(ID::make_chip_array_id("ChipArray1"),
                                 blk.chip_idx(), blk.chip_idy()),
                blk.core_idx(), blk.core_idy()));
        };

        auto add_input_data_block = [&](auto blk) {
            // TODO : 修正非0 Phase的情况
            auto core = get_core(blk);
            core->addiblock(0, ID::make_data_block_id(core->get_id(), blk.id()),
                            blk, 0, this->_case_name);
        };

        auto add_output_data_block = [&](auto blk) {
            // output_block 没有core信息，只有start_addr能够用来识别数据
            auto core = get_core(blk);
            core->addoblock(blk.phases(0),
                            ID::make_data_block_id(core->get_id(), blk.id()),
                            blk, 0, this->_case_name);
        };

        for (auto &blk : static_blocks) {
            add_input_data_block(blk);
        }

        for (auto &blk : dynamic_blocks) {
            if (blk.io_type() == tianjic_ir::OUTPUT_DATA) {
                add_output_data_block(blk);
            } else {
                add_input_data_block(blk);
            }
        }
    }

    shared_ptr<Core> config_core(shared_ptr<Chip> p_chip,
                                 tianjic_ir::assembly::CoreConfig core_config,
                                 int phase_group_id) {
        shared_ptr<Core> core =
            make_shared<Core>(p_chip->get_id(), core_config.core_x(),
                              core_config.core_y(), phase_group_id, _ctx);
        auto &static_prim_lists = core_config.static_prim_list();

        phase_id = 0;
        for (auto &static_prim_list : static_prim_lists) {
            auto visitor = make_shared<MemoryVisitor>();
            core->add_pi_group(
                config_pi_group(core, static_prim_list, visitor));
            _visitor_master->set_visitor(core->get_id(), phase_id++, *visitor);
        }

        core->set_memory_visitor(
            _visitor_master->get_visitor_map(core->get_id()));
        return core;
    }

    PIGroup
    config_pi_group(shared_ptr<Core> p_core,
                    tianjic_ir::assembly::StaticPrimList static_prim_list,
                    shared_ptr<MemoryVisitor> visitor) {
        PIGroup t;

        as_pipeline_en = false;
        if (static_prim_list.has_axon() && static_prim_list.has_soma1()) {
            std::string o_id = std::to_string(static_prim_list.axon().pic()) +
                               "_o_" + std::to_string(phase_id);
            axon_o_id = ID::make_data_block_id(p_core->get_id(), o_id);
            axon_o_base_addr = static_prim_list.axon().o_base_addr() << 2;
        }

        sr_pipeline_en = false;
        if (static_prim_list.has_soma1() && static_prim_list.has_router()) {
            if (static_prim_list.router().p09().soma_in_en()) {
                sr_pipeline_en = true;
            }
        }

        if (static_prim_list.has_soma1()) {
            t.add_soma1(primConfig(p_core, static_prim_list.soma1(), visitor));
        } else {
            t.add_soma1(nullptr);
        }

        if (static_prim_list.has_axon()) {
            t.add_axon(primConfig(p_core, static_prim_list.axon(), visitor));
        } else {
            t.add_axon(nullptr);
        }

        if (static_prim_list.has_router())
            t.add_router(
                primConfig(p_core, static_prim_list.router(), visitor));
        else
            t.add_router(nullptr);

        sr_pipeline_en = false;
        if (static_prim_list.has_soma2()) {
            t.add_soma2(primConfig(p_core, static_prim_list.soma2(), visitor));
        } else {
            t.add_soma2(nullptr);
        }

        return t;
    }

    string _case_name;
    shared_ptr<ChipArray> _chip_array;
    shared_ptr<Context> _ctx;
    shared_ptr<MemoryVisitorMaster> _visitor_master;
    map<string, tianjic_ir::DynamicBlock> _dynamic_data_blocks;
    map<string, tianjic_ir::StaticBlock> _static_data_blocks;
};

shared_ptr<Prim06_Parameter>
Simulator::paraConfig(tianjic_ir::assembly::P06 p) {
    shared_ptr<Prim06_Parameter> para = make_shared<Prim06_Parameter>();
    para->length_in = p.length_in();
    para->length_out = p.length_out();
    para->length_ciso = p.length_ciso();
    para->num_in = p.num_in();
    para->num_out = p.num_out();
    para->num_ciso = p.num_ciso();
    para->x1_precision = precision_map[p.x1_precision()];
    para->out_precision = precision_map[p.out_precision()];
    para->bit_shift_num = p.bit_shift_num();
    para->real_length_in_en = p.real_length_in_en();
    para->real_num_in = p.real_num_in();
    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}

shared_ptr<Prim25_Parameter>
Simulator::paraConfig25(tianjic_ir::assembly::PX5 p) {
    shared_ptr<Prim25_Parameter> para = make_shared<Prim25_Parameter>();

    para->x1_precision = precision_map[p.x1_precision()];
    para->out_precision = precision_map[p.out_precision()];
    para->niy = p.niy();
    para->nix = p.nix();
    para->nif = p.nif();
    para->nof = p.nof();
    para->nkx = p.nkx();
    para->nky = p.nky();
    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->pad_top = p.pad_top();
    para->pad_down = p.pad_down();
    para->pad_left = p.pad_left();
    para->pad_right = p.pad_right();

    para->compare_init = 0;
    int size = p.compare_init_size();
    for (int i = 0; i < size; ++i) {
        uint32_t x = p.compare_init(i);
        if (size == 4) {
            x = (uint8_t)p.compare_init(i);
        } else if (size == 16) {
            x = p.compare_init(i) >= 0 ? p.compare_init(i)
                                       : 4 + p.compare_init(i);
        }
        para->compare_init = (para->compare_init << (32 / size)) | x;
    }
    para->bit_shift_num = p.bit_shift_num();

    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}
shared_ptr<Prim05_Parameter>
Simulator::paraConfig(tianjic_ir::assembly::PX5 p) {
    shared_ptr<Prim05_Parameter> para = make_shared<Prim05_Parameter>();

    para->x1_precision = precision_map[p.x1_precision()];
    para->out_precision = precision_map[p.out_precision()];
    para->niy = p.niy();
    para->nix = p.nix();
    para->nif = p.nif();
    para->nof = p.nof();
    para->nkx = p.nkx();
    para->nky = p.nky();
    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->pad_top = p.pad_top();
    para->pad_down = p.pad_down();
    para->pad_left = p.pad_left();
    para->pad_right = p.pad_right();

    para->compare_init = 0;
    int size = p.compare_init_size();
    for (int i = 0; i < size; ++i) {
        uint32_t x = p.compare_init(i);
        if (size == 4) {
            x = (uint8_t)p.compare_init(i);
        } else if (size == 16) {
            x = p.compare_init(i) >= 0 ? p.compare_init(i)
                                       : 4 + p.compare_init(i);
        }
        para->compare_init = (para->compare_init << (32 / size)) | x;
    }

    para->bit_shift_num = p.bit_shift_num();
    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}

shared_ptr<Prim07_Parameter>
Simulator::paraConfig(tianjic_ir::assembly::P07 p) {
    shared_ptr<Prim07_Parameter> para = make_shared<Prim07_Parameter>();

    para->group_num = p.group_num();
    para->neuron_real_num = p.neuron_real_num();
    para->lut_data_width = p.lut_data_width();
    para->x1_precision = precision_map[p.x1_precision()];
    para->x2_precision = precision_map[p.x2_precision()];
    para->bit_shift_num = p.bit_shift_num();
    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}
shared_ptr<Prim08_Parameter>
Simulator::paraConfig(tianjic_ir::assembly::P08 p) {
    shared_ptr<Prim08_Parameter> para = make_shared<Prim08_Parameter>();

    para->neuron_num = p.neuron_num();
    para->group_num = p.group_num();
    para->seed = p.seed();
    para->Vth0 = p.vth0();
    para->Vth_adpt_en = p.vth_adpt_en();
    para->Vth_alpha = p.vth_alpha();
    para->Vth_beta = p.vth_beta();
    para->Vth_Incre = p.vth_incre();
    para->VR = p.vr();
    para->VL = p.vl();
    para->Vleaky_adpt_en = p.vleaky_adpt_en();
    para->Vleaky_alpha = p.vleaky_alpha();
    para->Vleaky_beta = p.vleaky_beta();
    para->dV = p.dv();
    para->Ref_len = p.ref_len();
    para->Tw_cnt = p.tw_cnt();
    para->Vinit = p.vinit();
    para->Tw_len = p.tw_len();
    para->Tw_en = p.tw_en();
    para->VM_const_en = p.vm_const_en();
    para->VM_const = p.vm_const();
    para->VM_len = p.vm_len();
    para->Vtheta_const_en = p.vtheta_const_en();
    para->Vtheta_const = p.vtheta_const();
    para->Vtheta_len = p.vtheta_len();
    para->ref_cnt_const_en = p.ref_cnt_const_en();
    para->ref_cnt_const = p.ref_cnt_const();
    para->reset_mode = p.reset_mode();
    para->fire_type = p.fire_type();
    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}
shared_ptr<Prim26_Parameter>
Simulator::paraConfig(tianjic_ir::assembly::P26 p) {
    shared_ptr<Prim26_Parameter> para = make_shared<Prim26_Parameter>();
    para->length_in = p.length_in();
    para->length_out = p.length_out();
    para->length_ciso = p.length_ciso();
    para->num_in = p.num_in();
    para->num_out = p.num_out();
    para->num_ciso = p.num_ciso();
    para->x1_precision = precision_map[p.x1_precision()];
    para->out_precision = precision_map[p.out_precision()];
    para->bit_shift_num = p.bit_shift_num();
    as_pipeline_en = p.row_pipeline_en();
    if (as_pipeline_en)
        pipeline_num = p.row_pipeline_num();
    return para;
}
shared_ptr<Primitive>
Simulator::primConfig(shared_ptr<Core> p_core,
                      tianjic_ir::assembly::RouterPrim router,
                      shared_ptr<MemoryVisitor> visitor) {
    bool is_relay = router.p09().cxy();
    auto broadcast_type = ROUTER::BROADCAST_TYPE(is_relay);

    auto dx = router.p09().nx();
    auto dy = router.p09().ny();
    auto relay_number = is_relay ? (router.p09().relay_number() + 1) : 0;
    auto packet_type = ROUTER::PACKET_TYPE(router.p09().packet_size_mode());
    auto packet_size = packet_type == ROUTER::SINGLE_PACK ? 1 : 8;
    int dout_length = ((router.p09().addr_dout_length() + 1)) << 4;
    shared_ptr<Prim09_Parameter> para = make_shared<Prim09_Parameter>(
        broadcast_type, dx, dy, relay_number, router.p09().receive_number() + 1,
        packet_type, router.p09().send_en(), router.p09().receive_en(),
        (0x8000 + router.p09().addr_din_base()) << 2,
        (router.p09().addr_din_length() + 1) << 3, phase_id);

    shared_ptr<Primitive> prim = make_shared<Prim09>(para);

    std::string send_id = router.has_send_block()
                              ? router.send_block()
                              : ("09_send_" + std::to_string(phase_id));
    std::string recv_id = router.has_receive_block()
                              ? router.receive_block()
                              : ("09_recv_" + std::to_string(phase_id));

    int head_n = 0;
    int data_addr = 0;

    if (router.p09().send_en() && !sr_pipeline_en) {
        data_addr = out_base + (router.p09().addr_dout_base() << 2);
        DataBlock block(p_core->get_id(), send_id, nullptr, data_addr,
                        dout_length);
        p_core->init_data_block(block);
        prim->add_input_id(block.get_id());
        if (router.has_send_block()) {
            auto &blk = _static_data_blocks[router.send_block()];
            p_core->addiblock(phase_id, block.get_id(), blk, 9,
                              this->_case_name);
        }
    }

    int header_addr = out_base + (router.p09().addr_rhead_base() << 2);

    int data_length = 0;
    for (auto head : router.p09().router_heads()) {
        uint8_t *raw;
        int header_length;
        int packet_num;
        if (head.has_enable()) {
            HeadAdvanced *pheader = new HeadAdvanced();
            pheader->base.S = head.is_instant_request();
            pheader->base.T = head.packet_size_mode();
            pheader->base.P = head.is_packet_finish();
            pheader->base.Q = head.relay_type();
            pheader->base.X = head.dx();
            pheader->base.Y = head.dy();
            pheader->base.A = head.destination_addr();

            pheader->A_offset = head.destination_offset();
            pheader->Const = head.destination_const();
            pheader->pack_per_Rhead = head.pack_per_router_head();
            pheader->EN = head.enable();
            header_length = 8;
            raw = reinterpret_cast<uint8_t *>(pheader);
            packet_num = pheader->pack_per_Rhead + 1;
        } else {
            HeadBase *pheader = new HeadBase();
            pheader->A = head.destination_addr();
            pheader->X = head.dx();
            pheader->Y = head.dy();
            pheader->S = head.is_instant_request();
            pheader->P = head.is_packet_finish();
            pheader->Q = head.relay_type();
            pheader->T = head.packet_size_mode();
            header_length = 4;
            raw = reinterpret_cast<uint8_t *>(pheader);
            packet_num = 1;
        }

        DataBlock block(p_core->get_id(),
                        std::string("packet_header_") + to_string(head_n) +
                            "_" + to_string(phase_id),
                        raw, header_addr, header_length);
        p_core->init_data_block(block);
        prim->add_input_id(block.get_id());

        for (int i = 0; i < packet_num; ++i) {
            int packet_length = packet_size;
            DataBlock block(p_core->get_id(),
                            std::string("data_packet_") + to_string(head_n) +
                                "_" + to_string(i) + "_" + to_string(phase_id),
                            nullptr, data_addr + data_length, packet_length,
                            packet_size, soma_o_id);
            p_core->init_data_block(block);
            prim->add_input_id(block.get_id());

            data_length += packet_size;
            if (!sr_pipeline_en)
                data_length %= dout_length;  // 考虑源多播
        }

        header_addr += header_length;
        head_n++;
    }

    return prim;
}
shared_ptr<Primitive> Simulator::primConfig(shared_ptr<Core> p_core,
                                            tianjic_ir::assembly::SomaPrim soma,
                                            shared_ptr<MemoryVisitor> visitor) {
    shared_ptr<Primitive> prim;
    // TODO: unify 25 05

    auto init_data_block = [&](auto data_id, auto start, auto length) {
        DataBlock block(p_core->get_id(), data_id, nullptr, start, length);
        p_core->init_data_block(block);
        return block.get_id();
    };

    auto add_x_data_block = [&](auto data_id, auto start, auto length) {
        auto input_id = ID();
        int size = length;
        if (as_pipeline_en) {
            input_id = axon_o_id;
            size = length * pipeline_ny / pipeline_num;
            start -= axon_o_base_addr;
        }
        DataBlock block(p_core->get_id(), data_id, nullptr, start, length, size,
                        input_id);
        p_core->init_data_block(block);
        prim->add_input_id(block.get_id());
    };

    auto add_input_data_block = [&](auto data_id, auto start, auto length) {
        prim->add_input_id(init_data_block(data_id, start, length));
    };

    auto add_output_data_block = [&](auto data_id, auto start, auto length,
                                     bool sel = true) {
        if (sr_pipeline_en && sel) {
            if (start >= (0x9000 << 2))
                start = 0x9000 << 2;
            DataBlock block(p_core->get_id(), data_id, nullptr, start, length,
                            length);
            p_core->init_data_block(block);
            prim->add_output_id(block.get_id());
            soma_o_id = block.get_id();
        } else {
            prim->add_output_id(init_data_block(data_id, start, length));
        }
    };

    auto add_inout_data_block = [&](auto data_id, auto start, auto length) {
        auto id = init_data_block(data_id, start, length);
        prim->add_input_id(id);
        prim->add_output_id(id);
    };

    auto data_block_id = [&](auto suffix, int addr) {
        return std::to_string(soma.pic()) + "_" + std::to_string(phase_id) +
               "_" + suffix + "_" + std::to_string(addr);
    };
    std::string id_05, id_06, id_06_ciso, id_07, id_08, id_09, id_26;
    switch (soma.pic()) {
        case 0x05: {
            if (soma.px5().pic_mode() == 1)
                prim = make_shared<Prim25>(paraConfig25(soma.px5()));
            else
                prim = make_shared<Prim05>(paraConfig(soma.px5()));

            pipeline_ny = soma.px5().niy();

            if (soma.px5().has_x1_base_addr())
                add_x_data_block(
                    soma.has_x1_block()
                        ? soma.x1_block()
                        : data_block_id("x1", soma.px5().x1_base_addr()),
                    soma.px5().x1_base_addr() << 2,
                    soma.px5().x1_addr_length() << 2);
            if (soma.px5().has_o_base_addr())
                add_output_data_block(
                    data_block_id("o", soma.px5().o_base_addr()),
                    soma.px5().o_base_addr() << 2,
                    soma.px5().o_addr_length() << 2);
            break;
        }

        case 0x06: {
            prim = make_shared<Prim06>(paraConfig(soma.p06()));
            id_06 = soma.has_x1_block()
                        ? soma.x1_block()
                        : data_block_id("x1", soma.p06().x1_base_addr());
            id_06_ciso =
                soma.has_ciso_block()
                    ? soma.ciso_block()
                    : data_block_id("ciso", soma.p06().ciso_base_addr());
            if (soma.p06().in_ciso_pipe_sel() == 0) {
                pipeline_ny = soma.p06().num_in();
                if (soma.p06().has_x1_base_addr())
                    add_x_data_block(id_06, soma.p06().x1_base_addr() << 2,
                                     soma.p06().x1_addr_length() << 2);
                if (soma.p06().has_ciso_base_addr())
                    add_input_data_block(id_06_ciso,
                                         soma.p06().ciso_base_addr() << 2,
                                         soma.p06().ciso_addr_length() << 2);
            } else {
                pipeline_ny = soma.p06().num_ciso();
                if (soma.p06().has_x1_base_addr())
                    add_input_data_block(id_06, soma.p06().x1_base_addr() << 2,
                                         soma.p06().x1_addr_length() << 2);
                if (soma.p06().has_ciso_base_addr())
                    add_x_data_block(id_06_ciso,
                                     soma.p06().ciso_base_addr() << 2,
                                     soma.p06().ciso_addr_length() << 2);
            }

            if (soma.p06().has_o_base_addr())
                add_output_data_block(
                    data_block_id("o", soma.p06().o_base_addr()),
                    soma.p06().o_base_addr() << 2,
                    soma.p06().o_addr_length() << 2);
            break;
        }

        case 0x07: {
            prim = make_shared<Prim07>(paraConfig(soma.p07()));
            pipeline_ny = soma.p07().group_num();

            if (soma.p07().has_x1_base_addr())
                add_x_data_block(
                    soma.has_x1_block()
                        ? soma.x1_block()
                        : data_block_id("x1", soma.p07().x1_base_addr()),
                    soma.p07().x1_base_addr() << 2,
                    soma.p07().x1_addr_length() << 2);
            if (soma.p07().has_lut_base_addr())
                add_input_data_block(
                    soma.has_lut_block()
                        ? soma.lut_block()
                        : data_block_id("lut", soma.p07().lut_base_addr()),
                    soma.p07().lut_base_addr() << 2,
                    soma.p07().lut_addr_length() << 2);
            if (soma.p07().has_o_base_addr())
                add_output_data_block(
                    data_block_id("o", soma.p07().o_base_addr()),
                    soma.p07().o_base_addr() << 2,
                    soma.p07().o_addr_length() << 2);
            break;
        }

        case 0x08: {
            prim = make_shared<Prim08>(paraConfig(soma.p08()));
            pipeline_ny = soma.p08().group_num();

            if (soma.p08().has_uin_base_addr())
                add_x_data_block(soma.has_uin_block() ? soma.uin_block()
                                                      : "08_uin",
                                 soma.p08().uin_base_addr() << 2,
                                 soma.p08().uin_addr_length() << 2);

            if (soma.p08().has_vm_base_addr())
                add_input_data_block(soma.has_vm_block() ? soma.vm_block()
                                                         : "08_vm",
                                     soma.p08().vm_base_addr() << 2,
                                     soma.p08().vm_addr_length() << 2);

            if (soma.p08().has_s_base_addr())
                add_output_data_block("08_s", soma.p08().s_base_addr() << 2,
                                      soma.p08().s_addr_length() << 2);

            if (soma.p08().has_v_base_addr())
                add_inout_data_block(soma.has_v_block() ? soma.v_block()
                                                        : "08_v",
                                     soma.p08().v_base_addr() << 2,
                                     soma.p08().v_addr_length() << 2);

            if (soma.p08().has_vtheta_base_addr())
                add_inout_data_block(
                    soma.has_vtheta_block() ? soma.vtheta_block() : "08_vtheta",
                    soma.p08().vtheta_base_addr() << 2,
                    soma.p08().vtheta_addr_length() << 2);

            if (soma.p08().has_para_base_addr())
                add_output_data_block("08_para",
                                      soma.p08().para_base_addr() << 2,
                                      soma.p08().para_addr_length() << 2);

            break;
        }

        case 0x26: {
            prim = make_shared<Prim26>(paraConfig(soma.p26()));
            id_26 = soma.has_x1_block()
                        ? soma.x1_block()
                        : ("26_x1_" + std::to_string(phase_id) + "_" +
                           std::to_string(soma.p26().x1_addr_length()));
            pipeline_ny = soma.p26().num_in();

            if (soma.p26().has_x1_base_addr())
                add_x_data_block(id_26, soma.p26().x1_base_addr() << 2,
                                 soma.p26().x1_addr_length() << 2);
            if (soma.p26().has_o_base_addr())
                add_output_data_block(
                    "26_o" + std::to_string(soma.p26().o_base_addr()),
                    soma.p26().o_base_addr() << 2,
                    soma.p26().o_addr_length() << 2,
                    !soma.p26().out_ciso_sel());
            if (soma.p26().has_ciso_base_addr())
                add_output_data_block(
                    "26_ciso" + std::to_string(soma.p26().ciso_base_addr()),
                    soma.p26().ciso_base_addr() << 2,
                    soma.p26().ciso_addr_length() << 2,
                    soma.p26().out_ciso_sel());

            break;
        }

        default:
            break;
    }

    return prim;
}

shared_ptr<Prim02_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P02 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim02_Parameter> para = make_shared<Prim02_Parameter>();

    para->x1_precision = precision_map[p.x1_precision()];
    para->avg_pooling_en = p.avg_pooling_en();
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->constant_b = p.constant_b();

    para->bias_length = p.bias_length();

    para->niy = shape.niy();
    para->nix = shape.nix();
    para->nif = shape.nf();
    para->nkx = shape.nkx();
    para->nky = shape.nky();

    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();

    para->has_pad = p.has_pad();
    para->pad_top = p.pad_top();
    para->pad_down = p.pad_down();
    para->pad_left = p.pad_left();
    para->pad_right = p.pad_right();
    return para;
}
shared_ptr<Prim03_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P03 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim03_Parameter> para = make_shared<Prim03_Parameter>();
    para->x1_precision = precision_map[p.x1_precision()];
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->bias_length = p.bias_length();
    para->ny = shape.niy();
    para->nx = shape.nix();
    para->nif = shape.nf();

    para->n_branch = shape.n_branch();
    // para->n_branch = 0;

    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->constant_b = p.constant_b();
    para->tensor_en = p.tensor_en();
    return para;
}
shared_ptr<Prim04_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P04 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim04_Parameter> para = make_shared<Prim04_Parameter>();
    para->x1_precision = precision_map[p.x1_precision()];
    para->x2_precision = precision_map[p.x2_precision()];
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->nif = shape.nr();
    para->nof = shape.nf();
    para->constant_b = p.constant_b();
    return para;
}

shared_ptr<Prim81_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P81 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim81_Parameter> para = make_shared<Prim81_Parameter>();
    para->x1_precision = precision_map[p.x1_precision()];
    para->x2_precision = precision_map[p.x2_precision()];
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->niy = shape.niy();
    para->nix = shape.nix();
    para->nif = shape.nr();
    para->nof = shape.nf();
    para->nkx = shape.nkx();
    para->nky = shape.nky();
    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->pad_top = p.pad_top();
    para->pad_down = p.pad_down();
    para->pad_left = p.pad_left();
    para->pad_right = p.pad_right();
    para->dilate_x = p.dilate_x();
    para->dilate_y = p.dilate_y();

    return para;
}
shared_ptr<Prim83_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P83 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim83_Parameter> para = make_shared<Prim83_Parameter>();
    para->tensor_en = p.tensor_en();
    para->x1_precision = precision_map[p.x1_precision()];
    para->constant_a = p.constant_a();
    para->constant_b = p.constant_b();
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->bias_length = p.bias_length();

    para->ny = shape.niy();
    para->nx = shape.nix();
    para->nif = shape.nf();
    para->n_branch = shape.n_branch();

    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    return para;
}

shared_ptr<Prim41_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P41 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim41_Parameter> para = make_shared<Prim41_Parameter>();
    para->x1_precision = precision_map[p.x1_precision()];
    para->x2_precision = precision_map[p.x2_precision()];
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->niy = shape.niy();
    para->nix = shape.nix();
    para->nif = shape.nr();
    para->nof = shape.nf();
    para->nkx = shape.nkx();
    para->nky = shape.nky();
    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->pad_top = p.pad_top();
    para->pad_down = p.pad_down();
    para->pad_left = p.pad_left();
    para->pad_right = p.pad_right();
    para->dilate_x = p.dilate_x();
    para->dilate_y = p.dilate_y();
    return para;
}

shared_ptr<Prim43_Parameter> Simulator::paraConfig(tianjic_ir::assembly::P43 p,
                                                   tianjic_ir::Shape shape) {
    shared_ptr<Prim43_Parameter> para = make_shared<Prim43_Parameter>();
    para->x1_precision = precision_map[p.x1_precision()];
    para->x2_precision = precision_map[p.x2_precision()];
    para->bias_type = p.bias_type() == tianjic_ir::BiasType::CONSTANT ? 0 : 2;
    para->bias_length = p.bias_length();
    para->ny = shape.niy();
    para->nx = shape.nix();
    para->nif = shape.nf();
    para->x2_length = p.x2_length();
    para->n_branch = shape.n_branch();
    para->stride_x = p.stride_x();
    para->stride_y = p.stride_y();
    para->constant_b = p.constant_b();
    para->tensor_en = p.tensor_en();
    return para;
}

shared_ptr<Primitive> Simulator::primConfig(shared_ptr<Core> p_core,
                                            tianjic_ir::assembly::AxonPrim axon,
                                            shared_ptr<MemoryVisitor> visitor) {
    shared_ptr<Primitive> prim;
    if (as_pipeline_en) {
        pipeline_length = axon.o_addr_length() * 4;

        // 用于计算流水前的大小
        axon.set_o_addr_length(axon.o_addr_length() * pipeline_ny /
                               pipeline_num);
    }
    // nof 对齐
    switch (axon.pic()) {
        case 0x02: {
            prim = make_shared<Prim02>(paraConfig(axon.p02(), axon.shape()));
            break;
        }
        case 0x03: {
            prim = make_shared<Prim03>(paraConfig(axon.p03(), axon.shape()));
            break;
        }
        case 0x04: {
            prim = make_shared<Prim04>(paraConfig(axon.p04(), axon.shape()));
            break;
        }
        case 0x41: {
            prim = make_shared<Prim41>(paraConfig(axon.p41(), axon.shape()));
            break;
        }
        case 0x43: {
            prim = make_shared<Prim43>(paraConfig(axon.p43(), axon.shape()));
            break;
        }
        case 0x81: {
            prim = make_shared<Prim81>(paraConfig(axon.p81(), axon.shape()));
            break;
        }
        case 0x83: {
            prim = make_shared<Prim83>(paraConfig(axon.p83(), axon.shape()));
            break;
        }
        default:
            break;
    }

    auto init_data_block = [&](auto data_id, auto start, auto length,
                               auto size) {
        DataBlock block(p_core->get_id(), data_id, nullptr, start, length,
                        size);
        p_core->init_data_block(block);
        return block.get_id();
    };
    auto add_input_data_block = [&](auto data_id, auto start, auto length) {
        prim->add_input_id(init_data_block(data_id, start, length, length));
    };
    auto add_output_data_block = [&](auto data_id, auto start, auto length) {
        if (!as_pipeline_en) {
            prim->add_output_id(
                init_data_block(data_id, start, length, length));
        } else {
            prim->add_output_id(
                init_data_block(data_id, start, pipeline_length, length));
        }
    };

    auto data_block_id = [&](auto suffix) {
        return std::to_string(axon.pic()) + "_" + std::to_string(phase_id) +
               suffix;
    };
    // 考虑用SFINAE写得更统一
    // 考虑怎么连接datablock
    std::string x1_id =
        axon.has_x1_block() ? axon.x1_block() : data_block_id("x1");
    std::string x2_id =
        axon.has_x2_block() ? axon.x2_block() : data_block_id("x2");
    std::string bias_id =
        axon.has_bias_block() ? axon.bias_block() : data_block_id("bias");
    // std::string o_id = axon.output_block_size() > 0 ? axon.output_block(0) :
    // std::to_string(axon.pic()) + "_o_" + std::to_string(phase_id);
    std::string o_id =
        std::to_string(axon.pic()) + "_o_" + std::to_string(phase_id);

    if (axon.has_x1_base_addr())
        add_input_data_block(x1_id, axon.x1_base_addr() << 2,
                             axon.x1_addr_length() << 2);

    if (axon.has_x2_base_addr())
        add_input_data_block(x2_id, axon.x2_base_addr() << 2,
                             axon.x2_addr_length() << 2);

    if (axon.has_bias_base_addr())
        add_input_data_block(bias_id, axon.bias_base_addr() << 2,
                             axon.bias_addr_length() << 2);

    add_output_data_block(o_id, axon.o_base_addr() << 2,
                          axon.o_addr_length() << 2);
    return prim;
}

#endif  // SIMULATOR_H
