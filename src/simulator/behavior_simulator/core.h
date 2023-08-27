// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef CORE_H
#define CORE_H

#include "spdlog/spdlog.h"
#include "src/compiler/ir/msg.pb.h"
#include "src/simulator/behavior_simulator/context.h"
#include "src/simulator/behavior_simulator/ioclient.h"
#include "src/simulator/behavior_simulator/memory_visitor.h"
#include "src/simulator/behavior_simulator/primitive/primitive.h"
#include "src/simulator/behavior_simulator/space.h"
#include "top/global_config.h"
#include <algorithm>
#include <functional>
#include <list>

class PIGroup
{
 public:
    PIGroup(const shared_ptr<Primitive> &axon = nullptr,
            const shared_ptr<Primitive> &soma1 = nullptr,
            const shared_ptr<Primitive> &router = nullptr,
            const shared_ptr<Primitive> &soma2 = nullptr) {
        pi_list[0] = axon;
        pi_list[1] = soma1;
        pi_list[2] = router;
        pi_list[3] = soma2;
    }

    void add_axon(const shared_ptr<Primitive> &pi) {
        if (pi == nullptr)
            return;
        assert(pi->get_type() == Primitive::AXON);
        pi_list[0] = pi;
    }
    void add_soma1(const shared_ptr<Primitive> &pi) {
        if (pi == nullptr)
            return;
        assert(pi->get_type() == Primitive::SOMA);
        pi_list[1] = pi;
    }
    void add_router(const shared_ptr<Primitive> &pi) {
        if (pi == nullptr)
            return;
        assert(pi->get_type() == Primitive::ROUTER);
        pi_list[2] = pi;
    }
    void add_soma2(const shared_ptr<Primitive> &pi) {
        if (pi == nullptr)
            return;
        assert(pi->get_type() == Primitive::SOMA);
        pi_list[3] = pi;
    }

    shared_ptr<Primitive> axon() const { return pi_list[0]; }
    shared_ptr<Primitive> soma1() const { return pi_list[1]; }
    shared_ptr<Primitive> router() const { return pi_list[2]; }
    shared_ptr<Primitive> soma2() const { return pi_list[3]; }

 private:
    array<shared_ptr<Primitive>, 4> pi_list;
};

class Core : public Space
{
 public:
    Core(ID chip_id, uint32_t x, uint32_t y, uint32_t group_id,
         shared_ptr<Context> ctx)
        : Space(ID::make_core_id(chip_id, x, y), ctx), _x(x), _y(y),
          _phase_group_id(group_id) {
        _client.bind(get_id(), ctx);
    }

    ~Core() override;

    void set_memory_visitor(const map<size_t, MemoryVisitor> &visitor) {
        _visitor = visitor;
    }

    void execute() override {
        spdlog::get("logger")->info("execute in core");
        auto exec = bind(&Context::execute, context, get_id(), placeholders::_1,
                         placeholders::_2);

        for (int i = 0; i < context->n_step; ++i) {
            int32_t phase_num = 0;

            if (i == 0)
                _client.do_irequest(phase_num);

            for_each(
                _pi_group_list.begin(), _pi_group_list.end(),
                [&](const PIGroup &pi_group) {
                    exec(pi_group.axon(), phase_num);
                    exec(pi_group.soma1(), phase_num);
                    exec(pi_group.router(), phase_num);
                    exec(pi_group.soma2(), phase_num);

                    if (GlobalConfig::TEST_MODE == tianjic_ir::CASE_OUTPUT) {
                        _client.do_orequest(phase_num);
                    } else {
                        _visitor[phase_num].serialize_fstream(
                            get_id(),
                            get_id().get_chip_id().get_chip_xy().first,
                            get_id().get_chip_id().get_chip_xy().second,
                            _phase_group_id, _x, _y, i, phase_num, context);
                    }

                    ++phase_num;
                });
        }
    }

    void add_pi_group(const PIGroup &pi) { _pi_group_list.push_back(pi); }

    void init_data_block(const DataBlock &block) {
        context->init_data_block(block);
    }

    void addiblock(const int &phase_id, const ID &id,
                   const tianjic_ir::DynamicBlock &blk, const int &pic,
                   const string &case_name) {
        _client.addRequest(phase_id, id, blk, pic, case_name);
    }

    void addiblock(const int &phase_id, const ID &id,
                   const tianjic_ir::StaticBlock &blk, const int &pic,
                   const string &case_name) {
        _client.addRequest(phase_id, id, blk, pic, case_name);
    }

    void addoblock(const int &phase_id, const ID &id,
                   const tianjic_ir::DynamicBlock &blk, const int &pic,
                   const string &case_name) {
        _visitor[phase_id - 1].add_output_segment(
            {blk.start_addr() << 2, blk.length() << 2}, blk.id());
        _client.addRequest(phase_id - 1, id, blk, pic, case_name);
    }

 private:
    list<PIGroup> _pi_group_list;
    uint32_t _x;
    uint32_t _y;
    uint32_t _phase_group_id;

    // TODO: 这个写法感觉有点问题，是拷贝不是引用
    map<size_t, MemoryVisitor> _visitor;

    iostreamer::MemoryAccessClient _client;
};

Core::~Core() {}

#endif  // CORE_H
