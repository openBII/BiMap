// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef CHIP_H
#define CHIP_H

#include "src/simulator/behavior_simulator/core.h"
#include "src/simulator/behavior_simulator/space.h"
#include <map>

class Chip : public Space
{
 public:
    Chip(ID chip_array_id, uint64_t x, uint64_t y, uint32_t step_group_id,
         shared_ptr<Context> ctx)
        : Space(ID::make_chip_id(chip_array_id, x, y), ctx), _x(x), _y(y),
          _step_group_id(step_group_id) {}
    ~Chip() override;

    void execute() override {
        spdlog::get("logger")->info("execute in chip");
        if (_core_list.size() > 1) {
            vector<thread> pool;
            for (auto it = _core_list.begin(); it != _core_list.end(); ++it) {
                thread t(&Core::execute, &(*(*it).second));
                pool.push_back(move(t));
            }
            for_each(pool.begin(), pool.end(), [](thread &t) { t.join(); });
        } else {
            (*_core_list.begin()).second->execute();
        }
    }

    void add_core(shared_ptr<Core> p_core) {
        _core_list.insert(
            pair<ID, shared_ptr<Core>>(p_core->get_id(), (p_core)));
    }

 private:
    map<ID, shared_ptr<Core>> _core_list;
    uint32_t _x;
    uint32_t _y;
    uint32_t _step_group_id;
};

Chip::~Chip() {}

#endif  // CHIP_H
