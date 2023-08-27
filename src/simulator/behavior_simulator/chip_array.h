// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef CHIPARRAY_H
#define CHIPARRAY_H

#include "src/simulator/behavior_simulator/chip.h"
#include "src/simulator/behavior_simulator/space.h"

class ChipArray : public Space
{
 public:
    ChipArray(string id = "", shared_ptr<Context> ctx = nullptr)
        : Space(ID::make_chip_array_id(id), ctx) {}
    ~ChipArray() override {}

    void execute() override {
        spdlog::get("logger")->info("execute in chip array");

        if (_chip_list.size() > 1) {
            // 多线程版本
            vector<thread> pool;
            for (auto it = _chip_list.begin(); it != _chip_list.end(); ++it) {
                thread t(&Chip::execute, &(*(*it).second));
                pool.push_back(move(t));
            }
            for_each(pool.begin(), pool.end(), [](thread &t) { t.join(); });
        } else {
            // 单线程版本
            (*_chip_list.begin()).second->execute();
        }
    }

    void add_chip(shared_ptr<Chip> p_chip) {
        _chip_list.insert(pair<ID, shared_ptr<Chip>>(p_chip->get_id(), p_chip));
    }

 private:
    map<ID, shared_ptr<Chip>> _chip_list;
};

#endif  // CHIPARRAY_H
