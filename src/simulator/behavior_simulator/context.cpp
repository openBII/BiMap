// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "src/simulator/behavior_simulator/context.h"
#include <thread>

void Context::execute(const ID &core_id, const shared_ptr<Primitive> &pi,
                      uint32_t phase_num) {
    if (pi == nullptr)
        return;

    vector<DataBlock> input_data;
    vector<DataBlock> output_data;

    auto input_list = pi->get_input_id_list();
    for_each(input_list.begin(), input_list.end(), [&](const ID &id) {
        input_data.push_back(_memory->read_memory_block(id));
    });
    auto output_list = pi->get_output_id_list();
    for_each(output_list.begin(), output_list.end(), [&](const ID &id) {
        output_data.push_back(_memory->read_memory_block(id));
    });
    if (pi->get_type() == Primitive::ROUTER) {
        while (!_network->route(
            core_id, input_data, output_data,
            static_pointer_cast<Prim09_Parameter>(pi->get_parameters()),
            phase_num)) {
            this_thread::yield();
        }
    } else {
        pi->execute(input_data, output_data);
    }

    for_each(output_data.begin(), output_data.end(),
             [&](const DataBlock &data) { _memory->write_memory_block(data); });
}
