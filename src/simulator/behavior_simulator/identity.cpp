// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "identity.h"

pair<uint32_t, uint32_t> ID::get_core_xy() const {
    assert(_type == CORE || _type == RESOURCE);
    string core_id = get_core_id().get_module_id_str();
    uint32_t core_x = stoul(core_id.substr(0, core_id.find_first_of('_')));
    uint32_t core_y = stoul(core_id.substr(core_id.find_first_of('_') + 1));
    return pair<uint32_t, uint32_t>(core_x, core_y);
}
pair<uint32_t, uint32_t> ID::get_chip_xy() const {
    assert(_type == CORE || _type == RESOURCE || _type == CHIP);
    string chip_id = get_chip_id().get_module_id_str();
    uint32_t chip_x = stoul(chip_id.substr(0, chip_id.find_first_of('_')));
    uint32_t chip_y = stoul(chip_id.substr(chip_id.find_first_of('_') + 1));
    return pair<uint32_t, uint32_t>(chip_x, chip_y);
}

ID IDUtil::find_offset_core_id(ID core_id, int32_t dx, int32_t dy) {
    assert(core_id.is_core());

    pair<uint32_t, uint32_t> local = core_id.get_core_xy();
    int32_t raw_core_x = int32_t(local.first);
    int32_t raw_core_y = int32_t(local.second);

    local = core_id.get_chip_xy();
    int32_t raw_chip_x = int32_t(local.first);
    int32_t raw_chip_y = int32_t(local.second);

    string chip_array_id = core_id.get_chip_array_id().get_module_id_str();

    int32_t chip_dx = dx / COREINCHIPROW;
    int32_t chip_dy = dy / COREINCHIPCOLUMN;

    int32_t new_core_x = raw_core_x + dx % COREINCHIPROW;
    int32_t new_core_y = raw_core_y + dy % COREINCHIPCOLUMN;

    if (new_core_x >= COREINCHIPROW) {
        new_core_x -= COREINCHIPROW;
        ++chip_dx;
    } else if (new_core_x < 0) {
        new_core_x += COREINCHIPROW;
        --chip_dx;
    }

    if (new_core_y >= COREINCHIPCOLUMN) {
        new_core_y -= COREINCHIPCOLUMN;
        ++chip_dy;
    } else if (new_core_y < 0) {
        new_core_y += COREINCHIPCOLUMN;
        --chip_dy;
    }

    int32_t new_chip_x = raw_chip_x + chip_dx;
    int32_t new_chip_y = raw_chip_y + chip_dy;

    // fpga < 0
    // assert(new_chip_x >= 0);
    // assert(new_chip_y >= 0);
    if (new_chip_x < 0 || new_chip_y < 0) {
        return ID::make_fpga_id();
    }

    return ID::make_core_id(
        ID::make_chip_id(ID::make_chip_array_id(chip_array_id),
                         uint32_t(new_chip_x), uint32_t(new_chip_y)),
        uint32_t(new_core_x), uint32_t(new_core_y));
}
