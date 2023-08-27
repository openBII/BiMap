// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "virtual_memory.h"
#include "src/simulator/behavior_simulator/util.h"
#include <spdlog/spdlog.h>
#include <stdlib.h>

using namespace std;

void Memory::write(size_t address, size_t length, shared_ptr<uint8_t> data) {
    // address:1Bytes align
    if (!(address + length <= MEM_SIZE)) {
        throw runtime_error("write memory start:" + to_string(address) +
                            " length:" + to_string(length) +
                            " > MEM_SIZE:" + to_string(MEM_SIZE));
    }
    if (data != nullptr) {
        memcpy(_mem + address, data.get(), length * sizeof(uint8_t));
    }
}
shared_ptr<uint8_t> Memory::read(size_t address, size_t length) const {
    // address:1Bytes align
    assert(address + length <= MEM_SIZE);
    uint8_t *p_data = new uint8_t[length];
    memcpy(p_data, _mem + address, length * sizeof(uint8_t));
    return shared_ptr<uint8_t>(p_data, [](uint8_t *p) { delete[] p; });
}

DataBlock VirtualMemory::read_memory_block(const ID &block_id) const {
    DataBlock block = Block_get_datablock(block_id);
    if (block.start() == MEM_SIZE) {
        return block;
    }
    ID core_id = block.get_id().get_core_id();
    if (block.input_block_id().valid()) {
        auto p_data = new_array<uint8_t>(block.size());
        memcpy(p_data.get(),
               Block_get_datablock(block.input_block_id()).get_data().get() +
                   block.start(),
               block.size());

        return DataBlock(core_id, block.get_id().get_module_id_str(), p_data,
                         block.start(), block.length(), block.size(),
                         block.input_block_id());
    } else {
        assert(block.start() + block.length() <= MEM_SIZE);
        auto b = DataBlock(
            core_id, block.get_id().get_module_id_str(),
            Memory_get_memory(core_id).read(block.start(), block.length()),
            block.start(), block.length(), block.size(),
            block.input_block_id());
        return b;
    }
}
void VirtualMemory::write_memory_block(const DataBlock &block) {

    Block_insert_datablock(block);
    if (block.start() >= MEM_SIZE)  // MEM3的情况为9
    {
        return;
    }
    shared_ptr<uint8_t> ptr = block.get_data();
    if (ptr != nullptr) {
        // calculate pipline area
        if (block.size() > block.length()) {
            size_t total_len = block.size();
            size_t goal_len = block.length();

            uint8_t *source = block.get_data().get();
            ptr = new_array<uint8_t>(goal_len);
            uint8_t *rebuild_bank = ptr.get();

            size_t remaind_len = total_len % goal_len;
            size_t p_remaind_offser = total_len / goal_len * goal_len;
            size_t last_bank_remain_len = goal_len - remaind_len;
            size_t p_last_bank_remain_offser =
                p_remaind_offser - last_bank_remain_len;

            memcpy(rebuild_bank, source + p_remaind_offser, remaind_len);
            memcpy(rebuild_bank + remaind_len,
                   source + p_last_bank_remain_offser, last_bank_remain_len);
        }
        Memory_get_memory_ref(block.get_id().get_core_id())
            .write(block.start(), block.length(), ptr);
        // spdlog::get("logger")->warn("write");
        // spdlog::get("logger")->warn(block.get_id().get_module_id_str());
    }
}

void VirtualMemory::init_memory_block(const DataBlock &block) {
    ID core_id = block.get_id().get_core_id();
    if (!Memory_memory_is_exist(core_id)) {
        Memory_create_core_memory(core_id);
    }
    write_memory_block(block);
}

void VirtualMemory::Block_insert_datablock(const DataBlock &block) {

    map<ID, DataBlock>::iterator obj, end;
    {
        unique_readguard<RWLock> _lock(_blocks_rwlock);
        obj = _blocks.find(block.get_id());
        end = _blocks.end();
    }

    if (obj == end) {
        unique_writeguard<RWLock> _lock(_blocks_rwlock);
        _blocks.insert(pair<ID, DataBlock>(block.get_id(), block));
    } else {
        obj->second = block;
    }
}

const DataBlock &VirtualMemory::Block_get_datablock(const ID &block_id) const {
    unique_readguard<RWLock> _lock(_blocks_rwlock);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) {
        // block不存在
        throw runtime_error("read_memory_block : id not find : " +
                            string(__FILE__) + " - " + to_string(__LINE__));
    } else {
        return it->second;
    }
}

DataBlock &VirtualMemory::Block_get_datablock_ref(const ID &block_id) {
    unique_readguard<RWLock> _lock(_blocks_rwlock);
    auto it = _blocks.find(block_id);
    if (it == _blocks.end()) {
        // block不存在
        throw runtime_error("read_memory_block : id not find : " +
                            string(__FILE__) + " - " + to_string(__LINE__));
    } else {
        return it->second;
    }
}
