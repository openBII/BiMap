// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef CONTEXT_H
#define CONTEXT_H

#include "src/simulator/behavior_simulator/identity.h"
#include "src/simulator/behavior_simulator/noc.h"
#include "src/simulator/behavior_simulator/primitive/primitive.h"
#include "src/simulator/behavior_simulator/rwlock.h"
#include "src/simulator/behavior_simulator/virtual_memory.h"
#include <map>
#include <memory>

using namespace std;

// template<typename T>
// class Singleton{
// public:
//     static T& get_instance(){
//         static T instance;
//         return instance;
//     }
//     virtual ~Singleton(){}
//     Singleton(const Singleton&)=delete;
//     Singleton& operator =(const Singleton&)=delete;
// protected:
//     Singleton(){}
// };

class Space;
class Core;

class Context
{
 public:
    Context()
        : _memory(make_unique<VirtualMemory>()), _network(make_unique<NoC>()) {}

    shared_ptr<uint8_t> read(const ID &core_id, size_t address,
                             size_t length) const {
        return _memory->read(core_id, address, length);
    }
    void execute(const ID &core_id, const shared_ptr<Primitive> &pi,
                 uint32_t phase_num);

    void init_data_block(const DataBlock &block) {
        _memory->init_memory_block(block);
    }

    DataBlock read_memory_block(const ID &block_id) const {
        return _memory->read_memory_block(block_id);
    }
    DataBlock &get_memory_block(const ID &block_id) const {
        return _memory->Block_get_datablock_ref(block_id);
    }

    void write_memory_block(const DataBlock &block) {
        _memory->write_memory_block(block);
    }

    Core *getCoreById(ID id) { return (Core *)(_identity[id]); }

    void addIdentity(ID id, Space *identity) { _identity[id] = identity; }

    void set_seed(int seed) { _seed = seed; }

    int seed() { return _seed; }

    shared_ptr<uint8_t> get_mem3_data(const ID &core_id) {
        return _memory->get_mem3_data(core_id);
    }

 public:
    int n_step;
    unique_ptr<NoC> _network;

 private:
    int _seed;
    map<ID, Space *> _identity;
    unique_ptr<VirtualMemory> _memory;
};

#endif  // CONTEXT_H
