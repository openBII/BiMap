// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include "src/simulator/behavior_simulator/data_block.h"
#include "src/simulator/behavior_simulator/rwlock.h"
#include <map>

using namespace std;

// CORE的物理内存大小的宏定义
#define MEM_SIZE 0x9000 * sizeof(uint32_t)

class Memory
{
 public:
    Memory() { _mem = new uint8_t[MEM_SIZE](); }
    Memory(const Memory &mem) {
        _mem = new uint8_t[MEM_SIZE]();
        memcpy(_mem, mem._mem, MEM_SIZE);
    }
    Memory(Memory &&mem) : _mem(mem._mem) { mem._mem = nullptr; }
    Memory &operator=(const Memory &mem) {
        memcpy(_mem, mem._mem, MEM_SIZE);
        return *this;
    }
    Memory &operator=(Memory &&mem) {
        delete[] _mem;
        _mem = mem._mem;
        mem._mem = nullptr;
        return *this;
    }
    ~Memory() { delete[] _mem; }
    void write(size_t address, size_t length, shared_ptr<uint8_t> data);
    shared_ptr<uint8_t> read(size_t address, size_t length) const;

 protected:
    uint8_t *_mem;
};

class VirtualMemory
{
 public:
    // read：VirtualMemory的read接口仅被memory_visitor调用，提供打印输出结果时的数据读取
    shared_ptr<uint8_t> read(const ID &core_id, size_t address,
                             size_t length) const {
        assert(address + length <= MEM_SIZE);
        return Memory_get_memory(core_id).read(address, length);
    }
    // read_memory_block：根据数据块的id，读取一个数据块
    DataBlock read_memory_block(const ID &block_id) const;
    // write_memory_block：写入一个数据块
    void write_memory_block(const DataBlock &block);
    // init_memory_block：写入一个数据块，仅在初始化阶段被调用
    void init_memory_block(const DataBlock &block);

    // Block_get_datablock_ref：根据给定id读取一个数据块，返回数据块的引用
    DataBlock &Block_get_datablock_ref(const ID &block_id);

    shared_ptr<uint8_t> get_mem3_data(const ID &core_id) {
        unique_readguard<RWLock> _lock(_memory_rwlock);
        for (auto &blk : _blocks) {
            if (blk.second.start() / 4 == 0x9000 &&
                blk.first.get_core_id() == core_id)
                return blk.second.get_data();
        }
    }

 private:
    // Memory_memory_is_exist：判断_memorys中是否已经创建了指定core
    // id的Memory对象，对读写锁代码段进行了封装
    bool Memory_memory_is_exist(const ID &core_id) {
        unique_readguard<RWLock> _lock(_memory_rwlock);
        return _memories.find(core_id) != _memories.end();
    }
    // Memory_create_core_memory：在_memorys中创建core的Memory对象，对读写锁代码段进行了封装
    void Memory_create_core_memory(const ID &core_id) {
        unique_writeguard<RWLock> _lock(_memory_rwlock);
        _memories.insert(pair<ID, Memory>(core_id, Memory()));
    }
    // Memory_get_memory：根据给定code id获取core的实际Memory对象
    const Memory &Memory_get_memory(const ID &core_id) const {
        unique_readguard<RWLock> _lock(_memory_rwlock);
        return _memories.find(core_id)->second;
    }
    // Memory_get_memory_ref：根据给定code
    // id获取core的实际Memory对象，返回Memory的引用
    Memory &Memory_get_memory_ref(const ID &core_id) {
        unique_readguard<RWLock> _lock(_memory_rwlock);
        return _memories[core_id];
    }
    // Block_insert_datablock：向_blocks中存入一个数据块
    void Block_insert_datablock(const DataBlock &block);
    // Block_get_datablock：根据给定id读取一个数据块
    const DataBlock &Block_get_datablock(const ID &block_id) const;

    //_memory_rwlock：变量_memories的读写锁
    mutable RWLock _memory_rwlock;
    //_blocks_rwlock：变量_blocks的读写锁
    mutable RWLock _blocks_rwlock;
    //_memories：保存所有core的memory对象，core的id作为字典的key
    map<ID, Memory> _memories;
    //_blocks：保存所有的数据块DataBlock对象，DataBlock对象的id作为字典的key
    map<ID, DataBlock> _blocks;
};

#endif  // VIRTUAL_MEMORY_H
