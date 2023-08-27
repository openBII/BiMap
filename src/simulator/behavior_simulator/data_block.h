// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef DATA_BLOCK_H
#define DATA_BLOCK_H

#include "src/simulator/behavior_simulator/formatter.h"
#include "src/simulator/behavior_simulator/identity.h"
#include <memory>
#include <string.h>

using namespace std;

// HeadBase：一对一包头的
struct HeadBase {
    uint32_t S : 1;
    uint32_t T : 1;
    uint32_t P : 1;
    uint32_t Q : 1;
    int32_t X : 8;
    int32_t Y : 8;
    uint32_t A : 12;
};

// HeadBase：一对多包头
struct HeadAdvanced {
    HeadBase base;
    uint32_t pack_per_Rhead : 12;  // 0表示1
    uint32_t A_offset : 12;        // 0表示1
    uint32_t Const : 7;            // 0表示1
    uint32_t EN : 1;
};

class DataBlock
{
 public:
    // data必须是堆内存，new/malloc动态申请的
    DataBlock(ID core_id, string id, uint8_t *data, size_t start, size_t length,
              ID input_source_id = ID());
    DataBlock(ID core_id, string id, shared_ptr<uint8_t> data, size_t start,
              size_t length, ID input_source_id = ID());
    DataBlock(ID core_id, string id, size_t start, size_t length,
              ID input_source_id = ID());
    DataBlock(ID core_id, string id, uint8_t *data, size_t start, size_t length,
              size_t size, ID input_source_id = ID());
    DataBlock(ID core_id, string id, shared_ptr<uint8_t> data, size_t start,
              size_t length, size_t size, ID input_source_id = ID());
    DataBlock(ID core_id, string id, size_t start, size_t length, size_t size,
              ID input_source_id = ID());
    DataBlock(ID core_id, string id, const DataBlock &block, size_t start);
    DataBlock(const DataBlock &block);
    DataBlock &operator=(const DataBlock &) = default;
    ~DataBlock() {}
    shared_ptr<uint8_t> get_data() const { return _data; }
    void set_data(shared_ptr<uint8_t> p) {
        _data = p;
        auto x = _data.get();
    }
    size_t size() const { return _size; }
    size_t start() const { return _start; }
    size_t length() const { return _length; }
    void length(int len) { _length = len; }
    const ID get_id() const { return _id; }
    ID input_block_id() const { return _input_source_id; }

 private:
    ID _id;
    shared_ptr<uint8_t> _data;
    size_t _start;
    size_t _length;  // memory len bytes
    size_t _size;    // real bytes
    ID _input_source_id;
    Formatter _formatter;
};

#endif  // DATA_BLOCK_H
