// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "data_block.h"

// data必须是堆内存，new/malloc动态申请的
DataBlock::DataBlock(ID core_id, string id, uint8_t *data, size_t start,
                     size_t length, ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)),
      _data(data, [](uint8_t *p) { delete[] p; }), _start(start),
      _length(length), _size(length), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, shared_ptr<uint8_t> data,
                     size_t start, size_t length, ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)), _data(data), _start(start),
      _length(length), _size(length), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, size_t start, size_t length,
                     ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)), _data(nullptr), _start(start),
      _length(length), _size(length), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, uint8_t *data, size_t start,
                     size_t length, size_t size, ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)),
      _data(data, [](uint8_t *p) { delete[] p; }), _start(start),
      _length(length), _size(size), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, shared_ptr<uint8_t> data,
                     size_t start, size_t length, size_t size,
                     ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)), _data(data), _start(start),
      _length(length), _size(size), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, size_t start, size_t length,
                     size_t size, ID input_source_id)
    : _id(ID::make_data_block_id(core_id, id)), _data(nullptr), _start(start),
      _length(length), _size(size), _input_source_id(input_source_id) {}

DataBlock::DataBlock(ID core_id, string id, const DataBlock &block,
                     size_t start)
    : _id(ID::make_data_block_id(core_id, id)), _start(start),
      _length(block._length), _size(block._size),
      _input_source_id(block._input_source_id), _formatter(block._formatter) {

    // 这里用length不用size可能有bug, 也可能是为了避免bug
    if (block._data != nullptr) {
        uint8_t *p_new_data = new uint8_t[_length]();
        uint8_t *raw = block._data.get();
        memcpy(p_new_data, raw, _length);
        _data.reset(p_new_data, [](uint8_t *p) { delete[] p; });
    }
}

DataBlock::DataBlock(const DataBlock &block)
    : _id(block._id), _start(block._start), _length(block._length),
      _size(block._size), _input_source_id(block._input_source_id),
      _formatter(block._formatter) {

    if (block._data != nullptr) {
        uint8_t *p_new_data = new uint8_t[_size]();
        memcpy(p_new_data, block._data.get(), _size);
        _data.reset(p_new_data, [](uint8_t *p) { delete[] p; });
    }
}
