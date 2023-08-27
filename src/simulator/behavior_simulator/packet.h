// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PACKET_H
#define PACKET_H

#include "src/simulator/behavior_simulator/data_block.h"
#include "src/simulator/behavior_simulator/primitive/prim_09.h"
#include <vector>

class Packet
{
 public:
    struct Head {
        ID source_id;             // 数据包的源core的ID
        ID destination_id;        // 数据包的目的core的ID
        bool broadcast_or_relay;  // 是否多播或中继，即物理包头的Q位
        ROUTER::PACKET_TYPE
            packet_type;  // 包头的类型，多包/单包，即物理包头的T位
        size_t offset;  // 接收存储位置的偏移量，即物理包头的A位
        bool stop;      // 是否停止，即物理包头的P位
        uint32_t recv_end_phase;  // 接收端在第几个phase接收
        int block_id;
        bool operator==(const Head &other) {
            return source_id == other.source_id &&
                   destination_id == other.destination_id &&
                   offset == other.offset && block_id == other.block_id;
        }
    };

    Head get_head() const { return _header; }
    DataBlock get_data() const { return _data; }
    Packet(Head header, DataBlock data) : _header(header), _data(data) {}
    bool operator==(const Packet &other) { return _header == other._header; }

 private:
    Head _header;
    DataBlock _data;
};

class DataPacketUtil
{
 public:
    // pack：将DataBlock根据router_parameter封装为Packet
    static vector<Packet>
    pack(const vector<DataBlock> &data,
         const shared_ptr<const Prim09_Parameter> &router_parameter);
    // pack：对Packet根据router_parameter重新封装，在多播或中继是调用，修改Packet中源坐标和目的坐标
    static vector<Packet>
    repack(const vector<Packet> &packets,
           const shared_ptr<const Prim09_Parameter> &router_parameter);
    // unpack：对Packet根据router_parameter解封装为DataBlock
    static vector<DataBlock>
    unpack(const vector<Packet> &packets,
           const shared_ptr<const Prim09_Parameter> &router_parameter);

 protected:
    // calc_address：对于一对多包头类型，计算每个包中的偏移量A字段
    static size_t calc_address(size_t packet_num, size_t start,
                               size_t const_num, size_t offset) {
        // packet_num:0表示1
        // const_num:0表示1
        // offset:0表示1
        return start +
               (packet_num / (const_num + 1)) * (offset + 1 + const_num) +
               packet_num % (const_num + 1);
    }
};

#endif  // PACKET_H
