// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_09_H
#define PRIM_09_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

class ROUTER
{
 public:
    enum BROADCAST_TYPE {
        NORMAL = 0,
        MULTICAST = 1,
        RELAY = 2,
    };

    enum PACKET_TYPE {
        SINGLE_PACK,
        MULTI_PACK,
    };
};

// 所有参数为实际值，不-1
struct Prim09_Parameter : public Parameter {

    Prim09_Parameter(ROUTER::BROADCAST_TYPE _multicast_relay_or_not,
                     int32_t _dx, int32_t _dy, size_t relay_packets_num,
                     size_t _received_stop_num,
                     ROUTER::PACKET_TYPE _hearder_multipack,
                     bool _send_en = false, bool _recv_en = false,
                     uint32_t _recv_address = 0, uint32_t _din_length = 0,
                     uint32_t _recv_end_phase = 0)
        : multicast_relay_or_not(_multicast_relay_or_not), dx(_dx), dy(_dy),
          multicast_relay_num(relay_packets_num),
          received_stop_num(_received_stop_num),
          hearder_multipack(_hearder_multipack), send_en(_send_en),
          recv_en(_recv_en), recv_address(_recv_address),
          din_length(_din_length), recv_end_phase(_recv_end_phase) {}

    ROUTER::BROADCAST_TYPE multicast_relay_or_not;
    int32_t dx;
    int32_t dy;
    size_t multicast_relay_num;
    size_t received_stop_num;
    ROUTER::PACKET_TYPE hearder_multipack;
    bool send_en;
    bool recv_en;
    uint32_t recv_address;
    uint32_t din_length;
    uint32_t recv_end_phase;
};

class Prim09 : public Primitive
{
 public:
    Prim09(shared_ptr<Prim09_Parameter> para) : Primitive(ROUTER, para) {}

 private:
};
#endif  // PRIM_09_H
