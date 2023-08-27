// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef NOC_H
#define NOC_H

#include "src/simulator/behavior_simulator/identity.h"
#include "src/simulator/behavior_simulator/packet.h"
#include "src/simulator/behavior_simulator/rwlock.h"
#include <algorithm>
#include <list>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;

class NoC
{
 public:
    class
        RouterState  // NoC类内部使用的状态类，外部只有Context.execute中使用到了RouterState::bool
    {
     public:
        friend NoC;
        enum STATE {
            SUCCESS,
            RECEIVED,
            RELAY,
            SEND,
            INIT,
        };

        RouterState() : _state(INIT) {}
        STATE get_state() const { return _state; }

        operator bool() const { return _state == SUCCESS; }

     private:
        RouterState(STATE _s) : _state(_s) {}
        STATE _state;
    };
    // get_core_state：从state_pool中获取core状态
    RouterState get_core_state(const ID &core_id);
    // route：完成core的数据收发、多播中继
    RouterState route(const ID &core_id, const vector<DataBlock> &in_blocks,
                      vector<DataBlock> &out_blocks,
                      const shared_ptr<const Prim09_Parameter> &para,
                      uint32_t phase_num);

    std::string extract(ID core_id, int phase_num, int block_id,
                        int packet_num) {
        vector<Packet> packets;
        {
            list<Packet>::iterator last_pack_it;
            {
                unique_writeguard<RWLock> _lock(_pool_rwlock);
                auto fpga_id = ID::make_fpga_id();

                auto pool = packet_pool[fpga_id][phase_num];
                for (auto &packet : pool) {
                    if (packet.get_head().source_id == core_id &&
                        packet.get_head().block_id == block_id) {
                        packets.push_back(packet);
                        packet_pool[fpga_id][phase_num].remove(packet);
                    }
                }
                assert(packet_num == packets.size());
                // 不删除元素，应该没问题 ？
                // 要删 用来区分不同块
            }
        }

        std::sort(packets.begin(), packets.end(),
                  [](const Packet &a, const Packet &b) {
                      return a.get_head().offset < b.get_head().offset;
                  });

        std::string s;
        for_each(packets.begin(), packets.end(), [&](const Packet &pack) {
            auto blk = pack.get_data();
            auto len = blk.length();
            auto data = blk.get_data().get();
            s += std::string(data, data + len);
        });
        return s;
    }

 private:
    // send函数重载一：在route函数中被调用，完成数据发送的具体操作
    void send(const vector<DataBlock> &in_blocks,
              const shared_ptr<const Prim09_Parameter> &para) {
        vector<Packet> send_packets = DataPacketUtil::pack(in_blocks, para);
        send(send_packets);
    }
    // send函数重载二：在send函数重载一中被调用，对读写锁代码段进行了封装，并将发送的数据包存储到packet_pool中
    void send(const vector<Packet> &send_packets) {
        unique_writeguard<RWLock> _lock(_pool_rwlock);
        for_each(send_packets.begin(), send_packets.end(),
                 [this](const Packet &pack) {
                     packet_pool[pack.get_head().destination_id]
                                [pack.get_head().recv_end_phase]
                                    .push_back(pack);
                 });
    }
    // set_state：在state_pool中设置core的状态，且对读写锁代码段进行了封装
    void set_state(const ID &core_id, RouterState::STATE state) {
        unique_writeguard<RWLock> _lock(_state_rwlock);
        state_pool[core_id] = state;
    }
    // multicast_relay_packet_num：获取目前packet_pool中core在phase_num所接收到的数据包中包头的Q位为1的有多少包
    size_t multicast_relay_packet_num(const ID &core_id, uint32_t phase_num);
    // stop_packet_num：获取目前packet_pool中core在phase_num所接收到的数据包中包头的P位为1的有多少包
    size_t stop_packet_num(const ID &core_id, uint32_t phase_num);

    // packet_pool：保存所有core所有phase收到的数据包
    unordered_map<ID, unordered_map<uint32_t, list<Packet>>, Hash_ID>
        packet_pool;
    //_pool_rwlock：变量packet_pool的读写锁
    mutable RWLock _pool_rwlock;
    // state_pool：保存所有core的当前路由状态
    unordered_map<ID, RouterState, Hash_ID> state_pool;
    //_state_rwlock：变量state_pool的读写锁
    mutable RWLock _state_rwlock;
};

#endif  // NOC_H
