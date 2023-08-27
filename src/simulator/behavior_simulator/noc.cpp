// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "noc.h"

using namespace std;

NoC::RouterState NoC::get_core_state(const ID &core_id) {
    unordered_map<ID, NoC::RouterState, Hash_ID>::iterator it;
    {
        unique_readguard<RWLock> _lock(_state_rwlock);
        it = state_pool.find(core_id);
        if (it != state_pool.end()) {
            return it->second;
        }
    }
    {
        unique_writeguard<RWLock> _lock(_state_rwlock);
        return state_pool[core_id];
    }
}

NoC::RouterState NoC::route(const ID &core_id,
                            const vector<DataBlock> &in_blocks,
                            vector<DataBlock> &out_blocks,
                            const shared_ptr<const Prim09_Parameter> &para,
                            uint32_t phase_num) {
    // 先对para进行处理，主要判断多播中继等条件是否满足，不满足直接返回，使context调度
    // 若满足条件再继续执行，返回succes

    NoC::RouterState current_state = get_core_state(core_id);

    switch (current_state.get_state()) {
        case NoC::RouterState::INIT: {
            if (para->send_en) {
                send(in_blocks, para);
            }
            set_state(core_id, NoC::RouterState::SEND);
        }

        case NoC::RouterState::SEND: {
            if (para->multicast_relay_or_not !=
                ROUTER::BROADCAST_TYPE::NORMAL) {
                // 判断是否先收到足够的中继多播包
                size_t num = multicast_relay_packet_num(core_id, phase_num);
                if (num < para->multicast_relay_num) {
                    return NoC::RouterState(NoC::RouterState::SEND);
                } else {
                    vector<Packet> multicast_relay_packets;
                    if (para->multicast_relay_or_not ==
                        ROUTER::BROADCAST_TYPE::MULTICAST) {
                        unique_readguard<RWLock> _lock(_pool_rwlock);
                        for_each(
                            packet_pool[core_id][phase_num].begin(),
                            packet_pool[core_id][phase_num].end(),
                            [=, &multicast_relay_packets](const Packet &pack) {
                                if (pack.get_head().broadcast_or_relay &&
                                    (multicast_relay_packets.size() <
                                     para->multicast_relay_num))
                                    multicast_relay_packets.push_back(pack);
                            });
                    } else if (para->multicast_relay_or_not ==
                               ROUTER::BROADCAST_TYPE::RELAY) {
                        {
                            vector<list<Packet>::iterator> indices;
                            unique_writeguard<RWLock> _lock(_pool_rwlock);

                            int32_t i = 0;
                            for (auto it =
                                     packet_pool[core_id][phase_num].begin();
                                 it != packet_pool[core_id][phase_num].end();
                                 ++it, ++i) {
                                if (it->get_head().broadcast_or_relay &&
                                    (multicast_relay_packets.size() <
                                     para->multicast_relay_num)) {
                                    multicast_relay_packets.push_back(*it);
                                    indices.push_back(it);
                                }
                            }
                            for (auto it : indices) {
                                packet_pool[core_id][phase_num].erase(it);
                            }
                        }
                    }
                    send(DataPacketUtil::repack(multicast_relay_packets, para));
                }
            }
            set_state(core_id, NoC::RouterState::RELAY);
        }

        case NoC::RouterState::RELAY: {
            if (para->recv_en) {
                if (stop_packet_num(core_id, phase_num) <
                    para->received_stop_num)
                    return NoC::RouterState(NoC::RouterState::RELAY);

                vector<Packet> received_packets;
                {

                    list<Packet>::iterator last_pack_it;
                    {
                        unique_writeguard<RWLock> _lock(_pool_rwlock);
                        int32_t count = para->received_stop_num;
                        last_pack_it = packet_pool[core_id][phase_num].begin();

                        for (auto it = packet_pool[core_id][phase_num].begin();
                             it != packet_pool[core_id][phase_num].end();
                             ++it) {
                            if (count > 0) {
                                received_packets.push_back(*it);
                            }
                            if (it->get_head().stop) {
                                --count;
                                if (count == 0) {
                                    last_pack_it = ++it;
                                    break;
                                }
                            }
                        }
                        packet_pool[core_id][phase_num].erase(
                            packet_pool[core_id][phase_num].begin(),
                            last_pack_it);
                    }
                }
                out_blocks = DataPacketUtil::unpack(received_packets, para);
            }

            set_state(core_id, NoC::RouterState::RECEIVED);
        }
        case NoC::RouterState::RECEIVED:
            break;

        default:
            throw runtime_error(
                "switch (current_state.get_state()) to default\n");
    }

    {
        unique_writeguard<RWLock> _lock(_state_rwlock);
        state_pool.erase(core_id);
    }
    return NoC::RouterState(NoC::RouterState::SUCCESS);
}

size_t NoC::multicast_relay_packet_num(const ID &core_id, uint32_t phase_num) {
    unique_readguard<RWLock> _lock(_pool_rwlock);
    if (packet_pool.find(core_id) == packet_pool.end()) {
        return 0;
    } else {
        size_t num = count_if(packet_pool[core_id][phase_num].begin(),
                              packet_pool[core_id][phase_num].end(),
                              [](const Packet &pack) {
                                  return pack.get_head().broadcast_or_relay;
                              });
        return num;
    }
}

size_t NoC::stop_packet_num(const ID &core_id, uint32_t phase_num) {
    unique_readguard<RWLock> _lock(_pool_rwlock);
    if (packet_pool.find(core_id) == packet_pool.end()) {
        return 0;
    } else {
        return count_if(
            packet_pool[core_id][phase_num].begin(),
            packet_pool[core_id][phase_num].end(),
            [](const Packet &pack) { return pack.get_head().stop; });
    }
}
