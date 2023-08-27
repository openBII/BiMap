// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "packet.h"

vector<Packet> DataPacketUtil::pack(
    const vector<DataBlock> &data,
    const shared_ptr<const Prim09_Parameter> &router_parameter) {
    vector<Packet> new_packets;
    size_t total_pack_num = 0;
    size_t current_remain_pack_num = 0;
    function<size_t(size_t)> func;

    ID destination_id;
    shared_ptr<HeadBase> pheader;
    for (auto it = data.begin(); it != data.end(); it++) {
        if (current_remain_pack_num == 0) {
            if (it->get_id().is_packet_header()) {
                pheader = reinterpret_pointer_cast<HeadBase>(it->get_data());

                assert(pheader->T == (router_parameter->hearder_multipack ==
                                              ROUTER::PACKET_TYPE::MULTI_PACK
                                          ? 1
                                          : 0));

                //  先判断包头大小，是一对一还是一对多
                if (it->length() == 4) {
                    func = bind(&DataPacketUtil::calc_address, placeholders::_1,
                                size_t(pheader->A), 0, 0);
                    current_remain_pack_num = total_pack_num = 1;
                } else if (it->length() == 8) {
                    shared_ptr<HeadAdvanced> pheader_Advanced =
                        reinterpret_pointer_cast<HeadAdvanced>(pheader);

                    if (pheader_Advanced->EN == 0) {
                        it += pheader_Advanced->pack_per_Rhead + 1;
                        continue;
                    }

                    func = bind(&DataPacketUtil::calc_address, placeholders::_1,
                                size_t(pheader_Advanced->base.A),
                                size_t(pheader_Advanced->Const),
                                size_t(pheader_Advanced->A_offset));
                    current_remain_pack_num = total_pack_num =
                        pheader_Advanced->pack_per_Rhead + 1;
                }
                destination_id = IDUtil::find_offset_core_id(
                    it->get_id().get_core_id(), pheader->X, pheader->Y);
            }
        } else {
            Packet::Head header;

            header.broadcast_or_relay = (pheader->Q == 1) ? true : false;
            header.packet_type = router_parameter->hearder_multipack;

            header.offset = func(total_pack_num - current_remain_pack_num);
            header.destination_id = destination_id;
            header.source_id = it->get_id().get_core_id();

            // 仅输出块使用
            header.block_id = pheader->A;

            --current_remain_pack_num;
            header.stop = current_remain_pack_num == 0 ? true : false;

            header.recv_end_phase = router_parameter->recv_end_phase;

            new_packets.push_back(Packet(header, *it));
        }
    }
    return new_packets;
}
vector<Packet> DataPacketUtil::repack(
    const vector<Packet> &packets,
    const shared_ptr<const Prim09_Parameter> &router_parameter) {
    vector<Packet> new_packets;
    for_each(packets.begin(), packets.end(), [&](const Packet &pack) {
        auto _head = pack.get_head();
        _head.source_id = _head.destination_id;

        _head.destination_id = IDUtil::find_offset_core_id(
            _head.destination_id, router_parameter->dx, router_parameter->dy);

        new_packets.push_back(Packet(_head, pack.get_data()));
    });
    return new_packets;
}
vector<DataBlock> DataPacketUtil::unpack(
    const vector<Packet> &packets,
    const shared_ptr<const Prim09_Parameter> &router_parameter) {
    vector<DataBlock> output;
    for_each(packets.begin(), packets.end(), [&](const Packet &pack) {
        auto data = pack.get_data();
        auto head = pack.get_head();
        if (head.packet_type == ROUTER::PACKET_TYPE::MULTI_PACK) {
            output.push_back(DataBlock(
                head.destination_id, data.get_id().get_module_id_str() + "recv",
                data,
                router_parameter->recv_address +
                    (head.offset * 8) % router_parameter->din_length));
        } else if (head.packet_type == ROUTER::PACKET_TYPE::SINGLE_PACK) {
            output.push_back(
                DataBlock(head.destination_id,
                          data.get_id().get_module_id_str() + "recv", data,
                          router_parameter->recv_address +
                              head.offset % router_parameter->din_length));
        }
    });
    return output;
}
