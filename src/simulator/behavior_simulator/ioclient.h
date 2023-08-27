// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef IO_CLIENT_H
#define IO_CLIENT_H

#include "spdlog/spdlog.h"
#include "src/compiler/ir/data.pb.h"
#include "src/runtime/io_streamer/client.h"
#include "src/simulator/behavior_simulator/identity.h"
#include <cstring>
#include <iostream>
#include <map>

using std::begin;
using std::cout;
using std::end;
using std::endl;
using std::map;
using std::shared_ptr;
using tianjic_ir::DynamicBlock;
using tianjic_ir::Request;
using tianjic_ir::StaticBlock;

namespace iostreamer {
class MemoryAccessClient : public Client
{
 public:
    shared_ptr<Context> _ctx;
    ID _core_id;
    map<int, list<Request>> _requests;
    map<int, list<Request>> _o_requests;

    ID memory_block_id;

    // 记一个字符串到对象的反向字典
    map<string, ID> _ids;

    void bind(ID core_id, shared_ptr<Context> ctx) {
        _ctx = ctx;
        _core_id = core_id;
    }

    void addRequest(const int &phase_id, const ID &id, const DynamicBlock &blk,
                    const int &pic, const string &case_name) {
        // TODO（zero): calc data length

        Request req;
        // req.set_id(_n_req++);
        req.set_request_type(blk.io_type());
        req.set_block_id(blk.start_addr());
        req.set_case_name(case_name);
        req.set_storage_path("behavior_out");

        req.set_socket_id(0);
        req.set_nth(-1);
        req.set_total_blocks(blk.socket_size());

        req.set_seed(_ctx->seed());
        req.set_precision(blk.precision());
        req.mutable_shape()->CopyFrom(blk.shape());
        req.set_block_size(blk.length());
        auto bp = blk.begin_position();
        req.mutable_begin_position()->CopyFrom(bp);
        // TODO(zero): fix dynamic input
        if (_o_requests.find(phase_id) == _o_requests.end()) {
            _o_requests[phase_id] = list<Request>();
        }

        _o_requests[phase_id].push_back(req);

        _ids[id.get_id_str()] = id;
    }

    void addRequest(const int &phase_id, const ID &id, const StaticBlock &blk,
                    const int &pic, const string &case_name) {
        Request req;
        req.set_id(id.get_id_str());
        req.set_request_type(tianjic_ir::STATIC_DATA);
        req.set_case_name(case_name);
        // req.set_block_id(-1);
        req.set_seed(_ctx->seed());
        req.set_precision(blk.precision());
        // auto shape = req.mutable_shape();
        // shape->CopyFrom(blk.shape());

        if (_requests.find(phase_id) == _requests.end()) {
            _requests[phase_id] = list<Request>();
        }

        _requests[phase_id].push_back(req);
        _ids[id.get_id_str()] = id;
    }

    void do_orequest(int phase_id) {
        spdlog::get("logger")->info("do request");
        // no input data
        if (_o_requests.find(phase_id) == _o_requests.end())
            return;

        // connect server
        spdlog::get("logger")->info("waiting for server ...");
        while (!is_connect) {
            do_connect();
        }

        spdlog::get("logger")->info("connected");

        const int bs = 1 << 17;
        int i = 0;
        for (auto req : _o_requests[phase_id]) {
            if (req.request_type() != tianjic_ir::OUTPUT_DATA)
                continue;

            const string data = _ctx->_network->extract(
                _core_id, phase_id, req.block_id(), req.block_size() / 2);
            assert(data.length() < bs);

            if (data == "") {
                spdlog::get("console")->error("This phase no out");
                continue;
            }
            req.set_data(std::move(data));

            char sendbuf[bs];
            spdlog::get("logger")->info("size {}", req.ByteSizeLong());
            req.SerializeToArray(sendbuf, req.ByteSizeLong());
            spdlog::get("logger")->info("send request {}", req.DebugString());
            send(sock_cli, sendbuf, req.ByteSizeLong() + 1, 1);
        }
        do_listen();
    }

    void do_irequest(int phase_id) {
        spdlog::get("logger")->info("do request");
        // no input data
        if (_requests.find(phase_id) == _requests.end())
            return;

        // connect server
        spdlog::get("logger")->info("waiting for server ...");
        while (!is_connect) {
            do_connect();
        }

        spdlog::get("logger")->info("connected");

        const int kBufSize = 1024;
        for (auto req : _requests[phase_id]) {
            if (req.request_type() == tianjic_ir::OUTPUT_DATA)
                continue;
            char sendbuf[kBufSize];
            spdlog::get("logger")->info("size {}", req.ByteSizeLong());
            req.SerializeToArray(sendbuf, req.ByteSizeLong());
            spdlog::get("logger")->info("send request {}", req.DebugString());

            send(sock_cli, sendbuf, req.ByteSizeLong() + 1, 1);

            this->memory_block_id = _ids[req.id()];
            do_listen();
        }
    }

    void handle_response() {
        spdlog::get("logger")->info("handling");
        char recvbuf[BUFFER_SIZE];
        // 第一个int表示数据量
        recv(sock_cli, recvbuf, 4, 0);
        int total_len = ((int *)recvbuf)[0];
        if (total_len == -1) {
            spdlog::get("console")->error("Behavior NO BLOCK DATA");
            total_len = 0;
            // 和server输出同步
        } else if (total_len == -2) {
            return;
        }

        DataBlock &blk = _ctx->get_memory_block(memory_block_id);
        shared_ptr<uint8_t> data(new uint8_t[total_len]);
        auto ptr = data.get();

        while (total_len) {
            spdlog::get("logger")->info("total len {}", total_len);
            int len = recv(sock_cli, recvbuf, BUFFER_SIZE, 0);
            spdlog::get("logger")->info("recved");
            memcpy(ptr, recvbuf, len);
            ptr += len;
            total_len -= len;
        }
        blk.set_data(data);
        _ctx->write_memory_block(blk);
        spdlog::get("logger")->info("all data received");
    }
};
};  // namespace iostreamer
#endif
