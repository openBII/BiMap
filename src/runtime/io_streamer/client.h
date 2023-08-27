// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <iostream>
#include "spdlog/spdlog.h"
#include "src/runtime/io_streamer/interface.h"
#include "src/compiler/ir/msg.pb.h"

using namespace std;

#define MYPORT 7000
#define BUFFER_SIZE (1 << 16)

namespace iostreamer
{

    class Client
    {
    private:
        fd_set rfds;
        struct timeval tv;
        int retval, maxfd;
        struct sockaddr_in serv_addr;

    protected:
        int sock_cli;
        Client();

        bool is_connect;

        void do_close();

        void do_connect();

        void do_listen();

        virtual void handle_response() = 0;
    };
};
