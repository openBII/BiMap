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
#include <thread>
#include <list>
#include "spdlog/spdlog.h"
#include "src/compiler/ir/msg.pb.h"

#define PORT 7000
#define IP "127.0.0.1"
#define BUFF_SIZE  1024
using namespace std;

namespace iostreamer
{
    class TcpServerSelect
    {
    private:
        int max_fd;             // select()函数监听的最大文件描述符的数量
        fd_set reads;           //保存原始的监听描述符结构
        fd_set copy_read;       //复制一个原始的监听描述符解雇
        struct timeval timeout; //设置select函数监听的超时时间
    public:
        //构造函数
        TcpServerSelect(const int fd);
        //创建监听
        void addSocketFD(const int fd);
        //移除监听
        void deleteSocketFD(const int fd);
        //等待select()函数
        int selectWait(list<int> &ver_fd);

    };

    class Server
    {
    private:
        struct sockaddr_in serv_addr; //服务器中服务端的网络地址结构
        struct sockaddr_in clnt_addr; //服务端中客户端的网络地址结构
        socklen_t clnt_addr_len;      //服务器中客户端的网络地址结构的长度
        int serv_sock; //服务器中服务端的socket描述符

    protected:
        Server(const string &port);
        TcpServerSelect tcp_select;
        int do_accept();

    public:
        void start();
        void stop();
        virtual void handle_request(int fd, char* buf) = 0;
    };

    class IOstreamerServer: public Server{
        private: 
            string get_filename(const Request& req){
                spdlog::info(req.filename());
                int core_id = req.core_id();
                int phase_id = req.phase_id();
                return req.filename();
            }
            int get_filesize(FILE * file){
                fseek(file, 0, SEEK_END);
                int fsize = ftell(file);
                fseek(file, 0, SEEK_SET);
                return fsize;
            }
        public:
            IOstreamerServer(const string &port): Server(port){}

            void handle_request(int fd, char* buf){
                Request req;
                
                spdlog::info("before parser");
                req.ParseFromArray(buf, BUFF_SIZE);

                auto file = fopen(get_filename(req).c_str(), "rb");
                int fsize = get_filesize(file);

                char sendbuf[1<<20];
                ((int*)sendbuf)[0] = fsize;
                fread(&sendbuf[4], 1, fsize, file);

                if (-1 == send(fd, sendbuf, 4+fsize, MSG_NOSIGNAL))  // MSG_NOSIGNAL 禁用异常
                {
                    tcp_select.deleteSocketFD(fd);
                    close(fd);
                    spdlog::info("close connection {}", fd);
                }
                
            }
    };
}