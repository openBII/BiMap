// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "spdlog/spdlog.h"
#include "src/runtime/io_streamer/client.h"

namespace iostreamer
{
    Client::Client() : maxfd(0), is_connect(false)
    {
        ///定义sockfd
        sock_cli = socket(AF_INET, SOCK_STREAM, 0);
        maxfd = sock_cli;
        ///定义sockaddr_in
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(MYPORT);                 ///服务器端口
        serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); ///服务器ip

        tv.tv_sec = 5;
        tv.tv_usec = 0;
    }

    void Client::do_close()
    {
        // close(sock_cli);
        is_connect = false;
    }

    void Client::do_connect()
    {
        if (connect(sock_cli, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
        {
            spdlog::get("logger")->info("waiting server ...");
            sleep(2);
            return;
        }
        is_connect = true;
    }



    void Client::do_listen()
    {
        //连接服务器，成功返回0，错误返回-1
        while (1)
        {
            /*把可读文件描述符的集合清空*/
            FD_ZERO(&rfds);
            /*把标准输入的文件描述符加入到集合中*/
            FD_SET(0, &rfds);
            /*把当前连接的文件描述符加入到集合中*/
            FD_SET(sock_cli, &rfds);
            /*找出文件描述符集合中最大的文件描述符*/

            //找到就绪的文件描述符
            retval = select(sock_cli + 1, &rfds, NULL, NULL, &tv);
            if (retval == -1)
            {
                spdlog::get("logger")->error("select error\n");
                break;
            }
            else if (retval == 0)
            {
                spdlog::get("logger")->debug("select timeout\n");
                continue;
            }
            else
            {
                /*服务器发来了消息*/
                if (FD_ISSET(sock_cli, &rfds))
                {
                    handle_response();
                    break;
                }
            }
        }
    }
};
