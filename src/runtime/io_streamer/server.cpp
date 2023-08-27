// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "server.h"

namespace iostreamer
{
        //构造函数
        TcpServerSelect::TcpServerSelect(const int fd) : max_fd(fd)
        {
            //先设置监听的位
            FD_ZERO(&reads);
            FD_SET(max_fd, &reads);
            timeout.tv_sec = 5;
            timeout.tv_usec = 0;
        }

        //创建监听
        void TcpServerSelect::addSocketFD(const int fd)
        {
            //构造这个监听位
            FD_SET(fd, &reads);
            if (fd > max_fd)
            {
                //如果新的描述符大于原来的描述符，那么就替换
                max_fd = fd;
            }
            spdlog::debug("新增监听位成功！{}", fd);
        }

        //移除监听
        void TcpServerSelect::deleteSocketFD(const int fd)
        {
            FD_CLR(fd, &reads);
            spdlog::debug("移除监听 {}", fd);
        }

        //等待select()函数
        int TcpServerSelect::selectWait(list<int> &ver_fd)
        {
            //调用在调用的时候，已经传入了初始的监听数量和初始化了一个监听位
            copy_read = reads;
            //设置监听的时间

            int select_res;
            if ((select_res = select(max_fd + 1, &copy_read, NULL, NULL, &timeout)) == -1)
            {
                //调用select函数发生了错误
                //退出监听，结束程序
                return -1;
            }
            if (select_res == 0)
            {
                // select函数监听到达超时时间，要返回点什么
                return 0;
            }

            //循环获取监听存在变化的位
            for (int i = 0; i < max_fd + 1; ++i)
            {
                if (FD_ISSET(i, &copy_read))
                {
                    //把存在变化的监听位放在容器中
                    ver_fd.push_back(i);
                }
            }
            return 1;
        }

    Server::Server(const string &port): 
        serv_sock(socket(AF_INET, SOCK_STREAM, 0)),
        tcp_select(TcpServerSelect(serv_sock))
    {
        //绑定服务端的socket到相应的网络地址结构中
        if (serv_sock == -1)
        {
            spdlog::error("服务器创建监听套接字失败 socket() error!");
            assert(false);
        }

        //绑定服务端的socket到相应的网络地址结构中
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;                 // tcp/ipv4
        serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);  // 任意client 地址
        serv_addr.sin_port = htons(atoi(port.c_str())); // 大端序

        int on = 1;
        setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(int));

        if (bind(serv_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
        {
            stop();
            spdlog::error("服务器绑定套接字失败 bind() error!");
            assert(false);
        }

        //开始监听服务端的套接字
        if (listen(serv_sock, 5) == -1)
        {
            stop();
            spdlog::debug("服务端监听套接字失败 listen() error!");
            assert(false);
        }
    }

    int Server::do_accept()
    {
        if (serv_sock == -1)
        {
            spdlog::error("服务器未开启监听 socket Accept() error!");
            return 0;
        }
        clnt_addr_len = sizeof(clnt_addr);
        return accept(serv_sock, (struct sockaddr *)&clnt_addr, &clnt_addr_len);
    }

    void Server::start()
    {
        spdlog::debug("serv_sock: {}", serv_sock);
        list<int> ver_fd;
        while (1)
        {
            ver_fd.clear();
            int select_res = tcp_select.selectWait(ver_fd);
            if (select_res == -1)
            {
                break;
            }
            else if (select_res == 0)
            {
                continue;
            }

            for (auto fd : ver_fd)
            {
                if (fd == serv_sock)
                {
                    int new_connection = do_accept();
                    tcp_select.addSocketFD(new_connection);
                    spdlog::debug("new connection {}", new_connection);
                }
                else
                {
                    char buf[BUFF_SIZE];
                    if (0 >= recv(fd, buf, BUFF_SIZE, 0))
                    {
                        // socket 不可用
                        tcp_select.deleteSocketFD(fd);
                        close(fd);
                        spdlog::debug("close connection {}", fd);
                    }
                    else
                    {
                        handle_request(fd, buf);
                    }
                }
            }
        }
    }

    void Server::stop()
    {
        if (serv_sock > 0)
        {
            spdlog::debug("close server socket");
            close(serv_sock);
            serv_sock = -1;
        }
    }
}