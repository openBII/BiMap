# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import socket
import select
import src.compiler.ir.msg_pb2 as pb
import src.compiler.ir.data_pb2 as tj_data
import src.compiler.ir.basic_pb2 as tj_basic
import struct
import data_generator
import numpy as np
import signal
import logging

logging.basicConfig(filename='io_server.log', level=logging.DEBUG)

def cc_handler(signum, frame):
    print("broken pipe")
    print(signum)



signal.signal(signal.SIGPIPE, cc_handler)


class Server():
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = ('localhost', 7000)
        self.sock.bind(server_address)

        s_send_buffer_size = self.sock.getsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF)
        s_recv_buffer_size = self.sock.getsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF)
        print("socket send buffer size[old] is %d" % s_send_buffer_size)
        print("socket receive buffer size[old] is %d" % s_recv_buffer_size)
        self.sock.listen(500)

        self.socket_blocks = {}

    def handle_output(self, req):

        if req.socket_id not in self.socket_blocks:
            self.socket_blocks[req.socket_id] = []
        self.socket_blocks[req.socket_id].append((req))

        if  len(self.socket_blocks[req.socket_id]) == req.total_blocks:
            print ("socket_{} received".format( req.socket_id))
            nx,ny,nf = -1, -1, -1
            for r in self.socket_blocks[req.socket_id]:
                nx = max(nx, r.shape.nx + r.begin_position.nx)
                ny = max(ny, r.shape.ny + r.begin_position.ny)
                nf = max(nf, r.shape.nf + r.begin_position.nf)

            socket = np.zeros((ny, nx, nf), np.int32)

            for r in self.socket_blocks[req.socket_id]:
                data = r.data
                length = len(data)
                shape = (r.shape.ny, r.shape.nx, r.shape.nf)
                if r.precision == tj_basic.Precision.INT_32:
                    length >>= 2
                    data = np.array(struct.unpack (length*"i", data), np.int32)
                elif r.precision == tj_basic.Precision.INT_8: 
                    data = np.array(struct.unpack (length*"b", data), np.int8)
                else:
                    data = np.array(struct.unpack (length*"b", data), np.uint8)
                    if r.precision == tj_basic.Precision.TERNARY: 
                        data = np.array([ [x&0b11, (x>>2)&0b11, (x>>4)&0b11, (x>>6)&0b11] for x in data], np.int8).reshape(-1)
                x = r.begin_position.nx
                x_ = r.begin_position.nx + r.shape.nx                
                y = r.begin_position.ny
                y_ = r.begin_position.ny + r.shape.ny              
                f = r.begin_position.nf
                f_ = r.begin_position.nf + r.shape.nf

                if len(data) != shape[0]*shape[1]*shape[2]:
                    logging.warning("data truncate")
                data = data.reshape ((shape[0], shape[1], -1))
                data = data[:, :, 0:shape[2]]
                socket[y:y_, x:x_, f:f_] = data
            socket = socket.reshape(-1)
            with open("temp/{}/{}/o_{}_{}_{}.dat.txt".format(req.case_name, req.storage_path, 0, req.socket_id, req.nth+1), "w") as f:
                for x in socket:
                    # if r.precision == tj_basic.Precision.INT_8: 
                    #     f.write(struct.pack('>b', x) + "\n")
                    # elif r.precision == tj_basic.Precision.UINT_8: 
                    #     f.write(struct.pack('>B', x) + "\n")
                    # else:
                        f.write(str(x) + "\n")
                        # f.write(struct.pack('>i', x) + "\n")


            self.socket_blocks[req.socket_id] = []
            print("data output")

    def close(self):
        self.sock.close()

    def accept(self):
        epoll = select.epoll()
        epoll.register(self.sock.fileno(), select.EPOLLIN)
        fd_to_socket = {self.sock.fileno(): self.sock}
        while True:
            # try : 
                events = epoll.poll(5)
                if not events:
                    print("epoll 超时，重新轮询")

                for fd, event in events:
                    socket = fd_to_socket[fd]
                    if socket == self.sock:
                        conn, addr = self.sock.accept()
                        conn.setblocking(False)
                        epoll.register(conn.fileno(), select.EPOLLIN)
                        fd_to_socket[conn.fileno()] = conn
                    elif event & select.EPOLLHUP:
                        epoll.unregister(fd)
                        # 关闭客户端的文件句柄
                        fd_to_socket[fd].close()
                        # 在字典中删除与已关闭客户端相关的信息
                        del fd_to_socket[fd]
                    elif event & select.EPOLLIN:
                        data = b''
                        buf = socket.recv(1 << 16)
                        while buf != b'':
                            data += buf 
                            try:
                                buf = socket.recv(1 << 16)
                            except:
                                buf = b''

                        print("recv {} bytes".format(len(buf)))
                        if data != b'':
                            req = pb.Request()
                            req.ParseFromString(data)
                            print(
                                "---------------------{}-------------------".format(req.case_name))
                            if req.request_type == tj_data.IOType.OUTPUT_DATA:
                                self.handle_output(req)
                                data = struct.pack("i", -2);
                                socket.sendall(data)
                            else:
                                data = data_generator.fetch(req)
                                if  (len(data) != 0):
                                    data = struct.pack("i", len(data)) + data
                                else:
                                    data = struct.pack("i", -1) + data;
                                try:
                                    socket.sendall(data)
                                except Exception as e:
                                    print("fail to send, close connection")
                                    epoll.unregister(fd)
                                    fd_to_socket[fd].close()
                                    del fd_to_socket[fd]
                                    print("close ", fd)
                        else:
                            print("no remaining data, close connection")
                            epoll.unregister(fd)
                            fd_to_socket[fd].close()
                            del fd_to_socket[fd]
                            print("close ", fd)
                    else:
                        print("else", fd, event)

            # except Exception as e:
            #     logging.error(e)

if __name__ == '__main__':
    server = Server()
    server.accept()
