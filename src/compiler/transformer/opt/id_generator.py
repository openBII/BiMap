# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import Union

from src.compiler.transformer.opt.socket_type import SocketType
from src.compiler.transformer.task_model.task_block_type import TaskBlockType
from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class IDGenerator():
    '''ID生成器

    可以生成task ID, edge ID, 输入socket ID, 输出socket ID
    '''
    task_num = -1
    edge_num = -1
    input_socket_num = -1
    output_socket_num = -1

    @staticmethod
    def get_next_task_id() -> int:
        '''获取一个新的task ID
        '''
        IDGenerator.task_num += 1
        return IDGenerator.task_num

    @staticmethod
    def set_base_task_id(base: Union[int, TaskGraph]):
        '''设置初始的task ID

        可设置为固定数字或根据输入任务图自动设置成任务图中最大的task ID
        '''
        IDGenerator.task_num = -1
        if isinstance(base, TaskGraph):
            for block_id in base.blocks:
                if block_id > IDGenerator.task_num:
                    IDGenerator.task_num = block_id
        else:
            IDGenerator.task_num = base

    @staticmethod
    def get_next_edge_id() -> int:
        '''获取一个新的edge ID
        '''
        IDGenerator.edge_num += 1
        return IDGenerator.edge_num

    @staticmethod
    def set_base_edge_id(base: Union[int, TaskGraph]):
        '''设置初始的edge ID

        可设置为固定数字或根据输入任务图自动设置成任务图中最大的edge ID
        '''
        if isinstance(base, TaskGraph):
            for edge_id in base.edges:
                if edge_id > IDGenerator.edge_num:
                    IDGenerator.edge_num = edge_id
        else:
            IDGenerator.edge_num = base

    @staticmethod
    def get_next_socket_id(type: SocketType) -> int:
        '''获取一个新的socket ID

        Args:
            type: socket的类型, 输入或输出
        '''
        if type == SocketType.INPUT:
            IDGenerator.input_socket_num += 1
            return IDGenerator.input_socket_num
        elif type == SocketType.OUTPUT:
            IDGenerator.output_socket_num += 1
            return IDGenerator.output_socket_num
        else:
            raise ValueError('Wrong socket type')

    @staticmethod
    def set_base_socket_id(base: Union[int, TaskGraph], type: SocketType = None):
        '''设置初始的edge ID

        可设置为固定数字或根据输入任务图自动设置成任务图中最大的socket ID
        '''
        if isinstance(base, TaskGraph):
            for block_id in base.blocks:
                block = base.blocks[block_id]
                if hasattr(block, '_socket_id'):
                    socket_id = block.socket_id
                    if socket_id is not None:
                        if block.type == TaskBlockType.INPUT:
                            if socket_id > IDGenerator.input_socket_num:
                                IDGenerator.input_socket_num = socket_id
                        elif block.type == TaskBlockType.OUTPUT:
                            if socket_id > IDGenerator.output_socket_num:
                                IDGenerator.output_socket_num = socket_id
                        else:
                            raise ValueError('Wrong socket type')
        else:
            if type == SocketType.INPUT:
                IDGenerator.input_task_num = base
            elif type == SocketType.OUTPUT:
                IDGenerator.output_task_num = base
            else:
                raise ValueError('Wrong socket type')
