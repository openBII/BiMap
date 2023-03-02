# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import List

from src.compiler.transformer.opt.id_generator import IDGenerator
from src.compiler.transformer.opt.socket_type import SocketType
from src.compiler.transformer.task_model.task_graph_basics import (
    Edge, InputTaskBlock, OutputTaskBlock, STaskBlock, TaskGraph)


class IONodeInserter:
    '''在任务图中插入输入输出任务块
    '''

    def __init__(self, task_graph: TaskGraph) -> None:
        self.task_graph = task_graph
        IDGenerator.set_base_task_id(self.task_graph)
        IDGenerator.set_base_edge_id(self.task_graph)

    def insert(self):
        self.insert_input_nodes()
        self.insert_output_nodes()

    def insert_input_nodes(self):
        '''插入输入任务块

        所有不是由其他计算任务块产生的且不是静态的存储任务块会被连接到一个输入任务块上,
        如果多个这样的存储任务块来自于ONNX计算图中的同一个数据, 这些存储任务块会被连接到同一个输入任务块上
        '''
        new_blocks: List[InputTaskBlock] = list()
        new_edges: List[Edge] = list()
        for block_id in self.task_graph.blocks:
            block = self.task_graph.blocks[block_id]
            if isinstance(block, STaskBlock):
                if block.is_input() and not block.has_data:
                    inserted, input_block = IONodeInserter.has_inserted(
                        new_blocks, block.original_data.name)
                    input_edge_id = IDGenerator.get_next_edge_id()
                    if not inserted:
                        input_block_id = IDGenerator.get_next_task_id()
                        input_socket_id = IDGenerator.get_next_socket_id(
                            SocketType.INPUT)
                        input_block = InputTaskBlock(
                            id=input_block_id, socket_id=input_socket_id, s_task_block=block, edge_id=input_edge_id)
                        input_block.original_node = block.original_data
                        new_blocks.append(input_block)
                    else:
                        input_block.add_interface_to_output_cluster(
                            input_edge_id)
                        input_block.create_input_cluster_for_s_task_block(
                            block, input_edge_id)
                    input_edge = Edge(
                        id=input_edge_id, src_block_id=input_block.id, dst_block_id=block_id)
                    new_edges.append(input_edge)
        self.task_graph.add_blocks(new_blocks)
        self.task_graph.add_edges(new_edges)

    @staticmethod
    def has_inserted(blocks: List[InputTaskBlock], name: str):
        '''某个输入或输出任务块是否已经被创建
        '''
        for block in blocks:
            if block.original_node.name == name:
                return True, block
        return False, None

    def insert_output_nodes(self):
        '''插入输出任务块

        所有不会被计算任务块使用的存储任务块会被连接到一个输出任务块上
        '''
        new_blocks = list()
        new_edges = list()
        for block_id in self.task_graph.blocks:
            block = self.task_graph.blocks[block_id]
            if isinstance(block, STaskBlock):
                if block.is_output():
                    output_block_id = IDGenerator.get_next_task_id()
                    output_socket_id = IDGenerator.get_next_socket_id(
                        SocketType.OUTPUT)
                    output_edge_id = IDGenerator.get_next_edge_id()
                    output_block = OutputTaskBlock(
                        id=output_block_id, socket_id=output_socket_id, s_task_block=block, edge_id=output_edge_id)
                    output_edge = Edge(
                        id=output_edge_id, src_block_id=block_id, dst_block_id=output_block_id)
                    new_blocks.append(output_block)
                    new_edges.append(output_edge)
        self.task_graph.add_blocks(new_blocks)
        self.task_graph.add_edges(new_edges)
