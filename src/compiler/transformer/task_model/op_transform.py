# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.onnx.onnx_basics import ONNXGraph
from src.compiler.transformer.task_model.task_graph_basics import (CTaskBlock,
                                                                   Edge,
                                                                   STaskBlock,
                                                                   TaskGraph)


class OpTransformer:
    @staticmethod
    def op_transform(task_graph: TaskGraph, onnx_graph: ONNXGraph):
        '''实现ONNX算子到任务组的转换

        一个ONNX算子在任务图中一般会被转换成一个计算任务块和相应的输入存储任务块组成的任务组

        Args:
            - task_graph: 空的任务图
            - onnx_graph: ONNX计算图
        '''
        for node_name in onnx_graph.nodes:
            node = onnx_graph.get_node(node_name)
            s_task_blocks = list()
            input_task_block = None
            has_bias = False
            for data_name in node.inputs:
                block_id = task_graph.ctx.get_block_counter()
                s_task_block = STaskBlock(
                    block_id, onnx_graph.get_data(data_name), node.get_op_type())
                task_graph.blocks.update({s_task_block.get_id(): s_task_block})
                s_task_blocks.append(s_task_block.get_id())
                if len(node.inputs) == 3 and s_task_block.has_data and len(s_task_block.data.dims) == 1:
                    has_bias = True
                if not s_task_block.has_data:
                    input_task_block = s_task_block
            assert input_task_block is not None
            output_data = onnx_graph.get_data(node.get_output())
            block_id = task_graph.ctx.get_block_counter()
            c_task_block = CTaskBlock(
                block_id, node, output_data, input_task_block, has_bias)
            c_task_block.create_empty_output_cluster()
            task_graph.blocks.update({c_task_block.get_id(): c_task_block})
            task_graph.groups.update({c_task_block.get_id(): s_task_blocks})
            for s_task_block_id in s_task_blocks:  # 生成组内连接
                edge_id = task_graph.ctx.get_edge_counter()
                edge = Edge(edge_id, s_task_block_id, c_task_block.get_id())
                s_task_block = task_graph.get_block(s_task_block_id)
                s_task_block.create_output_cluster(edge_id)
                c_task_block.create_input_cluster(
                    edge_id, s_task_block.get_shape())
                task_graph.edges.update({edge_id: edge})
            task_graph.ctx.create_mapping(
                c_task_block.get_id(), node_name)  # 在context中建立mapping
            if onnx_graph.is_output_node(node_name):  # 添加输出块
                block_id = task_graph.ctx.get_block_counter()
                s_task_block = STaskBlock(block_id, output_data, 'output')
                task_graph.blocks.update({block_id: s_task_block})
                edge_id = task_graph.ctx.get_edge_counter()
                edge = Edge(edge_id, c_task_block.get_id(), block_id)
                task_graph.edges.update({edge_id: edge})
                c_task_block.fill_output_cluster(edge_id)
                c_task_block_output_cluster = c_task_block.get_output_cluster()
                s_task_block.create_input_cluster(
                    edge_id, c_task_block_output_cluster.get_shape())
                # 将输出存储任务块加入相连的组
                task_graph.groups[c_task_block.id].append(s_task_block.id)
