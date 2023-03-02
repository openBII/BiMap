# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.onnx.onnx_basics import ONNXGraph
from src.compiler.transformer.task_model.task_graph_basics import (Edge,
                                                                   TaskGraph)


class GroupConnector:
    @staticmethod
    def group_connect(task_graph: TaskGraph, onnx_graph: ONNXGraph):
        '''根据ONNX计算图上的算子连接生成任务组之间的连接

        Args:
            task_graph: 算子转换后得到的任务图
            onnx_graph: 初始的ONNX计算图
        '''
        for c_block_id in task_graph.groups:
            original_node_name = task_graph.ctx.mapping_dict[c_block_id]
            c_block = task_graph.get_block(c_block_id)
            for output_node_name in onnx_graph.get_output_nodes_names(original_node_name):
                next_c_block_id = task_graph.ctx.mapping_dict.inverse[output_node_name]
                # 下一个计算任务块的所有输入
                input_s_blocks_ids = task_graph.groups[next_c_block_id]
                for input_s_block_id in input_s_blocks_ids:
                    input_s_block = task_graph.get_block(input_s_block_id)
                    if c_block.get_original_node().get_output() == input_s_block.original_data.get_name():
                        edge_id = task_graph.ctx.get_edge_counter()
                        edge = Edge(edge_id, c_block_id, input_s_block_id)
                        task_graph.edges.update({edge_id: edge})
                        c_block.add_interface_to_output_cluster(edge_id)
                        c_block_output_cluster_shape = c_block.get_output_cluster().get_shape()
                        input_s_block.create_input_cluster(
                            edge_id, c_block_output_cluster_shape)
