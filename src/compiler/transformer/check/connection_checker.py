# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class ConnectionChecker:
    @staticmethod
    def check(task_graph: TaskGraph):
        '''检查边连接是否正确
        '''
        for edge_id in task_graph.edges:
            edge = task_graph.get_edge(edge_id)
            src_block_id = edge.get_src_block_id()
            dst_block_id = edge.get_dst_block_id()
            src_block = task_graph.get_block(src_block_id)
            dst_block = task_graph.get_block(dst_block_id)
            assert src_block.has_output_connection(edge_id)
            assert dst_block.has_input_connection(edge_id)
