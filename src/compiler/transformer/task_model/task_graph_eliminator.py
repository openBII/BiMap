# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import Sequence

from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class TaskGraphEliminator:
    @staticmethod
    def eliminate(task_graph: TaskGraph, eliminated_blocks_ids: Sequence[int],
                  eliminated_groups_ids: Sequence[int], eliminated_edges_ids: Sequence[int]):
        '''任务图删除方法

        在任务图中删掉任务块, 边, 组信息和ONNX计算图到任务图的映射信息

        Args:
            - task_graph: 任务图
            - eliminated_blocks_ids: 待删除的任务块ID列表
            - eliminated_groups_ids: 待删除的组ID列表
            - eliminated_edges_ids: 待删除的边ID列表
        '''
        eliminated_blocks_ids = list(set(eliminated_blocks_ids))
        eliminated_groups_ids = list(set(eliminated_groups_ids))
        eliminated_edges_ids = list(set(eliminated_edges_ids))
        for block_id in eliminated_blocks_ids:
            del task_graph.blocks[block_id]
        for group_id in eliminated_groups_ids:
            del task_graph.groups[group_id]
            del task_graph.ctx.mapping_dict[group_id]
        for edge_id in eliminated_edges_ids:
            del task_graph.edges[edge_id]
