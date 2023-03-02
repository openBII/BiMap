# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class MappingChecker:
    @staticmethod
    def check(task_graph: TaskGraph):
        '''检查ONNX计算图和任务图之间的结点映射是否正确
        '''
        for c_block_id, node_name in task_graph.ctx.mapping_dict.items():
            c_block = task_graph.get_block(c_block_id)
            assert node_name == c_block.get_original_node().get_name()
