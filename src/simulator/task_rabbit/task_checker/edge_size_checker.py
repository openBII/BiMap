# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from task_rabbit.task_model.task_block import TaskBlock
from task_rabbit.task_model.task_visitor import TaskVisitor


class EdgeSizeChecker(TaskVisitor):
    """
    EdgeSizeChecker统计连接前后任务块的部分的size是否一致
    """

    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_edges = []

    def visit(self, task: TaskBlock):
        for cluster in task.output_clusters:
            for e in cluster.all_enable_edges:
                out_task = e.out_task
                input_size = out_task.get_input_edge_size(e)
                if input_size != cluster[e].size:
                    self.passed = False
                    if len(e.rearrange_info) != 0:
                        if input_size.volume == cluster[e].size.volume:
                            self.passed = True
                    assert self.passed, 'Errors in the output clusters of task block {:d}'.format(
                        task.id)
                    entry = (task.id, e)
                    self.invalid_edges.append(entry)
        for cluster in task.input_clusters:
            for e in cluster.all_enable_edges:
                in_task = e.in_task
                input_size = in_task.get_output_edge_size(e)
                if input_size != cluster[e].size:
                    self.passed = False
                    if len(e.rearrange_info) != 0:
                        if input_size.volume == cluster[e].size.volume:
                            self.passed = True
                    assert self.passed, 'Errors in the input clusters of task block {:d}'.format(
                        task.id)
                    entry = (task.id, e)
                    self.invalid_edges.append(entry)
