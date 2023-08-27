# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_visitor import TaskVisitor


class CountVisitor(TaskVisitor):
    """
    CountVisitor统计每种类型的任务块出现了几次
    """

    def __init__(self):
        super().__init__()
        self.task_count = {}

    def visit(self, task: TaskBlock):
        if task.task_type not in self.task_count:
            self.task_count[task.task_type] = 0
        self.task_count[task.task_type] += 1
