# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from task_rabbit.task_model.task_visitor import TaskVisitor


class ShapeChecker(TaskVisitor):
    """
    检查节点shape和输入输出簇数量、shape
    """

    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_tasks = []

    def visit(self, task):
        task.check()
