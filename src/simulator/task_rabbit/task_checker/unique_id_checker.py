# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/




from task_rabbit.task_model.task_visitor import TaskVisitor


class UniqueIDChecker(TaskVisitor):
    """
    检查图结点ID是否唯一
    """
    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_tasks = []
        self._ids = set()

    def visit(self, task):
        if task.id in self._ids:
            self.passed = False
            self.invalid_tasks.append(task.id)
        self._ids.add(task.id)
