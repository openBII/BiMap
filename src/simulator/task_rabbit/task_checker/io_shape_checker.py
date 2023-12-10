# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from task_rabbit.task_model.task_visitor import TaskVisitor


class IOShapeChecker(TaskVisitor):
    """
    IOShapeChecker检查每个任务节点的边累加，是否满足边簇的shape需求
    """

    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_tasks = []

    def visit(self, task):
        for cluster in task.input_clusters:
            # 没有输入边簇的存储任务块也会在构造函数里创建一个空的边簇
            if len(cluster.all_enable_edges) == 0:
                continue  # 这些存储任务块的边簇不进行检查
            if cluster.check_filled_properly() != 0:
                self.passed = False
                self.invalid_tasks.append(task.id)
