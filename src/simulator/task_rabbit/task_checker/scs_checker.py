# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_visitor import TaskVisitor


class SCSChecker(TaskVisitor):
    """
    SCSChecker检查是否计算任务块输出连接的都是存储任务块
    存储任务块输出连接的都是计算任务块
    """

    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_tasks = []

    def visit_S(self, task: STaskBlock):
        for out_task in task.out_tasks:
            if not (TaskBlockType.is_compute_task(out_task.task_type) or
                    out_task.task_type is TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_C(self, task: CTaskBlock):
        for out_task in task.out_tasks:
            if not TaskBlockType.is_storage_task(out_task.task_type):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SIC(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CC,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SIC2D(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CC2D,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SIFC(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CVM,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SI(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CADD,
                                          TaskBlockType.CVVH,
                                          TaskBlockType.CAX,
                                          TaskBlockType.CVS,
                                          TaskBlockType.CCMPB,
                                          TaskBlockType.CCMPS,
                                          TaskBlockType.CLUT,
                                          TaskBlockType.CLIF,
                                          TaskBlockType.CAVG,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SW(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CVM,
                                          TaskBlockType.CC,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_SW2D(self, task):
        for out_task in task.out_tasks:
            if out_task.task_type not in (TaskBlockType.CC2D,
                                          TaskBlockType.OUTPUT):
                self.passed = False
                self.invalid_tasks.append(task.id)

    def visit_INPUT(self, task):
        for out_task in task.out_tasks:
            if not TaskBlockType.is_storage_task(out_task.task_type):
                self.passed = False
                self.invalid_tasks.append(task.id)
