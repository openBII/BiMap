# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from numpy import ndarray

from src.simulator.task_rabbit.task_model.task_visitor import TaskVisitor


class DataChecker(TaskVisitor):
    """
    检查SW/SB数据块的data：
    1）是否存在 
    2）是否符合shape要求（体现在新存储任务生成）
    3）是否达到量化要求
    """

    def __init__(self):
        super().__init__()
        self.passed = True
        self.invalid_tasks = []

    def visit_SW(self, task):
        self._check_data_type_and_size(task)
        self._check_quantized(task)

    def visit_SB(self, task):
        self._check_data_type_and_size(task)
        self._check_quantized(task)

    def visit_SWFC(self, task):
        self._check_data_type_and_size(task)
        self._check_quantized(task)

    def visit_SW2D(self, task):
        self._check_data_type_and_size(task)
        self._check_quantized(task)

    def _check_data_type_and_size(self, task):
        data = task.data
        shape = task.shape
        if data is None or type(data) is not ndarray or \
                shape.volume != data.size:
            self.passed = False
            self.invalid_tasks.append(task.id)

    def _check_quantized(self, task):
        # 没实现
        pass
