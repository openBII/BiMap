# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import math
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision


class Storage():
    """
    Storage 类负责评估TaskBlock的存储
    """

    def __init__(self, precision: Precision, shape: Shape,
                 storage_alignment_num: int,
                 first_store_dimension: list,
                 pipeline_num: int):
        self._local_storage = None
        self._precision = precision
        self._shape = shape
        self._storage_alignment_num = storage_alignment_num
        self._first_store_dimension = first_store_dimension
        self._pipeline_num = pipeline_num

    @property
    def precision_align_num(self):
        if self._precision == Precision.TERNARY:
            return 4
        elif self._precision == Precision.INT_32:
            return 0.25
        elif (self._precision == Precision.INT_8) or \
                (self._precision == Precision.UINT_8):
            return 1
        else:
            return 1
            raise ValueError("Doesn\'t suppport this type")

    @property
    def storage_alignment_num(self):
        # if self._type == TaskBlockType.SW:
        #     return 32
        # else:
        #     return 16
        return self._storage_alignment_num

    @property
    def first_store_dimension(self):
        # if self._type == TaskBlockType.SI:
        #     return [3, 2, 1, None, None, None]
        # elif self._type == TaskBlockType.SIC:
        #     return [2, 1, None, 3, None, None]
        # elif self._type == TaskBlockType.SIC2D:
        #     return [2, 3, None, 1, None, None]
        # elif self._type == TaskBlockType.SIFC:
        #     return [None, None, None, 1, None, None]
        # elif self._type == TaskBlockType.SB:
        #     return [None, None, 1, None,  None, None]
        # elif self._type == TaskBlockType.SO:
        #     return [3, 2, 1, None, None, None]
        # elif self._type == TaskBlockType.SW:
        #     return [None, None, 1, 2, 4, 3]
        # # elif self._type == TaskBlockType.SW2D:
        # #     return [None, None, 1, 3, 4, 2]
        # else:
        #     return [None, None, None, None, None, None]
        return self._first_store_dimension

    @property
    def local_storage(self) -> int:
        if self._local_storage is not None:
            return self._local_storage

        # 首先考虑补零问题(除了SW和SW2C都是16B对齐)，涉及到精度和优先存储维度问题
        # 比如某存储块，按16B对齐，优先存储维度是nf，精度是三值
        # num_in指的是每16B(或32B)中存储多少个当前精度的数，此时num_in = 16 * 4 = 128
        num_in = self._storage_alignment_num * self.precision_align_num
        # 再将nf维度对128进行ceil取整补零
        first_store_dimension_real_store = math.ceil(
            self._shape[self._first_store_dimension] / num_in) * num_in

        real_store = (self._shape.volume // self._shape[self._first_store_dimension] *
                      first_store_dimension_real_store) // self.precision_align_num


        if self._shape.ny > 0:
            if self._pipeline_num == 0:
                self._pipeline_num = self._shape.ny
            real_store = real_store // self._shape.ny * self._pipeline_num

        return real_store
