# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from enum import Enum
from typing import List

from src.simulator.task_rabbit.task_model.task_block_state import TaskState


class RearrangeInfoType(Enum):
    IDENTITY = 0
    RESHAPE = 1
    PERMUTE = 2
    ROTATE = 3
    SHIFT = 4
    SCALE = 5
    SHEAR = 6
    AFFINE = 7
    REFLECT = 8
    PROJECT = 9
    SHUFFLE = 10


class RearrangeInfo:
    """
    RearrangeInfo类描述边连接中进行的数据重排操作
    rearrange_type表示数据重排类型
    rearrange_matrix是一个二维的位置变换矩阵
    """

    def __init__(self, rearrange_type: RearrangeInfoType,
                 rearrange_matrix=None):
        self.rearrange_type: RearrangeInfoType = rearrange_type
        self.rearrange_matrix = []
        if rearrange_matrix is not None:
            self.rearrange_matrix = rearrange_matrix


class Edge():
    """
    Edge 类负责描述TaskBlock的连接
    """

    def __init__(self, in_task, out_task,
                 rearrange_info: List[RearrangeInfo] = None):
        # in_task与out_task为TaskBlock对象
        # 这里不存ID是希望在没有TaskGraph这个全局性的变量的情况下
        # 链式的遍历结点
        self._in_task = in_task
        self._out_task = out_task

        # 一条边可以有多组数据重排信息
        self._rearrange_info = []
        if rearrange_info is not None:
            self._rearrange_info = rearrange_info

        # 当前边的状态，如enabled状态
        self._state = TaskState.ENABLE

        # self.id = ''

    @property
    def edge_id(self) -> int:
        return id(self)

    @property
    def in_task(self):
        return self._in_task

    @in_task.setter
    def in_task(self, other):
        self._in_task = other

    @property
    def out_task(self):
        return self._out_task

    @out_task.setter
    def out_task(self, other):
        self._out_task = other

    @property
    def rearrange_info(self) -> List[RearrangeInfo]:
        return self._rearrange_info

    @rearrange_info.setter
    def rearrange_info(self, other: List[RearrangeInfo]):
        self._rearrange_info = other

    def is_enable(self) -> bool:
        return self._state == TaskState.ENABLE

    def enable(self) -> None:
        """
        将Edge置于enable状态
        """
        self._state = TaskState.ENABLE

    def disable(self) -> None:
        """
        将Edge置于disable状态
        """
        self._state = TaskState.DISABLE

    def add_rearrange_info(self, value: RearrangeInfo) -> None:
        self._rearrange_info.append(value)

    def destroy(self) -> None:
        self._in_task = None
        self._out_task = None
        self._rearrange_info = []
        self._state = TaskState.DISABLE

    def __str__(self):
        return str(self._in_task) + '-->' + str(self._out_task)

    def __hash__(self):
        return hash(self.edge_id)

    def __eq__(self, other):
        return self.edge_id == other.edge_id

    def __contains__(self, item):
        return self._in_task == item or self._out_task == item
