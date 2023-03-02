# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class TaskBlockType(Enum):
    '''任务块类型的枚举类
    '''
    SI = 0
    SW = 1
    SIC2D = 2
    SIFC = 3
    SIC = 4
    SB = 5
    SWFC = 6
    SW2D = 7
    CADD = 8
    CVVH = 9
    CVM = 10
    CC = 11
    CAX = 12
    CC2D = 13
    CVS = 14
    CCMPB = 15
    CAVG = 16
    CCMPS = 17
    CLUT = 18
    CLIF = 19
    INPUT = 50
    OUTPUT = 51
    TASK_NULL = 100

    @staticmethod
    def get_type(value: int):
        for type in TaskBlockType:
            if type.value == value:
                return type

    @staticmethod
    def get_name(value: int):
        return TaskBlockType.get_type(value).name

    @staticmethod
    def is_input(value: int):
        type = TaskBlockType.get_type(value)
        if type in (TaskBlockType.SI, TaskBlockType.SIC2D, TaskBlockType.SIC, TaskBlockType.SIFC):
            return True
        else:
            return False

    @staticmethod
    def is_storage(value: int):
        if value <= TaskBlockType.SW2D.value:
            return True
        else:
            return False

    @staticmethod
    def is_computation(value: int):
        return not(TaskBlockType.is_storage(value))
