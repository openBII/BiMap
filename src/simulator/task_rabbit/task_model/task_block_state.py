from enum import Enum


class TaskState(Enum):
    """
    TaskState 枚举类任务块可能的状态
    """
    ENABLE = 0
    DISABLE = 1
