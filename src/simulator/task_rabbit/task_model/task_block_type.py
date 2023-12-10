from enum import Enum


class TaskBlockType(Enum):
    """
    TaskBlockType 枚举类记录TaskBlock的类型
    """

    SI = 0
    SIC2D = 1
    SIFC = 2
    SIC = 3
    SW = 4
    SB = 5
    SWFC = 6
    SW2D = 7

    SEPERATOR = 20

    CADD = 21
    CVVH = 22
    CVM = 23
    CC = 24
    CAX = 25
    CC2D = 26
    CVS = 27
    CCMPB = 28
    CCMPS = 29
    CLUT = 30
    CLIF = 31
    CAVG = 32

    INPUT = 50
    OUTPUT = 51

    @staticmethod
    def is_compute_task(task_type):
        return (task_type.value > TaskBlockType.SEPERATOR.value and
                task_type.value < TaskBlockType.INPUT.value)

    @staticmethod
    def is_storage_task(task_type):
        return task_type.value < TaskBlockType.SEPERATOR.value

    @staticmethod
    def is_soma_task(task_type):
        return task_type.value in (TaskBlockType.CCMPB.value,
                                   TaskBlockType.CCMPS.value,
                                   TaskBlockType.CLUT.value,
                                   TaskBlockType.CLIF.value)

    @staticmethod
    def is_axon_task(task_type):
        return TaskBlockType.is_compute_task(task_type) and \
            (not TaskBlockType.is_soma_task(task_type))

    @staticmethod
    def is_io_task(task_type):
        return task_type.value in (TaskBlockType.INPUT.value,
                                   TaskBlockType.OUTPUT.value)

    @staticmethod
    def is_static_task(task_type):
        return task_type.value in (TaskBlockType.SW.value,
                                   TaskBlockType.SB.value,
                                   TaskBlockType.SWFC.value,
                                   TaskBlockType.SW2D.value)

    def __str__(self):
        return self.name
