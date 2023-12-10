import math
from copy import copy

from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


def ceil(num):
    return math.ceil(num / 32)


class CTaskBlock(TaskBlock):
    """
    CTaskBlock类描述一个计算任务块
    """

    def __init__(self, task_id: int, shape: Shape,
                 task_type: TaskBlockType, precision: Precision,
                 bias_type: BiasType = BiasType.NONE):
        super().__init__(task_id, shape, precision)
        self.bias_type = bias_type
        self._type = task_type

        self._halide_func = None

    def _construct_storage(self) -> None:
        self._storage = 0

    def _construct_computation(self) -> None:
        self._computation = 0

    def accept(self, visitor):
        visitor.visit_C(self)

    # def copy_like(self) -> TaskBlock:
    #     data = deepcopy(self._data)
    #     new_task_block = SITaskBlock(IDGenerator.get_next_task_id(),
    #                                  copy(self.shape),
    #                                  self.precision, data)
    #     return new_task_block
    

