from copy import copy


from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_model.tick import Tick


class InputTaskBlock(TaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision, socket_id: int = 0):
        super().__init__(task_id, shape, precision)
        self.socket_id = socket_id
        self._type = TaskBlockType.INPUT

    def _construct_computation(self) -> None:
        self._computation = 0

    def _construct_storage(self) -> None:
        self._storage = 0

    def accept(self, visitor):
        visitor.visit_INPUT(self)

    def fire(self, tick_num: int):
        for i in range(tick_num):
            for edge in self._output_edges:
                tick = Tick(self._id, i, 0)
                edge.add_tick(tick)

    # def copy_like(self) -> TaskBlock:
    #     new_task_block = InputTaskBlock(copy(self.shape),
    #                                     IDGenerator.get_next_task_id(),
    #                                     self.precision)
    #     new_task_block.socket_id = self.socket_id
    #     return new_task_block
