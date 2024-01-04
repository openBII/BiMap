from typing import Tuple
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class OutputTaskBlock(TaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision, socket_id: int = 0):
        super().__init__(task_id, shape, precision)
        self.socket_id = socket_id
        self._type = TaskBlockType.OUTPUT

    def _construct_computation(self) -> None:
        self._computation = 0

    def _construct_storage(self) -> None:
        self._storage = 0

    def accept(self, visitor):
        visitor.visit_OUTPUT(self)

    def consume(self) -> Tuple[int, int, int]:
        start_time = float("inf")
        available_time = 0
        for edge in self._input_edges:
            received_tick = edge.consume_tick()
            # find the time of the earliest input as the start time of this task
            if received_tick.time < start_time:
                start_time = received_tick.time
            if received_tick.time > available_time:
                available_time = received_tick.time
        return start_time, available_time - start_time, received_tick.iteration

    # def copy_like(self) -> TaskBlock:
    #     new_task_block = OutputTaskBlock(copy(self.shape),
    #                                      IDGenerator.get_next_task_id(),
    #                                      self.precision)
    #     new_task_block.socket_id = self.socket_id
    #     return new_task_block
