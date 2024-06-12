from typing import List
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_model.tick import Tick
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision


class VTaskBlock(TaskBlock):
    """
    Virtual Task Block, which comes from edge splitting
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        super().__init__(task_id, shape, precision)
        self._type = TaskBlockType.VIRTUAL

    def _construct_storage(self) -> None:
        self._storage = 0

    def _construct_computation(self) -> None:
        self._computation = 0

    def accept(self, visitor):
        visitor.visit_V(self)

    def fire(self, ticks: List[Tick]):
        assert len(self._output_edges) == 1
        edge = self._output_edges[0]
        while len(ticks) != 0:
            edge.add_tick(ticks.pop(0))

    def consume(self) -> List[Tick]:
        assert len(self._input_edges) == 1
        edge = self._input_edges[0]
        ticks: List[Tick] = []
        while not edge._output_ticks.empty():
            received_tick = edge.consume_tick()
            ticks.append(received_tick)
        return ticks
    
    def transfer_ticks(self):
        self.fire(self.consume())