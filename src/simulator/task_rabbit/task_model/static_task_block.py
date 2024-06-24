from queue import Queue
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_model.tick import Tick
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision


class StaticTaskBlock(TaskBlock):
    """
    Static Task Block, which automatically produces #iteration tokens
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        super().__init__(task_id, shape, precision)
        self._type = TaskBlockType.STATIC
        self._ticks: Queue[Tick] = Queue()

    def _construct_storage(self) -> None:
        self._storage = 0

    def _construct_computation(self) -> None:
        self._computation = 0

    @property
    def activated(self):
        return not self._ticks.empty()

    def accept(self, visitor):
        visitor.visit_STATIC(self)

    # TODO(huanyu): 由DRAM搬运到SRAM的Static Block, 可能会被换出, 此时的生存周期需要显式管理
    def init_ticks(self, tick_num: int):
        for iteration in range(tick_num):
            tick = Tick(self.id, iteration, 0)
            self._ticks.put(tick)

    def consume(self):
        tick = self._ticks.get()
        return tick.time, tick.iteration

    def fire(self, iteration: int, time: int):
        for edge in self._output_edges:
            tick = Tick(self._id, iteration, time)
            edge.add_tick(tick)
