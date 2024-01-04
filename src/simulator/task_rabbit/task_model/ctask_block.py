from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_model.tick import Tick
from typing import List, Tuple


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

    def fire(self, iteration: int, time: int, consumed_ticks: List[Tick]):
        for tick in consumed_ticks:
            if tick.callback is not None:
                tick.callback(tick.task_id, iteration, time)
        for edge in self._output_edges:
            tick = Tick(self._id, iteration, time)
            edge.add_tick(tick)

    def consume(self) -> Tuple[int, int, List[Tick]]:
        # 处理一下没有输入边的情况
        start_time = 0
        consumed_ticks = []
        for edge in self._input_edges:
            received_tick = edge.consume_tick()
            consumed_ticks.append(received_tick)
            # find the time of the latest input as the start time of this task
            if received_tick.time > start_time:
                start_time = received_tick.time
        return start_time, received_tick.iteration, consumed_ticks

    # def copy_like(self) -> TaskBlock:
    #     data = deepcopy(self._data)
    #     new_task_block = SITaskBlock(IDGenerator.get_next_task_id(),
    #                                  copy(self.shape),
    #                                  self.precision, data)
    #     return new_task_block
    

