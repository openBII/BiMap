from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from typing import Tuple, Callable
from src.simulator.resource_simulator.st_model.tick import Tick


class STaskBlock(TaskBlock):
    """
    STaskBlock类描述一个存储任务块
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        super().__init__(task_id, shape, precision)
        self._pipeline_area = shape 
        self._type = TaskBlockType.SI

    @property
    def pipeline_num(self) -> Shape:
        return self._pipeline_num

    @pipeline_num.setter
    def pipeline_num(self, value: Shape):
        '''
        设置流水区域，流水更新后，需要重新构建存储信息
        raises:
            ValueError: 当设置的行流水参数大于任务块y方向大小时，抛出异常
        '''
        self._pipeline_num = value
        self._construct_storage()

    def _construct_computation(self) -> None:
        self._computation = 0

    def _construct_storage(self) -> None:
        # 加入计算过程
        self._storage = 0

    def accept(self, visitor):
        visitor.visit_S(self)

    def fire(self, iteration: int, time: int, callback: Callable, start_callback: Callable = None):
        for edge in self._output_edges:
            tick = Tick(self._id, iteration, time, callback, start_callback)
            edge.add_tick(tick)

    def consume(self) -> Tuple[int, int, int, bool]:
        start_time = float("inf")
        available_time = 0
        input_flag = False
        for edge in self._input_edges:
            received_tick = edge.consume_tick()
            if received_tick.start_callback:
                input_flag = True
            # find the time of the earliest input as the start time of this task
            if received_tick.time < start_time:
                start_time = received_tick.time
            if received_tick.time > available_time:
                available_time = received_tick.time
        return start_time, available_time, received_tick.iteration, input_flag

    # def copy_like(self) -> TaskBlock:
    #     data = deepcopy(self._data)
    #     new_task_block = SITaskBlock(IDGenerator.get_next_task_id(),
    #                                  copy(self.shape),
    #                                  self.precision, data)
    #     return new_task_block