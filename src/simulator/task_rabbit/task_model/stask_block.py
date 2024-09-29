import torch
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from typing import Tuple, Callable
from src.simulator.resource_simulator.st_model.tick import Tick
from copy import deepcopy
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator


class STaskBlock(TaskBlock):
    """
    STaskBlock类描述一个存储任务块
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        super().__init__(task_id, shape, precision)
        self._pipeline_area = shape 
        self._type = TaskBlockType.SI
        self.data: torch.Tensor = None

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
        self._storage = self.shape.volume

    def accept(self, visitor):
        visitor.visit_S(self)

    def fire(self, iteration: int, time: int, callback: Callable, 
             start_callback: Callable = None):
        for edge in self.enabled_output_edges:
            tick = Tick(self._id, iteration, time, callback, start_callback)
            tick.start_time = time
            edge.add_tick(tick)

    def consume(self, is_pipeline: bool = False) -> Tuple[int, int, int, bool]:
        start_time = float("inf")
        available_time = 0
        input_flag = False
        for edge in self.enabled_input_edges:
            received_tick = edge.consume_tick()
            if received_tick.start_callback:
                input_flag = True
            # find the time of the earliest input as the start time of this task
            if is_pipeline:
                if received_tick.start_time < start_time:
                    start_time = received_tick.start_time
                end_time = max(
                    received_tick.time, 
                    received_tick.compute_time + received_tick.latency)
                if end_time > available_time:
                    available_time = end_time
            else:
                if received_tick.time < start_time:
                    start_time = received_tick.time
                if received_tick.time > available_time:
                    available_time = received_tick.time
        return start_time, available_time, received_tick.iteration, input_flag

    def copy_like(self, shape: Shape = None) -> TaskBlock:
        new_task_block = STaskBlock(IDGenerator.get_next_task_id(),
                                    deepcopy(self.shape) if shape is None else shape,
                                    self.precision)
        return new_task_block