from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_model.tick import Tick
from typing import List, Tuple
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from copy import deepcopy


class CTaskBlock(TaskBlock):
    """
    CTaskBlock类描述一个计算任务块
    """

    def __init__(self, task_id: int, shape: Shape,
                 task_type: TaskBlockType, precision: Precision,
                 bias_type: BiasType = BiasType.NONE,
                 constant: float = None):
        super().__init__(task_id, shape, precision)
        self.bias_type = bias_type
        self._type = task_type
        self.constant = constant
        self._halide_func = None

    def _construct_storage(self) -> None:
        self._storage = 0

    def _construct_computation(self) -> None:
        self._computation = 0

    @property
    def out_shape(self):
        return Shape(ny=self.shape.ny, nx=self.shape.nx, nf=self.shape.nf)

    def accept(self, visitor):
        visitor.visit_C(self)

    def fire(self, iteration: int, end_time: float, is_pipeline: bool = False,
             start_time: float = None, latency: float = None):
        for edge in self.enabled_output_edges:
            if is_pipeline:
                tick = Tick(self._id, iteration, start_time + latency)
                tick.compute_time = end_time
                tick.start_time = start_time + latency
            else:
                tick = Tick(self._id, iteration, end_time)
            edge.add_tick(tick)

    def callback(self, time: float, consumed_ticks: List[Tick], 
                 duration: float):
        for tick in consumed_ticks:
            if tick.callback is not None:
                tick.callback(tick.task_id, tick.iteration, time)
            if tick.edge_callback is not None:
                tick.edge_callback(tick, time - duration - tick.time)

    def consume(self, is_pipeline: bool = False) -> Tuple[int, int, List[Tick]]:
        # 处理一下没有输入边的情况
        start_time = 0
        end_time = 0
        consumed_ticks: List[Tick] = []
        for edge in self.enabled_input_edges:
            received_tick = edge.consume_tick()
            consumed_ticks.append(received_tick)
            # Time of the first packet arrived
            if received_tick.start_time > start_time:
                start_time = received_tick.start_time
            if received_tick.time > end_time:
                end_time = received_tick.time
        if is_pipeline:
            return start_time, end_time, received_tick.iteration, consumed_ticks
        else:
            return end_time, received_tick.iteration, consumed_ticks
    
    def put_back(self, ticks: List[Tick]):
        assert len(ticks) == len(self.enabled_input_edges)
        for i, edge in enumerate(self.enabled_input_edges):
            edge.put_tick_back(ticks[i])

    def copy_like(self, shape: Shape = None) -> TaskBlock:
        new_task_block = CTaskBlock(IDGenerator.get_next_task_id(),
                                    deepcopy(self.shape) if shape is None else shape,
                                    self.task_type,
                                    self.precision,
                                    self.bias_type)
        return new_task_block
