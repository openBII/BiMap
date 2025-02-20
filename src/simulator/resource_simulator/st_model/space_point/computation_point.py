from src.simulator.resource_simulator.sync.sync_table import SyncTable
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.recorder import ComputationRecorder
from src.simulator.resource_simulator.evaluation_model.computation_evaluator import ComputationEvaluator, MACArrayEvaluator, VectorUnitEvaluator, HybridPrecisionMACArrayEvaluator
from src.simulator.resource_simulator.config.computation_config import ComputationConfig


class ComputationPoint(STPoint):
    def __init__(self, config: ComputationConfig):
        super().__init__()
        self.config = config
        self.recorder = ComputationRecorder()
        self.evaluator = ComputationEvaluator(self.config)

    def __repr__(self):
        return repr(self.config)
        
    def process(self, task_id: int, sync_table: SyncTable):
        if self._tasks[self._pc].id == task_id:
            task: CTaskBlock = self._tasks[self._pc]
            duration = self.evaluator(task)
            # The start time is obtained considering data dependencies
            if self._is_pipeline:
                start_time, communicate_end_time, iteration, consumed_ticks = task.consume(True)
                start_time, end_time = self.calculate_time(start_time, duration,
                                                           communicate_end_time)
            else:
                start_time, iteration, consumed_ticks = task.consume(False)
                start_time, end_time = self.calculate_time(start_time, duration)
            last_max_time = self.recorder.max_time
            self.recorder.record(task_id, iteration, start_time, end_time)
            # Here we can remove this record or not
            self.increment_pc()
            if not super().process(sync_table):
                task.put_back(consumed_ticks)
                # self.recorder.remove(task_id, iteration)
                self.recorder.reset(last_max_time)
                self.decrement_pc()
                return False
            # TODO: 计算如果可以和访存流水如何计算存储任务块的存活时间
            task.callback(self.recorder.max_time, consumed_ticks, duration)
            task.fire(iteration, self.recorder.max_time, self._is_pipeline,
                      start_time, self.evaluator._latency)
            return True
        else:
            return False
        
    def calculate_time(self, start_time: float, duration: float, 
                           communicate_end_time: float = None):
        allow_time = float(max(start_time, self.recorder.max_time))
        compute_end_time = allow_time + duration
        if self._is_pipeline:
            return (allow_time, 
                    float(max(compute_end_time, 
                              communicate_end_time + self.evaluator._latency)))
        else:
            return allow_time, float(compute_end_time)


class MACArrayPoint(ComputationPoint):
    def __init__(self, config: ComputationConfig):
        super().__init__(config)
        evaluator = MACArrayEvaluator(config)
        self.set_evaluator(evaluator)

class HybridPrecisionMACArrayPoint(ComputationPoint):
    def __init__(self, config: ComputationConfig):
        super().__init__(config)
        evaluator = HybridPrecisionMACArrayEvaluator(config)
        self.set_evaluator(evaluator)

class VectorPoint(ComputationPoint):
    def __init__(self, config: ComputationConfig):
        super().__init__(config)
        evaluator = VectorUnitEvaluator(config)
        self.set_evaluator(evaluator)