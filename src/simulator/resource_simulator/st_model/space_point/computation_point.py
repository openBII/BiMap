from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.recorder import ComputationRecorder
from src.simulator.resource_simulator.evaluation_model.computation_evaluator import ComputationEvaluator, MACArrayEvaluator, VectorUnitEvaluator
from src.simulator.resource_simulator.config.computation_config import ComputationConfig, MACArrayConfig, VectorUnitConfig


class ComputationPoint(STPoint):
    def __init__(self, config: ComputationConfig):
        super().__init__()
        self.config = config
        self.compute_recorder = ComputationRecorder()
        self.evaluator = ComputationEvaluator(self.config)

    def __repr__(self):
        return repr(self.config)
        
    def process(self, task_id: int):
        if self._tasks[self._pc].id == task_id:
            task = self._tasks[self._pc]
            duration = self.evaluator(task)
            # The start time is obtained considering data dependencies
            start_time, iteration, consumed_ticks = task.consume()
            self.compute_recorder.record(task_id, iteration, start_time, duration)
            self.increment_pc()
            # TODO: 计算如果可以和访存流水如何计算存储任务块的存活时间
            task.callback(self.compute_recorder.max_time, consumed_ticks, duration)
            task.fire(iteration, self.compute_recorder.max_time)
            return True, task
        else:
            return False, None
        

class MACArrayPoint(ComputationPoint):
    def __init__(self, config: MACArrayConfig):
        super().__init__(config)
        evaluator = MACArrayEvaluator(config)
        self.set_evaluator(evaluator)


class VectorPoint(ComputationPoint):
    def __init__(self, config: VectorUnitConfig):
        super().__init__(config)
        evaluator = VectorUnitEvaluator(config)
        self.set_evaluator(evaluator)