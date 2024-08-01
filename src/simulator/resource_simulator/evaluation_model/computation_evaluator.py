import math
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator, EvaluationMode
from src.simulator.resource_simulator.config.computation_config import ComputationConfig
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class ComputationEvaluator(Evaluator):
    def __init__(self, config: ComputationConfig, 
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(mode)
        self.config = config
    

class MACArrayEvaluator(ComputationEvaluator):
    def __init__(self, config: ComputationConfig, 
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        in_precision = task.in_precision
        precision = min(in_precision)
        assert precision in self.config, "Cannot process task in {:s}".format(precision.name)
        array = self.config[precision]
        if task.task_type == TaskBlockType.CVM:
            num_pe = math.prod(array)
            return task.shape.volume / num_pe
        else:
            raise NotImplementedError
        
    def eval_area(self):
        area = 0
        for _, size in self.config:
            area += math.prod(size) * 0.0007
        return area

class VectorUnitEvaluator(ComputationEvaluator):
    def __init__(self, config: ComputationConfig, 
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        in_precision = task.in_precision
        precision = min(in_precision)
        num_pe = self.config[precision][0]
        if task.task_type == TaskBlockType.CRELU:
            return task.shape.volume / num_pe
        elif task.task_type == TaskBlockType.CSSoftMax:
            return task.shape.volume / num_pe * 4
        elif task.task_type == TaskBlockType.CSoftMax:
            return task.shape.volume / num_pe * 3
        elif task.task_type == TaskBlockType.CADD:
            volume = 0
            for in_task in task.in_tasks:
                volume += in_task.shape.volume
            branch = volume / task.shape.volume
            return task.shape.volume / num_pe * (branch - 1)
        elif task.task_type == TaskBlockType.CLayerNorm:
            return task.shape.volume / num_pe * 5
        else:
            raise NotImplementedError
        
    def eval_area(self):
        area = 0
        for _, size in self.config:
            area += math.prod(size) * 0.0007
        return area
