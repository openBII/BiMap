from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
import numpy
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator, EvaluationMode
from src.simulator.resource_simulator.config.computation_config import MACArrayConfig, ComputationConfig, VectorUnitConfig


class ComputationEvaluator(Evaluator):
    def __init__(self, config: ComputationConfig, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(mode)
        self.config = config
    

class MACArrayEvaluator(ComputationEvaluator):
    def __init__(self, config: MACArrayConfig, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        in_precision = task.in_precision
        precision = min(in_precision)
        assert precision in self.config, "Cannot process task in {:s}".format(precision.name)
        num_pe = numpy.prod(self.config[precision])
        return task.shape.volume / num_pe
    

class VectorUnitEvaluator(ComputationEvaluator):
    def __init__(self, config: VectorUnitConfig, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)