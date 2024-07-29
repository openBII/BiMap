from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.evaluation_model.evaluator import EvaluationMode
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator


class MemoryEvaluator(Evaluator):
    def __init__(self, capacity: int,
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(mode)
        self.capacity = capacity

    def eval_area(self):
        return self.capacity / 1024 * 2.5

    def eval_by_model(self, input: TaskBlock):
        # TODO: evaluate memory operations
        pass
