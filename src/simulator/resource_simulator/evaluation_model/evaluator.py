from __future__ import annotations
from enum import Enum
from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode


class EvaluationMode(Enum):
    STATIC = 0
    DYNAMIC = 1


class Evaluator():
    def __init__(self, process_node: ProcessNode,
                 mode: EvaluationMode = EvaluationMode.STATIC,
                 latency: float = None) -> None:
        self.process_node = process_node
        self.mode = mode
        self._latency = latency

    def __call__(self, input):
        return self.eval(input)

    def eval(self, input):
        if self.mode == EvaluationMode.STATIC:
            return self.eval_by_model(input)
        elif self.mode == EvaluationMode.DYNAMIC:
            return self.eval_by_execution(input)
        else:
            raise ValueError('Unsupported evaluation mode')
    
    def eval_by_model(self, input):
        raise NotImplementedError
    
    def eval_by_execution(self, input):
        raise NotImplementedError
    
    def eval_area(self) -> float:
        raise NotImplementedError
    
    def eval_energy(self):
        raise NotImplementedError
