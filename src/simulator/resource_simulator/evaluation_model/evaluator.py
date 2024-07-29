from __future__ import annotations
from enum import Enum


class EvaluationMode(Enum):
    STATIC = 0
    DYNAMIC = 1


class Evaluator():
    def __init__(self, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        self.mode = mode

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
