from typing import Dict, Tuple, Sequence
from src.simulator.task_rabbit.task_model.precision import Precision


class ComputationConfig:
    def __init__(self) -> None:
        super().__init__()
        self.dict: Dict[Precision, Tuple[int]] = {}

    def __getitem__(self, precision: Precision):
        return self.dict[precision]
    
    def __iter__(self):
        for precision in self.dict:
            yield precision, self.dict[precision]

    def __setitem__(self, precision: Precision, size: Sequence[int]):
        if type(size) is int:
            size = (size, )
        self.dict[precision] = size

    def __contains__(self, precision: Precision):
        return precision in self.dict
    
    def __repr__(self) -> str:
        string  = ''
        for precision in self.dict:
            string += precision.name + ': ' + repr(self.dict[precision]) + '\n'
        return string
