from typing import Dict, Tuple, Sequence
from src.simulator.task_rabbit.task_model.precision import Precision


class ComputationConfig:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        pass


class MACArrayConfig(ComputationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.dict: Dict[Precision, Tuple[int]] = {}

    def __getitem__(self, precision: Precision):
        return self.dict[precision]

    def __setitem__(self, precision: Precision, size: Sequence[int]):
        self.dict[precision] = tuple(size)

    def __contains__(self, precision: Precision):
        return precision in self.dict
    
    def __repr__(self) -> str:
        string  = ''
        for precision in self.dict:
            string += precision.name + ': ' + repr(self.dict[precision]) + '\n'
        return string


class VectorUnitConfig(ComputationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.dict: Dict[str, int] = {}

    def __getitem__(self, function: str):
        return self.dict[function]

    def __setitem__(self, function: str, size: int):
        self.dict[function] = size

    def __repr__(self) -> str:
        string  = ''
        for function in self.dict:
            string += function + ': ' + repr(self.dict[function]) + '\n'
        return string

if __name__ == "__main__":
    config = MACArrayConfig()
    config[Precision.INT_8] = (4, 32)
    print(Precision.INT_8 in config)
    print(Precision.INT_16 in config)