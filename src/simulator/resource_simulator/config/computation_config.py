from typing import Dict, Tuple, Sequence, Union
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode


class ComputationInfo:
    def __init__(self, parallelism: Union[int, Sequence[int]], latency: Union[int, Dict]):
        self.parallelism = parallelism
        if type(latency) is int:
            self.latency = latency
        else:
            self.latency = {}
            for op in latency:
                if op == 'relu':
                    self.latency[TaskBlockType.CRELU] = latency[op]
                elif op == 'softmax':
                    self.latency[TaskBlockType.CSoftMax] = latency[op]
                elif op == 'add':
                    self.latency[TaskBlockType.CADD] = latency[op]
                else:
                    raise NotImplementedError(
                        'Operation ' + op + ' is not supported')
                
    def __repr__(self):
        return 'Parallelism: ' + repr(self.parallelism) + ' Latency: ' + repr(self.latency)


class ComputationConfig:
    def __init__(self) -> None:
        self.dict: Dict[Precision, ComputationInfo] = {}
        self.local_memory_latency: int = None
        self.local_memory_bandwidth: int = None
        self.process_node: ProcessNode = None

    def __getitem__(self, precision: Precision):
        return self.dict[precision]
    
    def __iter__(self):
        for precision in self.dict:
            yield precision, self.dict[precision]

    def __setitem__(self, precision: Precision, info: ComputationInfo):
        self.dict[precision] = info

    def __contains__(self, precision: Precision):
        return precision in self.dict
    
    def __repr__(self) -> str:
        string  = ''
        for precision in self.dict:
            string += precision.name + ': ' + repr(self.dict[precision]) + '\n'
        return string
