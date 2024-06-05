from __future__ import annotations
from typing import Dict
from src.simulator.resource_simulator.st_model.st_coord import LinkCoord


class BandwidthDict:
    def __init__(self) -> None:
        self.dict: Dict[LinkCoord, float] = {}

    def __setitem__(self, key: LinkCoord, value: float):
        self.dict[key] = value

    def __getitem__(self, key: LinkCoord):
        return self.dict[key]
    
    def __repr__(self) -> str:
        string = '\n'
        for link_coord in self.dict:
            string += repr(link_coord) + ': ' + repr(self.dict[link_coord]) + '\n'
        string = string[:-1]
        return string


class CommunicationConfig:
    def __init__(self, bandwidth: float | BandwidthDict = None) -> None:
        self.bandwidth = bandwidth


class CoreCommunicationConfig(CommunicationConfig):
    def __init__(self, buffer2array_input: float, 
                 buffer2array_weight: float, 
                 array2buffer: float, 
                 array2vector: float, 
                 buffer2vector_input: float,
                 buffer2vector_params: float,
                 vector2buffer: float) -> None:
        super().__init__()
        self.buffer2array_input = buffer2array_input
        self.buffer2array_weight = buffer2array_weight
        self.array2buffer = array2buffer
        self.array2vector = array2vector
        self.buffer2vector_input = buffer2vector_input
        self.buffer2vector_params = buffer2vector_params
        self.vector2buffer = vector2buffer
        