from __future__ import annotations
from typing import Tuple
from src.simulator.resource_simulator.evaluation_model.communication_evaluator import NetworkParameterDict


class CommunicationConfig:
    def __init__(self, bandwidth: float = None,
                 process_node = None,
                 size: Tuple[int] = None,
                 latency: float = 0) -> None:
        self.bandwidth = bandwidth
        self.latency = latency
        self.size = size
        self.process_node = process_node


class HybridCoreCommunicationConfig(CommunicationConfig):
    def __init__(self, sram_bandwidth: float, dram_bandwidth: float, 
                 flash_bandwidth: float, noc_bandwidth: float, nop_bandwidth: float, process_node: int, latency: float = 0):
        super().__init__(process_node=process_node)
        self.sram_bandwidth = sram_bandwidth
        self.dram_bandwidth = dram_bandwidth
        self.flash_bandwidth = flash_bandwidth
        self.noc_bandwidth = noc_bandwidth
        self.nop_bandwidth = nop_bandwidth
        self.latency = latency


class CoreCommunicationConfig(CommunicationConfig):
    def __init__(self, buffer2array_input: float, 
                 buffer2array_weight: float, 
                 array2buffer: float, 
                 array2vector: float, 
                 buffer2vector_input: float,
                 buffer2vector_params: float,
                 vector2buffer: float,
                 buffer2router: float,
                 router2buffer: float,
                 size: Tuple[int] = None,
                 latency: float = 0) -> None:
        super().__init__(size=size, latency=latency)
        self.buffer2array_input = buffer2array_input
        self.buffer2array_weight = buffer2array_weight
        self.array2buffer = array2buffer
        self.array2vector = array2vector
        self.buffer2vector_input = buffer2vector_input
        self.buffer2vector_params = buffer2vector_params
        self.vector2buffer = vector2buffer
        self.buffer2router = buffer2router
        self.router2buffer = router2buffer
        