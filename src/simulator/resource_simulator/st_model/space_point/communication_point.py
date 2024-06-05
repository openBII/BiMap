import heapq
from typing import List, Dict, Union
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, LinkCoord, Coord
from src.simulator.resource_simulator.st_model.tick import Tick
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.communication_evaluator import CommunicationEvaluator
from src.simulator.resource_simulator.config.communication_config import BandwidthDict, CoreCommunicationConfig, CommunicationConfig


class CommunicationPoint(STPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.bandwidth: Union[float, BandwidthDict] = self.config_handler(config)
        self._edge_map: Dict[Edge, List[MLCoord]] = {}
        self.communication_evaluator = CommunicationEvaluator(self.bandwidth)

    def __repr__(self):
        return 'bandwidth: ' + repr(self.bandwidth) + '\n'

    def config_handler(self, config: CommunicationConfig):
        return config.bandwidth

    @property
    def edge_map(self):
        return self._edge_map

    def process(self, edges: List[Edge]) -> List[Edge]:
        # TODO(huanyu): This method can only be applied to MESH topology now
        start_time_heap = []
        tick_dict: Dict[Edge, List[Tick]] = {}
        for edge in edges:
            tick = edge._consume()  # same edge may appear N times for N different iterations
            if edge in tick_dict:
                tick_dict[edge].append(tick)
            else:
                tick_dict.update({edge: [tick]})
            heapq.heappush(start_time_heap, (tick.time, (edge, tick.iteration)))
        for edge in tick_dict:
            ticks = tick_dict[edge]
            for tick in ticks:
                if self.communication_evaluator.edge_not_mapped((edge, tick.iteration)):
                    self.communication_evaluator.create_edge_map((edge, tick.iteration), self.edge_map[edge])
        finished_edges, finish_time = self.communication_evaluator.eval(start_time_heap)
        for edge in edges:
            ticks = tick_dict[edge]
            tick = ticks.pop(0)
            if (edge, tick.iteration) in finished_edges:
                edge._fire(tick, finish_time, self.communication_evaluator.recorder.correct_time)
            else:
                edge._put_back(tick, finish_time)
        for edge in finished_edges:
            edges.remove(edge[0])
        return edges
    

class CoreCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CoreCommunicationConfig):
        super().__init__(config)

    def config_handler(self, config: CoreCommunicationConfig):
        bandwidth_dict = BandwidthDict()
        bandwidth_dict[LinkCoord(Coord(0), Coord(1))] = config.buffer2array_input
        bandwidth_dict[LinkCoord(Coord(0), Coord(1), 1)] = config.buffer2array_weight
        bandwidth_dict[LinkCoord(Coord(1), Coord(0))] = config.array2buffer
        bandwidth_dict[LinkCoord(Coord(1), Coord(2))] = config.array2vector
        bandwidth_dict[LinkCoord(Coord(0), Coord(2))] = config.buffer2vector_input
        bandwidth_dict[LinkCoord(Coord(0), Coord(2), 1)] = config.buffer2vector_input
        bandwidth_dict[LinkCoord(Coord(2), Coord(0))] = config.vector2buffer
        return bandwidth_dict
    

class ChipStarCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)


class BoardCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)


class ServerCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        