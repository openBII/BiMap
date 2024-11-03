import heapq
from queue import PriorityQueue
from typing import List, Dict, Tuple
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.hop import Hop
from src.simulator.resource_simulator.st_model.tick import Tick
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.communication_evaluator import CommunicationEvaluator, SharedMemoryCommunicationEvaluator, CoreCommunicationEvaluator, NetworkParameterDict
from src.simulator.resource_simulator.config.communication_config import CoreCommunicationConfig, CommunicationConfig


class CommunicationPoint(STPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.bandwidth, self.latency = self.config_handler(config)
        self._edge_map: Dict[Edge, List[Tuple[MLCoord, int]]] = {}
        self.evaluator = CommunicationEvaluator(self.bandwidth, 
                                                process_node=config.process_node,
                                                size=config.size,
                                                latency=self.latency)

    def update_bandwidth(self, value: float):
        self.bandwidth = value
        self.evaluator.bandwidth = value

    def __repr__(self):
        return 'bandwidth: ' + repr(self.bandwidth) + '\n'

    def config_handler(self, config: CommunicationConfig):
        return config.bandwidth, config.latency

    @property
    def edge_map(self):
        return self._edge_map

    def process(self, edges: List[Edge], 
                deadline: float = None) -> Tuple[List[Edge], List[Edge], float]:
        # start_time_heap = []
        start_time_heap = PriorityQueue()
        tick_dict: Dict[Edge, List[Tick]] = {}
        for edge in edges:
            # same edge may appear N times for N different iterations
            tick = edge._consume()
            if edge in tick_dict:
                tick_dict[edge].append(tick)
            else:
                tick_dict.update({edge: [tick]})
            # heapq.heappush(start_time_heap, (tick.time, (edge, tick.iteration)))
            start_time_heap.put((tick.time, (edge, tick.iteration)))
        if deadline is None:
            copied_edge_map = self.evaluator.copy_edge_map()
        for edge in tick_dict:
            ticks = tick_dict[edge]
            for tick in ticks:
                if not self.evaluator.is_edge_mapped(edge, tick.iteration):
                    self.evaluator.create_edge_path(edge, tick.iteration, 
                                                    self.edge_map[edge])
        finished_edges: List[Tuple[Edge, int]]
        finished_edges, finish_time = self.evaluator(start_time_heap, deadline)
        if deadline is None:
            self.evaluator.edge_map = copied_edge_map
            for edge in edges:
                ticks = tick_dict[edge]
                tick = ticks.pop()  # 后进先出, 因为put_back时每次都是在队首放一个tick
                edge._put_back(tick, tick.time)
        else:
            for edge in edges:
                ticks = tick_dict[edge]
                tick = ticks.pop()  # 后进先出, 因为put_back时每次都是在队首放一个tick
                if (edge, tick.iteration) in finished_edges:
                    edge._fire(
                        tick, finish_time, 
                        self.evaluator.recorder.correct_time,
                        tick.start_time + self.evaluator.get_latency(edge),
                        self.evaluator.get_latency(edge))
                else:
                    edge._put_back(
                        tick, 
                        tick.time if tick.time > finish_time else finish_time)
        # unfinished_edges = copy(edges)
        # for edge in finished_edges:
        #     unfinished_edges.remove(edge[0])
        unfinished_edges = []
        # for entry in start_time_heap:
        #     unfinished_edges.append(entry[1][0])
        while not start_time_heap.empty():
            unfinished_edges.append(start_time_heap.get()[1][0])
        # unfinished_hash = 0
        # for edge in unfinished_edges:
        #     unfinished_hash += hash(edge)
        # another_unfinished_hash = 0
        # for edge in another_unfinished_edges:
        #     another_unfinished_hash += hash(edge)
        # assert unfinished_hash == another_unfinished_hash
        if deadline is not None:
            assert finish_time == deadline
        return (unfinished_edges, [edge[0] for edge in finished_edges], 
                finish_time)
    

class CoreCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        self.set_evaluator(CoreCommunicationEvaluator(self.bandwidth,
                                                      latency=self.latency))

    # def config_handler(self, config: CoreCommunicationConfig):
    #     bandwidth_dict = NetworkParameterDict()
    #     bandwidth_dict[Hop(Coord(0), Coord(1))] = config.buffer2array_input
    #     bandwidth_dict[Hop(Coord(0), Coord(1), 1)] = config.buffer2array_weight
    #     bandwidth_dict[Hop(Coord(1), Coord(0))] = config.array2buffer
    #     bandwidth_dict[Hop(Coord(1), Coord(2))] = config.array2vector
    #     bandwidth_dict[Hop(Coord(0), Coord(2))] = config.buffer2vector_input
    #     bandwidth_dict[Hop(Coord(0), Coord(2), 1)] = config.buffer2vector_input
    #     bandwidth_dict[Hop(Coord(2), Coord(0))] = config.vector2buffer
    #     bandwidth_dict[Hop(Coord(0), Coord(3))] = config.buffer2router
    #     bandwidth_dict[Hop(Coord(3), Coord(0))] = config.router2buffer
    #     return bandwidth_dict, config.latency
    

class SharedMemoryCommunicationPoint(CommunicationPoint):
    def __init__(self, config: CommunicationConfig, shared_memory_coord: Coord, 
                 arbitrator_coord: Coord):
        super().__init__(config)
        evaluator = SharedMemoryCommunicationEvaluator(self.bandwidth, 
                                                       shared_memory_coord, 
                                                       arbitrator_coord,
                                                       config.process_node,
                                                       config.size,
                                                       latency=self.latency)
        self.set_evaluator(evaluator)
     