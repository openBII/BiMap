from typing import Dict, Tuple, List
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.hop import Hop
import heapq
from enum import Enum
from src.simulator.resource_simulator.evaluation_model.recorder import CommunicationRecorder, CommunicationRecord


class EvaluationMode(Enum):
    STATIC = 0
    DYNAMIC = 1


class Evaluator():
    def __init__(self) -> None:
        pass

    def eval(self, input, mode: EvaluationMode = EvaluationMode.STATIC):
        if mode == EvaluationMode.STATIC:
            return self.eval_by_model(input)
        elif mode == EvaluationMode.DYNAMIC:
            return self.eval_by_execution(input)
        else:
            raise ValueError('unsupported evaluation mode')
    
    def eval_by_model(self):
        raise NotImplementedError
    
    def eval_by_execution(self):
        raise NotImplementedError


class CommunicationEvaluator(Evaluator):
    def __init__(self, bandwidth: float, edge_map: Dict[Edge, List[MLCoord]] = None) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.edge_map: Dict[Edge, List[Hop]] = {}
        if edge_map is not None:
            self.create_edge_map(edge_map)
        # results: {(Edge, iteration, Hop): CommunicationRecord}
        self.recorder = CommunicationRecorder()

    def edge_not_mapped(self, edge: Edge):
        path = self.edge_map[edge]
        return len(path) == 0

    def create_all_edge_map(self, edge_map: Dict[Edge, List[MLCoord]]):
        for edge in edge_map:
            ml_coords = edge_map[edge]
            self.create_edge_map(edge, ml_coords)

    def create_edge_map(self, edge: Edge, ml_coords: List[MLCoord]):
        src_ml_coord = ml_coords[0]
        for i in range(1, len(ml_coords)):
            dst_ml_coord = ml_coords[i]
            assert src_ml_coord.level == dst_ml_coord.level
            assert src_ml_coord.outer_coord == dst_ml_coord.outer_coord
            src_bottom_coord = src_ml_coord.bottom_coord
            dst_bottom_coord = dst_ml_coord.bottom_coord
            assert src_bottom_coord.dim == dst_bottom_coord.dim
            self.edge_map[edge] = []
            index = 0
            dim = src_bottom_coord.dim
            for i in range(dim):
                if src_bottom_coord[i] != dst_bottom_coord[i]:
                    index = i
                    break
            if src_bottom_coord[index] > dst_bottom_coord[index]:
                for i in range(src_bottom_coord[index], dst_bottom_coord[index], -1):
                    src_coord_list = list(src_bottom_coord)
                    dst_coord_list = list(src_bottom_coord)
                    src_coord_list[index] = i
                    dst_coord_list[index] = i - 1
                    hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list))
                    self.edge_map[edge].append(hop)
            else:
                for i in range(src_bottom_coord[index], dst_bottom_coord[index]):
                    src_coord_list = list(src_bottom_coord)
                    dst_coord_list = list(src_bottom_coord)
                    src_coord_list[index] = i
                    dst_coord_list[index] = i + 1
                    hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list))
                    self.edge_map[edge].append(hop)
            src_ml_coord = dst_ml_coord


    def eval_by_model(self, edge_heap: List[Tuple[int, Tuple[Edge, int]]]) -> Tuple[List[Edge], int]:
        '''
        Pseudo-code:
        while not SOME_EDGE_FINISHED:
            找到所有最先开始的边, 记录到min_edges中  O(nlogn)
            找到第二先开始的边, 记录second_min_start_time O(logn)
            for edge in min_edges:  O(n)
                获取edge的第一个hop
                将edge加入hop_dict[hop]对应的列表中, 相当于对所有edge按第一个hop分组
            for hop in hop_dict:
                for edge in hop_dict[hop]:
                    

        Args:
        - edge_heap: Heap[(start time, (Edge, iteration))]

        Returns:
        - finished_edges: [Edge]
        - finish_time: float
        '''
        finished_edges: List[Edge] = []
        while len(finished_edges) == 0:  # 每次评估一个hop
            min_start_time, min_edge = heapq.heappop(edge_heap)  # 最先可以开始的边
            if len(edge_heap) != 0:
                second_min_start_time, second_min_edge = heapq.heappop(edge_heap)  # 第二先可以开始的边
            else:
                second_min_start_time = float('inf')
                second_min_edge = None
            min_edges = [min_edge]  # 所有同一时间且最先开始的边
            while second_min_start_time == min_start_time:
                min_edges.append(second_min_edge)
                second_min_start_time, second_min_edge = heapq.heappop(edge_heap)
            # 对找到的最先开始的边根据第一个Hop是否重合进行分组
            hop_dict: Dict[Hop, List[Tuple[Edge, int]]] = {}
            for edge in min_edges:
                first_hop = self.edge_map[edge[0]].pop(0)
                if not first_hop in hop_dict:
                    hop_dict[first_hop] = [edge]
                else:
                    hop_dict[first_hop].append(edge)
            # 分组进行评估
            min_end_time = float('inf')  # 当前轮次评估中最先结束的Hop
            for hop in hop_dict:
                edges = hop_dict[hop]
                num_edges = len(edges)
                real_bandwidth = self.bandwidth / num_edges
                for edge in edges:
                    if (*edge, hop) in self.recorder:
                        start_time = self.recorder[(*edge, hop)].start_time
                        last_percent = self.recorder[(*edge, hop)].percent
                    else:
                        start_time = min_start_time
                        last_percent = 0
                    duration = edge[0].flux * (1 - last_percent) / real_bandwidth
                    end_time = min_start_time + duration
                    if end_time < min_end_time:
                        min_end_time = end_time
                    self.recorder.update((*edge, hop), CommunicationRecord(start_time, end_time, last_percent))
            for hop in hop_dict:
                for edge in hop_dict[hop]:
                    deadline = min_end_time if min_end_time < second_min_start_time else second_min_start_time
                    record = self.recorder[(*edge, hop)]
                    if record.end_time > deadline:
                        record.end_time = deadline
                        record.percent += (deadline - min_start_time) * real_bandwidth / edge[0].flux
                        # 未处理完的hop需要放回到edge_map
                        self.edge_map[edge].insert(0, hop)
                    else:
                        record.percent = 1
                    # 真正的评估结果
                    self.recorder.update((*edge, hop), record)
                    if record.percent == 1 and len(self.edge_map[edge[0]]) == 0:  # 某edge评估结束
                        finished_edges.append(edge[0])
                        # 评估的结束时间
                        finish_time = record.end_time
                    else:
                        # 将未完成的边重新加入堆中
                        heapq.heappush(edge_heap, (record.end_time, edge))
            if second_min_edge is not None:
                heapq.heappush(edge_heap, (second_min_start_time, second_min_edge))
        return finished_edges, finish_time
