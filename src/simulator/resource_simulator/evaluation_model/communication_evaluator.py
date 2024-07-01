from __future__ import annotations
from typing import Dict, Tuple, List
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.hop import Hop, HopDict
import heapq
from copy import deepcopy
from src.simulator.resource_simulator.evaluation_model.recorder import CommunicationRecorder, CommunicationRecord
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator, EvaluationMode


class BandwidthDict:
    def __init__(self) -> None:
        self.dict: Dict[Hop, float] = {}

    def __setitem__(self, hop: Hop, bandwidth: float):
        self.dict[hop] = bandwidth

    def __getitem__(self, hop: Hop):
        for key in self.dict:
            if key.src == hop.src and key.dst == hop.dst and key.link_id == hop.link_id:
                return self.dict[key]
    
    def __repr__(self) -> str:
        string = '\n'
        for hop in self.dict:
            string += repr(hop) + ': ' + repr(self.dict[hop]) + '\n'
        string = string[:-1]
        return string


class CommunicationEvaluator(Evaluator):
    def __init__(self, bandwidth: float, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(mode)
        self.bandwidth = bandwidth
        self.edge_map: Dict[Tuple[Edge, int], List[Hop]] = {}
        # results: {(Edge, iteration, Hop): CommunicationRecord}
        self.recorder = CommunicationRecorder()

    def __call__(self, input, deadline=None):
        return self.eval(input, deadline)

    def eval(self, input, deadline=None):
        if self.mode == EvaluationMode.STATIC:
            return self.eval_by_model(input, deadline)
        elif self.mode == EvaluationMode.DYNAMIC:
            return self.eval_by_execution(input, deadline)
        else:
            raise ValueError('Unsupported evaluation mode')

    def is_edge_mapped(self, edge: Edge, iteration: int):
        if (edge, iteration) in self.edge_map:
            if len(self.edge_map[(edge, iteration)]) == 0:
                return False
            else:
                return True
        else:
            return False

    def create_edge_path(self, edge: Edge, iteration: int, ml_coords: List[Tuple[MLCoord, int]]):
        """
        Link ID规定由dst coord中指明
        两个坐标之间默认Link ID相同
        """
        self.edge_map[(edge, iteration)] = []
        src_ml_coord = ml_coords[0]
        for i in range(1, len(ml_coords)):
            dst_ml_coord = ml_coords[i]
            assert src_ml_coord[0].level == dst_ml_coord[0].level
            assert src_ml_coord[0].outer_coord == dst_ml_coord[0].outer_coord
            src_bottom_coord = src_ml_coord[0].bottom_coord
            dst_bottom_coord = dst_ml_coord[0].bottom_coord
            assert src_bottom_coord.dim == dst_bottom_coord.dim
            self.generate_hops(edge, iteration, src_bottom_coord, dst_bottom_coord, dst_ml_coord[1])
            # index = 0
            # dim = src_bottom_coord.dim
            # for i in range(dim):
            #     if src_bottom_coord[i] != dst_bottom_coord[i]:
            #         index = i
            #         break
            # if src_bottom_coord[index] > dst_bottom_coord[index]:
            #     for i in range(src_bottom_coord[index], dst_bottom_coord[index], -1):
            #         src_coord_list = list(src_bottom_coord)
            #         dst_coord_list = list(src_bottom_coord)
            #         src_coord_list[index] = i
            #         dst_coord_list[index] = i - 1
            #         hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list))
            #         self.edge_map[(edge, iteration)].append(hop)
            # else:
            #     for i in range(src_bottom_coord[index], dst_bottom_coord[index]):
            #         src_coord_list = list(src_bottom_coord)
            #         dst_coord_list = list(src_bottom_coord)
            #         src_coord_list[index] = i
            #         dst_coord_list[index] = i + 1
            #         hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list))
            #         self.edge_map[(edge, iteration)].append(hop)
            src_ml_coord = dst_ml_coord

    def append_hop(self, edge: Edge, iteration: int, hop: Hop):
        self.edge_map[(edge, iteration)].append(hop)

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, link_id: int):
        """
        This method may need to be overriden in custom communication evaluators.
        The default method is for MESH topology.

        Args:
            src: Coord, source coordinate
            dst: Coord, destination coordinate

        Returns:
            hops: List[Hop], a list of hops between src and dst 
        """
        index = 0
        dim = src.dim
        for i in range(dim):
            if src[i] != dst[i]:
                index = i
                break
        if src[index] > dst[index]:
            for i in range(src[index], dst[index], -1):
                src_coord_list = list(src)
                dst_coord_list = list(src)
                src_coord_list[index] = i
                dst_coord_list[index] = i - 1
                hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list), id=link_id)
                self.append_hop(edge, iteration, hop)
        else:
            for i in range(src[index], dst[index]):
                src_coord_list = list(src)
                dst_coord_list = list(src)
                src_coord_list[index] = i
                dst_coord_list[index] = i + 1
                hop = Hop(src=Coord(src_coord_list), dst=Coord(dst_coord_list), id=link_id)
                self.append_hop(edge, iteration, hop)

    def get_bandwidth(self, hop: Hop):
        if type(self.bandwidth) in [float, int]:
            return self.bandwidth
        elif isinstance(self.bandwidth, BandwidthDict):
            return self.bandwidth[hop]
        else:
            raise TypeError("Unsupported type of Attribute: bandwidth")
        
    def all_edges_reach_deadline(self, edge_heap: List[Tuple[int, Tuple[Edge, int]]], deadline: float):
        if deadline is None:
            return False
        for entry in edge_heap:
            if entry[0] != deadline:
                return False
        return True
    
    def copy_recorder(self):
        recorder = {}
        for key, record in self.recorder:
            recorder[key] = deepcopy(record)
        return recorder

    def eval_by_model(self, edge_heap: List[Tuple[int, Tuple[Edge, int]]], extern_deadline: float = None) -> Tuple[List[Edge], int]:
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
        finished_edges: List[Tuple[Edge, int]] = []
        if extern_deadline is None:
            recorder = self.copy_recorder()
        while len(finished_edges) == 0:  # 每次评估一个hop
            if self.all_edges_reach_deadline(edge_heap, extern_deadline):
                break
            min_start_time, min_edge = heapq.heappop(edge_heap)  # 最先可以开始的边
            if extern_deadline is not None:
                if min_start_time > extern_deadline:
                    return [], extern_deadline
            if len(edge_heap) != 0:
                second_min_start_time, second_min_edge = heapq.heappop(edge_heap)  # 第二先可以开始的边
            else:
                second_min_start_time = float('inf')
                second_min_edge = None
            min_edges = [min_edge]  # 所有同一时间且最先开始的边
            while second_min_start_time == min_start_time:
                min_edges.append(second_min_edge)
                if len(edge_heap) != 0:
                    second_min_start_time, second_min_edge = heapq.heappop(edge_heap)
                else:
                    break
            if second_min_start_time != min_start_time and second_min_edge is not None:
                heapq.heappush(edge_heap, (second_min_start_time, second_min_edge))
            else:
                second_min_start_time = float('inf')
            # 对找到的最先开始的边根据第一个Hop是否重合进行分组
            hop_dict = HopDict()
            for edge in min_edges:
                first_hop = self.edge_map[edge].pop(0)
                if not first_hop in hop_dict:
                    hop_dict[first_hop] = [edge]
                else:
                    hop_dict[first_hop].append(edge)
            # 分组进行评估
            min_end_time = float('inf')  # 当前轮次评估中最先结束的Hop
            for hop in hop_dict:
                edges = hop_dict[hop]
                num_edges = len(edges)
                real_bandwidth = self.get_bandwidth(hop) / num_edges
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
            # deadline = min_end_time if min_end_time < second_min_start_time else second_min_start_time
            if extern_deadline is not None:
                deadline = min(min_end_time, second_min_start_time, extern_deadline)
            else:
                deadline = min(min_end_time, second_min_start_time)
            for hop in hop_dict:
                edges = hop_dict[hop]
                num_edges = len(edges)
                real_bandwidth = self.get_bandwidth(hop) / num_edges
                for edge in edges:
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
                    if record.percent == 1 and len(self.edge_map[edge]) == 0:  # 某edge评估结束
                        finished_edges.append(edge)
                        # 评估的结束时间
                        finish_time = record.end_time
                    else:
                        # 将未完成的边重新加入堆中
                        heapq.heappush(edge_heap, (record.end_time, edge))
        if extern_deadline is None:
            self.recorder.recorder_time = recorder
        else:
            if len(finished_edges) != 0:
                assert finish_time == extern_deadline
            else:
                finish_time = extern_deadline
        return finished_edges, finish_time


class CoreCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: BandwidthDict, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(bandwidth, mode)

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, link_id: int):
        if (Coord(3) in [src, dst]) and (Coord(0) not in [src, dst]):
            self.append_hop(edge, iteration, Hop(src, Coord(0), link_id))
            self.append_hop(edge, iteration, Hop(Coord(0), dst, link_id))
        else:
            self.append_hop(edge, iteration, Hop(src, dst, link_id))
    

class SharedMemoryCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float, shared_memory_coord: Coord, arbitrator_coord: Coord, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(bandwidth, mode)
        self.shared_memory_coord = shared_memory_coord
        self.arbitrator_coord = arbitrator_coord

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, link_id: int):
        if self.arbitrator_coord in [src, dst]:
            self.append_hop(edge, iteration, Hop(src, dst, link_id))
        else:
            if self.shared_memory_coord in [src, dst]:
                self.append_hop(edge, iteration, Hop(src, self.arbitrator_coord, link_id))
                self.append_hop(edge, iteration, Hop(self.arbitrator_coord, dst, link_id))
            else:
                self.append_hop(edge, iteration, Hop(src, self.arbitrator_coord, link_id))
                self.append_hop(edge, iteration, Hop(self.arbitrator_coord, self.shared_memory_coord, link_id))
                self.append_hop(edge, iteration, Hop(self.shared_memory_coord, self.arbitrator_coord, link_id))
                self.append_hop(edge, iteration, Hop(self.arbitrator_coord, dst, link_id))


class BoardCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(bandwidth, mode)


class ServerCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float, mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(bandwidth, mode)


if __name__ == "__main__":
    a = Coord(2)
    b = Coord(3)
    print((Coord(3) in [a, b]) and (Coord(0) not in [a, b]))