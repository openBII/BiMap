from __future__ import annotations
import os
import heapq
import math
import copy
from typing import Dict, Tuple, List, Set
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.hop import Hop, HopDict
from src.simulator.resource_simulator.evaluation_model.recorder import CommunicationRecorder, CommunicationRecord
from src.simulator.resource_simulator.evaluation_model.recorder import BookSimCommRecorder, BooksimCommunicationRecord
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator, EvaluationMode
from queue import PriorityQueue
from src.simulator.resource_simulator.evaluation_model.utils import create_overlap_groups
from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode
from ext.booksim2.api.booksim_sync import BookSim2Sync

class NetworkParameterDict:
    def __init__(self) -> None:
        self.dict: Dict[Hop, float] = {}

    def __setitem__(self, hop: Hop, parameter: float):
        self.dict[hop] = float(parameter)

    def __getitem__(self, hop: Hop):
        return self.dict[hop]
    
    def __repr__(self) -> str:
        string = '\n'
        for hop in self.dict:
            string += repr(hop) + ': ' + repr(self.dict[hop]) + '\n'
        string = string[:-1]
        return string


class CommunicationEvaluator(Evaluator):
    def __init__(self, bandwidth: float,
                 process_node: ProcessNode,
                 mode: EvaluationMode = EvaluationMode.STATIC,
                 size: Tuple[int] = None,
                 latency: float = 0) -> None:
        super().__init__(process_node, mode)
        self.bandwidth = bandwidth
        self.latency = latency
        self.size = size
        self.edge_map: Dict[Tuple[Edge, int], List[Hop]] = {}
        self.recorder = CommunicationRecorder()
        self._latency_dict: Dict[Edge, float] = {}

    def __call__(self, input, deadline=None):
        return self.eval(input, deadline)

    def eval(self, input, deadline=None):
        if self.mode == EvaluationMode.STATIC:
            return self.eval_by_model(input, deadline)
        elif self.mode == EvaluationMode.DYNAMIC:
            return self.eval_by_execution(input, deadline)
        else:
            raise ValueError('Unsupported evaluation mode')
        
    def eval_area(self):
        return math.prod(self.size) * 0.025

    def is_edge_mapped(self, edge: Edge, iteration: int):
        if (edge, iteration) in self.edge_map:
            if len(self.edge_map[(edge, iteration)]) == 0:
                return False
            else:
                return True
        else:
            return False
        
    def copy_edge_map(self):
        edge_map: Dict[Tuple[Edge, int], List[Hop]] = {}
        for key in self.edge_map:
            edge_map[key] = [] 
            for hop in self.edge_map[key]:
                edge_map[key].append(Hop(hop.src, hop.dst, hop.link_id))
        return edge_map

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

        # Calculate the latency of the given edge
        if edge not in self._latency_dict:
            self._latency_dict[edge] = self._get_latency(
                self.edge_map[(edge, iteration)])
            
    def get_latency(self, edge: Edge):
        return self._latency_dict[edge]

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
            if hop.src == hop.dst:
                return float("inf")
            else:
                return self.bandwidth
        elif isinstance(self.bandwidth, NetworkParameterDict):
            return self.bandwidth[hop]
        else:
            raise TypeError("Unsupported type of Attribute: bandwidth")
        
    def get_hop_latency(self, hop: Hop):
        if type(self.latency) in [float, int]:
            if hop.src == hop.dst:
                return 0
            else:
                return self.latency
        elif isinstance(self.latency, NetworkParameterDict):
            return self.latency[hop]
        else:
            raise TypeError("Unsupported type of Attribute: latency")
        
    def _get_latency(self, path: List[Hop]):
        latency = 0
        for hop in path:
            latency += self.get_hop_latency(hop)
        return latency
        
    def all_edges_reach_deadline(self, 
                                 edge_heap: List[Tuple[int, Tuple[Edge, int]]], 
                                 deadline: float, 
                                 edges_cannot_proceed: Set[Tuple[Edge, int]]):
        if deadline is None:
            return False
        # for entry in edge_heap:
        #     if entry[0] < deadline and entry[1] not in edges_cannot_proceed:
        #         return False
        entry_list = []
        while not edge_heap.empty():
            entry = edge_heap.get()
            entry_list.append(entry)
        for entry in entry_list:
            edge_heap.put(entry)
        for entry in entry_list:
            if entry[0] < deadline and entry[1] not in edges_cannot_proceed:
                return False
        return True
    
    def copy_recorder(self):
        recorder = {}
        for key in self.recorder.recorder_time:
            record = self.recorder.recorder_time[key]
            recorder[key] = CommunicationRecord(record.start_time, record.end_time, record.percent)
        return recorder
    
    def backup_recorder(self):
        pass

    def eval_by_model(self, edge_heap: PriorityQueue, 
                      extern_deadline: float = None) -> Tuple[List[Edge], int]:
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
        
        # print("\n[eval_by_model]", self.__class__.__name__, os.getpid())
        # print("[edge_heap_init]", edge_heap.queue)
                    
        finished_edges: List[Tuple[Edge, int]] = []
        if extern_deadline is None:
            recorder = self.copy_recorder()
        
        while len(finished_edges) == 0:  # 每次评估一个hop
            # if self.all_edges_reach_deadline(edge_heap, extern_deadline):
            #     break
            # min_start_time, min_edge = heapq.heappop(edge_heap)  # 最先可以开始的边
            min_start_time, min_edge = edge_heap.get()
            if extern_deadline is not None:
                if min_start_time > extern_deadline:
                    # heapq.heappush(edge_heap, (min_start_time, min_edge))
                    edge_heap.put((min_start_time, min_edge))
                    return [], extern_deadline
            # if len(edge_heap) != 0:
            #     second_min_start_time, second_min_edge = heapq.heappop(edge_heap)  # 第二先可以开始的边
            if not edge_heap.empty():
                second_min_start_time, second_min_edge = edge_heap.get()  # 第二先可以开始的边
            else:
                second_min_start_time = float('inf')
                second_min_edge = None
            min_edges = [min_edge]  # 所有同一时间且最先开始的边
            while second_min_start_time == min_start_time:
                min_edges.append(second_min_edge)
                # if len(edge_heap) != 0:
                #    second_min_start_time, second_min_edge = heapq.heappop(edge_heap)
                if not edge_heap.empty():
                    second_min_start_time, second_min_edge = edge_heap.get()
                else:
                    break
            if second_min_start_time != min_start_time and second_min_edge is not None:
                # heapq.heappush(edge_heap, (second_min_start_time, second_min_edge))
                edge_heap.put((second_min_start_time, second_min_edge))
            else:
                second_min_start_time = float('inf')
            # 判断边之间是否会发生重合
            overlap_groups = create_overlap_groups(min_edges, self.edge_map)
            
            # 分组进行评估
            min_end_time = float('inf')  # 当前轮次评估中最先结束的edge
            for key_edge, neighbor_edges in overlap_groups.items():
                num_edges = len(neighbor_edges)
                # FIXME
                real_bandwidth = self.get_bandwidth(self.edge_map[key_edge][0]) / num_edges
                latency = self.latency * len(self.edge_map[key_edge])

                if key_edge in self.recorder:
                    start_time = self.recorder[key_edge].start_time
                    last_percent = self.recorder[key_edge].percent
                else:
                    start_time = min_start_time
                    last_percent = 0
                duration = key_edge[0].flux * (1 - last_percent) / real_bandwidth
                # print("[Comm-Eval]", duration, "sel", self.edge_map[key_edge][0], "original bw", self.get_bandwidth(self.edge_map[key_edge][0]), "num_edges", num_edges, "flux", key_edge[0].flux, "real bw" , real_bandwidth)
                
                end_time = min_start_time + duration
                if last_percent == 0:
                    end_time += latency
                if end_time < min_end_time:
                    min_end_time = end_time
                self.recorder.update(key_edge, CommunicationRecord(start_time, end_time, last_percent))
            
            # Calcaulate Deadline
            if extern_deadline is not None:
                deadline = min(min_end_time, second_min_start_time, extern_deadline)
            else:
                deadline = min(min_end_time, second_min_start_time)
            
            edges_cannot_proceed = set()


            for key_edge, neighbor_edges in overlap_groups.items():
                num_edges = len(neighbor_edges)
                # FIXME
                real_bandwidth = self.get_bandwidth(self.edge_map[key_edge][0]) / num_edges
                latency = self.latency * len(self.edge_map[key_edge])

                record = self.recorder[key_edge]
                if record.end_time > deadline:
                    if (deadline - latency <= min_start_time and
                        record.percent == 0):
                        record.end_time = min_start_time
                        record.percent = 0
                        edges_cannot_proceed.add(key_edge)
                    else:
                        record.end_time = deadline
                        if record.percent == 0:
                            record.percent += (deadline - min_start_time - latency) * real_bandwidth / key_edge[0].flux
                        else:
                            record.percent += (deadline - min_start_time) * real_bandwidth / key_edge[0].flux
                    edge_heap.put((record.end_time, key_edge))
                else:  # 当前edge处理完成
                    record.percent = 1
                    finished_edges.append(key_edge)
                    # 评估的结束时间
                    finish_time = record.end_time
                # 真正的评估结果
                self.recorder.update(key_edge, record)
            if self.all_edges_reach_deadline(edge_heap, extern_deadline,
                                             edges_cannot_proceed):
                break
        if extern_deadline is None:
            self.recorder.recorder_time = recorder
        else:
            if len(finished_edges) != 0:
                assert finish_time == extern_deadline
            else:
                finish_time = extern_deadline
        return finished_edges, finish_time


class CoreCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float,
                 mode: EvaluationMode = EvaluationMode.STATIC,
                 latency: float = 0) -> None:
        super().__init__(bandwidth, mode, latency=latency)

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, link_id: int):
        if (Coord(3) in [src, dst]) and (Coord(0) not in [src, dst]):
            self.append_hop(edge, iteration, Hop(src, Coord(0), link_id))
            self.append_hop(edge, iteration, Hop(Coord(0), dst, link_id))
        else:
            self.append_hop(edge, iteration, Hop(src, dst, link_id))

    def eval_area(self):
        return 0
    

class SharedMemoryCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float, shared_memory_coord: Coord, 
                 arbitrator_coord: Coord, process_node, size: Tuple[int] = None,
                 mode: EvaluationMode = EvaluationMode.STATIC,
                 latency: float = 0) -> None:
        super().__init__(bandwidth, process_node, mode, size, latency)
        self.shared_memory_coord = shared_memory_coord
        self.arbitrator_coord = arbitrator_coord

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, 
                      link_id: int):
        if src == dst or self.arbitrator_coord in [src, dst]:
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


class HybridCoreCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, bandwidth: float, process_node: ProcessNode, 
                 mode: EvaluationMode = EvaluationMode.STATIC, 
                 size: Tuple[int] = None, latency: float = 0) -> None:
        super().__init__(bandwidth, process_node, mode, size, latency)

    def generate_hops(self, edge: Edge, iteration: int, src: Coord, dst: Coord, 
                      link_id: int):
        # TODO: Take NoC & NoP Bandwidth into Consideration
        self.append_hop(edge, iteration, Hop(src, dst, link_id))


class BookSimCommunicationEvaluator(CommunicationEvaluator):
        
    def __init__(self, bandwidth: float, process_node: ProcessNode, 
                 mode: EvaluationMode, 
                 size: Tuple[int] = None, latency: float = 0) -> None:
        super().__init__(bandwidth, process_node, mode, size, latency)
        
        # BookSim Simulator API
        self.noc_simulator = BookSim2Sync()
        
        # The Booksim defaults to the first package being initialed in cycle 0.
        # So, self.noc_time_offset, as an offset that implements clock alignment, is needed.
        self.noc_time_offset = -1
        
        # Cache all the pkts that executed by the BookSim simulator but larger than the
        # current deadline (extern_deadline) in BiMap.
        self.noc_uncommitted_pkts = []
        
        # (src,dst) -> edge
        self.src_dst_2_edge_map = {}
    
    def copy_booksim_recorder(self):
        recorder = {}
        for key in self.recorder.recorder_time:
            record = self.recorder.recorder_time[key]
            recorder[key] = BooksimCommunicationRecord(record.start_time, record.end_time, record.recv_pkts, record.goal_pkts)
        return recorder
    
    def info_to_list(self, eject_info):
        num_list = [int(s) for s in eject_info.split() if s.isdigit()]
        assert len(num_list) % 4 == 0
        grounped_pkg_list = [num_list[i:i+4] for i in range(0,len(num_list),4)]
        grounped_pkg_list.sort(key=lambda x: x[1])
        return grounped_pkg_list
    
    def eval_by_execution(self, edge_heap: PriorityQueue, 
                      extern_deadline: float = None) -> Tuple[List[Edge], int]:
        
        """ HINT:
            Because of the edge design of MLDSE, we need to first pre-execute a simulation 
                to get the expected performance before giving the real performance. 
            In this process, (extern_deadline is None) marks the first pre-simulation process.
        """
        # Function Definition and Initialization
        finished_edges: List[Tuple[Edge, int]] = []
        finish_time = 0
        if extern_deadline is None:
            recorder = self.copy_booksim_recorder()
        else:
            recorder = None
        # - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        print("\n[eval_by_execution]", self.__class__.__name__, os.getpid())
        print("[edge_heap_init]", edge_heap.queue, "[recoder_init]", recorder)
        
        # Flits Injection
        for egde_item in edge_heap.queue:
            start_time = egde_item[0]
            edge = egde_item[1]
            if edge in self.recorder.recorder_time.keys():
                # The edge is recorded
                # print("-> [old]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
                pass
            elif extern_deadline is None: 
                # The edge is new and can inject, but not record
                # print("-> [new]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
                # Inject (HINT: Edge-grained Sending Now!)
                src_idx = self.edge_map[edge][0].src[0] * 8 + self.edge_map[edge][0].src[1]
                dst_idx = self.edge_map[edge][-1].dst[0] * 8 + self.edge_map[edge][-1].dst[1]
                pkt_size = edge[0].flux
                noc_bandwidth = self.get_bandwidth(self.edge_map[edge][0])
                flit_num = math.ceil(pkt_size / noc_bandwidth)
                for flit_idx in range(flit_num):
                    self.noc_simulator.inject(1, src_idx, dst_idx, 1)
                    # Print for Debugging
                    # if flit_idx == 0 or flit_idx == flit_num - 1:
                    #     print("-> [e2e] [{}]".format(flit_idx), src_idx, "->", dst_idx, pkt_size)
                # Update State & Recoder
                if self.noc_time_offset == -1: self.noc_time_offset = start_time
                self.src_dst_2_edge_map.update({(src_idx, dst_idx): edge})
            else:
                # Record the edge
                # print("-> [ejected]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
                src_idx = self.edge_map[edge][0].src[0] * 8 + self.edge_map[edge][0].src[1]
                dst_idx = self.edge_map[edge][-1].dst[0] * 8 + self.edge_map[edge][-1].dst[1]
                pkt_size = edge[0].flux
                noc_bandwidth = self.get_bandwidth(self.edge_map[edge][0])
                flit_num = math.ceil(pkt_size / noc_bandwidth)
                self.recorder.update(edge, BooksimCommunicationRecord(start_time, extern_deadline, 0, flit_num)) # -1 marks uncommitted edge
                
        # Fake Simulation to obtain the estimated final_time
        if extern_deadline is None:
            eject_info = self.noc_simulator.run_ahead()
            # print("-> [eject_info]", eject_info)
            
            grounped_pkg_list = self.info_to_list(eject_info)
            end_time_list = {}
            for [t_i, t_e, src, dst] in grounped_pkg_list:
                if (src, dst) not in end_time_list.keys():
                    end_time_list.update({(src, dst): t_e + self.noc_time_offset})
                else:
                    end_time_list.update({(src, dst): max(end_time_list[(src, dst)], t_e + self.noc_time_offset)})
            
            # print("-> [end_time_list]", end_time_list)
            # print("-> [src_dst_2_edge_map]", self.src_dst_2_edge_map)
            
            min_edge_time = 1e24
            min_edge = None
            for (src, dst), edge_time in end_time_list.items():
                if edge_time < min_edge_time:
                    min_edge_time = edge_time
                    min_edge = self.src_dst_2_edge_map[(src, dst)]
            
            finished_edges = [min_edge]
            finish_time = min_edge_time
        
        # Real Simulation
        while extern_deadline != None:
            
            # Process Unconmitted Pkts
            last_noc_uncommitted_pkts = copy.deepcopy(self.noc_uncommitted_pkts)
            self.noc_uncommitted_pkts = []
            for [t_i, t_e, src, dst] in last_noc_uncommitted_pkts:
                if t_e + self.noc_time_offset > extern_deadline:
                    self.noc_uncommitted_pkts.append([t_i, t_e, src, dst])
                else:
                    # print([t_i, t_e, src, dst], "executed @", t_e + self.noc_time_offset)
                    mapped_edge = self.src_dst_2_edge_map[(src, dst)]
                    mapped_record = self.recorder.recorder_time[mapped_edge]
                    mapped_record.recv_pkts += 1
                    if mapped_record.recv_pkts == mapped_record.goal_pkts:
                        finished_edges.append(mapped_edge)
                        finish_time = max(t_e + self.noc_time_offset, finish_time)
                        self.recorder.update(mapped_edge, BooksimCommunicationRecord(t_i, finish_time, mapped_record.recv_pkts, mapped_record.goal_pkts))
                    
            if len(self.noc_uncommitted_pkts) > 0: break
            
            # Next Execute & Get Unsaved Pkts
            self.noc_simulator.run_step()
            eject_info = self.noc_simulator.info
            
            # Extract all the number from the str eject_info
            grounped_pkg_list = self.info_to_list(eject_info)
            
            # State Update
            for [t_i, t_e, src, dst] in grounped_pkg_list:
                if t_e + self.noc_time_offset > extern_deadline:
                    # Need to run in the next round
                    self.noc_uncommitted_pkts.append([t_i, t_e, src, dst])
                else:
                    # Received a packet [Valid]
                    # print([t_i, t_e, src, dst], "executed @", t_e + self.noc_time_offset)
                    mapped_edge = self.src_dst_2_edge_map[(src, dst)]
                    mapped_record = self.recorder.recorder_time[mapped_edge]
                    mapped_record.recv_pkts += 1
                    if mapped_record.recv_pkts == mapped_record.goal_pkts:
                        finished_edges.append(mapped_edge)
                        finish_time = max(t_e + self.noc_time_offset, finish_time)
                        self.recorder.update(mapped_edge, BooksimCommunicationRecord(t_i, finish_time, mapped_record.recv_pkts, mapped_record.goal_pkts))
            
            # Logging cached pkts
            if len(self.noc_uncommitted_pkts) != 0:
                # print("[Cached Pkts]", self.noc_uncommitted_pkts)
                break
            
            # check if all the recorder are done
            end = True
            for key_edge_i, record_i in self.recorder.recorder_time.items():
                if record_i.recv_pkts != record_i.goal_pkts: end = False
            
            if end: break
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        total_edge = len(edge_heap.queue)
        for i in range(total_edge):
            _, lst_edge = edge_heap.get()
            if lst_edge not in finished_edges:
                edge_heap.put((finish_time, lst_edge))
        
        if extern_deadline is None:
            self.recorder.recorder_time = recorder
        else:
            if len(finished_edges) != 0:
                assert finish_time == extern_deadline
            else:
                finish_time = extern_deadline
        
        print("[record]", self.recorder.recorder_time)
        print("[finished_edges]", finished_edges)
        print("[finish_time]", finish_time)
        print("[edge_heap]", edge_heap.queue)
        
        return finished_edges, finish_time
    
    def eval_by_model(self, edge_heap: PriorityQueue, 
                      extern_deadline: float = None) -> Tuple[List[Edge], int]:
        
        # Function Definition and Initialization
        finished_edges: List[Tuple[Edge, int]] = []
        finish_time = 0
        if extern_deadline is None:
            recorder = self.copy_booksim_recorder()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        # print("\n[eval_by_model]", self.__class__.__name__, os.getpid())
        # print("[edge_heap_init]", edge_heap.queue)
        
        # Flits Injection
        for egde_item in edge_heap.queue:
            start_time = egde_item[0]
            edge = egde_item[1]
            if edge in self.recorder.recorder_time.keys():
                # The edge is recorded
                print("-> [old]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
            elif extern_deadline is None: 
                # The edge is new and can inject, but not record
                print("-> [new]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
            else:
                # Record the edge
                print("-> [ejected]", start_time, "-", extern_deadline, edge, self.edge_map[edge])
        
        # Goal: Simulate until one edge is finished or extern_deadline is reached
        while len(finished_edges) == 0:
            min_start_time, min_edge = edge_heap.get()
            if extern_deadline is not None:
                if min_start_time > extern_deadline:
                    # heapq.heappush(edge_heap, (min_start_time, min_edge))
                    edge_heap.put((min_start_time, min_edge))
                    return [], extern_deadline
            if not edge_heap.empty():
                second_min_start_time, second_min_edge = edge_heap.get()  # 第二先可以开始的边
            else:
                second_min_start_time = float('inf')
                second_min_edge = None
            min_edges = [min_edge]
            while second_min_start_time == min_start_time:
                min_edges.append(second_min_edge)
                if not edge_heap.empty():
                    second_min_start_time, second_min_edge = edge_heap.get()
                else:
                    break
            if second_min_start_time != min_start_time and second_min_edge is not None:
                edge_heap.put((second_min_start_time, second_min_edge))
            else:
                second_min_start_time = float('inf')
            overlap_groups = create_overlap_groups(min_edges, self.edge_map)
            
            min_end_time = float('inf')  # 当前轮次评估中最先结束的edge
            for key_edge, neighbor_edges in overlap_groups.items():
                num_edges = len(neighbor_edges)
                # FIXME
                real_bandwidth = self.get_bandwidth(self.edge_map[key_edge][0]) / num_edges
                latency = self.latency * len(self.edge_map[key_edge])

                if key_edge in self.recorder:
                    start_time = self.recorder[key_edge].start_time
                    last_percent = self.recorder[key_edge].recv_pkts / self.recorder[key_edge].goal_pkts
                else:
                    start_time = min_start_time
                    last_percent = 0
                duration = key_edge[0].flux * (1 - last_percent) / real_bandwidth
                
                end_time = min_start_time + duration
                if last_percent == 0:
                    end_time += latency
                if end_time < min_end_time:
                    min_end_time = end_time
                
                if key_edge in self.recorder:
                    recv_pkt = last_percent * self.recorder[key_edge].goal_pkts
                    goal_pkt = self.recorder[key_edge].goal_pkts
                else:
                    recv_pkt = 0
                    goal_pkt = key_edge[0].flux
                self.recorder.update(key_edge, BooksimCommunicationRecord(start_time, end_time, recv_pkt, goal_pkt))
            
            # Calcaulate Deadline
            if extern_deadline is not None:
                deadline = min(min_end_time, second_min_start_time, extern_deadline)
            else:
                deadline = min(min_end_time, second_min_start_time)
            
            edges_cannot_proceed = set()
            for key_edge, neighbor_edges in overlap_groups.items():
                num_edges = len(neighbor_edges)
                real_bandwidth = self.get_bandwidth(self.edge_map[key_edge][0]) / num_edges
                latency = self.latency * len(self.edge_map[key_edge])
                record = self.recorder[key_edge]
                if record.end_time > deadline:
                    if (deadline - latency <= min_start_time and
                        record.recv_pkts == 0):
                        record.end_time = min_start_time
                        record.recv_pkts = 0
                        record.goal_pkts = key_edge[0].flux
                        edges_cannot_proceed.add(key_edge)
                    else:
                        record.end_time = deadline
                        if record.recv_pkts == 0:
                            record.recv_pkts += (deadline - min_start_time - latency) * real_bandwidth
                        else:
                            record.recv_pkts += (deadline - min_start_time) * real_bandwidth
                    edge_heap.put((record.end_time, key_edge))
                else:
                    record.recv_pkts = record.goal_pkts
                    finished_edges.append(key_edge)
                    finish_time = record.end_time
                self.recorder.update(key_edge, record)
            if self.all_edges_reach_deadline(edge_heap, extern_deadline,
                                             edges_cannot_proceed):
                break
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if extern_deadline is None:
            self.recorder.recorder_time = recorder
        else:
            if len(finished_edges) != 0:
                assert finish_time == extern_deadline
            else:
                finish_time = extern_deadline
        print("[record]", self.recorder.recorder_time)
        print("[finished_edges]", finished_edges)
        print("[finish_time]", finish_time)
        print("[edge_heap]", edge_heap.queue)
        return finished_edges, finish_time

if __name__ == "__main__":
    a = Hop(Coord(0), Coord(1))
    b = Hop(Coord(0), Coord(1))
    d = NetworkParameterDict()
    d[a] = 1
    print(d[b])