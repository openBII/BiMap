from __future__ import annotations
from typing import Set, Union, List, Dict, Tuple
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.computation_point import ComputationPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.vtask_block import VTaskBlock
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.st_model.st_coord import MLCoord
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.sync.sync_table import SyncTable
from copy import deepcopy


class Scheduler():
    def __init__(self, st_matrix: STMatrix, st_context: STContext, task_graph: TaskGraph, sync_table: SyncTable, initial_tasks: Set[TaskBlock] = None) -> None:
        self._activated_task_id = set()
        self.init(initial_tasks)
        self._activated_edges = []
        self._st_matrix = st_matrix
        self._st_context = st_context
        self._task_graph = task_graph
        self._sync_table = sync_table

    def init(self, tasks: Set[TaskBlock]):
        for task in tasks:
            if task.is_enable():
                self._activated_task_id.add(task.id)
        
    # def add_activated_tasks(self):
    #     new_activated_tasks = set()
    #     for task_id in self._activated_task_id:
    #         task = self._task_graph.get_node(task_id)
    #         if task.activated:
    #             new_activated_tasks.add(task_id)
    #         out_task: TaskBlock
    #         for out_task in task.out_tasks:
    #             if out_task.activated:
    #                 new_activated_tasks.add(out_task.id)
    #     self._activated_task_id = new_activated_tasks
    #     for task_id in self._activated_task_id:
    #         if self._activated_task_id[task_id]:
    #             task = self._task_graph.get_node(task_id)
    #             if task.activated:
    #                 self._activated_task_id.update({task.id: False})
    #             else:
    #                 finished_tasks.append(task_id)
    #             for out_task in task.out_tasks:
    #                 if out_task.activated:
    #                     self._activated_task_id.update({out_task.id: False})
    #     for task_id in finished_tasks:
    #         del self._activated_task_id[task_id]
    
    def add_activated_edges(self, task: TaskBlock):
        for edge in task.output_edges:
            if edge.input_activated and edge.is_enable():
                self._activated_edges.append(edge)

    def schedule(self):
        # 每次处理完所有激活的任务后处理边
        while (len(self._activated_task_id) != 0 or len(self._activated_edges) != 0):
            self.schedule_tasks()
            self.schedule_edges()
            self.remove_finished_tasks()

    def remove_finished_tasks(self):
        task_set = deepcopy(self._activated_task_id)
        for task_id in task_set:
            task = self._task_graph.get_node(task_id)
            if not task.activated:
                self._activated_task_id.remove(task_id)

    def schedule_tasks(self):
        # 对所有activated的任务不断遍历直到没有任务可以被完成
        flag = True
        while flag:
            flag = False
            for task_id in self._activated_task_id:
                # if not self._activated_task_id[task_id]:
                task = self._task_graph.get_node(task_id)
                if isinstance(task, VTaskBlock):
                    self.process_virtual_task(task)
                else:
                    ml_coord = self._st_context.get_ml_coord(task_id)
                    space_point = self._st_matrix.get_element(ml_coord)
                    # assert space_point is not None, "Space Point is None"
                    # 硬件处理当前任务的所有能被处理的iteration
                    flag = self.process_all_iterations(task, space_point)
                    # while True:
                    #     if isinstance(task, VTaskBlock):
                    #         ticks = task.consume()
                    #         task.fire(ticks)
                    #         state = True
                    #     else:
                    #         state, task = st_point.process(task_id)
                    #     # self._activated_task_id.update({task_id: state})
                    #     if state:
                    #         FLAG = True
                    #         self.add_activated_edges(task)
                    #     else:  # 当前任务后续iteration无法被处理
                    #         break
                    #     if not task.activated:  # 当前任务所有iteration均被处理完成
                    #         break
            # if not :
            #     break

    def process_virtual_task(self, task: VTaskBlock):
        if not task.activated:
            return
        task.transfer_ticks()
        self.add_activated_edges(task)

    def process_all_iterations(self, task: TaskBlock, space_point: Union[ComputationPoint, MemoryPoint]):
        flag = False
        state = True
        # state = False 当前任务后续iteration无法被处理
        # task.activated = False 当前任务所有iteration均被处理完成
        while state and task.activated:
            state = space_point.process(task.id, self._sync_table)
            if state:
                flag = True
                self.add_activated_edges(task)
        return flag

    def schedule_edges(self):
        # XXX(huanyu): 现在的实现方式是评估两次, 第一次获得最短时间
        min_finish_time = float("inf")
        classified_edges = self.classify_edges()
        for i, edge_dict in enumerate(classified_edges):
            if i == 0:
                network_id: int
                for network_id in edge_dict:
                    edges = edge_dict[network_id]
                    _, _, finish_time = self._st_matrix.communication_networks[network_id].process(edges)
                    min_finish_time = min(finish_time, min_finish_time)
            else:
                for network_coord in edge_dict:
                    container_coord, network_id = network_coord
                    edges = edge_dict[network_coord]
                    container: STMatrix = self._st_matrix.get_element(container_coord)
                    _, _, finish_time = container.communication_networks[network_id].process(edges)
                    min_finish_time = min(finish_time, min_finish_time)
        self.correct_edge_evaluation(classified_edges, min_finish_time)

    def correct_edge_evaluation(self, edges: List[Dict[Tuple[MLCoord, int] | int, List[Edge]]], time: float):
        unfinished_edges = []
        for i, edge_dict in enumerate(edges):
            if i == 0:
                network_id: int
                for network_id in edge_dict:
                    edges = edge_dict[network_id]
                    unfinished, finished, _ = self._st_matrix.communication_networks[network_id].process(edges, time)
                    self.add_activated_tasks(finished)
                    unfinished_edges.extend(unfinished)
            else:
                for network_coord in edge_dict:
                    container_coord, network_id = network_coord
                    edges = edge_dict[network_coord]
                    container: STMatrix = self._st_matrix.get_element(container_coord)
                    unfinished, finished, _ = container.communication_networks[network_id].process(edges, time)
                    self.add_activated_tasks(finished)
                    unfinished_edges.extend(unfinished)
        self._activated_edges = unfinished_edges

    def add_activated_tasks(self, edges: Edge):
        for edge in edges:
            out_task: TaskBlock = edge.out_task
            if out_task.activated and out_task.is_enable():
                self._activated_task_id.add(out_task.id)

    def classify_edges(self):
        '''
        将激活的边分成两级
        1. 边应该被哪个空间层次的硬件处理
        2. 边应该被当前空间层次的哪个硬件处理

        Returns:
        - classified_edges: [{0: [edge0, edge1, ...]}, {MLCoord0: [edge2, edge3, ...]; MLCoord1: [edge4, edge5, ...]}, ...]
        '''
        space_level = self._st_matrix._space_level
        classified_edges: List[Dict[Union[Tuple[MLCoord, int], int], List[Edge]]] = [{} for _ in range(space_level)]
        for edge in self._activated_edges:
            level = self._st_context.get_level(edge)
            network_id = self._st_context.get_network_id(edge)
            container_coord = self._st_context.get_container_coord(edge)
            network_coord = (container_coord, network_id)
            if level == 1:
                if not network_id in classified_edges[0]:
                    classified_edges[0].update({network_id: [edge]})
                else:
                    classified_edges[0][network_id].append(edge)
            else:
                if not network_coord in classified_edges[level - 1]:
                    classified_edges[level - 1].update({network_coord: [edge]})
                else:
                    classified_edges[level - 1][network_coord].append(edge)
        return classified_edges
