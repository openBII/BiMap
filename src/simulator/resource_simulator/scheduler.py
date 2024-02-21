from collections import OrderedDict
from typing import Set
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.vtask_block import VTaskBlock
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.input_type import InputType


class Scheduler():
    def __init__(self, st_matrix: STMatrix, st_context: STContext, task_graph: TaskGraph, initial_tasks: Set[TaskBlock] = None) -> None:
        self._activated_task_id = OrderedDict()
        self.init(initial_tasks)
        self._activated_edges = []
        self._st_matrix = st_matrix
        self._st_context = st_context
        self._task_graph = task_graph

    def init(self, tasks: Set[TaskBlock]):
        for task in tasks:
            self._activated_task_id.update({task.id: False})
        
    def add_activated_tasks(self):
        finished_tasks = []
        for task_id in self._activated_task_id:
            if self._activated_task_id[task_id]:
                task = self._task_graph.get_node(task_id)
                if task.activated:
                    self._activated_task_id.update({task.id: False})
                else:
                    finished_tasks.append(task_id)
                for out_task in task.out_tasks:
                    if out_task.activated:
                        self._activated_task_id.update({out_task.id: False})
        for task_id in finished_tasks:
            del self._activated_task_id[task_id]

    def get_next_activated_task(self) -> int:
        return self._activated_task_id.pop()
    
    def add_activated_edges(self, task: TaskBlock):
        for edge in task.output_edges:
            if edge.input_activated:
                self._activated_edges.append(edge)

    def schedule(self, input_type: InputType):
        # 每次处理完所有激活的任务后处理边
        if input_type == InputType.BATCH:
            self.schedule_initial_tasks()
        self.schedule_edges()
        self.add_activated_tasks()
        while len(self._activated_task_id) != 0:
            self.schedule_tasks()
            self.schedule_edges()
            self.add_activated_tasks()

    def schedule_initial_tasks(self):
        for task_id in self._activated_task_id:
            ml_coord = self._st_context.get_ml_coord(task_id)
            st_point = self._st_matrix.get_element(ml_coord)
            while True:
                _, task = st_point.process(task_id)
                self.add_activated_edges(task)
                if not task.activated:
                    break
            self._activated_task_id.update({task_id: True})

    def schedule_tasks(self):
        # 对所有activated的任务不断遍历直到没有任务可以被完成
        while True:
            FLAG = False
            for task_id in self._activated_task_id:
                if not self._activated_task_id[task_id]:
                    ml_coord = self._st_context.get_ml_coord(task_id)
                    st_point = self._st_matrix.get_element(ml_coord)
                    if st_point is None:
                        raise Exception("st_point is None")
                    # 硬件处理当前任务
                    task = self._task_graph.get_node(task_id)
                    if isinstance(task, VTaskBlock):
                        ticks = task.consume()
                        task.fire(ticks)
                        state = True
                    else:
                        state, task = st_point.process(task_id)
                    self._activated_task_id.update({task_id: state})
                    if state:
                        FLAG = True
                        self.add_activated_edges(task)
            if not FLAG:
                break

    def schedule_edges(self):
        classified_edges = self.classify_edges()
        unfinished_edges = []
        for i, edge_dict in enumerate(classified_edges):
            if i == 0:
                if 0 in edge_dict:
                    unfinished_edges.extend(self._st_matrix.process(edge_dict[0]))
            else:
                for coord in edge_dict:
                    edges = edge_dict[coord]
                    hardware = self._st_matrix.get_element(coord)
                    unfinished_edges.extend(hardware.process(edges))
        self._activated_edges = unfinished_edges

    def classify_edges(self):
        '''
        将激活的边分成两级
        1. 边应该被哪个空间层次的硬件处理
        2. 边应该被当前空间层次的哪个硬件处理

        Returns:
        - classified_edges: [{0: [edge0, edge1, ...]}, {MLCoord0: [edge2, edge3, ...]; MLCoord1: [edge4, edge5, ...]}, ...]
        '''
        space_level = self._st_matrix._space_level
        classified_edges = [{} for _ in range(space_level)]
        for edge in self._activated_edges:
            first_coord = self._st_context.get_first_ml_coord(edge)
            level = first_coord.level
            if level == 1:
                if not 0 in classified_edges[0]:
                    classified_edges[0].update({0: [edge]})
                else:
                    classified_edges[0][0].append(edge)
            else:
                if not first_coord.outer_coord in classified_edges[level - 1]:
                    classified_edges[level - 1].update({first_coord.outer_coord: [edge]})
                else:
                    classified_edges[level - 1][first_coord.outer_coord].append(edge)
        return classified_edges
            