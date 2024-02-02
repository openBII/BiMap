#!/usr/bin/env python
# coding: utf-8

"""
StMatrix类是一种自定义的list，表示一个时空矩阵，保存STPoint
具体的说，在tianjicX架构中，表示一个Core model
"""
import heapq
from typing import Dict, List, Union
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.st_model.matrix_config import MatrixConfig
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.evaluation_model.evaluator import CommunicationEvaluator


# 为什么不选择直接继承dict的方式
class STMatrix():
    def __init__(self, dim, space_level=0, evaluator: CommunicationEvaluator = None):
        # {key: STMatrix} or {key: STPoint}
        # TODO: 改成OrderedDict
        self._container: Dict[MLCoord, Union[STMatrix, STPoint]] = {}

        self.dim = dim  # 本层次的维度
        self._space_level = space_level
        
        self.config = MatrixConfig()

        self._edge_map: Dict[Edge, List[MLCoord]] = {}
        self.evaluator = evaluator

    def __getitem__(self, ml_coord: MLCoord):
        """
        返回ml_coord指定位置的STMatrix或STPoint或STPoint中的item
        有点复杂，先不写了
        """
        pass
        # if space_coord.level == self._space_level:
        #     pass

    def get(self, ml_coord: MLCoord, item_id=None):
        """
        返回ml_coord指明的位置的item
        ml_coord指明的位置时空上需要精确到最细粒度
        """
        if self._space_level == 0:
            if not CoordinatesProperty.is_finest_level(ml_coord):
                raise ValueError('The MLCoord is not at the highest level.')

        top_coord = ml_coord.space_coord.top_coord
        if top_coord not in self._container:
            return None

        return self._container[top_coord].get(ml_coord.space_sub_level,
                                              item_id)

    def get_element(self, ml_coord: MLCoord):
        """
        返回ml_coord指明的位置的element
        """
        if ml_coord.top_coord not in self._container:
            return None
    
        if ml_coord.level == 1:
            return self._container[ml_coord.top_coord]

        return self._container[ml_coord.top_coord].get_element(ml_coord.inner_coord)

    def get_space(self, space_coord: Coord):
        """
        返回space_coord指定的空间结构
        如果其指定的是最细粒度空间结构，则返回STPoint
        如果space_coord不包含在STMatrix中，返回None
        """
        top_coord = space_coord.top_coord
        if len(space_coord) == 1:
            return self._container.get(top_coord)

        if top_coord not in self._container:
            return None

        return self._container[top_coord].get_space(space_coord.inner_coord)

    def get_time(self, time_coord: Coord):
        """
        返回time_coord指定的空间结构
        如果其指定的是最细粒度空间结构，则返回STPoint
        如果space_coord不包含在STMatrix中，返回None
        目前因为空间上的不同质性，这个玩意不好写，先不管了
        """
        pass

    def add_element(self, coord: Coord, element):
        """
        在本层次中coord指明的位置添加一个element
        """
        assert coord.dim == self.dim
        
        # 重复了请用replace接口
        if coord in self._container:
            raise ValueError('The element already exists.')

        self._container[coord] = element

    def add_multi_level_element(self, ml_coord: MLCoord, element):
        """
        在跨层次中coord指明的位置添加一个element
        """

        pass
        
    def add_task(self, ml_coord: MLCoord, task):
        """
        在最低层次中的位置加入一个task
        """
        if ml_coord.level != self._space_level:
            raise ValueError('The space level of the task is not the same as the space level of the matrix.')
        
        if ml_coord.top_coord not in self._container:
            raise ValueError('The coord is not in the matrix.')
        

        next_matrix = self._container[ml_coord.top_coord]
        # 递归调用
        if isinstance(next_matrix, STPoint):
            next_matrix.add_task(task)
        else:
            next_matrix.add_task(ml_coord.inner_coord, task)
        
    # def get_all(self, scope: STScope) -> List:
    #     '''
    #     找到该范围内的所有item
    #     '''
    #     pass

    # FIXME I can't just pop a space structure
    def pop(self, ml_coord: MLCoord, item_id=None):
        """
        清空在ml_coord指明的位置的item，并返回该item
        """
        top_coord = ml_coord.space_coord.top_coord
        if top_coord not in self._container:
            raise ValueError('The coordinate is not in the container.')

        coord = ml_coord.space_sub_level
        if not hasattr(coord, 'space_coord') and len(coord) == 0:
            return self._container.pop(top_coord)
        return self._container[top_coord].pop(coord, item_id)

    def exist(self, ml_coord: MLCoord):
        """
        判断ml_coord指定的位置该位置是否存在
        不着急实现
        """
        pass

    def create(self, ml_coord: MLCoord):
        """
        创建ml_coord指定的位置
        不着急实现
        """
        pass

    def is_empty(self, ml_coord: MLCoord):
        """
        判断scope指定位置是否为空
        ml_coord指明的位置时空上需要精确到最细粒度
        """
        top_coord = ml_coord.space_coord.top_coord
        if top_coord not in self._container:
            return True
        return self._container[top_coord].is_empty(ml_coord.space_sub_level)

    def remove_space(self, space_coord):
        """
        在STMatrix中删除space_coord指定的空间
        不着急实现
        """
        pass

    def compress(self):
        '''
        压缩空的空间和时间，不着急实现
        '''
        pass

    @property
    def container(self):
        return self._container

    def accept(self, visitor):
        visitor.visit_matrix(self)

    def __iter__(self):
        for key, item in self._container.items():
            yield key, item

    @property
    def edge_map(self):
        return self._edge_map

    def add_edge(self, edge: Edge, path: List[MLCoord]):
        coord_level = path[0].level
        if coord_level == 1:
            self._edge_map.update({edge: path})
        else:
            inner_matrix = self._container[path[0].top_coord]
            inner_path = []
            for coord in path:
                inner_path.append(coord.inner_coord)
            inner_matrix.add_edge(edge, inner_path)

    def process(self, edges: List[Edge]) -> List[Edge]:
        # 每个iteration会将self.evaluator.edge_map中的hop信息消耗掉
        # XXX(huanyu): 可以考虑用循环队列
        if len(self.evaluator.edge_map) == 0:
            self.evaluator.create_all_edge_map(self._edge_map)
        for edge in edges:
            if self.evaluator.edge_not_mapped(edge):
                self.evaluator.create_edge_map(edge, self.edge_map[edge])
        start_time_heap = []
        tick_dict = {}
        for edge in edges:
            tick = edge.consume_tick()
            tick_dict.update({edge: tick})
            heapq.heappush(start_time_heap, (tick.time, (edge, tick.iteration)))
        finished_edges, finish_time = self.evaluator.eval(start_time_heap)
        for edge in edges:
            tick = tick_dict[edge]
            edge.fire(tick, finish_time)
        for edge in finished_edges:
            edges.remove(edge)
        return edges


# st_matrix = STMatrix()
# s_coord = Coord((1, 2, 3, 4))
# t_coord = Coord((4, 3, 2))
# ml_coord = MLCoord(s_coord, t_coord)
# st_matrix.add_element(ml_coord, 'haha')
# print(st_matrix.is_empty(ml_coord))
# print(st_matrix.is_empty(MLCoord((1, 2, 3, 4), (4, ))))
# print(st_matrix.get_space(s_coord))
# print(st_matrix.get(ml_coord, 'haha'))
# ml_coord2 = MLCoord(s_coord, Coord((4, 3)), False)
# print(st_matrix.pop(ml_coord2))

# from resource_simulator.st_model.st_point import STPoint
# from task_rabbit.task_model import TaskBlockType, TaskBlock
# from src.data_structure.shape import Shape

# shape = Shape(256, 256, 64, 3, 3, 3)
# position = Shape()
# task_block1 = TaskBlock(TaskBlockType.CLIF, shape, position)
# task_block2 = TaskBlock(TaskBlockType.CC, shape, position)
# task_block3 = TaskBlock(TaskBlockType.CADD, shape, position)
