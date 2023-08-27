# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from typing import Dict, Set

from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.shape import Shape


class EdgeInfo():
    """
    EdgeInfo类表示边簇中一个对应具体边的接口
    """

    def __init__(self, position=Shape(), size=Shape()):
        self.position = position  # type: Shape
        self.size = size          # type: Shape


class EdgeCluster():
    '''
    描述任务块的边簇
    采用简化冗余shape的存储方式
    即如果边簇的shape与其相应的任务块形状总是一致
    则边簇内部记录的形状为None, 外部获取边簇的形状时，直接返回任务块形状
    '''

    def __init__(self, shape: Shape = None):
        # shape: Shape
        # edges: Dict[Edge:EdgeInfo]

        self._shape: Shape = shape  # 边簇的形状
        # 如果Edge传输的数据是边簇所有的数据
        # 则为了节省操作代价，EdgeInfo设为None
        self._edges: Dict[Edge, EdgeInfo] = {}  # 边簇包含的边

    @property
    def shape(self) -> Shape:
        return self._shape

    @shape.setter
    def shape(self, shape: Shape):
        self._shape = shape

    @property
    def in_tasks(self) -> Set:
        return {e.in_task for e in self._edges.keys() if e.is_enable()}

    @property
    def out_tasks(self) -> Set:
        return {e.out_task for e in self._edges.keys() if e.is_enable()}

    @property
    def all_enable_edges(self):
        return [e for e in self._edges.keys() if e.is_enable()]

    @property
    def all_edges(self):
        return self._edges.keys()

    def get_edge_position(self, edge):
        assert edge.is_enable()
        if self._edges[edge].position is None:
            return Shape()
        return self._edges[edge].position

    def get_edge_size(self, edge):
        assert edge.is_enable()
        if self._edges[edge].size is None:
            return self.shape
        return self._edges[edge].size

    def add_edge(self, position: Shape, size: Shape, edge: Edge) -> None:
        assert edge.is_enable()
        one_edge = EdgeInfo(position, size)
        self._edges[edge] = one_edge

    def remove(self, edge: Edge) -> EdgeInfo:
        return self._edges.pop(edge)

    def destroy(self) -> None:
        for e in self._edges.keys():
            e.destroy()
        self._edges.clear()

    def check_shape(self) -> None:
        '''
        检查边簇里每条边的形状信息
        raises:
            ValueError: 当形状不满足一些约束的时候，返回异常
        '''
        for e, edge_info in self._edges.items():
            if not e.is_enable():
                continue
            # 如果为None，则意味着省略记录了，不用检查
            if edge_info.size is not None and edge_info.position is not None \
                    and not self.shape >= edge_info.size + edge_info.position:
                raise ValueError(
                    'Edge\'s range shouldn\'t over the cluster\'s shape!')

    def check_filled_properly(self) -> int:
        '''
        检查边簇中的所有边是否恰好填充边簇的总形状
        returns:
            0: 表示刚好满了
            1: 表示满了且有一些重叠
            2: 表示没满
        raises:
            ValueError: 当形状不满足一些约束的时候，返回异常
        '''
        self.check_shape()
        checked = set()
        for k_1 in self.all_enable_edges:
            checked.add(k_1)
            for k_2 in self.all_enable_edges:
                if k_2 in checked:
                    continue
                if self.check_overlap(k_1, k_2):
                    return 1  # overlap
        if self.check_full():
            return 0  # just full
        return 2  # not full

    def check_overlap(self, edge_1, edge_2):
        '''
        检查边簇中的所有边表示的区域是否有重叠
        '''
        e_1_position = self.get_edge_position(edge_1)
        e_1_size = self.get_edge_size(edge_1)
        e_2_position = self.get_edge_position(edge_2)
        e_2_size = self.get_edge_size(edge_2)

        s1 = Shape.max(e_1_position, e_2_position)
        s2 = Shape.min(
            e_1_position + e_1_size, e_2_position + e_2_size)

        return Shape.max(e_1_position, e_2_position) < Shape.min(
            e_1_position + e_1_size, e_2_position + e_2_size)

    def check_full(self):
        '''
        检查边簇中的所有边是否能完全覆盖边簇代表的形状
        '''
        e_volume = self.shape.volume
        for k in self.all_enable_edges:
            e_volume -= self.get_edge_size(k).volume
        return e_volume == 0

    def __contains__(self, edge) -> bool:
        return edge in self._edges

    def __getitem__(self, edge) -> EdgeInfo:
        return self._edges[edge]
