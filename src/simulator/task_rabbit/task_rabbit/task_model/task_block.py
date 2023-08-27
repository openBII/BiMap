# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from abc import ABC, abstractmethod
from copy import copy
from typing import List, Set, Tuple

from src.simulator.task_rabbit.task_model.computation import Computation
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.edge_cluster import EdgeCluster
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.shape_constraints import \
    ShapeConstraints
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block_state import TaskState
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class TaskBlock(ABC):
    """
    TaskBlock类描述一个任务块
    """

    def __init__(self, task_id: int, shape: Shape, task_type: TaskBlockType, precision: Precision):
        # 基础属性
        self._id = task_id
        self._type = task_type
        self._state = TaskState.ENABLE
        self._shape = shape
        self._precision = precision

        # 计算存储信息
        self._storage: Storage = None
        self._computation: Computation = None

        # 输入输出边信息
        self._input_clusters: List[EdgeCluster] = []
        self._output_clusters: List[EdgeCluster] = []

        # 构建任务结点信息
        self.construct()

        # 检查Task Block信息是否合法
        self.check()

    def construct(self) -> None:
        self._construct_clusters()
        self._construct_computation()
        self._construct_storage()

    @abstractmethod
    def _construct_clusters(self) -> None:
        """
        构建边簇，由子类实现
        """
        raise NotImplementedError

    @abstractmethod
    def _construct_computation(self) -> None:
        """
        构建计算信息，由子类实现
        """
        raise NotImplementedError

    @abstractmethod
    def _construct_storage(self) -> None:
        """
        构建存储信息，由子类实现
        """
        raise NotImplementedError

    def check(self) -> None:
        """
        检查TaskBlock是否满足一些规则约束，包括检查自身的形状、检查边簇的形状
        """
        self._check_shape()
        self._check_clusters_shape()

    def _check_shape(self) -> None:
        """
        检查TaskBlock自身的形状是否满足约束
        TODO: 在这一层判断并抛出异常
        """
        self._shape.check(ShapeConstraints.ShapeConstraint[self._type])

    def _check_clusters_shape(self) -> None:
        """
        检查TaskBlock检查边簇的形状，由子类实现
        """
        raise NotImplementedError

    def copy_like(self):
        # TODO ???
        return self.copy_like()

    def is_enable(self) -> bool:
        """
        判断TaskBlock是不是处在enable状态
        """
        return self._state == TaskState.ENABLE

    def enable(self) -> None:
        """
        将TaskBlock置于enable状态
        """
        self._state = TaskState.ENABLE

        for edge_cluster in self._input_clusters:
            for e in edge_cluster.all_edges:
                if e.in_task.is_enable():
                    e.enable()
        for edge_cluster in self._output_clusters:
            for e in edge_cluster.all_edges:
                if e.out_task.is_enable():
                    e.enable()

    def disable(self) -> None:
        """
        将TaskBlock置于disable状态
        """
        self._state = TaskState.DISABLE

        for e in self.get_input_edges():
            e.disable()
        for e in self.get_output_edges():
            e.disable()

    # 形状相关的属性
    @property
    def shape(self) -> Shape:
        return self._shape

    @shape.setter
    def shape(self, shape: Shape) -> None:
        self._shape = shape
        self._check_shape()
        self._construct_storage()
        self._construct_computation()

    @property
    def precision(self) -> Precision:
        return self._precision

    # ID
    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, task_id: int) -> None:
        self._id = task_id

    # 输入输出边相关操作
    def set_input_shape(self, index: int, shape: Shape) -> None:
        """
        设置第index个输入边簇的形状
        """
        if not index < len(self._input_clusters):
            raise IndexError(
                'Index of {:d} is out of input clusters\' range'.format(index))
        self._input_clusters[index].shape = shape
        self._check_clusters_shape()

    def set_output_shape(self, index: int, shape: Shape) -> None:
        if not index < len(self._output_clusters):
            raise IndexError(
                'Index of {:d} is out of output clusters\' range'.format(
                    index))
        self._output_clusters[index].shape = shape
        self._check_clusters_shape()

    @property
    def task_type(self) -> TaskBlockType:
        return self._type

    # edges
    @property
    def input_clusters(self) -> List[EdgeCluster]:
        return self._input_clusters

    @input_clusters.setter
    def input_clusters(self, value):
        # 尽量不要用
        self._input_clusters = value

    @property
    def output_clusters(self) -> List[EdgeCluster]:
        return self._output_clusters

    @output_clusters.setter
    def output_clusters(self, value):
        # 尽量不要用
        self._output_clusters = value

    @property
    def in_tasks(self) -> Set:
        """
        获得此任务块的输入任务块的集合
        """
        tasks = set()
        for cluster in self._input_clusters:
            for task in cluster.in_tasks:
                tasks.add(task)
        return tasks

    @property
    def out_tasks(self) -> Set:
        """
        获得此任务块的输出任务块的集合
        """
        tasks = set()
        for cluster in self._output_clusters:
            for task in cluster.out_tasks:
                tasks.add(task)
        return tasks

    @property
    def all_in_tasks(self) -> Set:
        """
        包括disable的任务块
        """
        tasks = set()
        for cluster in self._input_clusters:
            for edge in cluster.all_edges:
                tasks.add(edge.in_task)
        return tasks

    @property
    def all_out_tasks(self) -> Set:
        """
        包括disable的任务块
        """
        tasks = set()
        for cluster in self._output_clusters:
            for edge in cluster.all_edges:
                tasks.add(edge.out_task)
        return tasks

    def add_input_edge(self, edge: Edge, index: int = 0, position: Shape = None,
                       size: Shape = None):
        assert index < len(self._input_clusters)
        if position is None:
            position = Shape()
        if size is None:
            size = self._input_clusters[index].shape
        self._input_clusters[index].add_edge(
            copy(position), copy(size), edge)

    def add_output_edge(self, edge: Edge, index: int = 0, position: Shape = None,
                        size: Shape = None):
        assert (index < len(self._output_clusters))
        if position is None:
            position = Shape()
        if size is None:
            size = self._output_clusters[index].shape
        self._output_clusters[index].add_edge(
            copy(position), copy(size), edge)

    def remove_input_task(self, task_id: int) -> None:
        """
        删除此任务块中，task_id任务块到此任务块的连接
        此操作不会对task_id任务块进行任何操作
        但会销毁相应的连接
        此操作不能撤销，因此不应被st_env暴露给外部
        """
        remove_e = []
        # 任务块A与任务块B之间可能有多条边的连接
        for i, cluster in enumerate(self._input_clusters):
            for e in cluster.all_edges:
                if e.in_task.id == task_id:
                    remove_e.append((i, e))
        for i, e in remove_e:
            self._input_clusters[i].remove(e)
            e.destroy()

    def remove_output_task(self, task_id: int) -> None:
        """
        删除此任务块中，此任务块到task_id任务块的连接
        此操作不会对task_id任务块进行任何操作
        此操作不能撤销，因此不应被st_env暴露给外部
        """
        remove_e = []
        for i, cluster in enumerate(self._output_clusters):
            for e in cluster.all_edges:
                if e.out_task.id == task_id:
                    remove_e.append((i, e))
        for i, e in remove_e:
            self._output_clusters[i].remove(e)
            e.destroy()

    def remove_task(self, task_id: int) -> None:
        """
        删除此任务块中，与task_id相关的任务块的连接
        此操作不会对task_id任务块进行任何操作
        此操作不能撤销，因此不应被st_env暴露给外部
        """
        self.remove_input_task(task_id)
        self.remove_output_task(task_id)

    def remove_input_edge(self, edge) -> None:
        for cluster in self._input_clusters:
            if edge in cluster:
                cluster.remove(edge)
                return

    def remove_output_edge(self, edge) -> None:
        for cluster in self._output_clusters:
            if edge in cluster:
                cluster.remove(edge)
                return

    def get_input_edges(self, index: int = None) -> List[Edge]:
        edges = set()
        if index is not None:
            return [e for e in self._input_clusters[index].all_enable_edges]
        else:
            edges = []
            for edge_cluster in self._input_clusters:
                edges.extend([e for e in edge_cluster.all_enable_edges])
            return edges

    def get_output_edges(self, index: int = None) -> List[Edge]:
        edges = set()
        if index is not None:
            return [e for e in self._output_clusters[index].all_enable_edges]
        else:
            edges = []
            for edge_cluster in self._output_clusters:
                edges.extend([e for e in edge_cluster.all_enable_edges])
            return edges

    def get_input_edge_position(self, edge) -> Shape:
        for cluster in self._input_clusters:
            if edge in cluster:
                return cluster.get_edge_position(edge)

    def get_input_edge_size(self, edge) -> Shape:
        for cluster in self._input_clusters:
            if edge in cluster:
                return cluster.get_edge_size(edge)

    def get_output_edge_position(self, edge) -> Shape:
        for cluster in self._output_clusters:
            if edge in cluster:
                return cluster.get_edge_position(edge)

    def get_output_edge_size(self, edge) -> Shape:
        for cluster in self._output_clusters:
            if edge in cluster:
                return cluster.get_edge_size(edge)

    def get_input_edge_cluster(self, edge) -> Tuple[int, EdgeCluster]:
        for index, cluster in enumerate(self._input_clusters):
            if edge in cluster:
                return index, cluster

    def get_output_edge_cluster(self, edge) -> Tuple[int, EdgeCluster]:
        for index, cluster in enumerate(self._output_clusters):
            if edge in cluster:
                return index, cluster

    # Evaluation
    def get_storage(self) -> int:
        if self._state == TaskState.DISABLE:
            return 0
        if self._storage is None:
            self._construct_storage()
        return self._storage.local_storage

    def get_computation(self) -> int:
        return self._computation.total_computation

    def get_image(self):
        return ImageTaskBlock(self._type)

    def destroy(self) -> None:
        '''
        销毁该任务块
        任务块相关的边也都会被销毁
        '''
        for cluster in self._input_clusters:
            cluster.destroy()
        for cluster in self._output_clusters:
            cluster.destroy()
        self._storage = None
        self._computation = None
        self._state = TaskState.DISABLE

    def accept(self, visitor):
        '''
        为访问者模式留下的接口
        '''
        pass

    def __str__(self):
        return str(self._id)

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if other is None:
            return False
        return self._id == other.id


class ImageTaskBlock(TaskBlock):
    pass
