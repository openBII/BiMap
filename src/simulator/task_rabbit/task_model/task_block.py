# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from abc import ABC, abstractmethod
from copy import copy
from typing import List, Set, Tuple

from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.task_block_state import TaskState
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType



class TaskBlock(ABC):
    """
    TaskBlock类描述一个任务块
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        # 基础属性
        self._id = task_id
        self._type = None
        self._state = TaskState.ENABLE
        self._shape = shape
        self._precision = precision

        # 计算存储信息
        self._storage: int = None
        self._computation: int = None

        # 输入输出边信息
        self._input_edges: List[Edge] = []
        self._output_edges: List[Edge] = []

        # 构建任务结点信息
        self.construct()

    def construct(self) -> None:
        self._construct_computation()
        self._construct_storage()

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

        for edge in self._input_edges:
            if edge.in_task.is_enable():
                edge.enable()
        for edge in self._output_edges:
            if edge.out_task.is_enable():
                edge.enable()

    def disable(self) -> None:
        """
        将TaskBlock置于disable状态
        """
        self._state = TaskState.DISABLE

        for edge in self._input_edges:
            edge.disable()
        for edge in self._output_edges:
            edge.disable()

    # 形状相关的属性
    @property
    def shape(self) -> Shape:
        return self._shape

    @shape.setter
    def shape(self, shape: Shape) -> None:
        self._shape = shape
        self._check_shape()  # FIXME
        self._construct_storage()
        self._construct_computation()

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, task_id: int) -> None:
        self._id = task_id


    @property
    def task_type(self) -> TaskBlockType:
        return self._type

    @property
    def input_edges(self) -> List[Edge]:
        return self._input_edges

    @input_edges.setter
    def input_edges(self, value):
        # 尽量不要用
        self._input_edges = value

    @property
    def output_edges(self) -> List[Edge]:
        return self._output_edges

    @output_edges.setter
    def output_edges(self, value):
        # 尽量不要用
        self._output_edges = value

    @property
    def in_tasks(self) -> Set:
        """
        获得此任务块的输入任务块的集合
        """
        tasks = set()
        for edge in self._input_edges:
            for task in edge.in_task:
                tasks.add(task)
        return tasks

    @property
    def out_tasks(self) -> Set:
        """
        获得此任务块的输出任务块的集合
        """
        tasks = set()
        for edge in self._output_edges:
            tasks.add(edge.out_task)
        return tasks

    @property
    def activated(self):
        """If all the input edges of this task block are activated, this task block will be activated.
        """
        return all(map(lambda x: x.output_activated, self._input_edges))

    # @property
    # def all_in_tasks(self) -> Set:
    #     """
    #     包括disable的任务块
    #     """
    #     tasks = set()
    #     for cluster in self._input_edges:
    #         for edge in cluster.all_edges:
    #             tasks.add(edge.in_task)
    #     return tasks

    # @property
    # def all_out_tasks(self) -> Set:
    #     """
    #     包括disable的任务块
    #     """
    #     tasks = set()
    #     for cluster in self._output_edges:
    #         for edge in cluster.all_edges:
    #             tasks.add(edge.out_task)
    #     return tasks

    def add_input_edge(self, edge: Edge):
        self._input_edges.append(edge)

    def add_output_edge(self, edge: Edge):
        self._output_edges.append(edge)

    def remove_input_task(self, task_id: int) -> None:
        """
        删除此任务块中，task_id任务块到此任务块的连接
        此操作不会对task_id任务块进行任何操作
        但会销毁相应的连接
        此操作不能撤销，因此不应被st_env暴露给外部
        """
        remove_e = []
        # 任务块A与任务块B之间可能有多条边的连接
        for e in self._input_edges:
            if e.in_task.id == task_id:
                remove_e.append(e)
        for e in remove_e:
            self._input_edges.remove(e)
            e.destroy()

    def remove_output_task(self, task_id: int) -> None:
        """
        删除此任务块中，此任务块到task_id任务块的连接
        此操作不会对task_id任务块进行任何操作
        此操作不能撤销，因此不应被st_env暴露给外部
        """
        remove_e = []
        for e in self._output_edges:
            if e.out_task.id == task_id:
                remove_e.append(e)
        for e in remove_e:
            self._output_edges.remove(e)
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
        if edge in self._input_edges:
            self._input_edges.remove(edge)

    def remove_output_edge(self, edge) -> None:
        if edge in self._output_edges:
            self._output_edges.remove(edge)

    # Evaluation
    def get_storage(self) -> int:
        if self._state == TaskState.DISABLE:
            return 0
        if self._storage is None:
            self._construct_storage()
        return self._storage.local_storage

    def get_computation(self) -> int:
        return self._computation

    def get_image(self):
        return ImageTaskBlock(self._type)

    def fire(self):
        raise NotImplementedError

    def consume(self):
        raise NotImplementedError
        
    def destroy(self) -> None:
        '''
        销毁该任务块
        任务块相关的边也都会被销毁
        '''
        for edge in self._input_edges:
            edge.destroy()
        for edge in self._output_edges:
            edge.destroy()
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
