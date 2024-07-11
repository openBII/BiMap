#!/usr/bin/env python
# coding: utf-8

"""
ActionModel 类负责各种动作的响应
"""

from copy import deepcopy
import logging
from typing import List, Union, Tuple, Iterable
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.vtask_block import VTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.action_model.splitter import Splitter, SplitType
from src.simulator.resource_simulator.action_model.replicater import Replicater
from src.simulator.resource_simulator.action_model.column_merger import ColumnMerger
from src.simulator.resource_simulator.action_model.column_deleter import ColumnDeleter
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord, PathCoord
from src.simulator.resource_simulator.sync.sync_task import SyncTask
from src.simulator.resource_simulator.sync.sync_table import SyncTable
from src.simulator.task_rabbit.task_model.precision import Precision


class ActionModel():
    def __init__(self, task_graph: TaskGraph, st_matrix: STMatrix, st_context: STContext, sync_table: SyncTable):
        self._task_graph = task_graph
        self._st_matrix = st_matrix
        self._context = st_context
        self._sync_table = sync_table

    def split_task(self, task_id: int, split_vector: SplitVector, is_static: bool = False, record: bool = True):
        task = self._task_graph[task_id]
        split_tasks: List[TaskBlock] = []
        for _ in range(split_vector.num_slices):
            if isinstance(task, StaticTaskBlock):
                new_task: TaskBlock = task.copy_like(split_vector.get_slice_shape(task.shape), is_static)
            else:
                new_task: TaskBlock = task.copy_like(split_vector.get_slice_shape(task.shape))
            if record:
                self._task_graph.add_node(new_task)
            split_tasks.append(new_task)
        return split_tasks
    
    def connect_tasks(self, in_tasks: Iterable[TaskBlock], out_tasks: Iterable[TaskBlock]):
        if len(in_tasks) > len(out_tasks):
            for i, in_task in enumerate(in_tasks):
                self._task_graph.connect(in_task.id, out_tasks[i % len(out_tasks)].id)
        else:
            for i, out_task in enumerate(out_tasks):
                self._task_graph.connect(in_tasks[i % len(in_tasks)].id, out_task.id)

    def _delete_edges(self, in_tasks: Iterable[TaskBlock], out_tasks: Iterable[TaskBlock]):
        for in_task in in_tasks:
            useless_edges = []
            for out_edge in in_task.output_edges:
                out_task: TaskBlock = out_edge.out_task
                if out_task in out_tasks:
                    out_task.input_edges.remove(out_edge)
                    useless_edges.append(out_edge)
            for edge in useless_edges:
                in_task.output_edges.remove(edge)
                del edge

    def copy_task(self, task_id: int, num: int = 1, is_static: bool = False):
        task = self._task_graph[task_id]
        copied_tasks: List[TaskBlock] = []
        for _ in range(num):
            if isinstance(task, StaticTaskBlock):
                new_task: TaskBlock = task.copy_like(is_static=is_static)
            else:
                new_task: TaskBlock = task.copy_like()
            self._task_graph.add_node(new_task)
            copied_tasks.append(new_task)
        if num == 1:
            return new_task
        return copied_tasks
    
    def split_and_copy_task(self, task_id: int, split_vector: SplitVector, num: int = 1, is_static: bool = False) -> List[TaskBlock]:
        split_tasks: List[TaskBlock] = self.split_task(task_id, split_vector, is_static)
        copied_tasks = []
        for task in split_tasks:
            if num == 1:
                copied_tasks.append(self.copy_task(task.id, num, is_static))
            else:
                copied_tasks.extend(self.copy_task(task.id, num, is_static))
        split_tasks.extend(copied_tasks)
        assert len(split_tasks) == split_vector.num_slices * (1 + num)
        return split_tasks
    
    def reverse_split(self, new_tasks:Union[List[List[TaskBlock]], List[TaskBlock]], old_tasks: List[TaskBlock]) -> None:
        '''
        split_task的逆操作
        将new_tasks删除
        将old_tasks置为enable
        '''
        if isinstance(new_tasks[0], TaskBlock):
            self.delete_tasks(new_tasks)
        else:
            for tasks in new_tasks:
                self.delete_tasks(tasks)

        self.enable_tasks(old_tasks)
    
    def split_mlp(self, input: STaskBlock, split_inputs: List[STaskBlock], weight: StaticTaskBlock, compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector, precision: Precision = None):
        new_tasks = []

        weight_on_chip: TaskBlock = self.copy_task(weight.id)
        split_weight = self.split_task(weight_on_chip.id, SplitVector(nf=split_vector.nf, nr=split_vector.nr))
        self.connect_tasks([weight], split_weight)

        split_mlp = self.split_task(compute.id, SplitVector(nf=split_vector.nf, nr=split_vector.nr))
        self.connect_tasks(split_weight, split_mlp)

        if len(split_inputs) >= split_vector.nr:
            assert len(split_inputs) % split_vector.nr == 0
            num_grouped_inputs = len(split_inputs) // split_vector.nr
            for j in range(split_vector.nr):
                for i in range(split_vector.nf):
                    self.connect_tasks(split_inputs[num_grouped_inputs * j:num_grouped_inputs * (j + 1)], [split_mlp[i + j * split_vector.nf]])
        else:
            assert split_vector.nr % len(split_inputs) == 0
            num_split = split_vector.nr // len(split_inputs)
            new_split_inputs = []
            last_compute = []
            for task in split_inputs:
                last_compute.extend(list(task.in_tasks))
                new_split_inputs.extend(self.split_task(task.id, SplitVector(nr=num_split)))
                # self.delete_task(task.id)
                self.disable_task(task.id)
            self.connect_tasks(last_compute, new_split_inputs)
            for i in range(split_vector.nf):
                self.connect_tasks(new_split_inputs, split_mlp[split_vector.nr * i:split_vector.nr * (i + 1)])
            new_tasks.append(new_split_inputs)

        split_mlp_output: List[TaskBlock] = self.split_and_copy_task(output.id, SplitVector(nf=split_vector.nf), num=split_vector.nr - 1)
        self.connect_tasks(split_mlp, split_mlp_output)
        new_tasks.append(split_weight)
        new_tasks.append(split_mlp)
        new_tasks.append(split_mlp_output)
        
        if split_vector.nr > 1:
            add_tasks: List[TaskBlock] = []
            for _ in range(split_vector.nr):
                add_task = CTaskBlock(IDGenerator.get_next_task_id(), Shape(nf=split_mlp_output[0].shape.nf, branch=split_vector.nr), TaskBlockType.CADD, precision if precision is not None else compute.precision)
                self._task_graph.add_node(add_task)
                add_tasks.append(add_task)
            self.connect_tasks(split_mlp_output, add_tasks)

            split_add_output = self.split_task(output.id, SplitVector(nf=split_vector.nf))
            self.connect_tasks(add_tasks, split_add_output)
            new_tasks.append(add_tasks)
            new_tasks.append(split_add_output)

        # self.delete_tasks([input, compute, weight_on_chip])
        self.delete_task(weight_on_chip.id)
        self.disable_tasks([input, compute, output])

        if len(output.out_tasks) != 0:
            if split_vector.nr > 1:
                for out_task in output.out_tasks:
                    self.connect_tasks(split_add_output, [out_task])
            else:
                for out_task in output.out_tasks:
                    self.connect_tasks(split_mlp_output, [out_task])

        # return new_split_inputs, split_weight, split_mlp, split_mlp_output, add_tasks, split_add_output
        return new_tasks
    
    def split_pointwise(self, input: STaskBlock, split_inputs: List[STaskBlock], compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector):
        new_tasks = []

        split_compute = self.split_task(compute.id, split_vector)

        if len(split_inputs) >= split_vector.nf:
            assert len(split_inputs) % split_vector.nf == 0
            self.connect_tasks(split_inputs, split_compute)
        else:
            assert split_vector.nf % len(split_inputs) == 0
            num_split = split_vector.nf // len(split_inputs)
            new_split_inputs = []
            last_compute = []
            for task in split_inputs:
                last_compute.extend(list(task.in_tasks))
                new_split_inputs.extend(self.split_task(task.id, SplitVector(nf=num_split)))
                # self.delete_task(task.id)
                self.disable_task(task.id)
            self.connect_tasks(last_compute, new_split_inputs)
            self.connect_tasks(new_split_inputs, split_compute)
            new_tasks.append(new_split_inputs)

        if output.shape.nr != 0:
            split_output = self.split_task(output.id, SplitVector(nx=split_vector.nx, ny=split_vector.ny, nr=split_vector.nf))
        else:
            split_output = self.split_task(output.id, split_vector)
        self.connect_tasks(split_compute, split_output)
        new_tasks.append(split_compute)
        new_tasks.append(split_output)

        if len(output.out_tasks) != 0:
            for out_task in output.out_tasks:
                self.connect_tasks(split_output, [out_task])

        # self.delete_tasks([compute, input])
        self.disable_tasks([compute, input, output])

        return new_tasks

    def split_reduction(self, input: STaskBlock, split_inputs: List[STaskBlock], compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector, precision: Precision = None):
        new_split_inputs, split_compute, split_output = self.split_pointwise(input, split_inputs, compute, output, split_vector)

        for task in split_output:
            task.shape.nx = 1 if task.shape.nx != 0 else 0
            task.shape.ny = 1 if task.shape.ny != 0 else 0
            if output.shape.nr != 0:
                task.shape.nr = 1
            else:
                task.shape.nf = 1

        add_task = CTaskBlock(IDGenerator.get_next_task_id(), Shape(nf=1, branch=split_vector.nf), TaskBlockType.CADD, precision if precision is not None else compute.precision)
        self._task_graph.add_node(add_task)
        self.connect_tasks(split_output, [add_task])
        add_output = self.copy_task(split_output[0].id)
        self.connect(add_task, add_output)

        return new_split_inputs, split_compute, split_output, add_task, add_output

    # def split_task(self, task_id, split_vector: Shape, split_funcs: List[SplitType]):
    #     if split_vector == 1:
    #         return  # 不需要拆分
    #     original_task = deepcopy(self._task_graph.get_node(task_id))
    #     if split_vector.nky != 1 or split_vector.nkx != 1:
    #         logging.warn(
    #             'Values in ky and kx dimensions of split vector will be ignored')
    #         split_vector.nky = 1
    #         split_vector.nkx = 1
    #     original_task_shape = original_task.shape
    #     if original_task_shape.ny == 1 or original_task_shape.ny == -1:
    #         if split_vector.ny != 1:
    #             logging.warn(
    #                 'Value in y dimension of split vector will be ignored')
    #             split_vector.ny = 1
    #     if original_task_shape.nx == 1 or original_task_shape.nx == -1:
    #         if split_vector.nx != 1:
    #             logging.warn(
    #                 'Value in x dimension of split vector will be ignored')
    #             split_vector.nx = 1
    #     if original_task_shape.nf == 1 or original_task_shape.nf == -1:
    #         if split_vector.ny != 1:
    #             logging.warn(
    #                 'Value in f dimension of split vector will be ignored')
    #             split_vector.nf = 1
    #     if original_task_shape.nr == 1 or original_task_shape.nr == -1:
    #         if split_vector.nr != 1:
    #             logging.warn(
    #                 'Value in r dimension of split vector will be ignored')
    #             split_vector.nr = 1
    #     new_node_ids = []
    #     if original_task.task_type in (TaskBlockType.CC, TaskBlockType.CC2D):
    #         # 对CCTaskBlock进行拆分
    #         splitter = Splitter(task_id, split_vector,
    #                             split_funcs, self._task_graph)
    #         splitter.split_node()
    #         new_node_ids_after_c_split = splitter.id_list
    #         new_node_ids.extend(splitter.id_list)
    #         # 生成SICTaskBlock的split intervals
    #         cc_split_intervals = splitter.result
    #         si_split_intervals = [[(1, 0)] for _ in range(6)]
    #         si_split_intervals[0] = deepcopy(cc_split_intervals[4])  # y = iy
    #         si_split_intervals[1] = deepcopy(cc_split_intervals[5])  # x = ix
    #         si_split_intervals[3] = deepcopy(cc_split_intervals[3])  # r = r
    #         # 这里是为了处理copy_node_without_edge中的zip
    #         si_split_intervals[4] = si_split_intervals[0]
    #         si_split_intervals[5] = si_split_intervals[1]
    #         # 由于f方向拆分导致的复制
    #         for _ in range(split_vector.nf - 1):
    #             si_split_intervals[3].extend(deepcopy(cc_split_intervals[3]))
    #         # 对SICTaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             task: STaskBlock
    #             if task.task_type in (TaskBlockType.SIC, TaskBlockType.SIC2D):
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(si_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #                 break  # 只有一个SICTaskBlock
    #         # 生成SWTaskBlock的split intervals
    #         sw_split_intervals = [[(1, 0)] for _ in range(6)]
    #         sw_split_intervals[2] = deepcopy(cc_split_intervals[2])
    #         sw_split_intervals[3] = deepcopy(cc_split_intervals[3])
    #         # 由于y和x方向拆分导致的复制
    #         for _ in range(split_vector.ny * split_vector.nx - 1):
    #             sw_split_intervals[2].extend(deepcopy(cc_split_intervals[2]))
    #         # 对SWTaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SW:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(sw_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #                 break  # 只有一个SWTaskBlock
    #         if split_vector.nr == 1:  # 输入通道拆分会使加bias的操作在CADD进行
    #             # 生成SBTaskBlock的split intervals
    #             sb_split_intervals = [[(1, 0)] for _ in range(6)]
    #             sb_split_intervals[2] = deepcopy(cc_split_intervals[2])
    #             # 由于y和x方向拆分导致的复制
    #             for _ in range(split_vector.ny * split_vector.nx - 1):
    #                 sb_split_intervals[2].extend(
    #                     deepcopy(cc_split_intervals[2]))
    #             # 对SBTaskBlock进行拆分
    #             for task in original_task.in_tasks:
    #                 if task.task_type == TaskBlockType.SB:
    #                     splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                         split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                     splitter.split_node(sb_split_intervals)
    #                     new_node_ids.extend(splitter.id_list)
    #                     break  # 只有一个SBTaskBlock
    #         # 消除冗余连接
    #         Splitter.remove_redundant_connections(
    #             new_node_ids_after_c_split, original_task, self._task_graph)
    #     elif original_task.task_type in (TaskBlockType.CCMPB, TaskBlockType.CCMPS, TaskBlockType.CAVG,
    #                                      TaskBlockType.CADD, TaskBlockType.CVVH, TaskBlockType.CVS, TaskBlockType.CAX):
    #         if split_vector.nr != 1:
    #             logging.warn(
    #                 '{:s} cannot be split in r dimension'.format(type(original_task).__name__))
    #             split_vector.nr = 1
    #         if original_task.task_type == TaskBlockType.CAVG:
    #             if (original_task.shape.nix == original_task.shape.nkx or original_task.shape.nx == 1):
    #                 if split_vector.nx != 1:
    #                     logging.warn(
    #                         'CAVGTaskBlock cannot be split in x dimension because kernel size equals to the input size.'
    #                         + ' The value in x dimension of split vector will be ignored.')
    #                     split_vector.nx = 1
    #             if (original_task.shape.niy == original_task.shape.nky or original_task.shape.ny == 1):
    #                 if split_vector.ny != 1:
    #                     logging.warn(
    #                         'CAVGTaskBlock cannot be split in y dimension because kernel size equals to the input size.'
    #                         + ' The value in y dimension of split vector will be ignored.')
    #                     split_vector.ny = 1
    #         # 对计算任务块进行拆分
    #         splitter = Splitter(task_id, split_vector,
    #                             split_funcs, self._task_graph)
    #         splitter.split_node()
    #         new_node_ids_after_c_split = splitter.id_list
    #         new_node_ids.extend(splitter.id_list)
    #         # 生成SITaskBlock的split intervals
    #         c_split_intervals = splitter.result
    #         si_split_intervals = [[(1, 0)] for _ in range(6)]
    #         si_split_intervals[0] = deepcopy(c_split_intervals[4])  # y = iy
    #         si_split_intervals[1] = deepcopy(c_split_intervals[5])  # x = ix
    #         si_split_intervals[2] = deepcopy(c_split_intervals[2])  # f = f
    #         # 这里是为了处理copy_node_without_edge中的zip
    #         si_split_intervals[4] = si_split_intervals[0]
    #         si_split_intervals[5] = si_split_intervals[1]
    #         # SITaskBlock不会发生复制
    #         # 对SITaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SI:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(si_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #         # 对bias进行拆分
    #         if original_task.bias_type == BiasType.VECTOR:
    #             # 生成SBTaskBlock的split intervals
    #             sb_split_intervals = [[(1, 0)] for _ in range(6)]
    #             sb_split_intervals[2] = deepcopy(c_split_intervals[2])
    #             # 由于y和x方向拆分导致的复制
    #             for _ in range(split_vector.ny * split_vector.nx - 1):
    #                 sb_split_intervals[2].extend(
    #                     deepcopy(c_split_intervals[2]))
    #             # 对SBTaskBlock进行拆分
    #             for task in original_task.in_tasks:
    #                 if task.task_type == TaskBlockType.SB:
    #                     splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                         split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                     splitter.split_node(sb_split_intervals)
    #                     new_node_ids.extend(splitter.id_list)
    #         # 消除冗余连接
    #         Splitter.remove_redundant_connections(
    #             new_node_ids_after_c_split, original_task, self._task_graph)
    #     elif original_task.task_type == TaskBlockType.CVM:
    #         if split_vector.ny != 1:
    #             logging.warn(
    #                 '{:s} cannot be split in y dimension'.format(type(original_task).__name__))
    #             split_vector.ny = 1
    #         if split_vector.nx != 1:
    #             logging.warn(
    #                 '{:s} cannot be split in x dimension'.format(type(original_task).__name__))
    #             split_vector.nx = 1
    #         # 对CVMTaskBlock进行拆分
    #         splitter = Splitter(task_id, split_vector,
    #                             split_funcs, self._task_graph)
    #         splitter.split_node()
    #         new_node_ids_after_c_split = splitter.id_list
    #         new_node_ids.extend(splitter.id_list)
    #         # 生成SIFCTaskBlock的split intervals
    #         cc_split_intervals = splitter.result
    #         si_split_intervals = [[(1, 0)] for _ in range(6)]
    #         si_split_intervals[0] = [(0, 1)]
    #         si_split_intervals[1] = [(0, 1)]
    #         si_split_intervals[3] = deepcopy(cc_split_intervals[3])  # r = r
    #         si_split_intervals[4] = si_split_intervals[0]
    #         si_split_intervals[5] = si_split_intervals[1]
    #         # 由于f方向拆分导致的复制
    #         for _ in range(split_vector.nf - 1):
    #             si_split_intervals[3].extend(deepcopy(cc_split_intervals[3]))
    #         # 对SIFCTaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SIFC:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(si_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #                 break  # 只有一个SIFCTaskBlock
    #         # 生成SWFCTaskBlock的split intervals
    #         sw_split_intervals = [[(1, 0)] for _ in range(6)]
    #         sw_split_intervals[2] = deepcopy(cc_split_intervals[2])
    #         sw_split_intervals[3] = deepcopy(cc_split_intervals[3])
    #         # 不会发生复制
    #         # 对SWTaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SWFC:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(sw_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #                 break  # 只有一个SWFCTaskBlock
    #         if split_vector.nr == 1:  # 输入通道拆分会使加bias的操作在CADD进行
    #             # 生成SBTaskBlock的split intervals
    #             sb_split_intervals = [[(1, 0)] for _ in range(6)]
    #             sb_split_intervals[2] = deepcopy(cc_split_intervals[2])
    #             # 不会发生复制
    #             # 对SBTaskBlock进行拆分
    #             for task in original_task.in_tasks:
    #                 if task.task_type == TaskBlockType.SB:
    #                     splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                         split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                     splitter.split_node(sb_split_intervals)
    #                     new_node_ids.extend(splitter.id_list)
    #                     break  # 只有一个SBTaskBlock
    #         # 消除冗余连接
    #         Splitter.remove_redundant_connections(
    #             new_node_ids_after_c_split, original_task, self._task_graph)
    #     elif original_task.task_type == TaskBlockType.CLUT:
    #         if split_vector.nr != 1:
    #             logging.warn(
    #                 '{:s} cannot be split in r dimension'.format(type(original_task).__name__))
    #             split_vector.nr = 1
    #         # 对CLUTTaskBlock进行拆分
    #         splitter = Splitter(task_id, split_vector,
    #                             split_funcs, self._task_graph)
    #         splitter.split_node()
    #         new_node_ids_after_c_split = splitter.id_list
    #         new_node_ids.extend(splitter.id_list)
    #         # 生成SITaskBlock的split intervals
    #         c_split_intervals = splitter.result
    #         si_split_intervals = [[(1, 0)] for _ in range(6)]
    #         si_split_intervals[0] = deepcopy(c_split_intervals[4])  # y = iy
    #         si_split_intervals[1] = deepcopy(c_split_intervals[5])  # x = ix
    #         si_split_intervals[2] = deepcopy(c_split_intervals[2])  # f = f
    #         # 这里是为了处理copy_node_without_edge中的zip
    #         si_split_intervals[4] = si_split_intervals[0]
    #         si_split_intervals[5] = si_split_intervals[1]
    #         # 不会发生复制
    #         # 对所有SITaskBlock进行拆分
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SI:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(si_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #                 break  # 只能有一个SITaskBlock
    #         # LUT不进行拆分只发生复制
    #         sb_split_intervals = [[(1, 0)] for _ in range(6)]
    #         sb_split_intervals[2] = [(0, original_task.lut_len)]
    #         # 由于y, x, f方向拆分导致的复制
    #         for _ in range(split_vector.ny * split_vector.nx * split_vector.nf - 1):
    #             sb_split_intervals[2].extend([(0, original_task.lut_len)])
    #         # 对SBTaskBlock进行复制
    #         for task in original_task.in_tasks:
    #             if task.task_type == TaskBlockType.SB:
    #                 splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
    #                                     split_funcs, self._task_graph)  # split_vector和split_funcs无效
    #                 splitter.split_node(sb_split_intervals)
    #                 new_node_ids.extend(splitter.id_list)
    #         # 消除冗余连接
    #         Splitter.remove_redundant_connections(
    #             new_node_ids_after_c_split, original_task, self._task_graph)
    #     elif original_task.task_type == TaskBlockType.CLIF:
    #         raise NotImplementedError('Splitting {:s} has not been implemented'.format(
    #             type(original_task).__name__))
    #     else:
    #         raise TypeError('{:s} cannot be split'.format(
    #             type(original_task).__name__))
    #     return new_node_ids

    # def old_split_task(self, task_id, split_vector, split_funcs):
    #     from src.simulator.resource_simulator.action_model.Splitter import Splitter
    #     Splitter = Splitter(task_id, split_vector,
    #                       split_funcs, self._task_graph)
    #     Splitter.split_node()

    def delete_task(self, task_id):
        return self._task_graph.delete_node(task_id)
    
    def delete_tasks(self, tasks: Iterable[TaskBlock]):
        for task in tasks:
            self.delete_task(task.id)

    def replicate_task(self, task_id):
        new_task = Replicater.copy_node(self._task_graph.get_node(task_id))
        self._task_graph.add_node(new_task)
        return new_task.id

    def fuse_task(self, task_id_list):
        pass

    def set_pipeline_num(self, task_id, pipeline_num):
        assert self._task_graph.get_node(task_id).type.is_storage_task
        self._task_graph.get_node(task_id).pipeline_num(pipeline_num)

    def enable_tasks(self, tasks: Iterable[TaskBlock]):
        for task in tasks:
            self.enable_task(task.id)

    def enable_task(self, task_id):
        self._task_graph.enable_node(task_id)

    def disable_tasks(self, tasks: Iterable[TaskBlock]):
        for task in tasks:
            self.disable_task(task.id)

    def disable_task(self, task_id):
        self._task_graph.disable_node(task_id)

    def merge_column(self, space_coord_list):
        # 空间上需要精确到最细粒度
        step_column_list = [self._st_matrix.get_space(space_coord_list[0])]
        for space_coord in space_coord_list[1:]:
            assert len(space_coord) == 4
            step_column_list.append(
                self._st_matrix.get_space(space_coord))
        ColumnMerger.merge_step(step_column_list, space_coord_list, self)
        for space_coord in space_coord_list[1:]:
            assert len(self.delete_column(space_coord)) == 0

    def delete_column(self, space_coord):
        space_column = self._st_matrix.pop(MLCoord(space_coord, Coord(())))
        d = ColumnDeleter(self._context)
        d.delete_step_phase(space_column, space_coord)
        return d.task_list

    def put_in(self, ml_coord: MLCoord, task_id: int):
        self._st_matrix.add_task(ml_coord, self._task_graph.get_node(task_id))
        self._context.put_task_to(ml_coord, task_id)

    def put_tasks_in(self, ml_coord: MLCoord, tasks: Iterable[TaskBlock]):
        for task in tasks:
            self.put_in(ml_coord, task.id)

    def sync(self, ml_coord: MLCoord, sync_id: int):
        space_point = self._st_matrix.get_element(ml_coord)
        task = space_point.get_last_task()
        sync_task = SyncTask(sync_id, task)
        if task is not None:
            self._sync_table.add(sync_task)
        self._st_matrix.add_task(ml_coord, sync_task)

    def take_out(self, ml_coord: MLCoord, task_id: int):
        task = self._st_matrix.pop(ml_coord, task_id)
        self._context.take_task_out(task.id)
        return task
    
    def take_tasks_out(self, ml_coord: MLCoord, tasks: Iterable[TaskBlock]):
        for task in tasks:
            self.take_out(ml_coord, task.id)

    # def take_out_task(self, task_id, ml_coord: MLCoord):
    #     # 删除坐标种的某个节点
    #     assert len(ml_coord.space_coord) == 4
    #     assert len(ml_coord.time_coord) == 3
    #     self._context.take_task_out(task_id, ml_coord)

    def move(self, src_coord, dml_coord, task_id=None):
        assert len(src_coord.space_coord) == len(dml_coord.space_coord)
        assert len(src_coord.time_coord) == len(dml_coord.time_coord)
        task = self.take_out(src_coord, task_id)
        if type(task) is dict:
            for v in task.values():
                self.put_in(v.id, dml_coord)
        else:
            self.put_in(task.id, dml_coord)
        return task

    def connect(self, src_task, src_index, src_info,
                dst_task, dst_index, dst_info):
        assert src_info.size == dst_info.size
        self._task_graph.connect(src_task.id, dst_task.id,
                                  source_cluster=src_index,
                                  destination_cluster=dst_index,
                                  source_position=src_info.position,
                                  destination_position=dst_info.position,
                                  packet_shape=src_info.size)

    def map_edge(self, edge: Edge, path: List[Union[MLCoord, Tuple[MLCoord, int], Tuple[MLCoord, int, int]]]):
        """
        对path的要求: 必须包含跳出当前层次的坐标, 跨域的坐标也必须包含, 跨域的ID默认为当前域的ID
        跨域说明: 1 -> 1 -> 2, 则1 -> 1在域1中, 1 -> 2在域2中; 2 -> 2 -> 1, 则2 -> 2在域2中, 2 -> 1在域1中
        假设chiplet -> chiplet -> DRAM, 则写入数据到DRAM为1 -> 1 -> 2, 从DRAM读数据为2 -> 2 -> 1
        """
        # 加入虚拟任务结点保证每条边上的所有坐标都在同一空间层次, 且在同一个互联域中
        if not edge.is_enable():
            raise ValueError("Disabled edge cannot be mapped")
        new_tasks = []
        new_edges = []
        path: PathCoord = PathCoord(path)
        assert not path.illegal, "Illegal edge path"
        IDGenerator.set_base_task_id(self._task_graph)
        task0: TaskBlock = edge.in_task
        task1 = None
        new_path = PathCoord()
        for i in range(len(path) - 1):
            ml_coord0 = path[i]
            ml_coord1 = path[i + 1]
            new_path.append(ml_coord0)
            if ml_coord1.ml_coord.level != ml_coord0.ml_coord.level:
                task1, new_edge = self.add_virtual_task(task0, edge)
                new_tasks.append(task1)
                # task1 = VTaskBlock(
                #     task_id=IDGenerator.get_next_task_id(),
                #     shape=task0.out_shape,
                #     precision=task0.precision
                #     )
                # new_edge = Edge(in_task=task0, out_task=task1)
                # self._task_graph.add_node(task1)
                # task1.add_input_edge(new_edge)
                # if edge in task0.output_edges:
                #     task0.output_edges.remove(edge)
                # task0.add_output_edge(new_edge)
                task0 = task1
                # 低级到高级
                if ml_coord1.ml_coord.level < ml_coord0.ml_coord.level:
                    partial_new_tasks, partial_new_edges = self.split_same_level_edges(new_edge, new_path)
                    new_tasks.extend(partial_new_tasks)
                    new_edges.extend(partial_new_edges)
                    new_ml_coord = deepcopy(ml_coord0)
                    while new_ml_coord.ml_coord.level != ml_coord1.ml_coord.level:
                        new_ml_coord.ml_coord = new_ml_coord.ml_coord.outer_coord
                    new_ml_coord.network_id = ml_coord1.network_id  # 一定与dst所在的域相同
                    new_ml_coord.link_id = ml_coord1.link_id  # 一定与dst的Link ID相同
                    new_path = PathCoord([new_ml_coord])
                # 高级到低级
                else:
                    new_ml_coord = deepcopy(ml_coord1)
                    while new_ml_coord.ml_coord.level != ml_coord0.ml_coord.level:
                        new_ml_coord.ml_coord = new_ml_coord.ml_coord.outer_coord
                    if new_ml_coord.ml_coord in new_path:
                        assert (new_path[new_ml_coord.ml_coord].network_id == new_ml_coord.network_id and 
                                new_path[new_ml_coord.ml_coord].link_id == new_ml_coord.link_id), "Illegal edge coordinate"
                        partial_new_tasks, partial_new_edges = self.split_same_level_edges(new_edge, new_path)
                        new_tasks.extend(partial_new_tasks)
                        new_edges.extend(partial_new_edges)
                    else:
                        new_path.append(new_ml_coord)
                        partial_new_tasks, partial_new_edges = self.split_same_level_edges(new_edge, new_path)
                        new_tasks.extend(partial_new_tasks)
                        new_edges.extend(partial_new_edges)
                    new_path = PathCoord()
        if task1 is not None:
            new_edge = Edge(in_task=task1, out_task=edge.out_task)
            task1.add_output_edge(new_edge)
            new_path.append(ml_coord1)
            out_task: TaskBlock = edge.out_task
            # out_task.input_edges.remove(edge)
            out_task.add_input_edge(new_edge)
            partial_new_tasks, partial_new_edges = self.split_same_level_edges(new_edge, new_path)
            new_tasks.extend(partial_new_tasks)
            new_edges.extend(partial_new_edges)
            # del edge
            edge.disable()
            # del path
            return new_tasks, new_edges
        # 说明path上所有坐标都在同一层级
        else:
            return self.split_same_level_edges(edge, path)

    def add_virtual_task(self, task0: TaskBlock, edge: Edge):
        task1 = VTaskBlock(
            task_id=IDGenerator.get_next_task_id(),
            shape=task0.out_shape,
            precision=task0.precision
            )
        new_edge = Edge(in_task=task0, out_task=task1)
        self._task_graph.add_node(task1)
        task1.add_input_edge(new_edge)
        # if edge in task0.output_edges:
        #     task0.output_edges.remove(edge)
        task0.add_output_edge(new_edge) 
        return task1, new_edge

    def split_same_level_edges(self, edge: Edge, path: PathCoord):
        """
        拆分同一层次的edge: 坐标所在的容器不同
        """
        new_tasks = []
        new_edges = []
        assert path.same_level, "All coordinates on this path should be at the same level"
        # IDGenerator.set_base_task_id(self._task_graph)
        task0: TaskBlock = edge.in_task
        task1 = None
        new_path = PathCoord()
        for i in range(len(path) - 1):
            element0 = path[i]
            element1 = path[i + 1]
            new_path.append(element0)
            # 同容器跨互联域
            if element0.network_id != element1.network_id:
                task1, new_edge = self.add_virtual_task(task0, edge)
                new_tasks.append(task1)
                new_edges.append(new_edge)
                self._st_matrix.add_edge(new_edge, new_path)
                self._context.put_edge_to(new_path, new_edge)
                new_element = deepcopy(element0)
                new_element.network_id = element1.network_id
                new_path = PathCoord([new_element])
                task0 = task1

        if task1 is not None:
            new_edge = Edge(in_task=task1, out_task=edge.out_task)
            new_edges.append(new_edge)
            task1.add_output_edge(new_edge)
            new_path.append(element1)
            out_task: TaskBlock = edge.out_task
            out_task.input_edges.remove(edge)
            out_task.add_input_edge(new_edge)
            self._st_matrix.add_edge(new_edge, new_path)
            self._context.put_edge_to(new_path, new_edge)
            del edge
            del path
        # 说明path上所有hop都在同一domain
        else:
            self._st_matrix.add_edge(edge, path)
            self._context.put_edge_to(path, edge)
            return [], [edge]

        return new_tasks, new_edges

    def reverse_map_edge(self, original_edge: Edge, new_tasks: List[TaskBlock], new_edges: List[Edge]):
        original_edge.enable()
        for task in new_tasks:
            self._task_graph.delete_node(task.id)
        for edge in new_edges:
            path = self._context.take_edge_out(edge)
            self._st_matrix.take_edge_out(edge, path)

    def reverse_map_edges(self, edges: List[Edge], new_tasks: List[List[TaskBlock]], new_edges: List[List[Edge]]):
        for edge, new_task, new_edge in zip(edges, new_tasks, new_edges):
            self.reverse_map_edge(edge, new_task, new_edge)


if __name__ == "__main__":
    from src.simulator.resource_simulator.st_model.st_point import STPoint


    task_graph = TaskGraph()
    task0 = VTaskBlock(0)
    task1 = VTaskBlock(1)
    edge = Edge(task0, task1)
    task0.add_output_edge(edge)
    task1.add_input_edge(edge)
    task_graph.add_node(task0)
    task_graph.add_node(task1)
    st_matrix = STMatrix(dim=2, space_level=2)
    st_matrix_0_0 = STMatrix(dim=1, space_level=1)
    st_matrix_0_1 = STMatrix(dim=1, space_level=1)
    st_matrix_1_1 = STMatrix(dim=1, space_level=1)
    st_matrix_1_0 = STMatrix(dim=1, space_level=1)
    st_point_0_0_0 = STPoint()
    st_point_0_0_1 = STPoint()
    st_point_1_0_0 = STPoint()
    st_point_1_0_1 = STPoint()
    st_matrix_0_0.add_element(coord=Coord(0), element=st_point_0_0_0)
    st_matrix_0_0.add_element(coord=Coord(1), element=st_point_0_0_1)
    st_matrix_1_0.add_element(coord=Coord(0), element=st_point_1_0_0)
    st_matrix_1_0.add_element(coord=Coord(1), element=st_point_1_0_1)
    st_matrix.add_element(coord=Coord((0, 0)), element=st_matrix_0_0)
    st_matrix.add_element(coord=Coord((0, 1)), element=st_matrix_0_1)
    st_matrix.add_element(coord=Coord((1, 1)), element=st_matrix_1_1)
    st_matrix.add_element(coord=Coord((1, 0)), element=st_matrix_1_0)
    st_context = STContext()
    actor = ActionModel(task_graph, st_matrix, st_context)
    path = [MLCoord(Coord((0, 0)), Coord(0)), MLCoord(Coord((0, 0)), Coord(1)), MLCoord(Coord((0, 1))), MLCoord(Coord((1, 1))), MLCoord(Coord((1, 0)), Coord(1)), MLCoord(Coord((1, 0)), Coord(0))]
    actor.map_edge(edge=edge, path=path)
    