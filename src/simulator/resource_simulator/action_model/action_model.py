#!/usr/bin/env python
# coding: utf-8

"""
ActionModel 类负责各种动作的响应
"""

from copy import deepcopy
import logging
from typing import List, Union
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.action_model.splitter import Splitter, SplitType
from src.simulator.resource_simulator.action_model.replicater import Replicater
from src.simulator.resource_simulator.action_model.column_merger import ColumnMerger
from src.simulator.resource_simulator.action_model.column_deleter import ColumnDeleter
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord



class ActionModel():
    def __init__(self, task_graph: TaskGraph, st_matrix: STMatrix, st_context: STContext):
        self._task_graph = task_graph
        self._st_matrix = st_matrix
        self._context = st_context

    def split_task(self, task_id, split_vector: Shape, split_funcs: List[SplitType]):
        if split_vector == 1:
            return  # 不需要拆分
        original_task = deepcopy(self._task_graph.get_node(task_id))
        if split_vector.nky != 1 or split_vector.nkx != 1:
            logging.warn(
                'Values in ky and kx dimensions of split vector will be ignored')
            split_vector.nky = 1
            split_vector.nkx = 1
        original_task_shape = original_task.shape
        if original_task_shape.ny == 1 or original_task_shape.ny == -1:
            if split_vector.ny != 1:
                logging.warn(
                    'Value in y dimension of split vector will be ignored')
                split_vector.ny = 1
        if original_task_shape.nx == 1 or original_task_shape.nx == -1:
            if split_vector.nx != 1:
                logging.warn(
                    'Value in x dimension of split vector will be ignored')
                split_vector.nx = 1
        if original_task_shape.nf == 1 or original_task_shape.nf == -1:
            if split_vector.ny != 1:
                logging.warn(
                    'Value in f dimension of split vector will be ignored')
                split_vector.nf = 1
        if original_task_shape.nr == 1 or original_task_shape.nr == -1:
            if split_vector.nr != 1:
                logging.warn(
                    'Value in r dimension of split vector will be ignored')
                split_vector.nr = 1
        new_node_ids = []
        if isinstance(original_task, (CCTaskBlock, CC2DTaskBlock)):
            # 对CCTaskBlock进行拆分
            splitter = Splitter(task_id, split_vector,
                                split_funcs, self._task_graph)
            splitter.split_node()
            new_node_ids_after_c_split = splitter.id_list
            new_node_ids.extend(splitter.id_list)
            # 生成SICTaskBlock的split intervals
            cc_split_intervals = splitter.result
            si_split_intervals = [[(1, 0)] for _ in range(6)]
            si_split_intervals[0] = deepcopy(cc_split_intervals[4])  # y = iy
            si_split_intervals[1] = deepcopy(cc_split_intervals[5])  # x = ix
            si_split_intervals[3] = deepcopy(cc_split_intervals[3])  # r = r
            # 这里是为了处理copy_node_without_edge中的zip
            si_split_intervals[4] = si_split_intervals[0]
            si_split_intervals[5] = si_split_intervals[1]
            # 由于f方向拆分导致的复制
            for _ in range(split_vector.nf - 1):
                si_split_intervals[3].extend(deepcopy(cc_split_intervals[3]))
            # 对SICTaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, (SICTaskBlock, SIC2DTaskBlock)):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(si_split_intervals)
                    new_node_ids.extend(splitter.id_list)
                    break  # 只有一个SICTaskBlock
            # 生成SWTaskBlock的split intervals
            sw_split_intervals = [[(1, 0)] for _ in range(6)]
            sw_split_intervals[2] = deepcopy(cc_split_intervals[2])
            sw_split_intervals[3] = deepcopy(cc_split_intervals[3])
            # 由于y和x方向拆分导致的复制
            for _ in range(split_vector.ny * split_vector.nx - 1):
                sw_split_intervals[2].extend(deepcopy(cc_split_intervals[2]))
            # 对SWTaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, SWTaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(sw_split_intervals)
                    new_node_ids.extend(splitter.id_list)
                    break  # 只有一个SWTaskBlock
            if split_vector.nr == 1:  # 输入通道拆分会使加bias的操作在CADD进行
                # 生成SBTaskBlock的split intervals
                sb_split_intervals = [[(1, 0)] for _ in range(6)]
                sb_split_intervals[2] = deepcopy(cc_split_intervals[2])
                # 由于y和x方向拆分导致的复制
                for _ in range(split_vector.ny * split_vector.nx - 1):
                    sb_split_intervals[2].extend(
                        deepcopy(cc_split_intervals[2]))
                # 对SBTaskBlock进行拆分
                for task in original_task.in_tasks:
                    if isinstance(task, SBTaskBlock):
                        splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                            split_funcs, self._task_graph)  # split_vector和split_funcs无效
                        splitter.split_node(sb_split_intervals)
                        new_node_ids.extend(splitter.id_list)
                        break  # 只有一个SBTaskBlock
            # 消除冗余连接
            Splitter.remove_redundant_connections(
                new_node_ids_after_c_split, original_task, self._task_graph)
        elif isinstance(original_task, (CCMPBTaskBlock, CCMPSTaskBlock, CAVGTaskBlock,
                                        CADDTaskBlock, CVVHTaskBlock, CVSTaskBlock, CAXTaskBlock)):
            if split_vector.nr != 1:
                logging.warn(
                    '{:s} cannot be split in r dimension'.format(type(original_task).__name__))
                split_vector.nr = 1
            if isinstance(original_task, CAVGTaskBlock):
                if (original_task.shape.nix == original_task.shape.nkx or original_task.shape.nx == 1):
                    if split_vector.nx != 1:
                        logging.warn(
                            'CAVGTaskBlock cannot be split in x dimension because kernel size equals to the input size.'
                            + ' The value in x dimension of split vector will be ignored.')
                        split_vector.nx = 1
                if (original_task.shape.niy == original_task.shape.nky or original_task.shape.ny == 1):
                    if split_vector.ny != 1:
                        logging.warn(
                            'CAVGTaskBlock cannot be split in y dimension because kernel size equals to the input size.'
                            + ' The value in y dimension of split vector will be ignored.')
                        split_vector.ny = 1
            # 对计算任务块进行拆分
            splitter = Splitter(task_id, split_vector,
                                split_funcs, self._task_graph)
            splitter.split_node()
            new_node_ids_after_c_split = splitter.id_list
            new_node_ids.extend(splitter.id_list)
            # 生成SITaskBlock的split intervals
            c_split_intervals = splitter.result
            si_split_intervals = [[(1, 0)] for _ in range(6)]
            si_split_intervals[0] = deepcopy(c_split_intervals[4])  # y = iy
            si_split_intervals[1] = deepcopy(c_split_intervals[5])  # x = ix
            si_split_intervals[2] = deepcopy(c_split_intervals[2])  # f = f
            # 这里是为了处理copy_node_without_edge中的zip
            si_split_intervals[4] = si_split_intervals[0]
            si_split_intervals[5] = si_split_intervals[1]
            # SITaskBlock不会发生复制
            # 对SITaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, SITaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(si_split_intervals)
                    new_node_ids.extend(splitter.id_list)
            # 对bias进行拆分
            if original_task.bias_type == BiasType.VECTOR:
                # 生成SBTaskBlock的split intervals
                sb_split_intervals = [[(1, 0)] for _ in range(6)]
                sb_split_intervals[2] = deepcopy(c_split_intervals[2])
                # 由于y和x方向拆分导致的复制
                for _ in range(split_vector.ny * split_vector.nx - 1):
                    sb_split_intervals[2].extend(
                        deepcopy(c_split_intervals[2]))
                # 对SBTaskBlock进行拆分
                for task in original_task.in_tasks:
                    if isinstance(task, SBTaskBlock):
                        splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                            split_funcs, self._task_graph)  # split_vector和split_funcs无效
                        splitter.split_node(sb_split_intervals)
                        new_node_ids.extend(splitter.id_list)
            # 消除冗余连接
            Splitter.remove_redundant_connections(
                new_node_ids_after_c_split, original_task, self._task_graph)
        elif isinstance(original_task, CVMTaskBlock):
            if split_vector.ny != 1:
                logging.warn(
                    '{:s} cannot be split in y dimension'.format(type(original_task).__name__))
                split_vector.ny = 1
            if split_vector.nx != 1:
                logging.warn(
                    '{:s} cannot be split in x dimension'.format(type(original_task).__name__))
                split_vector.nx = 1
            # 对CVMTaskBlock进行拆分
            splitter = Splitter(task_id, split_vector,
                                split_funcs, self._task_graph)
            splitter.split_node()
            new_node_ids_after_c_split = splitter.id_list
            new_node_ids.extend(splitter.id_list)
            # 生成SIFCTaskBlock的split intervals
            cc_split_intervals = splitter.result
            si_split_intervals = [[(1, 0)] for _ in range(6)]
            si_split_intervals[0] = [(0, 1)]
            si_split_intervals[1] = [(0, 1)]
            si_split_intervals[3] = deepcopy(cc_split_intervals[3])  # r = r
            si_split_intervals[4] = si_split_intervals[0]
            si_split_intervals[5] = si_split_intervals[1]
            # 由于f方向拆分导致的复制
            for _ in range(split_vector.nf - 1):
                si_split_intervals[3].extend(deepcopy(cc_split_intervals[3]))
            # 对SIFCTaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, SIFCTaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(si_split_intervals)
                    new_node_ids.extend(splitter.id_list)
                    break  # 只有一个SIFCTaskBlock
            # 生成SWFCTaskBlock的split intervals
            sw_split_intervals = [[(1, 0)] for _ in range(6)]
            sw_split_intervals[2] = deepcopy(cc_split_intervals[2])
            sw_split_intervals[3] = deepcopy(cc_split_intervals[3])
            # 不会发生复制
            # 对SWTaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, SWFCTaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(sw_split_intervals)
                    new_node_ids.extend(splitter.id_list)
                    break  # 只有一个SWFCTaskBlock
            if split_vector.nr == 1:  # 输入通道拆分会使加bias的操作在CADD进行
                # 生成SBTaskBlock的split intervals
                sb_split_intervals = [[(1, 0)] for _ in range(6)]
                sb_split_intervals[2] = deepcopy(cc_split_intervals[2])
                # 不会发生复制
                # 对SBTaskBlock进行拆分
                for task in original_task.in_tasks:
                    if isinstance(task, SBTaskBlock):
                        splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                            split_funcs, self._task_graph)  # split_vector和split_funcs无效
                        splitter.split_node(sb_split_intervals)
                        new_node_ids.extend(splitter.id_list)
                        break  # 只有一个SBTaskBlock
            # 消除冗余连接
            Splitter.remove_redundant_connections(
                new_node_ids_after_c_split, original_task, self._task_graph)
        elif isinstance(original_task, CLUTTaskBlock):
            if split_vector.nr != 1:
                logging.warn(
                    '{:s} cannot be split in r dimension'.format(type(original_task).__name__))
                split_vector.nr = 1
            # 对CLUTTaskBlock进行拆分
            splitter = Splitter(task_id, split_vector,
                                split_funcs, self._task_graph)
            splitter.split_node()
            new_node_ids_after_c_split = splitter.id_list
            new_node_ids.extend(splitter.id_list)
            # 生成SITaskBlock的split intervals
            c_split_intervals = splitter.result
            si_split_intervals = [[(1, 0)] for _ in range(6)]
            si_split_intervals[0] = deepcopy(c_split_intervals[4])  # y = iy
            si_split_intervals[1] = deepcopy(c_split_intervals[5])  # x = ix
            si_split_intervals[2] = deepcopy(c_split_intervals[2])  # f = f
            # 这里是为了处理copy_node_without_edge中的zip
            si_split_intervals[4] = si_split_intervals[0]
            si_split_intervals[5] = si_split_intervals[1]
            # 不会发生复制
            # 对所有SITaskBlock进行拆分
            for task in original_task.in_tasks:
                if isinstance(task, SITaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(si_split_intervals)
                    new_node_ids.extend(splitter.id_list)
                    break  # 只能有一个SITaskBlock
            # LUT不进行拆分只发生复制
            sb_split_intervals = [[(1, 0)] for _ in range(6)]
            sb_split_intervals[2] = [(0, original_task.lut_len)]
            # 由于y, x, f方向拆分导致的复制
            for _ in range(split_vector.ny * split_vector.nx * split_vector.nf - 1):
                sb_split_intervals[2].extend([(0, original_task.lut_len)])
            # 对SBTaskBlock进行复制
            for task in original_task.in_tasks:
                if isinstance(task, SBTaskBlock):
                    splitter = Splitter(task.id, Shape(1, 1, 1, 1, 1, 1),
                                        split_funcs, self._task_graph)  # split_vector和split_funcs无效
                    splitter.split_node(sb_split_intervals)
                    new_node_ids.extend(splitter.id_list)
            # 消除冗余连接
            Splitter.remove_redundant_connections(
                new_node_ids_after_c_split, original_task, self._task_graph)
        elif isinstance(original_task, CLIFTaskBlock):
            raise NotImplementedError('Splitting {:s} has not been implemented'.format(
                type(original_task).__name__))
        else:
            raise TypeError('{:s} cannot be split'.format(
                type(original_task).__name__))
        return new_node_ids

    def old_split_task(self, task_id, split_vector, split_funcs):
        from resource_simulator.action_model.Splitter import Splitter
        Splitter = Splitter(task_id, split_vector,
                          split_funcs, self._task_graph)
        Splitter.split_node()

    def delete_task(self, task_id):
        return self._task_graph.delete_node(task_id)

    def replicate_task(self, task_id):
        new_task = Replicater.copy_node(self._task_graph.get_node(task_id))
        self._task_graph.add_node(new_task)
        return new_task.id

    def fuse_task(self, task_id_list):
        pass

    def set_pipeline_num(self, task_id, pipeline_num):
        assert self._task_graph.get_node(task_id).type.is_storage_task
        self._task_graph.get_node(task_id).pipeline_num(pipeline_num)

    def enable_task(self, task_id):
        self._task_graph.get_node(task_id).enable()

    def disable_task(self, task_id):
        self._task_graph.get_node(task_id).disable()

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

    def put_in(self, ml_coord: MLCoord, task_id):
        self._st_matrix.add_task(ml_coord, self._task_graph.get_node(task_id))
        self._context.put_task_to(ml_coord, task_id)

    def take_out(self, ml_coord: MLCoord, task_id=None):
        task = self._st_matrix.pop(ml_coord, task_id)
        if type(task) is dict:
            for v in task.values():
                self._context.take_task_out(v.id, ml_coord)
        else:
            self._context.take_task_out(task.id, ml_coord)
        return task

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

    def map_edge(self, edge: Edge, path: List[MLCoord]):
        self._st_matrix.add_edge(edge, path)
        self._context.put_edge_to(path, edge)
