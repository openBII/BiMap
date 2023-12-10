#!/usr/bin/env python
# coding: utf-8

import logging
from typing import List, Union
from copy import deepcopy
from math import ceil, inf, floor
from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.edge import Edge
from enum import Enum
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class SplitType(Enum):
    # 拆分模式类
    Average = 1


class Splitter(object):
    def __init__(self, base_node_id: int, shape: Shape,
                 split_mode_list: Union[SplitType, List[SplitType]], graph: TaskGraph, aligned_num=None):
        self.graph = graph
        self.base_node = graph.get_node(base_node_id)
        self.shape = shape  # split num in each dimension
        if isinstance(split_mode_list, SplitType):
            self.split_mode_list = [split_mode_list] * 6
        else:
            assert len(split_mode_list) == 6
            self.split_mode_list = split_mode_list
        self.aligned_num = aligned_num

        self.id_list = []  # list of new node id
        self.node_position = {}  # new node's position in base node
        self.si_node_before_cadd = []  # new SI nodes right after CADD node
        self.result = []  # split vector's result

        self.check()      # only the dimension of y, x, f, r are accepted

    def split_node(self, split_intervals: List = None):
        # get split intervals of each dimension
        if split_intervals is not None:
            self.result = split_intervals
        else:
            self.get_split_func_result()
        if self.base_node.task_type in \
                [TaskBlockType.CC, TaskBlockType.CC2D, TaskBlockType.CVM] \
                and self.shape.nr > 1:
            # add CADDTaskBlock
            self.add_addition_blocks()
        self.copy_node_without_edge()
        # rebuild output
        self.build_output_edges()
        # rebuild input
        self.build_input_edges()
        # delete base node
        self.graph.delete_node(self.base_node.id)
        # remove reduction links between CADD and SI
        self.remove_reduction_links_between_new_node_and_si()

    @staticmethod
    def remove_redundant_connections(c_block_id_list: List[int], original_c_block: CTaskBlock, graph: TaskGraph):
        for id in c_block_id_list:
            block = graph.get_node(id)
            if type(block) is type(original_c_block):
                for ic in block.input_clusters:
                    all_in_edges = [e for e in ic.all_edges]
                    if len(all_in_edges) > 1:  # 出现了冗余连接
                        for idx in range(1, len(all_in_edges)):
                            edge = all_in_edges[idx]
                            Splitter.delete_edge(edge)  # 只保留第一条边
                        for pre_block in ic.pre_tasks:
                            pre_all_out_edges = [e for e in
                                                 pre_block.output_clusters[0].all_edges]
                            for edge in pre_all_out_edges:
                                if edge.post_task != block:
                                    Splitter.delete_edge(edge)

    def get_split_func_result(self):
        """
        get split intervals of each dimension
        """
        if self.base_node.shape.ny == -1:
            self.result.append([(1, 0)])
        else:
            self.result.append(self.split_func_oxy(self.base_node.shape.ny,
                                                   self.shape.ny,
                                                   self.split_mode_list[0]))
        if self.base_node.shape.nx == -1:
            self.result.append([(1, 0)])
        else:
            self.result.append(self.split_func_oxy(self.base_node.shape.nx,
                                                   self.shape.nx,
                                                   self.split_mode_list[1]))
        if self.base_node.shape.nf == -1:
            self.result.append([(1, 0)])
        else:
            self.result.append(self.split_func_f(self.base_node.shape.nf,
                                                 self.shape.nf,
                                                 self.split_mode_list[2]))
        if self.base_node.shape.nr == -1:
            self.result.append([(1, 0)])
        else:
            self.result.append(self.split_func_r(self.base_node.shape.nr,
                                                 self.shape.nr,
                                                 self.split_mode_list[3]))
        if self.base_node.task_type in [TaskBlockType.CC, TaskBlockType.CC2D,
                                        TaskBlockType.CCMPB,
                                        TaskBlockType.CCMPS, TaskBlockType.CAVG]:
            # C Task Block with kernel
            self.result.append(self.split_func_ixy(self.base_node.shape.niy,
                                                   self.shape.ny,
                                                   self.split_mode_list[0],
                                                   self.base_node.shape.nky,
                                                   self.base_node.pad_up,
                                                   self.base_node.pad_down,
                                                   self.base_node.stride_y))
            self.result.append(self.split_func_ixy(self.base_node.shape.nix,
                                                   self.shape.nx,
                                                   self.split_mode_list[1],
                                                   self.base_node.shape.nkx,
                                                   self.base_node.pad_left,
                                                   self.base_node.pad_right,
                                                   self.base_node.stride_x))
        else:
            self.result.append(self.result[0])
            self.result.append(self.result[1])

    def add_addition_blocks(self):
        """
        add addition blocks to change "base_node --> out_tasks"
        to "base_node --> SI --> CADD --> out_tasks"
        """
        new_shape_c = Shape(self.base_node.shape.ny,
                            self.base_node.shape.nx,
                            self.base_node.shape.nf, -1, 1, 1)
        new_c_block = CADDTaskBlock(new_shape_c,
                                    IDGenerator.get_next_task_id(),
                                    bias_type=BiasType.VECTOR)
        new_c_block.input_clusters = []
        new_c_block.output_clusters = []
        new_c_block.construct_clusters(self.shape.nr)
        self.id_list.append(new_c_block.id)
        # add to graph
        self.graph.add_node(new_c_block)
        # link new_c_block to original base_node's out task: new_c_block -> base_node's output s_block
        assert (len(self.base_node.output_clusters) == 1)
        new_c_block.output_clusters[0].shape = deepcopy(
            self.base_node.output_clusters[0].shape)
        base_oc_all_edges = [e for e in
                             self.base_node.output_clusters[0].all_edges]
        for edge in base_oc_all_edges:
            edge.pre_task = new_c_block
            edge_info = self.base_node.output_clusters[0].edges.pop(edge)
            new_c_block.output_clusters[0].add_edge(edge_info.position,
                                                    edge_info.size,
                                                    edge)
        # add SI
        for i in range(self.shape.nr):
            new_shape_s = Shape(deepcopy(self.base_node.shape.ny),
                                deepcopy(self.base_node.shape.nx),
                                deepcopy(self.base_node.shape.nf),
                                -1, -1, -1)
            new_s_block = SITaskBlock(new_shape_s,
                                      IDGenerator.get_next_task_id())  # IDGenerator初始值在initial pass设置
            self.id_list.append(new_s_block.id)
            self.si_node_before_cadd.append(new_s_block)
            # link new_s_block to base_node
            new_edge = Edge(self.base_node, new_s_block)
            position = Shape.init_position(new_shape_s)
            self.base_node.output_clusters[0].add_edge(position, deepcopy(
                self.base_node.output_clusters[0].shape), new_edge)
            new_s_block.input_clusters[0].add_edge(position, deepcopy(
                new_s_block.input_clusters[0].shape), new_edge)
            # add to graph
            self.graph.add_node(new_s_block)
            # link new_c_block with new_s_block
            new_edge = Edge(new_s_block, new_c_block)
            new_c_block.add_input_edge(i, position, deepcopy(
                new_c_block.input_clusters[0].shape), new_edge)
            new_s_block.output_clusters[0].add_edge(position, deepcopy(
                new_s_block.output_clusters[0].shape), new_edge)
        # link original base_node's input SBTaskBlock to CADDTaskBlock
        bias_node = None
        for in_task in self.base_node.in_tasks:
            if in_task.task_type == TaskBlockType.SB:
                bias_node = in_task  # have one SBTaskBlock at most
                break
        if bias_node is not None:
            for edge_cluster in bias_node.output_clusters:
                for edge in edge_cluster.all_edges:
                    if edge.post_task is self.base_node:
                        post_info = self.base_node.get_input_edge_info(edge)
                        self.base_node.remove_input_edge(edge)
                        edge.post_task = new_c_block
                        new_c_block.input_clusters[-1].add_edge(
                            post_info.position, post_info.size, edge)

    def copy_node_without_edge(self, shape: Shape = None):
        """
        copy nodes in each dimension
        """
        delta = [None] * 6
        # y和iy
        for oy, iy in zip(self.result[0], self.result[4]):
            delta[0], delta[4] = oy[1] - oy[0], iy[1] - iy[0]
            # x和ix
            for ox, ix in zip(self.result[1], self.result[5]):
                delta[1], delta[5] = ox[1] - ox[0], ix[1] - ix[0]
                for f in self.result[2]:
                    delta[2] = f[1] - f[0]
                    for r in self.result[3]:
                        delta[3] = r[1] - r[0]
                        new_node = self.base_node.copy_like()
                        self.graph.add_node(new_node)
                        self.id_list.append(new_node.id)
                        for idx in [0, 1, 2, 3, 6, 7]:  # y, x, f, r, iy, ix
                            idx_n = idx if idx < 4 else idx - 2
                            if self.base_node.shape[idx] != -1:
                                new_node.shape[idx] = delta[idx_n]
                        # additional parameters
                        if new_node.task_type in [TaskBlockType.CC,
                                                  TaskBlockType.CC2D,
                                                  TaskBlockType.CCMPB,
                                                  TaskBlockType.CCMPS]:
                            new_node.pad_up, new_node.pad_down = 0, 0
                            if oy == self.result[0][0]:
                                new_node.pad_up = self.base_node.pad_up
                            if oy == self.result[0][-1]:
                                new_node.pad_down = self.base_node.pad_down
                            new_node.pad_left, new_node.pad_right = 0, 0
                            if ox == self.result[1][0]:
                                new_node.pad_left = self.base_node.pad_left
                            if ox == self.result[1][-1]:
                                new_node.pad_right = self.base_node.pad_right
                        # reset clusters shape
                        self.reset_clusters_shape(new_node)
                        # record new node's position
                        if self.base_node.task_type in [TaskBlockType.SIC, TaskBlockType.SIC2D, TaskBlockType.SIFC]:
                            self.node_position[new_node] = \
                                (oy[0], ox[0], r[0], r[0], iy[0], ix[0])
                        else:
                            self.node_position[new_node] = \
                                (oy[0], ox[0], f[0], r[0], iy[0], ix[0])
                        # 数据拆分
                        if isinstance(self.base_node, STaskBlock):
                            if self.base_node.data is not None:
                                if self.base_node.task_type == TaskBlockType.SW:
                                    new_node.data = self.base_node.data[f[0]
                                        :f[1], r[0]:r[1], :, :]
                                elif self.base_node.task_type == TaskBlockType.SWFC:
                                    new_node.data = self.base_node.data[f[0]
                                        :f[1], r[0]:r[1]]
                                elif self.base_node.task_type == TaskBlockType.SB:
                                    new_node.data = self.base_node.data[f[0]:f[1]]
                                elif self.base_node.task_type == TaskBlockType.SI:
                                    new_node.data = self.base_node.data[oy[0]
                                        :oy[1], ox[0]:ox[1], f[0]:f[1]]
                                elif (self.base_node.task_type == TaskBlockType.SIC or
                                      self.base_node.task_type == TaskBlockType.SIC2D):
                                    new_node.data = self.base_node.data[oy[0]
                                        :oy[1], ox[0]:ox[1], r[0]:r[1]]
                                elif self.base_node.task_type == TaskBlockType.SIFC:
                                    new_node.data = self.base_node.data[r[0]:r[1]]
                                else:
                                    raise NotImplementedError

    def reset_clusters_shape(self, new_node: TaskBlock):
        """
        根据taskblock的shape重设连接的shape
        """
        i_len = len(new_node.input_clusters)
        o_len = len(new_node.output_clusters)
        if isinstance(new_node, CADDTaskBlock):
            if new_node.bias_type == BiasType.VECTOR:
                new_node.construct_clusters(
                    n_branch=self.base_node.n_branch - 1)
            else:
                new_node.construct_clusters(n_branch=self.base_node.n_branch)
        else:
            if self.base_node.task_type == TaskBlockType.SIFC:
                if self.base_node.input_clusters[0].shape.ny == 1:
                    new_node.construct_clusters(1)
                else:
                    new_node.construct_clusters()
            else:
                new_node.construct_clusters()
        for i in range(i_len):
            new_node.input_clusters[i + i_len].edges = new_node.input_clusters[
                i].edges
        for o in range(o_len):
            new_node.output_clusters[o + o_len].edges = \
                new_node.output_clusters[o].edges
        new_node.input_clusters = new_node.input_clusters[i_len:]
        new_node.output_clusters = new_node.output_clusters[o_len:]

    def build_output_edges(self):
        """
        build new node's output edges
        """
        for out_task in self.base_node.out_tasks:
            for ic_idx, ic in enumerate(out_task.input_clusters):
                all_edges = [e for e in ic.all_edges]
                for edge in all_edges:
                    if edge.pre_task is self.base_node:
                        base_info = self.base_node.get_output_edge_info(edge)
                        bp, bs = base_info.position, base_info.size
                        out_info = out_task.get_input_edge_info(edge)
                        op, os = out_info.position, out_info.size
                        assert os == bs
                        for new_node in self.node_position.keys():
                            np = self.node_position[new_node]
                            if len(new_node.output_clusters) != 1:
                                raise NotImplementedError
                            ns = new_node.output_clusters[0].shape
                            pre_info = deepcopy(base_info)
                            post_info = deepcopy(out_info)
                            for idx in range(4):
                                if bs[idx] != -1:
                                    pre_info.position[idx] = \
                                        max(bp[idx] - np[idx], 0)
                                    pre_info.size[idx] = max(
                                        min(bp[idx] + bs[idx],
                                            np[idx] + ns[idx])
                                        - max(bp[idx], np[idx]), 0)
                                    #
                                    post_info.position[idx] = max(
                                        np[idx] - bp[idx], 0) + op[idx]
                                    post_info.size[idx] = pre_info.size[idx]
                            if pre_info.size.volume > 0:
                                new_e = Edge(new_node, out_task)
                                new_node.add_output_edge(0, pre_info.position,
                                                         pre_info.size, new_e)
                                out_task.add_input_edge(ic_idx,
                                                        post_info.position,
                                                        post_info.size, new_e)

    def build_input_edges(self):
        """
        build new node's input edges
        """
        for ic_idx, ic in enumerate(self.base_node.input_clusters):
            all_edges = [e for e in ic.all_edges]
            for edge in all_edges:
                pre_task = edge.pre_task
                pre_task_out_cluster_idx = \
                    pre_task.get_output_edge_cluster_idx(edge)
                o_pre_info = pre_task.get_output_edge_info(edge)
                pp, ps = o_pre_info.position, o_pre_info.size
                base_info = self.base_node.get_input_edge_info(edge)
                bp, bs = base_info.position, base_info.size
                for new_node in self.node_position.keys():
                    np = self.node_position[new_node]
                    ns = new_node.input_clusters[ic_idx].shape
                    pre_info = deepcopy(o_pre_info)
                    post_info = deepcopy(base_info)
                    for idx in range(4):
                        n_idx = idx + 4 if idx < 2 else idx
                        if ps[idx] != -1:
                            post_info.position[idx] = \
                                max(bp[idx] - np[n_idx], 0)
                            post_info.size[idx] = max(
                                min(bp[idx] + bs[idx],
                                    np[n_idx] + ns[idx])
                                - max(bp[idx], np[n_idx]), 0)
                            #
                            pre_info.position[idx] = max(
                                np[n_idx] - bp[idx], 0) + pp[idx]
                            pre_info.size[idx] = post_info.size[idx]
                    if pre_info.size.volume > 0:
                        new_e = Edge(pre_task, new_node)
                        new_node.add_input_edge(ic_idx, post_info.position,
                                                post_info.size, new_e)
                        pre_task.add_output_edge(pre_task_out_cluster_idx,
                                                 pre_info.position,
                                                 pre_info.size, new_e)

    def remove_reduction_links_between_new_node_and_si(self):
        """
        remove reduction links between new node and si
        """
        for node in self.si_node_before_cadd:
            all_in_edges = [e for e in node.input_clusters[0].all_edges]
            choose_edge_nr = self.node_position[all_in_edges[0].pre_task][3]
            for idx in range(1, len(all_in_edges)):
                edge = all_in_edges[idx]
                if self.node_position[edge.pre_task][3] != choose_edge_nr:
                    self.delete_edge(edge)
            for pre_node in node.in_tasks:
                pre_all_out_edges = [e for e in
                                     pre_node.output_clusters[0].all_edges]
                for edge in pre_all_out_edges:
                    if edge.post_task != node:
                        self.delete_edge(edge)

    @staticmethod
    def delete_edge(edge: Edge):
        # remove the edge in pre_task
        edge.pre_task.get_output_edge_cluster(edge).remove(edge)
        # remove the edge in post_task
        edge.post_task.get_input_edge_cluster(edge).remove(edge)

    def check(self):
        """
        检查拆分向量和待拆分任务块
        """
        if self.shape.nky != 1 or self.shape.nkx != 1:
            logging.warn(
                'Values in ky and kx dimensions of split vector will be ignored')
            self.shape.nky = 1
            self.shape.nkx = 1
        for idx in range(4):
            if self.base_node.shape[idx] == -1 and self.shape[idx] != 1:
                raise ValueError('Cannot split -1 dimension')

    def split_func_f(self, total_length, split_num, mode, aligned_num=None):
        if mode == SplitType.Average:
            return self.ave_split_func_f(total_length, split_num, aligned_num)
        else:
            raise NotImplementedError

    def split_func_r(self, total_length, split_num, mode, aligned_num=None):
        if mode == SplitType.Average:
            return self.ave_split_func_r(total_length, split_num, aligned_num)
        else:
            raise NotImplementedError

    def split_func_oxy(self, total_length, split_num, mode, aligned_num=None):
        if mode == SplitType.Average:
            return self.ave_split_func_oxy(total_length, split_num,
                                           aligned_num)
        else:
            raise NotImplementedError

    def split_func_ixy(self, total_length, split_num, mode, kx, pad_left,
                       pad_right, sx, aligned_num=None):
        if mode == SplitType.Average:
            return self.ave_split_func_ixy(total_length, split_num, kx,
                                           pad_left, pad_right, sx,
                                           aligned_num)
        else:
            raise NotImplementedError

    def ave_split_func_f(self, total_length, split_num, aligned_num=None):
        if not 0 < total_length and total_length < inf:
            raise ValueError(
                'The split value: %d is not in range (0, inf)!' % total_length)
        if aligned_num:
            split_length = []
            num = ceil(total_length / split_num / aligned_num)
            iter_if = 0
            for i in range(split_num - 1):
                split_length.append((iter_if, iter_if + num * aligned_num))
                iter_if += num * aligned_num
            if iter_if < total_length:
                split_length.append((iter_if, total_length))
            else:
                raise ValueError(
                    'Failed to split! total: %d split_num: %d, aligned_num %d'
                    % (total_length, split_num, aligned_num))
        else:
            if split_num > total_length:
                raise ValueError('split number is beyond the total length')
            stride = total_length / split_num
            split_length = []
            end = 0
            used_length = stride
            for _ in range(split_num - 1):
                start = end
                end = ceil(used_length)
                used_length += stride
                split_length.append((start, end))
            split_length.append((end, total_length))
        return split_length

    def ave_split_func_r(self, total_length, split_num, aligned_num=None):
        return self.ave_split_func_f(total_length, split_num, aligned_num)

    def ave_split_func_oxy(self, total_length, split_num, aligned_num=None):
        return self.ave_split_func_f(total_length, split_num, aligned_num)

    def ave_split_func_ixy(self, total_length, split_num, kx, pad_left,
                           pad_right, sx, aligned_num=None):
        out_x = total_length + pad_left + pad_right
        ox = max(floor((out_x - kx) / sx) + 1, 0)
        pad_right -= out_x - (ox - 1) * sx - kx
        if total_length == 0:
            ox = 0
        window_split = self.ave_split_func_f(ox, split_num, aligned_num)
        split_length = []
        for start_window, end_window in window_split:
            start = max(start_window * sx - pad_left, 0)
            end = max((end_window - 1) * sx + kx - pad_left, 0)
            split_length.append((start, end))
        split_length[-1] = (
            split_length[-1][0], split_length[-1][1] - pad_right)
        return split_length
