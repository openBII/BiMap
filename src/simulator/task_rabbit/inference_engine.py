# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import logging
import os
from copy import copy
from typing import Dict, List, Tuple

import numpy as np

from task_rabbit.operators import *
from task_rabbit.task_model.task_graph import TaskGraph
from task_rabbit.task_model.bias_type import BiasType
from task_rabbit.task_model.edge import (RearrangeInfo,
                                                       RearrangeInfoType)
from task_rabbit.task_model.precision import Precision
from task_rabbit.task_model.shape import Shape
from task_rabbit.task_model.task_visitor import TaskVisitor
from top.config import GlobalConfig


class InferenceEngine(TaskVisitor):
    """
    对TaskGraph进行前向推理，得到执行结果
    """

    def __init__(self, task_graph: TaskGraph, precision=np.int32):
        super(InferenceEngine, self).__init__()
        self.task_graph = task_graph
        # 每个存储任务块中的ndarray精度
        # 其要大于每个存储任务块本身的精度，以完成等价的处理
        self.precision = precision
        # 每个任务块的结算结果 {task_id: data}
        self.out_data: Dict[int, np.ndarray] = {}
        # 运行了第几次
        self.inference_time = -1

    def inference(self, input_data: Dict[int, np.ndarray] = None):
        '''
        对task_graph进行前行推理, 推理的结果存放在self.out_data中
        该函数执行前需先对要推理的图进行拓扑排序
        Args:
            input_data: 如果是None,则Task Graph的起始结点应该有预存数据，预存数据推理
            如果是字典，则表示结点ID到数据的映射
            debug: 如果为True，则self.out_data记录每一个任务块的结果，用于调试；
            如果为False，则只记录输出结点的结果
        '''
        if input_data is not None:
            for node_id, data in input_data.items():
                assert isinstance(data, np.ndarray)
                shape = self.task_graph.get_node(node_id).shape
                assert len(data.shape) == shape.dim_num
                self.out_data[node_id] = data
        self.task_graph.accept(self)
        self.inference_time += 1

    def visit_SIC(self, task: SICTaskBlock):
        shape = task.shape
        out = task.data
        if out is None:
            out = np.zeros((shape.ny, shape.nx, shape.nr),
                           dtype=self.precision)
        assert out.shape == (shape.ny, shape.nx, shape.nr)
        assert out.dtype == self.precision
        self.connect_to_in_task(task.input_clusters[0], out)
        self.out_data[task.id] = out

    def visit_SIC2D(self, task: SIC2DTaskBlock):
        shape = task.shape
        if task.data is not None:
            out = task.data
        else:
            out = np.zeros((shape.ny, shape.nx, shape.nr),
                           dtype=self.precision)
        assert out.shape == (shape.ny, shape.nx, shape.nr)
        assert out.dtype == self.precision
        self.connect_to_in_task(task.input_clusters[0], out)
        self.out_data[task.id] = out

    def visit_SIFC(self, task: SIFCTaskBlock):
        shape = task.shape
        if task.data is not None:
            out = task.data
        else:
            out = np.zeros(shape.nr, dtype=self.precision)
        assert out.shape == (shape.nr, )
        assert out.dtype == self.precision
        self.connect_to_in_task(task.input_clusters[0], out)
        self.out_data[task.id] = out

    def visit_SI(self, task: SITaskBlock):
        shape = task.shape
        if shape.ny == 0 and shape.nx == 0:
            if task.data is not None:
                out = task.data
            else:
                out = np.zeros(shape.nf, dtype=self.precision)
            assert out.shape == (shape.nf, )
        else:
            # 应该没有SI数据为2维的情况
            if task.data is not None:
                out = task.data
            else:
                out = np.zeros((shape.ny, shape.nx, shape.nf),
                               dtype=self.precision)
            assert out.shape == (shape.ny, shape.nx, shape.nf)
        assert out.dtype == self.precision
        self.connect_to_in_task(task.input_clusters[0], out)
        self.out_data[task.id] = out

    def visit_SW(self, task: SWTaskBlock):
        shape = task.shape
        out = task.data
        assert out.shape == (shape.nf, shape.nr, shape.nky, shape.nkx)
        assert out.dtype == self.precision
        self.out_data[task.id] = out

    def visit_SB(self, task: SBTaskBlock):
        shape = task.shape
        out = task.data
        assert out.shape == (shape.nf, ) and out.dtype == self.precision
        self.out_data[task.id] = out

    def visit_SWFC(self, task: SWFCTaskBlock):
        shape = task.shape
        out = task.data
        assert out.shape == (shape.nf, shape.nr)
        assert out.dtype == self.precision
        self.out_data[task.id] = out

    def visit_SW2D(self, task: SW2DTaskBlock):
        shape = task.shape
        out = task.data
        assert out.shape == (shape.nf, shape.nr, shape.nky, shape.nkx)
        assert out.dtype == self.precision
        self.out_data[task.id] = out

    def visit_CADD(self, task: CADDTaskBlock):
        shape = task.shape
        if shape.ny == 0 and shape.nx == 0:
            # 一维的情况
            x = np.zeros((shape.nf, task.n_branch), dtype=self.precision)
        else:
            # 三维的情况
            x = np.zeros((shape.ny, shape.nx, shape.nf, task.n_branch),
                         dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)

        for i in range(len(task.input_clusters)):
            if list(task.input_clusters[i].in_tasks)[0].task_type == TaskBlockType.SB:
                self.connect_to_in_task(
                    task.input_clusters[i], bias)
            else:
                if shape.ny == 0 and shape.nx == 0:
                    self.connect_to_in_task(
                        task.input_clusters[i], x[:, i])
                else:
                    self.connect_to_in_task(
                        task.input_clusters[i], x[:, :, :, i])
        if shape.ny == 0 and shape.nx == 0:
            out = x.sum(axis=1) + bias  # forward
        else:
            out = x.sum(axis=3) + bias  # forward
        self.out_data[task.id] = out

    def visit_CAVG(self, task: CAVGTaskBlock):
        shape = task.shape
        x_with_pad = np.zeros((shape.niy + task.pad_up + task.pad_down,
                               shape.nix + task.pad_left + task.pad_right,
                               shape.nf), dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0],
                                x_with_pad[task.pad_up:,
                                           task.pad_left:, :])
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[1], bias)
        # forward
        # 和maxpool一样当kernel为1的时候slicing有问题
        # for y in range(shape.ny):
        #     r_y = range(y * task.stride_y, y * task.stride_y + shape.nky)
        #     for x in range(shape.nx):
        #         r_x = range(x * task.stride_x, x * task.stride_x + shape.nkx)
        #         out[y, x, :] = \
        #             x_with_pad[r_y, r_x, :].mean(axis=0).mean(axis=0)
        self.out_data[task.id] = avgpool_forward(
            x_with_pad, (shape.nky, shape.nkx), (task.stride_y, task.stride_x)) + bias

    def visit_CVVH(self, task: CVVHTaskBlock):
        # 张量点乘
        shape = task.shape
        x1 = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        x2 = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], x1)
        self.connect_to_in_task(task.input_clusters[1], x2)
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[2], bias)
        # forward
        out = x1 * x2 + bias
        assert out.shape == (shape.ny, shape.nx, shape.nf)
        self.out_data[task.id] = out

    def visit_CVM(self, task: CVMTaskBlock):
        shape = task.shape
        x = np.zeros(shape.nr, dtype=self.precision)
        weight = np.zeros((shape.nf, shape.nr), dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], x)
        self.connect_to_in_task(task.input_clusters[1], weight)
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[2], bias)
        out = weight.dot(x) + bias
        self.out_data[task.id] = out

    def visit_CC(self, task: CCTaskBlock):
        shape = task.shape
        x_with_pad = np.zeros((shape.niy + task.pad_up + task.pad_down,
                               shape.nix + task.pad_left + task.pad_right,
                               shape.nr), dtype=self.precision)
        weight = np.zeros((shape.nf, shape.nr, shape.nky, shape.nkx),
                          dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        out = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0],
                                x_with_pad[task.pad_up:,
                                           task.pad_left:, :])
        self.connect_to_in_task(task.input_clusters[1], weight)
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[2], bias)
        # forward
        out = conv_forward(x_with_pad, weight, bias,
                           (task.stride_y, task.stride_x),
                           (task.dilation_y, task.dilation_x))
        self.out_data[task.id] = out

    def visit_CAX(self, task: CAXTaskBlock):
        shape = task.shape
        x1 = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        vector_a = np.zeros(shape.nf, dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], x1)
        self.connect_to_in_task(task.input_clusters[1], vector_a)
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[2], bias)
        # forward
        out = x1 * vector_a + bias
        assert out.shape == (shape.ny, shape.nx, shape.nf)
        self.out_data[task.id] = out

    def visit_CC2D(self, task: CC2DTaskBlock):
        self.visit_CC(task)

    def visit_CVS(self, task: CVSTaskBlock):
        shape = task.shape
        x1 = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        bias = np.zeros(shape.nf, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], x1)
        if task.bias_type == BiasType.VECTOR:
            self.connect_to_in_task(task.input_clusters[1], bias)
        # forward
        out = x1 * task.constant_a + bias
        assert out.shape == (shape.ny, shape.nx, shape.nf)
        self.out_data[task.id] = out

    def visit_CCMPB(self, task: CCMPBTaskBlock):
        shape = task.shape
        if shape.niy == 0 and shape.nix == 0:
            x = np.zeros(shape.nf, dtype=self.precision)
            out = np.zeros(shape.nf, dtype=self.precision)
            self.connect_to_in_task(task.input_clusters[0], x)
            # 应该是先移位后截取
            # 有些数移位后截取是不会溢出的
            self.out_data[task.id] = np.clip(x >> (task.bit_shift_num), task.CMP,
                                             127)
        else:
            x_with_pad = np.zeros((shape.niy + task.pad_up + task.pad_down,
                                   shape.nix + task.pad_left + task.pad_right,
                                   shape.nf), dtype=self.precision)
            out = np.zeros((shape.ny, shape.nx, shape.nf),
                           dtype=self.precision)
            self.connect_to_in_task(task.input_clusters[0],
                                    x_with_pad[task.pad_up:,
                                               task.pad_left:, :])
            out = maxpool_forward(
                x_with_pad, (shape.nky, shape.nkx), (task.stride_y, task.stride_x))
            if task.precision == Precision.INT_8:
                self.out_data[task.id] = np.clip(out >> (task.bit_shift_num), task.CMP,
                                                 127)
            elif task.precision == Precision.INT_32:
                self.out_data[task.id] = np.clip(out >> (task.bit_shift_num), task.CMP,
                                                 2147483647)
            elif task.precision == Precision.TERNARY:
                self.out_data[task.id] = np.clip(
                    out >> (task.bit_shift_num), task.CMP, 1)
            else:
                raise NotImplementedError

    def visit_CCMPS(self, task: CCMPSTaskBlock):
        shape = task.shape
        x_with_pad = np.zeros((shape.niy + task.pad_up + task.pad_down,
                               shape.nix + task.pad_left + task.pad_right,
                               shape.nf), dtype=self.precision)
        out = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0],
                                x_with_pad[task.pad_up:,
                                           task.pad_left:, :])
        # forward
        for y in range(shape.ny):
            y_s = y * task.stride_y
            y_e = y * task.stride_y + shape.nky
            for x in range(shape.nx):
                x_s = x * task.stride_x
                x_e = x * task.stride_x + shape.nkx
                out[y, x, :] = np.minimum(
                    x_with_pad[y_s: y_e, x_s: x_e, :].max(axis=0).max(axis=0),
                    task.CMP)
        self.out_data[task.id] = np.clip(out >> (task.bit_shift_num), -128,
                                         127)

    def visit_CLUT(self, task: CLUTTaskBlock):
        shape = task.shape
        x = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
        lut = np.zeros(task.lut_len, dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], x)
        self.connect_to_in_task(task.input_clusters[1], lut)
        # forward
        iter = np.nditer(x, op_flags=['readwrite'])
        while not(iter.finished):
            iter[0] = lut[iter[0] if iter[0] >=
                          0 else iter[0] + 2 ** task.input_data_width]
            iter.iternext()
        self.out_data[task.id] = x

    def visit_CLIF(self, task: CLIFTaskBlock):
        shape = task.shape
        if shape.ny == 0 and shape.nx == 0:
            x = np.zeros(shape.nf, dtype=self.precision)
            v_init = np.zeros(shape.nf, dtype=self.precision)
            # ref_cnt = np.zeros((shape.ny, shape.nx, shape.nf),
            #                    dtype=self.precision)
            self.connect_to_in_task(task.input_clusters[0], x)
            self.connect_to_in_task(task.input_clusters[1], v_init)
            # self.connect_to_in_task(task.input_clusters[2], ref_cnt)
            out, v_mem = lif_forward(
                x, v_init, task.v_th_0, task.v_leaky_alpha, task.v_leaky_beta, task.v_leaky_adpt_en, task.v_reset)
            self.out_data[task.id] = out
            assert len(list(task.input_clusters[1].in_tasks)) == 1
            self.out_data[list(task.input_clusters[1].in_tasks)[0].id] = v_mem
        else:
            x = np.zeros((shape.ny, shape.nx, shape.nf), dtype=self.precision)
            v_init = np.zeros((shape.ny, shape.nx, shape.nf),
                              dtype=self.precision)
            # ref_cnt = np.zeros((shape.ny, shape.nx, shape.nf),
            #                    dtype=self.precision)
            self.connect_to_in_task(task.input_clusters[0], x)
            self.connect_to_in_task(task.input_clusters[1], v_init)
            # self.connect_to_in_task(task.input_clusters[2], ref_cnt)
            out, v_mem = lif_forward(
                x, v_init, task.v_th_0, task.v_leaky_alpha, task.v_leaky_beta, task.v_leaky_adpt_en, task.v_reset)
            assert out.shape == (shape.ny, shape.nx, shape.nf)
            assert v_mem.shape == (shape.ny, shape.nx, shape.nf)
            self.out_data[task.id] = out
            assert len(list(task.input_clusters[1].in_tasks)) == 1
            self.out_data[list(task.input_clusters[1].in_tasks)[0].id] = v_mem

    def visit_INPUT(self, task: InputTaskBlock):
        # 起始结点，不需要做什么是
        pass

    def visit_OUTPUT(self, task: OutputTaskBlock):
        shape = task.shape
        if shape.ny != 0 and shape.nx != 0 and shape.nf != 0 and shape.nr == 0:
            out = np.zeros((shape.ny, shape.nx, shape.nf),
                           dtype=self.precision)
        if shape.ny != 0 and shape.nx != 0 and shape.nf == 0 and shape.nr != 0:
            out = np.zeros((shape.ny, shape.nx, shape.nr),
                           dtype=self.precision)
        if shape.ny == 0 and shape.nx == 0 and shape.nf != 0 and shape.nr == 0:
            out = np.zeros(shape.nf, dtype=self.precision)
        if shape.ny == 0 and shape.nx == 0 and shape.nf == 0 and shape.nr != 0:
            out = np.zeros(shape.nr, dtype=self.precision)
        if shape.ny == 0 and shape.nx == 0 and shape.nf != 0 and shape.nr != 0 and shape.nky != 0 and shape.nkx != 0:
            out = np.zeros((shape.nf, shape.nr, shape.nky,
                           shape.nkx), dtype=self.precision)
        if shape.ny == 0 and shape.nx == 0 and shape.nf != 0 and shape.nr != 0 and shape.nky == 0 and shape.nkx == 0:
            out = np.zeros((shape.nf, shape.nr), dtype=self.precision)
        self.connect_to_in_task(task.input_clusters[0], out)
        self.out_data[task.id] = out

    @staticmethod
    def set_data(src_start: Shape, src_size: Shape, src_data: np.ndarray,
                 dst_start: Shape, dst_size: Shape, dst_data: np.ndarray):
        # ndarray的存储顺序为y x f
        assert src_size.volume == dst_size.volume
        src_range, dst_range = tuple(), tuple()
        for i in range(6):
            if src_size[i] != 0:
                src_range += (range(src_start[i],
                                    src_start[i] + src_size[i]), )
            if dst_size[i] != 0:
                dst_range += (range(dst_start[i],
                                    dst_start[i] + dst_size[i]), )

        # Global pooling + 全连接的时候一出现
        if len(dst_data.shape) == 1 and len(src_data.shape) == 3:
            dst_data[dst_range[0][0]:dst_range[0][1]
                     ] = src_data[0, 0, src_range[2][0]:src_range[2][1]]
        else:
            dst_data[np.ix_(*dst_range)] = src_data[np.ix_(*src_range)]

    def connect_to_in_task(self, cluster: EdgeCluster, dst_data: np.ndarray):
        '''
        将cluster表示的边簇的输入边连接的另一端任务块的数据传递到当前边簇
        将数据存储到dst_data中
        '''
        for edge in cluster.all_enable_edges:
            in_task_out = self.out_data[edge.in_task.id]

            pre_position = edge.in_task.get_output_edge_position(edge)
            pre_size = edge.in_task.get_output_edge_size(edge)
            post_position = edge.out_task.get_input_edge_position(edge)
            post_size = edge.out_task.get_input_edge_size(edge)

            # 计算数据重排
            src_data, src_postion, src_size = InferenceEngine.rearrange_data_and_info(
                edge.rearrange_info, in_task_out, pre_position, pre_size, dst_data.shape)
            InferenceEngine.set_data(src_postion, src_size,
                                     src_data, post_position,
                                     post_size, dst_data)

    @staticmethod
    def rearrange_data_and_info(rearrange_info_list: List[RearrangeInfo], in_data: np.ndarray,
                                in_position: Shape, in_size: Shape,
                                final_out_size: Tuple[int]):
        out_position, out_size = copy(in_position), copy(in_size)
        if in_data.size != in_size.volume:
            in_range = tuple()
            for i in range(6):
                if in_size[i] != 0:
                    in_range += (range(in_position[i],
                                 in_position[i] + in_size[i]), )
            out_data = in_data[np.ix_(*in_range)]
            out_position = Shape()
        else:
            out_data = in_data
        rearrange_size = InferenceEngine.get_rearrange_size(rearrange_info_list,
                                                            final_out_size)
        for idx, rearrange_info in enumerate(rearrange_info_list):
            if rearrange_info.rearrange_type == RearrangeInfoType.IDENTITY:
                continue
            elif rearrange_info.rearrange_type == RearrangeInfoType.RESHAPE:
                # 暂时只考虑了flatten的情况
                assert len(rearrange_info.rearrange_matrix) == 1
                if np.array(rearrange_size[idx]).prod() != out_data.size:
                    rearrange_size[idx][0] = out_data.size
                    out_position = Shape()
                out_data = InferenceEngine.reshape_ndarray(
                    out_data, rearrange_size[idx],
                    list(rearrange_info.rearrange_matrix[0]),
                    [])
                out_size = InferenceEngine.reshape_out_info(out_size)
            elif rearrange_info.rearrange_type == RearrangeInfoType.PERMUTE:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.ROTATE:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.SHIFT:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.SCALE:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.SHEAR:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.AFFINE:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.REFLECT:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.PROJECT:
                raise NotImplementedError
            elif rearrange_info.rearrange_type == RearrangeInfoType.SHUFFLE:
                raise NotImplementedError
            else:
                raise TypeError('invalid rearrange type')
        return out_data, out_position, out_size

    @staticmethod
    def get_rearrange_size(rearrange_info_list: List[RearrangeInfo], final_out_size: Tuple[int]):
        out_size = []
        if len(rearrange_info_list) == 1:
            out_size.append(list(final_out_size))
        return out_size

    @staticmethod
    def reshape_out_info(out_size: Shape):
        out_size.nf = out_size.ny * out_size.nx * out_size.nf
        out_size.ny = 0
        out_size.nx = 0
        # TODO 这里对复杂情况下的reshape情况不支持
        return out_size

    @staticmethod
    def reshape_ndarray(in_data: np.ndarray, size: List[int], dims_src: List[int],
                        dims_dst: List[int]):
        dims_src = InferenceEngine.change_dims_meaning(dims_src)
        if dims_dst != []:
            dims_dst = InferenceEngine.change_dims_meaning(dims_dst)
        in_shape = np.array(list(in_data.shape))
        assert len(in_shape) == len(dims_src)
        assert in_shape.prod() == np.array(size).prod()  # 检查前后元素数量是否相同
        out_data = in_data.transpose(dims_src[::-1]).flatten()
        return out_data

    @staticmethod
    def change_dims_meaning(dims: List[int]):
        sorted_dims = copy(dims)
        sorted_dims.sort()
        new_dims = []
        for value in dims:
            new_dims.append(sorted_dims.index(value))
        return new_dims

    def save_data(self, case_name: str, out_dir: str = None, debug: bool = False):
        '''
        将推理结果保存成文件
        Args:
            case_name: 用例名
            out_dir: 输出文件目录, 如果为None，则默认为 temp/case_name/task_out/
            如果self.debug为False, 则输出文件的名字为：o_{net_id}_{socket_id}_{frame_id}.dat
            其中net_id、socket_id、frame_id分别表示网络ID、输出结点ID和第几个输入数据。
            如果self.debug为True, 则输出文件的名字为：{task_id}.dat
            task_id为任务快结点ID
            debug: 如果为True，则保存self.out_data记录的每一个任务块的结果，用于调试；
            如果为False，则只保存输出结点的结果
        '''
        if out_dir is None:
            out_dir = GlobalConfig.Path['temp'] + case_name + '/' + \
                GlobalConfig.Path['task_out']
        os.makedirs(out_dir, exist_ok=True)

        for task_id, data in self.out_data.items():
            task_block = self.task_graph.get_node(task_id)
            data = data.reshape(-1)
            if not debug:
                if not isinstance(task_block, OutputTaskBlock):
                    continue
                assert data.dtype == np.int32
                out_path = out_dir + 'o_{:d}_{:d}_{:d}.dat'.format(
                    0, task_block.socket_id, self.inference_time)

                with open(out_path, "w") as f:
                    for x in data:
                        f.write(str(x) + "\n")
            else:
                out_path = out_dir + '{:d}.dat'.format(task_block.id)
                # if task_block.precision == Precision.INT_32:
                with open(out_path, "w") as f:
                    for x in data:
                        f.write(str(x) + "\n")

                # elif task_block.precision == Precision.INT_8:
                #     aligned_length = int(np.ceil(len(data) / 4) * 4)
                #     aligned_data = np.zeros(aligned_length, dtype=data.dtype)
                #     aligned_data[0:len(data)] = data
                #     with open(out_path, "w") as f:
                #         for j in range(len(aligned_data) // 4):
                #             i = 4 * j
                #             f.write(struct.pack(
                #                 '>bbbb', aligned_data[i+3], aligned_data[i+2], aligned_data[i+1], aligned_data[i]).hex() + "\n")
                # elif task_block.precision == Precision.TERNARY:
                #     aligned_length = int(np.ceil(len(data) / 16) * 16)
                #     aligned_data = np.zeros(aligned_length, dtype=data.dtype)
                #     aligned_data[0:len(data)] = data
                #     with open(out_path, "w") as f:
                #         for j in range(len(aligned_data) // 16):
                #             temp = 0
                #             for i in range(16):
                #                 temp += aligned_data[16 * j + i] << (2 * i)
                #             f.write(struct.pack('>i', temp).hex() + "\n")
                # else:
                #     raise NotImplementedError

        logging.info('Generate data of dynamic blocks in {:s}'.format(out_dir))


if __name__ == '__main__':
    from task_rabbit.task_graph_parser import TaskGraphParser

    case_path = GlobalConfig.Path['test_lib'] + 'mapping_lib/1C1P/cadd.map.txt'
    case_name = 'cadd'
    input_type = 'map'

    task_graph = TaskGraphParser.parse(case_path, 'map')

    ie = InferenceEngine(task_graph)
    ie.inference()

    logging.info('Finish the inference of Task Graph')

    out_dir = GlobalConfig.Path['temp'] + \
        case_name + '/' + GlobalConfig.Path['map_out']
    ie.save_data(case_name, out_dir, debug=True)
