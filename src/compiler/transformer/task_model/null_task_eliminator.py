# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from copy import deepcopy
from typing import Dict, List, Union

from src.compiler.transformer.onnx.onnx_op_type import ONNXOpType
from src.compiler.transformer.task_model.attribute_type import AttributeType
from src.compiler.transformer.task_model.rearrange_info_type import \
    RearrangeInfoType
from src.compiler.transformer.task_model.task_block_type import TaskBlockType
from src.compiler.transformer.task_model.task_graph_basics import (Attribute,
                                                                   CTaskBlock,
                                                                   Edge,
                                                                   TaskGraph,
                                                                   get_min,
                                                                   init_shape)
from src.compiler.transformer.task_model.task_graph_eliminator import \
    TaskGraphEliminator
from src.simulator.task_rabbit.task_model.precision import Precision


class NullTaskEliminator:
    @staticmethod
    def null_task_eliminate(task_graph: TaskGraph):
        '''完成TASK_NULL类型的任务块的转换

        目前包含的功能:
        - 添加数据重排信息
        - 处理量化信息
        - 处理Concat算子

        Args:
            task_graph: 完成了算子转换和任务组连接生成的任务图

        Raises:
            NotImplementedError: 当TASK_NULL任务块对应的原本的ONNX算子不属于特定类型集合时会抛出错误
        '''
        NullTaskEliminator.eliminate_flatten(task_graph)
        NullTaskEliminator.eliminate_concat(task_graph)
        NullTaskEliminator.eliminate_cut(task_graph)
        for _, block in task_graph.blocks.items():
            if isinstance(block, CTaskBlock) and block.get_type() == TaskBlockType.TASK_NULL.value:
                original_node = block.get_original_node()
                op_type = original_node.get_op_type()
                if op_type not in (ONNXOpType.Flatten.value, ONNXOpType.Cut.value, ONNXOpType.Concat.value):
                    raise NotImplementedError(
                        'Original node type of TASK_NULL block is {:s}, which cannot be handled'.format(op_type))

    @staticmethod
    def eliminate_flatten(task_graph: TaskGraph):
        '''转换TASK_NULL类型的任务块并且对应的原始算子为Flatten
        '''
        eliminated_blocks_ids = list()
        eliminated_groups_ids = list()
        eliminated_edges_ids = list()
        for _, block in task_graph.blocks.items():
            if isinstance(block, CTaskBlock) and block.get_type() == TaskBlockType.TASK_NULL.value:
                original_node = block.get_original_node()
                op_type = original_node.get_op_type()
                c_block_id = block.get_id()
                if op_type == ONNXOpType.Flatten.value:
                    input_s_block_id = task_graph.get_last_blocks_ids(c_block_id)[
                        0]
                    input_s_block = task_graph.get_block(input_s_block_id)
                    output_s_block_id = task_graph.get_next_blocks_ids(c_block_id)[
                        0]
                    output_s_block = task_graph.get_block(output_s_block_id)
                    edge_id = input_s_block.get_input_clusters()[0].get_interfaces()[
                        0].get_id()
                    edge = task_graph.get_edge(edge_id)
                    edge.set_dst_block_id(output_s_block_id)
                    # 默认Tensor排布为[y, x, f], 代表如何读取Tensor
                    edge.add_rearrange_info(
                        RearrangeInfoType.RESHAPE.value, [2, 1, 0])
                    output_s_block.rewrite_input_cluster(edge_id)
                    inner_edge_id = input_s_block.get_output_edges_ids()[0]
                    output_edge_id = block.get_output_edges_ids()[0]
                    eliminated_blocks_ids.append(c_block_id)
                    eliminated_blocks_ids.append(input_s_block_id)
                    eliminated_groups_ids.append(c_block_id)
                    eliminated_edges_ids.append(inner_edge_id)
                    eliminated_edges_ids.append(output_edge_id)
        TaskGraphEliminator.eliminate(task_graph, eliminated_blocks_ids,
                                      eliminated_groups_ids, eliminated_edges_ids)

    @staticmethod
    def eliminate_concat(task_graph: TaskGraph):
        '''转换TASK_NULL类型的任务块并且对应的原始算子为Concat
        '''
        eliminated_blocks_ids = list()
        eliminated_groups_ids = list()
        eliminated_edges_ids = list()
        for _, block in task_graph.blocks.items():
            if isinstance(block, CTaskBlock) and block.get_type() == TaskBlockType.TASK_NULL.value:  # TASK_NULL
                original_node = block.get_original_node()
                op_type = original_node.get_op_type()
                c_block_id = block.get_id()
                # TODO(huanyu): 这里Concat的顺序是没有保证的
                if op_type == ONNXOpType.Concat.value:
                    # 删除输入数据块和占位块
                    input_s_blocks_ids = task_graph.get_last_blocks_ids(
                        c_block_id)  # Concat的输入块
                    output_s_blocks_ids = task_graph.get_next_blocks_ids(
                        c_block_id)  # Concat的输出块
                    input_c_blocks_ids = list()
                    # 删除Concat的输入块的输入边和输出边
                    # 删除生成Concat输入的计算块输出边簇中的接口
                    for input_s_block_id in input_s_blocks_ids:
                        input_s_block = task_graph.get_block(input_s_block_id)
                        input_c_block_id = task_graph.get_last_blocks_ids(input_s_block_id)[
                            0]
                        input_c_block = task_graph.get_block(input_c_block_id)
                        output_cluster_of_input_c_block = input_c_block.get_output_cluster()
                        output_cluster_of_input_c_block.delete_interface(0)
                        input_edge_id = input_s_block.get_input_cluster(
                            0).get_interface(0).get_id()
                        inner_edge_id = input_s_block.get_output_edges_ids()[0]
                        eliminated_blocks_ids.append(input_s_block_id)
                        eliminated_edges_ids.append(inner_edge_id)
                        eliminated_edges_ids.append(input_edge_id)
                        input_c_blocks_ids.append(input_c_block_id)
                    # 删除Concat的输出块的输入边
                    # 删除Concat的输出块的输入边簇中的接口
                    for output_s_block_id in output_s_blocks_ids:
                        output_s_block = task_graph.get_block(
                            output_s_block_id)
                        input_cluster_of_output_s_block = output_s_block.get_input_cluster(
                            0)
                        edge_id = input_cluster_of_output_s_block.get_interface(
                            0).get_id()
                        eliminated_edges_ids.append(edge_id)
                        input_cluster_of_output_s_block.delete_interface(0)
                    # 重新生成连接
                    for output_s_block_id in output_s_blocks_ids:
                        output_s_block = task_graph.get_block(
                            output_s_block_id)
                        input_cluster_of_output_s_block = output_s_block.get_input_cluster(
                            0)
                        assert input_cluster_of_output_s_block.get_num_interfaces() == 0  # 接口已被清空
                        position = init_shape()
                        final_position = deepcopy(position)
                        for input_c_block_id in input_c_blocks_ids:
                            input_c_block = task_graph.get_block(
                                input_c_block_id)
                            output_cluster_of_input_c_block = input_c_block.get_output_cluster()
                            edge_id = task_graph.ctx.get_edge_counter()
                            edge = Edge(edge_id, input_c_block_id,
                                        output_s_block_id)
                            task_graph.edges.update({edge_id: edge})
                            output_cluster_of_input_c_block.add_interface(
                                position=init_shape(),
                                size=output_cluster_of_input_c_block.get_shape(),
                                edge_id=edge_id
                            )
                            input_cluster_of_output_s_block.add_interface(
                                position=deepcopy(position),
                                size=deepcopy(
                                    output_cluster_of_input_c_block.get_shape()),
                                edge_id=edge_id
                            )
                            final_position = NullTaskEliminator.calculate_final_position(
                                position, output_cluster_of_input_c_block.get_shape())
                            NullTaskEliminator.position_increment(
                                position, output_cluster_of_input_c_block.get_shape(), axes='f')
                        input_cluster_of_output_s_block.set_shape(
                            final_position)
                        assert NullTaskEliminator.concat_checker(
                            final_position, output_s_block.get_output_cluster().get_shape())
                    # 删除Concat块
                    eliminated_groups_ids.append(c_block_id)
                    eliminated_blocks_ids.append(c_block_id)
        TaskGraphEliminator.eliminate(task_graph, eliminated_blocks_ids,
                                      eliminated_groups_ids, eliminated_edges_ids)

    @staticmethod
    def position_increment(position: Dict, size: Dict, axes: Union[str, List[str]] = 'f'):
        '''根据position和size计算新的起始位置

        在给定轴i上position[i] = position[i] + size[i]

        Args:
            - position: 当前位置
            - size: 当前数据的形状
            - axes: 需要进行累加的轴
        '''
        if isinstance(axes, List):
            for axis in axes:
                position[axis] = position[axis] + size[axis]
        else:
            position[axes] = position[axes] + size[axes]

    @staticmethod
    def calculate_final_position(position: Dict, size: Dict) -> Dict:
        '''计算每拼接一个数据块后的最终位置

        为在所有维度上初始位置和形状的加和

        Args:
            - position: 当前位置
            - size: 当前数据的形状
        '''
        final_position = {}
        all_axes = ('y', 'x', 'f', 'r', 'ky', 'kx')
        for axis in all_axes:
            final_position[axis] = position[axis] + size[axis]
        return final_position

    @staticmethod
    def concat_checker(input_shape: Dict, output_shape: Dict) -> bool:
        '''检查concat转换后的结果是否正确

        比较input_shape和output_shape是否相同, 两者可能在r和f维度上交换
        '''
        if (input_shape['y'] == output_shape['y'] and
            input_shape['x'] == output_shape['x'] and
            input_shape['ky'] == output_shape['ky'] and
                input_shape['kx'] == output_shape['kx']):
            if output_shape['f'] == 0:
                return input_shape['f'] == output_shape['r']
            else:
                return input_shape['f'] == output_shape['f']
        else:
            return False

    @staticmethod
    def eliminate_cut(task_graph: TaskGraph):
        '''转换TASK_NULL类型的任务块并且对应的原始算子为量化移位操作

        如果当前任务块后面为CCMPB任务块, 则直接将量化信息记录到后一个任务块中并删除当前任务块, 否则将当前任务块转换成CCMPB任务块
        '''
        eliminated_blocks_ids = list()
        eliminated_groups_ids = list()
        eliminated_edges_ids = list()
        for _, block in task_graph.blocks.items():
            if isinstance(block, CTaskBlock) and block.get_type() == TaskBlockType.TASK_NULL.value:
                original_node = block.get_original_node()
                op_type = original_node.get_op_type()
                c_block_id = block.get_id()
                if op_type == ONNXOpType.Cut.value:
                    input_s_block_id = task_graph.get_last_blocks_ids(c_block_id)[
                        0]
                    input_s_block = task_graph.get_block(input_s_block_id)
                    output_s_block_id = task_graph.get_next_blocks_ids(c_block_id)[
                        0]
                    output_s_block = task_graph.get_block(output_s_block_id)
                    output_c_block = task_graph.get_next_c_blocks_of_c_block(c_block_id)[
                        0]
                    new_attr_value = original_node.get_attribute(
                        'bit_shift_num').get_value()
                    new_attr = Attribute(
                        type=AttributeType.BIT_SHIGT_NUM.value, precision=Precision.INT_32.value, value=new_attr_value)
                    output_s_block.set_precision(
                        Precision.INT_32.value)
                    if not output_c_block.get_type() == TaskBlockType.CCMPB.value:  # 如果不满足则将当前任务块替换成CCMPB
                        block.set_type(TaskBlockType.CCMPB.value)
                        block.set_shape(
                            {
                                'y': 1 if input_s_block.get_shape()['y'] == 1 and output_s_block.get_shape()['y'] == 0 else output_s_block.get_shape()['y'],
                                'x': 1 if input_s_block.get_shape()['x'] == 1 and output_s_block.get_shape()['x'] == 0 else output_s_block.get_shape()['x'],
                                'f': output_s_block.get_shape()['r'] if output_s_block.get_shape()['f'] == 0 else output_s_block.get_shape()['f'],
                                'ky': 1,
                                'kx': 1,
                                'iy': input_s_block.get_shape()['y'],
                                'ix': input_s_block.get_shape()['x']
                            }
                        )
                        attribute0 = Attribute(
                            type=AttributeType.KERNEL_X.value, precision=Precision.INT_32.value, value=1)
                        attribute1 = Attribute(
                            type=AttributeType.KERNEL_Y.value, precision=Precision.INT_32.value, value=1)
                        attribute2 = Attribute(
                            type=AttributeType.STRIDE_X.value, precision=Precision.INT_32.value, value=1)
                        attribute3 = Attribute(
                            type=AttributeType.STRIDE_Y.value, precision=Precision.INT_32.value, value=1)
                        attribute4 = Attribute(
                            type=AttributeType.PAD_TOP.value, precision=Precision.INT_32.value, value=0)
                        attribute5 = Attribute(
                            type=AttributeType.PAD_DOWN.value, precision=Precision.INT_32.value, value=0)
                        attribute6 = Attribute(
                            type=AttributeType.PAD_LEFT.value, precision=Precision.INT_32.value, value=0)
                        attribute7 = Attribute(
                            type=AttributeType.PAD_RIGHT.value, precision=Precision.INT_32.value, value=0)
                        attribute8 = Attribute(
                            type=AttributeType.CMP.value,
                            precision=Precision.INT_32.value,
                            value=get_min(block.get_precision()))
                        block.add_attributes(
                            {
                                'kernel_x': attribute0,
                                'kernel_y': attribute1,
                                'stride_x': attribute2,
                                'stride_y': attribute3,
                                'pad_top': attribute4,
                                'pad_down': attribute5,
                                'pad_left': attribute6,
                                'pad_right': attribute7,
                                'cmp': attribute8,
                                'bit_shift_num': new_attr
                            }
                        )
                        block.get_output_cluster().set_shape(
                            {
                                'y': block.get_shape()['y'],
                                'x': block.get_shape()['x'],
                                'f': block.get_shape()['f'],
                            }
                        )
                        block.get_output_cluster().get_interface(0).set_position(
                            {
                                'y': 0,
                                'x': 0,
                                'f': 0
                            }
                        )
                        block.get_output_cluster().get_interface(0).set_size(
                            block.get_output_cluster().get_shape()
                        )
                        if not (block.get_shape()['y'] == 1 and output_s_block.get_shape()['y'] == 0):
                            output_s_block.get_input_cluster(0).set_shape(
                                block.get_output_cluster().get_shape()
                            )
                            output_s_block.get_input_cluster(0).get_interface(0).set_size(
                                block.get_output_cluster().get_shape()
                            )
                    else:
                        output_c_block.add_attribute(new_attr, 'bit_shift_num')
                        edge_id = input_s_block.get_input_clusters()[0].get_interfaces()[
                            0].get_id()
                        edge = task_graph.get_edge(edge_id)
                        edge.set_dst_block_id(output_s_block_id)
                        output_s_block.rewrite_input_cluster(edge_id)
                        inner_edge_id = input_s_block.get_output_edges_ids()[0]
                        output_edge_id = block.get_output_edges_ids()[0]
                        eliminated_blocks_ids.append(c_block_id)
                        eliminated_blocks_ids.append(input_s_block_id)
                        eliminated_groups_ids.append(c_block_id)
                        eliminated_edges_ids.append(inner_edge_id)
                        eliminated_edges_ids.append(output_edge_id)
        TaskGraphEliminator.eliminate(task_graph, eliminated_blocks_ids,
                                      eliminated_groups_ids, eliminated_edges_ids)
