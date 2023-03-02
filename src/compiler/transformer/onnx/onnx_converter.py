# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import math
from copy import deepcopy
from typing import List

from src.compiler.transformer.constant import Constant
from src.compiler.transformer.onnx.onnx_attribute_dtype import \
    ONNXAttributeDataType
from src.compiler.transformer.onnx.onnx_basics import (ONNXGraph, ONNXNode,
                                                       create_onnx_attr,
                                                       create_onnx_node)
from src.compiler.transformer.onnx.onnx_data_type import ONNXDataType
from src.compiler.transformer.onnx.onnx_eliminator import ONNXEliminator
from src.compiler.transformer.onnx.onnx_op_type import ONNXOpType
from src.compiler.transformer.snn.snn_simplifier import SNNSimplifier


class ONNXConverter:
    '''对ONNX计算图进行一系列变换, 为ONNX计算图转换到Task Graph做准备

    需要经过一系列的passes:
    1. 化简global average pooling的pass
    2. 如果是SNN, 则通过SNNSimplifier做必要的计算图化简
    3. 重写数据类型的pass
    4. 统一flatten操作表示的pass
    5. 对transpose的优化pass
    6. 转换移位量化操作的pass
    7. 消除不需要的Clip算子的pass

    Attributes:
        graph: 解析后的ONNXGraph对象
        is_snn: 是否是SNN
        readable_file_path: 可读文件的保存路径
    '''

    def __init__(self, graph: ONNXGraph, is_snn: bool, readable_file_path: str = None) -> None:
        self.graph = graph
        self.is_snn = is_snn
        self.readable_file_path = readable_file_path

    def convert(self) -> ONNXGraph:
        self.simplify_global_avgpool()
        if self.is_snn:
            sim = SNNSimplifier(self.graph)
            sim.simplify()
        self.graph.topologize()
        self.rewrite_data_type()
        self.reshape_to_flatten()
        self.eliminate_fake_transpose()
        self.convert_clip_div_floor_clip()
        self.convert_last_clip()
        if self.readable_file_path is not None:
            self.graph.record(self.readable_file_path)
        return self.graph

    def simplify_global_avgpool(self):
        eliminated_data = list()
        eliminated_nodes = list()
        for node_name in self.graph.nodes:
            node = self.graph.nodes[node_name]
            if node.op_type == ONNXOpType.Mul.value:
                last_node = self.graph.get_last_node(node_name)
                if last_node.op_type == ONNXOpType.GlobalAveragePool.value:
                    next_node = self.graph.get_next_node(node_name)  # Clip
                    last_node.set_outputs(deepcopy(node.get_outputs()))
                    self.graph.input_connections[next_node.name] = deepcopy(
                        self.graph.input_connections[node_name])
                    self.graph.output_connections[last_node.name] = deepcopy(
                        self.graph.output_connections[node_name])
                    eliminated_data.append(node.get_inputs()[0])
                    second_next_node = self.graph.get_next_node(next_node.name)
                    # onnxsim把Mul和Div的常数合成一个了
                    if not node.inputs[1] == second_next_node.inputs[1]:
                        eliminated_data.append(node.get_inputs()[1])
                    eliminated_nodes.append(node_name)
        ONNXEliminator.eliminate(self.graph, eliminated_nodes, eliminated_data)

    def rewrite_data_type(self):
        for node_name in self.graph.nodes:
            node = self.graph.get_node(node_name)
            output_name = node.get_output()
            output_data = self.graph.get_data(output_name)
            if not node.data_type_rewritten:
                if ONNXOpType.op_change_dtype(node.get_op_type()):
                    for input_name in node.get_inputs():
                        data = self.graph.get_data(input_name)
                        if data.is_bias():
                            data.set_data_type(ONNXDataType.INT32.value)
                        else:
                            data.set_data_type(ONNXDataType.INT8.value)
                    output_data.set_data_type(ONNXDataType.INT32.value)
                    node.data_type_rewritten = True
                elif self.is_bit_shift(node):
                    self.rewrite_bit_shift_data_type(node)
                elif ONNXOpType.op_not_change_dtype(node.get_op_type()):
                    input_name = node.get_inputs()[0]
                    input_data = self.graph.get_data(input_name)
                    output_data.set_data_type(
                        input_data.get_data_type())  # 和输入数据类型相同
                elif ONNXOpType.is_spiking_neuron(node.op_type):
                    for input_name in node.get_inputs():
                        data = self.graph.data[input_name]
                        if self.graph.is_input_data(data.name):
                            data.set_data_type(ONNXDataType.INT28.value)
                    output_data.set_data_type(ONNXDataType.TERNERY.value)
                else:
                    raise NotImplementedError(
                        'Cannot rewrite data type of node {:s}'.format(node.get_op_type()))

    def get_4_nodes(self, node: ONNXNode) -> List[ONNXNode]:
        '''获取连续的4个结点
        '''
        node_name = node.get_name()
        second_node = self.graph.get_output_nodes(node_name)[0]
        third_node = self.graph.get_output_nodes(second_node.get_name())[0]
        fourth_node = self.graph.get_output_nodes(third_node.get_name())[0]
        return [node, second_node, third_node, fourth_node]

    def has_3_subsequent_nodes(self, node: ONNXNode) -> bool:
        '''判断某个结点是否有3个后续结点
        '''
        node_name = node.get_name()
        second_nodes = self.graph.get_output_nodes(node_name)
        if len(second_nodes) < 1:
            return False
        else:
            third_nodes = self.graph.get_output_nodes(
                second_nodes[0].get_name())
            if len(third_nodes) < 1:
                return False
            else:
                fourth_nodes = self.graph.get_output_nodes(
                    third_nodes[0].get_name())
                if len(fourth_nodes) < 1:
                    return False
                else:
                    return True

    def is_bit_shift(self, node: ONNXNode) -> bool:
        '''判断某个子图是否为移位量化操作

        需要连续出现Clip-Div-Floor-Clip
        '''
        if node.get_op_type() == ONNXOpType.Clip.value and self.has_3_subsequent_nodes(node):
            node0, node1, node2, node3 = self.get_4_nodes(node)
            node0_type = node0.get_op_type()
            node1_type = node1.get_op_type()
            node2_type = node2.get_op_type()
            node3_type = node3.get_op_type()
            if (node0_type == ONNXOpType.Clip.value and
                node1_type == ONNXOpType.Div.value and
                node2_type == ONNXOpType.Floor.value and
                    node3_type == ONNXOpType.Clip.value):
                # TODO(huanyu): 这里最好再判断一下两个Clip的范围
                assert node0.attributes['min'].get_value(
                ) == Constant.INT32_MIN.value
                assert node3.attributes['min'].get_value(
                ) == Constant.INT8_MIN.value
                return True
            else:
                return False
        else:
            False

    def rewrite_bit_shift_data_type(self, node: ONNXNode):
        '''重写移位量化操作的数据类型
        '''
        nodes = self.get_4_nodes(node)
        node0, node1, node2, node3 = nodes
        output0 = self.graph.get_data(node0.get_output())
        output1 = self.graph.get_data(node1.get_output())
        output2 = self.graph.get_data(node2.get_output())
        output3 = self.graph.get_data(node3.get_output())
        output0.set_data_type(ONNXDataType.INT32.value)
        output1.set_data_type(ONNXDataType.INT32.value)
        output2.set_data_type(ONNXDataType.INT32.value)
        output3.set_data_type(ONNXDataType.INT8.value)
        for n in nodes:
            n.data_type_rewritten = True

    def reshape_to_flatten(self):
        '''将reshape tensor为[1, -1]的Reshape算子转换成Flatten算子
        '''
        for _, node in self.graph.nodes.items():
            if node.get_op_type() == ONNXOpType.Reshape.value:
                for input_name in node.get_inputs():
                    input_data = self.graph.get_data(input_name)
                    if input_data.get_data() == [1, -1]:
                        node.set_op_type(ONNXOpType.Flatten.value)
                        node.get_inputs().remove(input_name)
                        del self.graph.data[input_name]

    def eliminate_fake_transpose(self):
        eliminated_data = list()
        eliminated_nodes = list()
        for node_name in self.graph.nodes:
            node = self.graph.get_node(node_name)
            if ONNXConverter.is_layout_converter(node):
                if not self.graph.is_output_node(node_name):
                    next_node = self.graph.get_output_nodes(node_name)[0]
                    if next_node.get_op_type() == ONNXOpType.Flatten.value:
                        input_data = self.graph.get_data(
                            next_node.get_inputs()[0])
                        next_node.set_inputs(deepcopy(node.get_inputs()))
                        self.graph.input_connections[next_node.get_name()] = deepcopy(
                            self.graph.input_connections[node_name])
                        last_node = self.graph.get_input_nodes(node_name)[0]
                        self.graph.output_connections[last_node.get_name()] = deepcopy(
                            self.graph.output_connections[node_name])
                        eliminated_data.append(input_data.get_name())
                        eliminated_nodes.append(node_name)
        ONNXEliminator.eliminate(self.graph, eliminated_nodes, eliminated_data)

    @staticmethod
    def is_layout_converter(node: ONNXNode):
        if node.get_op_type() == ONNXOpType.Transpose.value:
            if node.get_attribute('perm').value == [0, 2, 3, 1]:
                return True
            else:
                return False
        else:
            False

    def convert_clip_div_floor_clip(self):
        '''将Clip-Div-Floor-Clip子图转换成自定义的Cut算子
        '''
        cut_counter = 0
        eliminated_nodes = list()
        eliminated_data = list()
        new_nodes = list()
        for node_name in self.graph.nodes:
            node0 = self.graph.get_node(node_name)
            if node0.get_op_type() == ONNXOpType.Clip.value:
                if self.graph.is_output_node(node_name):
                    continue
                node1 = self.graph.get_output_nodes(node_name)[0]
                node1_name = node1.get_name()
                if node1.get_op_type() == ONNXOpType.Div.value:
                    node2 = self.graph.get_output_nodes(node1_name)[0]
                    node2_name = node2.get_name()
                    if node2.get_op_type() == ONNXOpType.Floor.value:
                        node3 = self.graph.get_output_nodes(node2_name)[0]
                        node3_name = node3.get_name()
                        if node3.get_op_type() == ONNXOpType.Clip.value:
                            eliminated_nodes.extend(
                                [node_name, node1_name, node2_name, node3_name])
                            inputs = deepcopy(node0.get_inputs())
                            outputs = deepcopy(node3.get_outputs())
                            input_nodes = deepcopy(
                                self.graph.get_input_nodes_names(node_name))
                            output_nodes = deepcopy(
                                self.graph.get_output_nodes_names(node3_name))
                            eliminated_data.extend(node1.get_inputs())
                            eliminated_data.extend(node2.get_inputs())
                            eliminated_data.extend(node3.get_inputs())
                            node1_inputs_names = node1.get_inputs()
                            for input_name in node1_inputs_names:
                                input_data = self.graph.get_data(input_name)
                                if input_data.is_static:
                                    bit_shift_num = int(
                                        math.log(input_data.get_data()[0], 2))
                            new_node_name = 'Cut_{:d}'.format(cut_counter)
                            cut_counter += 1
                            new_node = create_onnx_node(
                                new_node_name, ONNXOpType.Cut.value, inputs, outputs)
                            new_attr = create_onnx_attr(
                                'bit_shift_num', ONNXAttributeDataType.INT.value, bit_shift_num)
                            new_node.add_attribute(new_attr)
                            new_nodes.append(new_node)
                            self.graph.output_connections.update(
                                {input_nodes[0]: [new_node_name]})
                            self.graph.input_connections[output_nodes[0]].remove(
                                node3_name)
                            self.graph.input_connections[output_nodes[0]].append(
                                new_node_name)
                            self.graph.add_input_connections(
                                new_node_name, input_nodes)
                            self.graph.add_output_connections(
                                new_node_name, output_nodes)
        for new_node in new_nodes:
            self.graph.add_node(new_node)
        ONNXEliminator.eliminate(self.graph, eliminated_nodes, eliminated_data)

    def convert_last_clip(self):
        eliminated_nodes = list()
        eliminated_data = list()
        for node_name in self.graph.nodes:
            node = self.graph.get_node(node_name)
            if node.get_op_type() == ONNXOpType.Clip.value:
                if self.graph.is_output_node(node_name):
                    if node.get_attribute('min').get_value() == Constant.INT32_MIN.value:
                        eliminated_nodes.append(node_name)
                        eliminated_data.append(node.get_output())
                        for input_node_name in self.graph.get_input_nodes_names(node_name):
                            self.graph.output_connections[input_node_name] = tuple(
                            )
                else:
                    next_node = self.graph.get_output_nodes(node_name)[0]
                    if next_node.op_type in (ONNXOpType.Relu.value, ONNXOpType.MaxPool.value):
                        if node.get_attribute('min').get_value() == Constant.INT32_MIN.value:
                            next_node.inputs = node.inputs
                            eliminated_nodes.append(node_name)
                            eliminated_data.append(node.get_output())
                            for input_node_name in self.graph.get_input_nodes_names(node_name):
                                self.graph.output_connections[input_node_name] = self.graph.output_connections[node_name]
                            for output_node_name in self.graph.get_output_nodes_names(node_name):
                                self.graph.input_connections[output_node_name] = self.graph.input_connections[node_name]
        for node_name in eliminated_nodes:
            del self.graph.nodes[node_name]
            del self.graph.input_connections[node_name]
            del self.graph.output_connections[node_name]
        for data_name in eliminated_data:
            del self.graph.data[data_name]
