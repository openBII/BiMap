# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import struct
from collections import OrderedDict
from typing import Dict, List, Sequence, Union

from src.compiler.transformer.onnx.onnx_attribute_dtype import \
    ONNXAttributeDataType
from src.compiler.transformer.onnx.onnx_data_type import ONNXDataType

import onnx


def parse_raw_data(raw_data, data_type, shape) -> Union[List[float], List[int]]:
    '''解析ONNX中的raw数据
    '''
    if data_type == ONNXDataType.FLOAT.value:
        f_num = 1
        for d in shape:
            f_num = f_num * d
        code = 'f' * f_num
        float_tuple = struct.unpack(code, raw_data)
        return list(float_tuple)
    elif data_type == ONNXDataType.INT64.value:
        q_num = 1
        for d in shape:
            q_num = q_num * d
        code = 'q' * q_num
        long_tuple = struct.unpack(code, raw_data)
        return list(long_tuple)
    else:
        raise NotImplementedError('Raw data of {:s} cannot be parsed'.format(
            ONNXDataType.get_dtype_name(data_type)))


class ONNXData():
    '''ONNX数据

    Attributes:
        - name: 数据的名字, 计算图中的唯一标识ID
        - data_type: 数据类型, 与ONNXDataType中定义一致
        - shape: 数据形状
        - data: 静态数据
        - num_dims: 维度数
        - is_static: 是否包含静态数据
    '''

    def __init__(self, onnx_data=None):
        self.name: str = None
        self.data_type: int = None
        self.shape: List[int] = None
        self.data: List = None
        self.num_dims: int = None
        self.is_static: bool = False
        if onnx_data is not None:
            self.parse_onnx_data(onnx_data)

    def parse_onnx_data(self, onnx_data):
        if isinstance(onnx_data, onnx.ValueInfoProto):
            self.parse_onnx_value_info(onnx_data)
        else:
            self.parse_onnx_tensor(onnx_data)

    def parse_onnx_value_info(self, onnx_value_info):
        '''解析动态数据
        '''
        self.name = onnx_value_info.name
        self.data_type = onnx_value_info.type.tensor_type.elem_type
        self.shape = [
            dim.dim_value for dim in onnx_value_info.type.tensor_type.shape.dim]
        self.num_dims = len(self.shape)

    def parse_onnx_tensor(self, onnx_tensor):
        '''解析静态数据
        '''
        self.name = onnx_tensor.name
        self.data_type = onnx_tensor.data_type
        self.shape = [dim for dim in onnx_tensor.dims]
        self.num_dims = len(self.shape)
        self.is_static = True
        if onnx_tensor.raw_data is not None:
            self.data = parse_raw_data(
                onnx_tensor.raw_data, self.data_type, self.shape)
        else:
            if ONNXDataType.is_float(self.data_type):
                self.data = [f for f in onnx_tensor.float_data]
            elif ONNXDataType.is_int32(self.data_type):
                self.data = [i for i in onnx_tensor.int32_data]
            elif ONNXDataType.is_string(self.data_type):
                self.data = [s for s in onnx_tensor.string_data]
            elif ONNXDataType.is_double(self.data_type):
                self.data = [d for d in onnx_tensor.double_data]
            elif ONNXDataType.is_uint64(self.data_type):
                self.data = [u for u in onnx_tensor.uint64_data]
            elif ONNXDataType.is_int64(self.data_type):
                self.data = [i for i in onnx_tensor.int64_data]
            else:
                raise ValueError(
                    'Tensor of {:s} cannot be parsed'.format(ONNXDataType.get_dtype_name(self.data_type)))

    def get_name(self):
        return self.name

    def get_data_type(self):
        return self.data_type

    def set_data_type(self, data_type: int):
        self.data_type = data_type

    def get_shape(self):
        return self.shape

    def get_num_dims(self):
        return self.num_dims

    def is_bias(self):
        if self.num_dims == 1:
            return True
        else:
            return False

    def get_data(self):
        return self.data

    def __repr__(self) -> str:
        data_str = ('\tData ' + self.name + ' {\n\t\t' +
                    'data_type: ' + ONNXDataType.get_dtype_name(self.data_type) + '\n\t\t' +
                    'shape: ' + repr(self.shape) + '\n\t\t' +
                    'is_static: ' + repr(self.is_static) + '\n\t}\n')
        return data_str


class ONNXAttribute():
    '''ONNX结点包含的属性

    Attributes:
        - name: 属性的名字
        - data_type: 属性的数据类型, 与ONNXAttributeDataType中的定义一致
        - value: 属性值
    '''

    def __init__(self, onnx_attribute=None):
        self.name: str = ''
        self.data_type: int = 0
        self.value = 0
        if onnx_attribute is not None:
            self.parse_onnx_attribute(onnx_attribute)

    def parse_onnx_attribute(self, onnx_attribute):
        self.name = onnx_attribute.name
        self.data_type = onnx_attribute.type
        if self.data_type == ONNXAttributeDataType.FLOAT.value:  # float
            self.value = onnx_attribute.f
        elif self.data_type == ONNXAttributeDataType.INT.value:  # int
            self.value = onnx_attribute.i
        elif self.data_type == ONNXAttributeDataType.STRING.value:  # string
            self.value = onnx_attribute.s
        elif self.data_type == ONNXAttributeDataType.FLOATS.value:  # list of floats
            self.value = [f for f in onnx_attribute.floats]
        elif self.data_type == ONNXAttributeDataType.INTS.value:  # list of ints
            self.value = [i for i in onnx_attribute.ints]
        elif self.data_type == ONNXAttributeDataType.STRINGS.value:  # list of strings
            self.value = [s for s in onnx_attribute.strings]
        elif self.data_type == ONNXAttributeDataType.TENSOR.value:  # tensor
            self.value = ONNXData(onnx_attribute.t)
        else:
            raise NotImplementedError(
                'Attributes of {:s} cannot be parsed'.format(ONNXAttributeDataType.get_dtype_name(self.data_type)))

    def get_name(self):
        return self.name

    def set_name(self, name: str):
        self.name = name

    def get_data_type(self):
        return self.data_type

    def set_data_type(self, data_type: int):
        self.data_type = data_type

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def __repr__(self) -> str:
        attribute_str = ('\n\t\tAttribute ' + self.name + ' {\n\t\t\t' +
                         'data_type: ' + ONNXAttributeDataType.get_dtype_name(self.data_type) + '\n\t\t\t' +
                         'value: ' + repr(self.value) + '\n\t\t}')
        return attribute_str


class ONNXNode():
    '''ONNX结点

    Attributes:
        - name: 结点的名字, 计算图中的唯一标识ID
        - op_type: 结点类型
        - inputs: 结点的输入数据的名字
        - outputs: 结点的输出数据的名字
        - attributes: 结点的属性, key为属性名字, value为ONNXAttribute对象
        - data_type_rewritten: 是否已经重写过数据类型, 在ONNXConverter中会用到
    '''

    def __init__(self, onnx_node: onnx.NodeProto = None):
        self.name: str = None
        self.op_type: str = None
        self.inputs: List[str] = None
        self.outputs: List[str] = None
        self.attributes: Dict[str, ONNXAttribute] = {}
        self.data_type_rewritten = False
        if onnx_node is not None:
            self.parse_onnx_node(onnx_node)

    def parse_onnx_node(self, onnx_node):
        self.name = onnx_node.name
        self.op_type = onnx_node.op_type
        self.inputs = [s for s in onnx_node.input]
        self.outputs = [s for s in onnx_node.output]
        for attribute in onnx_node.attribute:
            onnx_attribute = ONNXAttribute(attribute)
            self.attributes.update({attribute.name: onnx_attribute})

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, name: str) -> ONNXAttribute:
        return self.attributes[name]

    def add_attribute(self, onnx_attribute: ONNXAttribute):
        self.attributes.update({onnx_attribute.get_name(): onnx_attribute})

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    def get_op_type(self) -> str:
        return self.op_type

    def set_op_type(self, op_type: str):
        self.op_type = op_type

    def get_inputs(self) -> List[str]:
        return self.inputs

    def set_inputs(self, inputs: List[str]):
        self.inputs = inputs

    def get_output(self) -> str:
        '''获取单输出结点的输出
        '''
        assert len(self.outputs) == 1, 'Unsupported number of outputs'
        return self.outputs[0]

    def get_outputs(self) -> List[str]:
        return self.outputs

    def set_outputs(self, outputs: List[str]):
        self.outputs = outputs

    def __repr__(self) -> str:
        node_str = ('\tNode ' + self.name + ' {\n\t\t' +
                    'op_type: ' + self.op_type + '\n\t\t' +
                    'inputs: ' + repr(self.inputs) + '\n\t\t' +
                    'outputs: ' + repr(self.outputs))
        for _, attribute in self.attributes.items():
            node_str += repr(attribute)
        return node_str


class ONNXGraph():
    '''ONNX计算图

    当输入ModelProto对象时, 会自动调用parse_onnx_graph方法完成解析

    Attributes:
        - nodes: name到ONNXNode对象的字典
        - data: name到ONNXData对象的字典
        - input_connections: key为结点的name, value为产生当前结点输入数据的所有结点的name组成的列表
        - output_connections: key为结点的name, value为使用当前结点输出数据的所有结点的name组成的列表
    '''

    def __init__(self, onnx_graph: onnx.GraphProto = None):
        self.nodes: Dict[str, ONNXNode] = {}
        self.data: Dict[str, ONNXData] = {}
        self.input_connections: Dict[str, List[str]] = {}
        self.output_connections: Dict[str, List[str]] = {}
        if onnx_graph is not None:
            self.parse_onnx_graph(onnx_graph)

    def create_connections(self):
        '''构建input_connections和output_connections
        '''
        for current_node_name in self.nodes:
            current_node = self.get_node(current_node_name)
            input_list = []
            output_list = []
            for node_name in self.nodes:
                node = self.get_node(node_name)
                for o in node.outputs:
                    if o in current_node.inputs:
                        input_list.append(node_name)
                for i in node.inputs:
                    if i in current_node.outputs:
                        output_list.append(node_name)
            self.input_connections.update(
                {current_node_name: list(set(input_list))})
            self.output_connections.update(
                {current_node_name: list(set(output_list))})

    def get_input_nodes(self, node_name: str) -> List[ONNXNode]:
        '''获得生成某个结点输入的所有结点
        '''
        return [self.nodes[input_node_name] for input_node_name in self.input_connections[node_name]]

    def get_output_nodes(self, node_name: str) -> List[ONNXNode]:
        '''获得使用某个结点输出的所有结点
        '''
        return [self.nodes[output_node_name] for output_node_name in self.output_connections[node_name]]

    def get_last_node(self, node_name: str) -> ONNXNode:
        last_node = self.get_input_nodes(node_name)[0]
        return last_node

    def get_next_node(self, node_name: str) -> ONNXNode:
        next_node = self.get_output_nodes(node_name)[0]
        return next_node

    def get_input_nodes_names(self, node_name) -> List[str]:
        return self.input_connections[node_name]

    def get_output_nodes_names(self, node_name) -> List[str]:
        return self.output_connections[node_name]

    def get_node(self, node_name) -> ONNXNode:
        return self.nodes[node_name]

    def add_node(self, node: ONNXNode):
        self.nodes.update({node.get_name(): node})

    def add_nodes(self, nodes: Sequence[ONNXNode]):
        for node in nodes:
            self.add_node(node)

    def add_input_connections(self, node_name, input_nodes_names):
        self.input_connections.update({node_name: input_nodes_names})

    def add_output_connections(self, node_name, output_nodes_names):
        self.output_connections.update({node_name: output_nodes_names})

    def parse_onnx_graph(self, onnx_graph: onnx.GraphProto):
        '''解析ONNX模型

        - 将node转换成ONNXNode对象
        - 将initializer转换成带有静态数据的ONNXData对象
        - 将input, output, value_info转换成ONNXData对象
        - 创建input_connections和output_connections
        '''
        for node in onnx_graph.node:
            onnx_node = ONNXNode(node)
            self.nodes.update({onnx_node.name: onnx_node})
        for tensor in onnx_graph.initializer:
            onnx_tensor = ONNXData()
            onnx_tensor.parse_onnx_tensor(tensor)
            self.data.update({onnx_tensor.name: onnx_tensor})
        for value_info in onnx_graph.input:
            if value_info.name not in self.data.keys():
                onnx_value_info = ONNXData()
                onnx_value_info.parse_onnx_value_info(value_info)
                self.data.update({onnx_value_info.name: onnx_value_info})
        for value_info in onnx_graph.value_info:
            onnx_value_info = ONNXData()
            onnx_value_info.parse_onnx_value_info(value_info)
            self.data.update({onnx_value_info.name: onnx_value_info})
        for value_info in onnx_graph.output:
            onnx_value_info = ONNXData()
            onnx_value_info.parse_onnx_value_info(value_info)
            self.data.update({onnx_value_info.name: onnx_value_info})
        self.create_connections()

    def get_static_data(self, data_name: str):
        data = self.data[data_name]
        assert data.data is not None, 'This data item is just a placeholder which does not have data reserved in the graph'
        return data.data

    def get_data(self, data_name: str) -> ONNXData:
        return self.data[data_name]

    def __repr__(self) -> str:
        graph_str = 'Graph {\n'
        for node_name, node in self.nodes.items():
            graph_str += repr(node)
            graph_str += '\n\t\tinput_nodes: ' + \
                repr(self.input_connections[node_name])
            graph_str += '\n\t\toutput_nodes: ' + \
                repr(self.output_connections[node_name])
            graph_str += '\n\t}\n'
        for _, data in self.data.items():
            graph_str += repr(data)
        graph_str += '}'
        return graph_str

    def record(self, path: str):
        '''将repr(ONNXGraph)的结果写入文件
        '''
        with open(path, 'w') as f:
            f.write(self.__repr__())

    def is_output_node(self, node_name: str):
        '''某个结点是否为输出结点

        输出结点的定义为没有后续结点使用该结点的输出数据
        '''
        return len(self.output_connections[node_name]) == 0

    def is_input_node(self, node_name):
        '''某个结点是否为输入结点

        输入结点的定义为没有前置结点生成该结点的输入数据, 即该结点的输入数据为计算图输入或静态数据
        '''
        return len(self.input_connections[node_name]) == 0

    def is_input_data(self, data_name: str):
        '''数据是否为输入数据

        这里的输入数据指不是由计算图中的结点生成的数据
        '''
        for node_name in self.nodes:
            if data_name in self.nodes[node_name].outputs:
                return False
        return True

    def topologize(self) -> None:
        '''将所有结点拓扑排序并重新存到nodes里

        Raises:
            Exception: 当前不支持带环的图
        '''
        in_degrees = {}
        for node_id in self.nodes:
            in_degrees[node_id] = len(self.input_connections[node_id])
        vertex_num = len(in_degrees)
        node_zero_in = [u for u in in_degrees if in_degrees[u] == 0]
        nodes_topo = OrderedDict()
        while node_zero_in:
            u = node_zero_in.pop()
            nodes_topo[u] = self.nodes[u]
            for out_task_id in self.output_connections[u]:
                in_degrees[out_task_id] -= 1
                if in_degrees[out_task_id] == 0:
                    node_zero_in.append(out_task_id)  # 再次筛选入度为0的顶点
        if len(nodes_topo) == vertex_num:  # 如果循环结束后存在非0入度的顶点说明图中有环, 不存在拓扑排序
            self.nodes = nodes_topo
        else:
            raise Exception('There\'s a circle in the task graph.')


def create_onnx_node(name: str, op_type: str, inputs: List[str], outputs: List[str]) -> ONNXNode:
    '''创建ONNXNode对象
    '''
    node = ONNXNode()
    node.set_name(name)
    node.set_op_type(op_type)
    node.set_inputs(inputs)
    node.set_outputs(outputs)
    return node


def create_onnx_attr(name: str, data_type: int, value) -> ONNXAttribute:
    '''创建ONNXAttribute对象
    '''
    attr = ONNXAttribute()
    attr.set_name(name)
    attr.set_data_type(data_type)
    attr.set_value(value)
    return attr
