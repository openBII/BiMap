# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from copy import deepcopy
from typing import Dict, List, Union

from bidict import bidict
from src.compiler.transformer.constant import Constant
from src.compiler.transformer.onnx.onnx_basics import (ONNXAttribute, ONNXData,
                                                       ONNXGraph, ONNXNode)
from src.compiler.transformer.onnx.onnx_data_type import ONNXDataType
from src.compiler.transformer.onnx.onnx_op_type import ONNXOpType
from src.compiler.transformer.task_model.attribute_type import AttributeType
from src.compiler.transformer.task_model.rearrange_info_type import \
    RearrangeInfoType
from src.compiler.transformer.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision


def init_shape() -> Dict:
    '''初始化形状
    '''
    return {'y': 0,
            'x': 0,
            'f': 0,
            'r': 0,
            'ky': 0,
            'kx': 0}


def get_min(precision: int):
    '''获取不同精度下的最小值
    '''
    if precision == Precision.INT_8.value:
        return Constant.INT8_MIN.value
    elif precision == Precision.TERNARY.value:
        return Constant.TERNARY_MIN.value
    else:
        raise NotImplementedError('Unsupported precision')


def convert_data_type(onnx_data_type: int) -> int:
    '''将ONNX计算图中的精度转换成任务图中的精度
    '''
    onnx_data_type = ONNXDataType.get_dtype(onnx_data_type)
    if onnx_data_type == ONNXDataType.FLOAT:
        return Precision.FLOAT_32.value
    elif onnx_data_type == ONNXDataType.UINT8:
        return Precision.UINT_8.value
    elif onnx_data_type == ONNXDataType.INT8:
        return Precision.INT_8.value
    elif onnx_data_type == ONNXDataType.UINT16:
        return Precision.UINT_16.value
    elif onnx_data_type == ONNXDataType.INT16:
        return Precision.INT_16.value
    elif onnx_data_type == ONNXDataType.INT32:
        return Precision.INT_32.value
    elif onnx_data_type == ONNXDataType.FLOAT16:
        return Precision.FLOAT_16.value
    elif onnx_data_type == ONNXDataType.UINT32:
        return Precision.UINT_32.value
    elif onnx_data_type == ONNXDataType.TERNERY:
        return Precision.TERNARY.value
    elif onnx_data_type == ONNXDataType.INT28:
        return Precision.INT_28.value
    else:
        raise ValueError(
            'Unsupported ONNX data type, which cannot be converted to precision in Task Graph')


class Tensor:
    '''张量类

    Attributes:
        - dims: 张量的维度
        - precision: 张量的精度
        - float_data: 张量精度为浮点数类型时数据存储在float_data中
        - int32_data: 张量精度为整数类型时数据存储在int32_data中
        - uint32_data: 张量精度为无符号整数类型时数据存储在uint32_data中

    Raises:
        - NotImplementedError: 不支持的精度类型
    '''

    def __init__(self, onnx_data: ONNXData = None) -> None:
        if onnx_data is not None:
            self.dims = onnx_data.shape
            self.precision = convert_data_type(onnx_data.get_data_type())
            self.float_data = None
            self.int32_data = None
            self.uint32_data = None
            if Precision.is_float(self.precision):
                self.float_data = onnx_data.data
            elif Precision.is_int32(self.precision):
                self.int32_data = onnx_data.data
            elif Precision.is_uint32(self.precision):
                self.uint32_data = onnx_data.data
            else:
                raise NotImplementedError(
                    'Unsupported data precision in Task Graph')
        else:
            self.dims = None
            self.precision = None
            self.float_data = None
            self.int32_data = None
            self.uint32_data = None

    def set_dims(self, dims):
        self.dims = dims

    def get_dims(self):
        return self.dims

    def set_precision(self, precision):
        self.precision = precision

    def get_precision(self):
        return self.precision

    def set_int32_data(self, data):
        self.int32_data = data

    def get_data(self):
        if Precision.is_float(self.precision):
            return self.float_data
        elif Precision.is_int32(self.precision):
            return self.int32_data
        elif Precision.is_uint32(self.precision):
            return self.uint32_data
        else:
            raise NotImplementedError(
                'Unsupported data precision in Task Graph')

    def __repr__(self) -> str:
        tensor_str = (
            '\n\t\t\tTensor {' +
            '\n\t\t\t\tDims: ' + repr(self.dims) +
            '\n\t\t\t\tPrecision: ' + Precision.get_precision_name(self.precision) +
            '\n\t\t\t\tData: ' + repr(self.get_data()) +
            '\n\t\t\t}'
        )
        return tensor_str


class Attribute():
    '''任务块中的属性类

    只会出现在计算任务块中

    Attributes:
        - type: 属性的类型, 对应AttributeType枚举类
        - precision: 属性的数据精度
        - int_value: 当精度为整数类型时属性值存储在int_value中
        - float_value: 当精度为浮点数类型时属性值存储在float_value中
    '''

    def __init__(self, type: int, precision: int, value: Union[int, float]) -> None:
        self.type = type
        self.precision = precision
        self.int_value = None
        self.float_value = None
        if Precision.is_int(self.precision):
            self.int_value = int(value)
        elif Precision.is_float(self.precision):
            self.float_value = value

    def get_type(self):
        return self.type

    def get_precision(self):
        return self.precision

    def get_value(self):
        if self.int_value is not None:
            return self.int_value
        else:
            return self.float_value

    def set_value(self, value):
        if self.int_value is not None:
            self.int_value = value
        else:
            self.float_value = value

    def __repr__(self) -> str:
        attribute_str = (
            '\n\t\tAttribute {' +
            '\n\t\t\tType: ' + AttributeType.get_attribute_name(self.type) +
            '\n\t\t\tPrecision: ' + Precision.get_precision_name(self.precision) +
            '\n\t\t\tValue: ' + str(self.get_value()) +
            '\n\t\t}'
        )
        return attribute_str


class EdgeInterface():
    '''边接口类

    边接口代表边簇中的一个slicing, 会对应一条实际的边

    Attributes:
        - position: 接口在边簇中的起始位置
        - size: 接口在边簇中的形状大小
        - edge_id: 对应的边的ID
    '''

    def __init__(self, position: Dict, size: Dict, edge_id: int) -> None:
        self.position = position
        self.size = size
        self.edge_id = edge_id

    def get_id(self):
        return self.edge_id

    def set_id(self, edge_id: int):
        self.edge_id = edge_id

    def get_position(self):
        return self.position

    def set_position(self, position: Dict):
        self.position.update(position)

    def get_size(self):
        return self.size

    def set_size(self, size: Dict):
        self.size.update(size)

    def delete_position(self):
        self.position = None

    def delete_size(self):
        self.size = None

    def __repr__(self) -> str:
        edge_interface_str = (
            '\n\t\t\t\tInterface {' +
            '\n\t\t\t\t\tPosition: ' + repr(self.position) +
            '\n\t\t\t\t\tSize: ' + repr(self.size) +
            '\n\t\t\t\t\tEdge ID: ' + str(self.edge_id) +
            '\n\t\t\t\t}'
        )
        return edge_interface_str


class RearrangeInfo():
    '''数据重排信息类

    出现在Edge类中

    Attributes:
        - type: 数据重排信息类型, 对应RearrangeInfoType枚举类
        - matrix: 数据重排矩阵, 可以包含一个或多个数据重排矩阵, 会被转换成Tensor对象的列表
    '''

    def __init__(self, type: int, matrix: List) -> None:
        self.type = type
        self.matrix: List[Tensor] = list()
        if isinstance(matrix[0], int):
            tensor = Tensor()
            tensor.set_dims([len(matrix)])
            tensor.set_precision(Precision.INT_32.value)
            tensor.set_int32_data(matrix)
            self.matrix.append(tensor)
        else:
            for mat in matrix:
                tensor = Tensor()
                tensor.set_dims([len(mat)])
                tensor.set_precision(Precision.INT_32.value)
                tensor.set_int32_data(mat)
                self.matrix.append(tensor)

    def get_type(self):
        return self.type

    def get_matrix(self) -> List[Tensor]:
        return self.matrix

    def __repr__(self) -> str:
        info_str = (
            '\n\t\tRearrange Info {' +
            '\n\t\t\tType: ' +
            RearrangeInfoType.get_rearrange_info_name(self.type)
        )
        for mat in self.matrix:
            info_str += repr(mat)
        info_str += '\n\t\t}'
        return info_str


class Edge():
    '''边类

    构建起任务块之间的连接

    Attributes:
        - id: 边的ID
        - src_block_id: 边的数据来源对应的任务块ID
        - dst_block_id: 使用边传递的数据的任务块ID
        - rearrange_info: 数据重排信息
    '''

    def __init__(self, id: int, src_block_id: int, dst_block_id: int, rearrange_info: List[RearrangeInfo] = None) -> None:
        self.id = id
        self.src_block_id = src_block_id
        self.dst_block_id = dst_block_id
        self.rearrange_info = list()
        if rearrange_info is not None:
            self.rearrange_info = rearrange_info

    def add_rearrange_info(self, type: int, matrix: List):
        rearrange_info = RearrangeInfo(type, matrix)
        self.rearrange_info.append(rearrange_info)

    def get_rearrange_info(self) -> List[RearrangeInfo]:
        return self.rearrange_info

    def get_src_block_id(self):
        return self.src_block_id

    def get_dst_block_id(self):
        return self.dst_block_id

    def set_dst_block_id(self, block_id: int):
        self.dst_block_id = block_id

    def set_src_block_id(self, block_id: int):
        self.src_block_id = block_id

    def get_id(self):
        return self.id

    def __repr__(self) -> str:
        edge_str = (
            '\n\tEdge ' + str(self.id) + ' {' +
            '\n\t\tFrom Block ' +
            str(self.src_block_id) + ' To Block ' + str(self.dst_block_id)
        )
        for rearrange_info in self.rearrange_info:
            edge_str += repr(rearrange_info)
        edge_str += '\n\t}'
        return edge_str


class EdgeCluster():
    '''边簇类

    边簇是任务块中的概念, 例如当任务块的某个输入来源于多条边传递的数据, 每条边会对应边簇中的一个边接口, 而所有的接口被包含在边簇之中

    Attributes:
        - shape: 边簇的形状, 一般对应数据的完整形状
        - interfaces: 包含的边接口组成的列表
        - num_interfaces: 边接口的数量
    '''

    def __init__(self, shape: Dict, positions: List[Dict] = None, sizes: List[Dict] = None, edge_ids: List[int] = None) -> None:
        '''构造函数

        通过fill方法用positions, sizes, edge_ids来构建一系列EdgeInterface
        '''
        self.shape = shape
        self.interfaces: List[EdgeInterface] = list()
        self.num_interfaces = None
        if positions is not None and sizes is not None and edge_ids is not None:
            self.fill(positions, sizes, edge_ids)

    def get_interfaces(self) -> List[EdgeInterface]:
        return self.interfaces

    def get_interface(self, num: int) -> EdgeInterface:
        return self.interfaces[num]

    def get_num_interfaces(self):
        return self.num_interfaces

    def get_shape(self) -> Dict:
        return self.shape

    def set_shape(self, shape_dict: Dict):
        self.shape.update(shape_dict)

    def fill(self, positions: List[Dict], sizes: List[Dict], edge_ids: List[int]):
        self.num_interfaces = len(edge_ids)
        for i in range(self.num_interfaces):
            self.interfaces.append(EdgeInterface(
                positions[i], sizes[i], edge_ids[i]))

    def delete_interface(self, num: int):
        del self.interfaces[num]
        self.num_interfaces = len(self.interfaces)

    def add_interface(self, position: Dict, size: Dict, edge_id: int):
        self.interfaces.append(EdgeInterface(position, size, edge_id))
        self.num_interfaces = len(self.interfaces)

    def __repr__(self) -> str:
        edge_cluster_str = (
            '\n\t\t\tCluster {' +
            '\n\t\t\t\tShape: ' + repr(self.shape)
        )
        for interface in self.interfaces:
            edge_cluster_str += repr(interface)
        edge_cluster_str += '\n\t\t\t}'
        return edge_cluster_str


class STaskBlock():
    '''存储任务块类

    Attributes:
        - id: 任务块ID
        - type: 任务块类型, 对应TaskBlockType枚举类
        - precision: 任务块精度, 代表任务块输出数据的精度, 对应Precision枚举类
        - shape: 任务块形状
        - input_clusters: 任务块输入边簇的列表
        - output_clusters: 任务块输出边簇的列表
        - data: 静态数据
        - has_data: 是否包含静态数据
        - original_data: 对应的ONNX计算图中的数据
        - pipeline_num: 流水的行数
        - has_pipeline_num: 是否代表流水缓冲数据
    '''

    def __init__(self, id: int, onnx_data: ONNXData = None, onnx_node_op_type: str = None):
        self.id = id
        self.type: int = None
        self.precision: int = None
        self.shape: Dict[str, int] = init_shape()
        self.input_clusters: List[EdgeCluster] = list()
        self.output_clusters: List[EdgeCluster] = list()
        self.data: Tensor = None
        self.has_data: bool = False
        self.original_data: ONNXData = onnx_data
        self.pipeline_num: int = None
        self.has_pipeline_num: bool = False
        if (onnx_data is not None and onnx_node_op_type is not None):
            self.transform_onnx_data(onnx_data, onnx_node_op_type)

    def is_input(self):
        return len(self.input_clusters) == 0

    def is_output(self):
        return len(self.output_clusters) == 0

    def has_input_connection(self, edge_id: int):
        for input_cluster in self.get_input_clusters():
            for interface in input_cluster.get_interfaces():
                if interface.get_id() == edge_id:
                    return True
        return False

    def has_output_connection(self, edge_id: int):
        for output_cluster in self.get_output_clusters():
            for interface in output_cluster.get_interfaces():
                if interface.get_id() == edge_id:
                    return True
        return False

    def get_type(self):
        return self.type

    def get_precision(self):
        return self.precision

    def set_precision(self, precision: int):
        self.precision = precision

    def set_type(self, type: int):
        self.type = type

    def get_id(self):
        return self.id

    def set_id(self, id: int):
        self.id = id

    def set_shape(self, shape: Dict[str, int]):
        self.shape.update(shape)

    def get_data(self) -> Tensor:
        return self.data

    def set_data(self, data: Tensor):
        self.data = data
        self.has_data = True

    def set_pipeline_num(self, pipeline_num: int):
        self.has_pipeline_num = True
        self.pipeline_num = pipeline_num

    def get_pipeline_num(self):
        return self.pipeline_num

    def get_input_clusters(self) -> List[EdgeCluster]:
        return self.input_clusters

    def get_input_cluster(self, num: int) -> EdgeCluster:
        return self.input_clusters[num]

    def get_output_clusters(self) -> List[EdgeCluster]:
        return self.output_clusters

    def get_output_cluster(self) -> EdgeCluster:
        return self.output_clusters[0]  # 目前不支持多输出

    def get_output_edges_ids(self) -> List[int]:
        output_edges_ids = list()
        for output_cluster in self.get_output_clusters():
            for interface in output_cluster.get_interfaces():
                edge_id = interface.get_id()
                output_edges_ids.append(edge_id)
        return list(set(output_edges_ids))

    def transform_onnx_data(self, onnx_data: ONNXData, onnx_node_op_type: str):
        '''将ONNXData转换成存储任务块

        不生成输入输出边簇信息, 边簇信息在算子-任务组转换和生成组间连接过程中逐步生成
        '''
        self.precision = convert_data_type(onnx_data.get_data_type())
        if onnx_data.is_static:
            self.has_data = True
            self.data = Tensor(onnx_data)
        self.type = self.infer_block_type(onnx_data, onnx_node_op_type)
        self.convert_shape(onnx_data.shape)

    def create_output_cluster(self, edge_id: int):
        positions = list()
        positions.append(init_shape())
        sizes = [self.shape]
        edge_ids = [edge_id]
        output_cluster = EdgeCluster(self.shape, positions, sizes, edge_ids)
        self.output_clusters.append(output_cluster)

    def create_empty_output_cluster(self):
        output_shape = deepcopy(self.shape)
        output_cluster = EdgeCluster(output_shape)
        self.output_clusters.append(output_cluster)

    def create_empty_input_cluster(self, cluster_shape: Dict[str, int]):
        input_cluster = EdgeCluster(cluster_shape)
        self.input_clusters.append(input_cluster)

    def add_interface_to_input_cluster(self, input_cluster_num: int,
                                       position: Dict[str, int], size: Dict[str, int],
                                       edge_id: int):
        input_cluster = self.get_input_cluster(input_cluster_num)
        input_cluster.add_interface(position, size, edge_id)

    def create_input_cluster(self, edge_id: int, c_block_output_cluster_shape: Dict[str, int]):
        positions = [init_shape()]
        sizes = [c_block_output_cluster_shape]
        edge_ids = [edge_id]
        input_cluster = EdgeCluster(
            c_block_output_cluster_shape, positions, sizes, edge_ids)
        self.input_clusters.append(input_cluster)

    def rewrite_input_cluster(self, edge_id: int):
        input_cluster = self.get_input_cluster(0)
        interface = input_cluster.get_interface(0)
        interface.set_id(edge_id)
        interface.size['y'] = self.shape['y']
        interface.size['x'] = self.shape['x']
        interface.size['f'] = self.shape['r'] if self.shape['r'] != 0 else self.shape['f']

    def get_shape(self) -> Dict[str, int]:
        return self.shape

    def convert_shape(self, onnx_data_shape: List[int]):
        '''通过ONNX计算图中的数据得到任务块的形状
        '''
        if self.type == TaskBlockType.SI.value:
            self.shape['y'] = 0 if len(
                onnx_data_shape) == 2 else onnx_data_shape[2]
            self.shape['x'] = 0 if len(
                onnx_data_shape) == 2 else onnx_data_shape[3]
            self.shape['f'] = onnx_data_shape[1]
        elif self.type == TaskBlockType.SIC2D.value or self.type == TaskBlockType.SIC.value:
            self.shape['y'] = onnx_data_shape[2]
            self.shape['x'] = onnx_data_shape[3]
            self.shape['r'] = onnx_data_shape[1]
        elif self.type == TaskBlockType.SIFC.value:
            self.shape['r'] = onnx_data_shape[1]
        elif self.type == TaskBlockType.SW.value:
            self.shape['f'] = onnx_data_shape[0]
            self.shape['r'] = onnx_data_shape[1]
            self.shape['ky'] = onnx_data_shape[2]
            self.shape['kx'] = onnx_data_shape[3]
        elif self.type == TaskBlockType.SWFC.value:
            self.shape['f'] = onnx_data_shape[0]
            self.shape['r'] = onnx_data_shape[1]
        elif self.type == TaskBlockType.SB.value:
            self.shape['f'] = onnx_data_shape[0]
        else:
            raise ValueError('Non-existent data block type')

    def infer_block_type(self, onnx_data: ONNXData, onnx_node_op_type: str):
        '''根据ONNX计算图中的数据和使用该数据的算子推理出任务块的类型
        '''
        if onnx_node_op_type == ONNXOpType.Conv.value:
            if self.has_data:
                if onnx_data.num_dims == 4:
                    return TaskBlockType.SW.value
                elif onnx_data.num_dims == 1:
                    return TaskBlockType.SB.value
                else:
                    raise NotImplementedError(
                        'Data block of this type cannot be created')
            else:
                return TaskBlockType.SIC.value
        elif onnx_node_op_type == ONNXOpType.Gemm.value:
            if self.has_data:
                if onnx_data.num_dims == 2:
                    return TaskBlockType.SWFC.value
                elif onnx_data.num_dims == 1:
                    return TaskBlockType.SB.value
                else:
                    raise NotImplementedError(
                        'Data block of this type cannot be created')
            else:
                return TaskBlockType.SIFC.value
        else:
            return TaskBlockType.SI.value

    def __repr__(self) -> str:
        block_str = (
            '\n\tSTaskBlock ' + str(self.id) + ' {' +
            '\n\t\tType: ' + TaskBlockType.get_name(self.type) +
            '\n\t\tPrecision: ' + Precision.get_precision_name(self.precision) +
            '\n\t\tShape: ' + repr(self.shape) +
            '\n\t\tHas Data: ' + repr(self.has_data) +
            '\n\t\tOriginal Data: ' + self.original_data.get_name()
        )
        block_str += '\n\t\tInput {'
        for cluster in self.input_clusters:
            block_str += repr(cluster)
        block_str += '\n\t\t}'
        block_str += '\n\t\tOutput {'
        for cluster in self.output_clusters:
            block_str += repr(cluster)
        block_str += '\n\t\t}'
        block_str += '\n\t}'
        return block_str


class CTaskBlock():
    '''计算任务块类

    Attributes:
        - id: 任务块ID
        - original_node: 对应的ONNX计算图中的算子结点
        - type: 任务块类型, 对应TaskBlockType枚举类
        - precision: 任务块精度, 代表任务块输出数据的精度, 对应Precision枚举类
        - shape: 任务块形状
        - input_clusters: 任务块输入边簇的列表
        - output_clusters: 任务块输出边簇的列表
        - has_bias: 是否有bias
    '''

    def __init__(self, id: int, onnx_node: ONNXNode = None, output_data: ONNXData = None, input_task_block: STaskBlock = None, has_bias=False):
        self.id = id
        self.original_node = onnx_node
        self.type = None
        self.precision = None
        self.shape = {}
        self.attributes = {}
        self.input_clusters: List[EdgeCluster] = list()
        self.output_clusters: List[EdgeCluster] = list()
        self.has_bias = has_bias
        if (onnx_node is not None and output_data is not None and input_task_block is not None):
            self.transform_onnx_node(onnx_node, output_data, input_task_block)

    def has_input_connection(self, edge_id: int):
        for input_cluster in self.get_input_clusters():
            for interface in input_cluster.get_interfaces():
                if interface.get_id() == edge_id:
                    return True
        return False

    def has_output_connection(self, edge_id: int):
        for output_cluster in self.get_output_clusters():
            for interface in output_cluster.get_interfaces():
                if interface.get_id() == edge_id:
                    return True
        return False

    def get_id(self):
        return self.id

    def set_id(self, id: int):
        self.id = id

    def get_type(self):
        return self.type

    def set_type(self, type: int):
        self.type = type

    def get_precision(self):
        return self.precision

    def set_precision(self, precision: int):
        self.precision = precision

    def get_shape(self) -> Dict:
        return self.shape

    def set_shape(self, shape: Dict[str, int]):
        self.shape.update(shape)

    def get_original_node(self):
        return self.original_node

    def get_input_clusters(self) -> List[EdgeCluster]:
        return self.input_clusters

    def get_input_cluster(self, num: int) -> EdgeCluster:
        return self.input_clusters[num]

    def get_output_clusters(self) -> List[EdgeCluster]:
        return self.output_clusters

    def get_output_cluster(self) -> EdgeCluster:
        return self.output_clusters[0]  # 目前不支持多输出

    def get_output_edges_ids(self) -> List[int]:
        output_edges_ids = list()
        for output_cluster in self.get_output_clusters():
            for interface in output_cluster.get_interfaces():
                edge_id = interface.get_id()
                output_edges_ids.append(edge_id)
        return list(set(output_edges_ids))

    def transform_onnx_node(self, onnx_node: ONNXNode, output_data: ONNXData, input_task_block: STaskBlock):
        '''将ONNXNode对象转换成计算任务块
        '''
        self.type = self.convert_node_type(onnx_node.get_op_type())
        self.precision = convert_data_type(output_data.get_data_type())
        if self.type != TaskBlockType.TASK_NULL.value:
            self.convert_attributes(onnx_node.get_attributes(
            ), onnx_node.get_op_type(), input_task_block)
        self.infer_shape(output_data, input_task_block)

    def create_input_cluster(self, edge_id: int, s_task_block_shape: Dict[str, int]):
        positions = list()
        positions.append(init_shape())
        sizes = [s_task_block_shape]
        edge_ids = [edge_id]
        input_cluster = EdgeCluster(
            s_task_block_shape, positions, sizes, edge_ids)
        self.input_clusters.append(input_cluster)

    def fill_output_cluster(self, edge_id: int):
        output_cluster = self.get_output_cluster()
        positions = [init_shape()]
        edge_ids = [edge_id]
        sizes = [output_cluster.get_shape()]
        output_cluster.fill(positions, sizes, edge_ids)

    def add_interface_to_output_cluster(self, edge_id: int):
        output_cluster = self.get_output_cluster()
        position = init_shape()
        size = deepcopy(output_cluster.get_shape())
        output_cluster.add_interface(position, size, edge_id)

    def add_interface_to_output_cluster_with_position_and_size(self, edge_id: int, position: Dict[str, int], size: Dict[str, int]):
        output_cluster = self.get_output_cluster()
        output_cluster.add_interface(position, size, edge_id)

    def add_interface_to_input_cluster(self, position: Dict[str, int], size: Dict[str, int], edge_id: int):
        input_cluster = self.get_input_cluster(0)
        input_cluster.add_interface(position, size, edge_id)

    def create_empty_output_cluster(self):
        output_shape = init_shape()
        output_shape['y'] = self.shape['y']
        output_shape['x'] = self.shape['x']
        output_shape['f'] = self.shape['f']
        output_cluster = EdgeCluster(output_shape)
        self.output_clusters.append(output_cluster)

    def create_empty_input_cluster(self, s_task_block_shape: Dict[str, int]):
        input_shape = init_shape()
        input_shape['y'] = s_task_block_shape['y']
        input_shape['x'] = s_task_block_shape['x']
        input_shape['f'] = s_task_block_shape['f']
        input_cluster = EdgeCluster(input_shape)
        self.input_clusters.append(input_cluster)

    def convert_attributes(self, onnx_attributes: Dict[str, ONNXAttribute], op_type: str, input_task_block: STaskBlock) -> None:
        input_shape = input_task_block.get_shape()
        if self.type == TaskBlockType.CCMPB.value:
            if op_type == ONNXOpType.Relu.value:
                attribute0 = Attribute(
                    type=AttributeType.CMP.value, precision=Precision.INT_32.value, value=0)
                attribute1 = Attribute(
                    type=AttributeType.KERNEL_X.value, precision=Precision.INT_32.value, value=1)
                attribute2 = Attribute(
                    type=AttributeType.KERNEL_Y.value, precision=Precision.INT_32.value, value=1)
                attribute3 = Attribute(
                    type=AttributeType.STRIDE_X.value, precision=Precision.INT_32.value, value=1)
                attribute4 = Attribute(
                    type=AttributeType.STRIDE_Y.value, precision=Precision.INT_32.value, value=1)
                self.attributes.update({
                    'cmp': attribute0,
                    'kernel_x': attribute1,
                    'kernel_y': attribute2,
                    'stride_X': attribute3,
                    'stride_y': attribute4
                })
            elif op_type == ONNXOpType.MaxPool.value:
                attribute = Attribute(type=AttributeType.CMP.value, precision=Precision.INT_32.value, value=get_min(
                    self.precision))
                self.attributes.update({'cmp': attribute})
            else:
                raise ValueError('Unmatched ONNX op type')
        elif self.type == TaskBlockType.CAVG.value:
            if op_type == ONNXOpType.GlobalAveragePool.value:
                attribute0 = Attribute(
                    type=AttributeType.KERNEL_X.value, precision=Precision.INT_32.value, value=input_shape['x'])
                attribute1 = Attribute(
                    type=AttributeType.KERNEL_Y.value, precision=Precision.INT_32.value, value=input_shape['y'])
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
                self.attributes.update(
                    {
                        'kernel_x': attribute0,
                        'kernel_y': attribute1,
                        'stride_x': attribute2,
                        'stride_y': attribute3,
                        'pad_top': attribute4,
                        'pad_down': attribute5,
                        'pad_left': attribute6,
                        'pad_right': attribute7
                    }
                )
            elif op_type == ONNXOpType.AveragePool.value:
                raise NotImplementedError
            else:
                raise ValueError('Unmatched ONNX op type')
        for _, onnx_attribute in onnx_attributes.items():
            self.convert_attribute(onnx_attribute)

    def convert_attribute(self, onnx_attribute: ONNXAttribute) -> None:
        onnx_attribute_name = onnx_attribute.get_name()
        values = onnx_attribute.get_value()
        if onnx_attribute_name == 'kernel_shape':
            attribute = Attribute(type=AttributeType.KERNEL_X.value, precision=Precision.INT_32.value,
                                  value=values[1])
            self.attributes.update({'kernel_x': attribute})
            attribute = Attribute(type=AttributeType.KERNEL_Y.value, precision=Precision.INT_32.value,
                                  value=values[0])
            self.attributes.update({'kernel_y': attribute})
        elif onnx_attribute_name == 'pads':
            attribute = Attribute(type=AttributeType.PAD_TOP.value, precision=Precision.INT_32.value,
                                  value=values[0])
            self.attributes.update({'pad_top': attribute})
            attribute = Attribute(type=AttributeType.PAD_DOWN.value, precision=Precision.INT_32.value,
                                  value=values[2])
            self.attributes.update({'pad_down': attribute})
            attribute = Attribute(type=AttributeType.PAD_LEFT.value, precision=Precision.INT_32.value,
                                  value=values[1])
            self.attributes.update({'pad_left': attribute})
            attribute = Attribute(type=AttributeType.PAD_RIGHT.value, precision=Precision.INT_32.value,
                                  value=values[3])
            self.attributes.update({'pad_right': attribute})
        elif onnx_attribute_name == 'strides':
            attribute = Attribute(type=AttributeType.STRIDE_X.value, precision=Precision.INT_32.value,
                                  value=values[1])
            self.attributes.update({'stride_x': attribute})
            attribute = Attribute(type=AttributeType.STRIDE_Y.value, precision=Precision.INT_32.value,
                                  value=values[0])
            self.attributes.update({'stride_y': attribute})
        elif onnx_attribute_name == 'dilations':
            attribute = Attribute(type=AttributeType.DILATION_X.value, precision=Precision.INT_32.value,
                                  value=values[1])
            self.attributes.update({'dilation_x': attribute})
            attribute = Attribute(type=AttributeType.DILATION_Y.value, precision=Precision.INT_32.value,
                                  value=values[0])
            self.attributes.update({'dilation_y': attribute})
        elif onnx_attribute_name == 'group':
            assert values == 1, 'Group convolution is not supported'
        elif onnx_attribute_name in ['alpha', 'beta', 'transB']:
            assert values == 1, 'MLP is the only supported type of Gemm'
        # LIF相关
        elif onnx_attribute_name == 'v_init':
            attribute = Attribute(type=AttributeType.VINIT.value, precision=Precision.INT_32.value,
                                  value=values)
            self.attributes.update({'v_init': attribute})
        elif onnx_attribute_name == 'v_leaky_adpt_en':
            attribute = Attribute(type=AttributeType.VLEAKY_ADPT_EN.value, precision=Precision.TERNARY.value,
                                  value=values)
            self.attributes.update({'v_leaky_adpt_en': attribute})
        elif onnx_attribute_name == 'v_leaky_alpha':
            attribute = Attribute(type=AttributeType.VLEAKY_ALPHA.value, precision=Precision.INT_32.value,
                                  value=round(values * 256))
            self.attributes.update({'v_leaky_alpha': attribute})
        elif onnx_attribute_name == 'v_leaky_beta':
            attribute = Attribute(type=AttributeType.VLEAKY_BETA.value, precision=Precision.INT_32.value,
                                  value=values)
            self.attributes.update({'v_leaky_beta': attribute})
        elif onnx_attribute_name == 'v_reset':
            attribute = Attribute(type=AttributeType.VR.value, precision=Precision.INT_32.value,
                                  value=values)
            self.attributes.update({'v_reset': attribute})
        elif onnx_attribute_name == 'v_th':
            attribute = Attribute(type=AttributeType.VTH0.value, precision=Precision.INT_32.value,
                                  value=values)
            self.attributes.update({'v_th': attribute})
        elif onnx_attribute_name == 'time_window_size':
            attribute = Attribute(type=AttributeType.TW_LEN.value, precision=Precision.INT_32.value,
                                  value=values)
            self.attributes.update({'time_window_size': attribute})
        else:
            raise NotImplementedError(
                'ONNX attribute {:s} cannot be converted'.format(onnx_attribute_name))

    def get_attribute(self, attribute_name: str) -> Attribute:
        return self.attributes[attribute_name]

    def add_attribute(self, attr: Attribute, attr_name: str):
        self.attributes.update({attr_name: attr})

    def add_attributes(self, attr_dict: Dict[str, Attribute]):
        self.attributes.update(attr_dict)

    def get_attributes(self):
        return self.attributes

    def init_shape(self) -> Dict:
        self.shape.update({
            'y': 0,
            'x': 0,
            'f': 0,
            'r': 0,
            'kx': 0,
            'ky': 0,
            'iy': 0,
            'ix': 0
        })

    def infer_shape(self, output_data: ONNXData, input_task_block: STaskBlock):
        self.init_shape()
        output_shape = output_data.get_shape()
        input_shape = input_task_block.get_shape()
        if self.type == TaskBlockType.CC.value:
            self.shape.update({
                'y': output_shape[2],
                'x': output_shape[3],
                'f': output_shape[1],
                'r': input_shape['r'],
                'kx': self.get_attribute('kernel_x').get_value(),
                'ky': self.get_attribute('kernel_y').get_value(),
                'iy': input_shape['y'],
                'ix': input_shape['x']
            })
        elif self.type == TaskBlockType.CCMPB.value:
            self.shape.update({
                'y': 0 if len(output_shape) == 2 else output_shape[2],
                'x': 0 if len(output_shape) == 2 else output_shape[3],
                'f': output_shape[1],
                'kx': self.get_attribute('kernel_x').get_value(),
                'ky': self.get_attribute('kernel_y').get_value(),
                'iy': input_shape['y'],
                'ix': input_shape['x']
            })
        elif self.type == TaskBlockType.CVM.value:
            self.shape.update({
                'f': output_shape[1],
                'r': input_shape['r']
            })
        elif self.type == TaskBlockType.CADD.value:
            self.shape.update(
                {
                    'y': output_shape[2],
                    'x': output_shape[3],
                    'f': output_shape[1],
                    'kx': 1,
                    'ky': 1
                }
            )
        elif self.type == TaskBlockType.CAVG.value:
            self.shape.update(
                {
                    'y': output_shape[2],
                    'x': output_shape[3],
                    'f': output_shape[1],
                    'kx': self.get_attribute('kernel_x').get_value(),
                    'ky': self.get_attribute('kernel_y').get_value(),
                    'iy': input_shape['y'],
                    'ix': input_shape['x']
                }
            )
        elif self.type == TaskBlockType.TASK_NULL.value:
            pass  # 保持默认值
        elif self.type == TaskBlockType.CLIF.value:
            self.shape.update(
                {
                    'y': 0 if len(output_shape) == 2 else output_shape[2],
                    'x': 0 if len(output_shape) == 2 else output_shape[3],
                    'f': output_shape[1]
                }
            )
        else:
            raise NotImplementedError('Cannot infer shape for task block type {:s}'.format(
                TaskBlockType.get_name(self.type)))

    def convert_node_type(self, op_type: str):
        if op_type == ONNXOpType.Conv.value:
            return TaskBlockType.CC.value
        elif op_type in (ONNXOpType.Relu.value, ONNXOpType.MaxPool.value):
            return TaskBlockType.CCMPB.value
        elif op_type == ONNXOpType.Gemm.value:
            return TaskBlockType.CVM.value
        elif op_type in (ONNXOpType.Flatten.value, ONNXOpType.Cut.value, ONNXOpType.Concat.value):
            return TaskBlockType.TASK_NULL.value
        elif op_type == ONNXOpType.Add.value:
            return TaskBlockType.CADD.value
        elif op_type == ONNXOpType.GlobalAveragePool.value:
            return TaskBlockType.CAVG.value
        elif op_type == ONNXOpType.LIF.value:
            return TaskBlockType.CLIF.value
        else:
            raise NotImplementedError(
                'Op type {:s} cannot be converted into task block'.format(op_type))

    def __repr__(self) -> str:
        block_str = (
            '\n\tCTaskBlock ' + str(self.id) + ' {' +
            '\n\t\tType: ' + TaskBlockType.get_name(self.type) +
            '\n\t\tPrecision: ' + Precision.get_precision_name(self.precision) +
            '\n\t\tShape: ' + repr(self.shape) +
            '\n\t\tHas Bias: ' + repr(self.has_bias) +
            '\n\t\tOriginal Node: ' +
            (self.original_node.get_name()
             if self.original_node is not None else 'None')
        )
        for _, attribute in self.attributes.items():
            block_str += repr(attribute)
        block_str += '\n\t\tInput {'
        for cluster in self.input_clusters:
            block_str += repr(cluster)
        block_str += '\n\t\t}'
        block_str += '\n\t\tOutput {'
        for cluster in self.output_clusters:
            block_str += repr(cluster)
        block_str += '\n\t\t}'
        block_str += '\n\t}'
        return block_str


class IOTaskBlock(CTaskBlock):
    '''输入输出任务块类

    Attributes:
        - socket_id: 输入或输出socket ID
    '''

    def __init__(self, id: int, socket_id: int, s_task_block: STaskBlock) -> None:
        super(IOTaskBlock, self).__init__(id=id)
        self.socket_id = socket_id
        self.precision = s_task_block.precision
        self.shape = None
        self.convert_shape(s_task_block.shape)

    def convert_shape(self, shape: Dict):
        self.shape = deepcopy(shape)
        if self.shape['r'] != 0:
            self.shape['f'] = self.shape['r']
            self.shape['r'] = 0


class InputTaskBlock(IOTaskBlock):
    '''输入任务块类

    与相连的存储任务块建立起联系, 多个共享同一块数据的存储任务块会连接到同一个输入任务块上
    '''

    def __init__(self, id: int, socket_id: int, s_task_block: STaskBlock, edge_id: int) -> None:
        super(InputTaskBlock, self).__init__(
            id=id, socket_id=socket_id, s_task_block=s_task_block)
        self.type = TaskBlockType.INPUT.value
        self.create_empty_output_cluster()
        self.add_interface_to_output_cluster(edge_id=edge_id)
        self.create_input_cluster_for_s_task_block(s_task_block, edge_id)

    def create_empty_output_cluster(self):
        output_shape = deepcopy(self.shape)
        output_cluster = EdgeCluster(output_shape)
        self.output_clusters.append(output_cluster)

    def create_input_cluster_for_s_task_block(self, s_task_block: STaskBlock, edge_id: int):
        s_task_block.create_empty_input_cluster(cluster_shape=self.shape)
        position = init_shape()
        s_task_block.add_interface_to_input_cluster(
            input_cluster_num=0, position=position, size=self.shape, edge_id=edge_id)


class OutputTaskBlock(IOTaskBlock):
    '''输出任务块类

    与相连的存储任务块建立起联系
    '''

    def __init__(self, id: int, socket_id: int, s_task_block: STaskBlock, edge_id: int) -> None:
        super(OutputTaskBlock, self).__init__(
            id=id, socket_id=socket_id, s_task_block=s_task_block)
        self.type = TaskBlockType.OUTPUT.value
        self.create_input_cluster(
            edge_id=edge_id, s_task_block_shape=s_task_block.shape)
        self.create_output_cluster_for_s_task_block(s_task_block, edge_id)

    def create_output_cluster_for_s_task_block(self, s_task_block: STaskBlock, edge_id: int):
        s_task_block.create_output_cluster(edge_id=edge_id)


class Context():
    '''上下文类

    主要用于记录转换过程中的一些信息

    Attributes:
        - mapping_dict: 任务图和ONNX计算图之间的映射
        - block_counter: 任务块计数
        - edge_counter: 边计数
    '''

    def __init__(self) -> None:
        self.mapping_dict = bidict()
        self.block_counter: int = -1
        self.edge_counter: int = -1

    def get_block_counter(self):
        '''生成新的任务块ID
        '''
        self.block_counter += 1
        return self.block_counter

    def get_edge_counter(self):
        '''生成新的边ID
        '''
        self.edge_counter += 1
        return self.edge_counter

    def create_mapping(self, c_task_block_id: int, onnx_node_name: str):
        self.mapping_dict.update({c_task_block_id: onnx_node_name})


class TaskGraph():
    '''任务图类

    Attributes:
        - blocks: ID到任务块对象的字典
        - edges: ID到边对象的字典
        - groups: key为任务组中计算任务块的ID, value为任务组中所有存储任务块的ID
        - ctx: 转换过程的上下文对象
        - onnx_graph: 原始的ONNX计算图
    '''

    def __init__(self, onnx_graph: ONNXGraph = None) -> None:
        self.blocks: Dict[int, Union[STaskBlock, CTaskBlock]] = {}
        self.edges: Dict[int, Edge] = {}
        self.groups: Dict[int, List[int]] = {}
        self.ctx = Context()
        self.onnx_graph = onnx_graph

    def add_block(self, block: Union[STaskBlock, CTaskBlock]):
        self.blocks.update({block.get_id(): block})

    def add_blocks(self, blocks: List[Union[STaskBlock, CTaskBlock]]):
        for block in blocks:
            self.blocks.update({block.get_id(): block})

    def add_edge(self, edge: Edge):
        self.edges.update({edge.get_id(): edge})

    def add_edges(self, edges: List[Edge]):
        for edge in edges:
            self.edges.update({edge.get_id(): edge})

    def add_group(self, c_block: CTaskBlock, s_blocks: List[STaskBlock]):
        s_blocks_ids = [s_block.get_id() for s_block in s_blocks]
        self.groups.update({c_block.get_id(): s_blocks_ids})

    def get_block(self, block_id: int) -> Union[STaskBlock, CTaskBlock]:
        return self.blocks[block_id]

    def get_edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]

    def get_group(self, c_block_id: int):
        return self.groups[c_block_id]

    def is_output(self, block_id: int):
        '''某个任务块是否是输出

        对于存储任务块, 如果存储没有被其他计算任务块使用则为输出
        对于计算任务块, 如果其输出结果没有被其他计算使用则为输出
        '''
        block = self.get_block(block_id)
        if isinstance(block, STaskBlock):
            return block.is_output()
        if isinstance(block, CTaskBlock):
            next_blocks_ids = self.get_next_blocks_ids(block_id)
            for next_block_id in next_blocks_ids:
                next_block = self.get_block(next_block_id)
                if next_block.is_output():
                    return True

    def is_input(self, block_id: int):
        '''某个任务块是否是输入

        对于存储任务块, 如果没有其他计算任务块生成其数据则为输入
        对于计算任务块, 如果其输入动态数据不是由其他计算生成则为输入
        '''
        block = self.get_block(block_id)
        if isinstance(block, STaskBlock):
            return block.is_input()
        if isinstance(block, CTaskBlock):
            last_block_ids = self.get_last_blocks_ids(block_id)
            for last_block_id in last_block_ids:
                last_block = self.get_block(last_block_id)
                if last_block.is_input() and TaskBlockType.is_input(last_block.type):
                    return True

    def get_next_blocks_ids(self, block_id: int) -> List[int]:
        next_blocks_ids = list()
        block = self.get_block(block_id)
        output_cluster = block.get_output_cluster()
        interfaces = output_cluster.get_interfaces()
        for interface in interfaces:
            edge_id = interface.get_id()
            edge = self.get_edge(edge_id)
            next_blocks_ids.append(edge.get_dst_block_id())
        return next_blocks_ids

    def get_next_c_blocks_ids_of_c_block(self, c_block_id: int) -> List[int]:
        if self.is_output(c_block_id):
            return None
        next_c_blocks_ids = list()
        next_s_blocks_ids = self.get_next_blocks_ids(c_block_id)
        for next_s_block_id in next_s_blocks_ids:
            next_c_block_id = self.get_next_blocks_ids(next_s_block_id)[0]
            next_c_block = self.get_block(next_c_block_id)
            assert isinstance(next_c_block, CTaskBlock)
            next_c_blocks_ids.append(next_c_block_id)
        return next_c_blocks_ids

    def get_next_c_blocks_of_c_block(self, c_block_id: int) -> List[CTaskBlock]:
        next_c_blocks = list()
        next_c_blocks_ids = self.get_next_c_blocks_ids_of_c_block(c_block_id)
        if next_c_blocks_ids is None:
            return None
        for next_c_block_id in next_c_blocks_ids:
            next_c_blocks.append(self.get_block(next_c_block_id))
        return next_c_blocks

    def get_last_blocks_ids(self, block_id: int) -> List[int]:
        block = self.get_block(block_id)
        last_blocks_ids = list()
        input_clusters = block.get_input_clusters()
        for input_cluster in input_clusters:
            interfaces = input_cluster.get_interfaces()
            for interface in interfaces:
                edge_id = interface.get_id()
                edge = self.get_edge(edge_id)
                last_blocks_ids.append(edge.get_src_block_id())
        return list(set(last_blocks_ids))

    def __repr__(self) -> str:
        graph_str = 'Task Graph {'
        for _, block in self.blocks.items():
            graph_str += repr(block)
        for _, edge in self.edges.items():
            graph_str += repr(edge)
        graph_str += '\n}'
        return graph_str

    def record(self, path: str):
        '''以可读的形式将任务图写入文件
        '''
        with open(path, 'w') as f:
            f.write(self.__repr__())
