# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from copy import deepcopy
from typing import Dict, Union

import src.compiler.ir.basic_pb2 as basic_ir
import src.compiler.ir.task_pb2 as task_ir
from src.compiler.transformer.ir_gen.bias_type import BiasType
from src.compiler.transformer.task_model.task_block_type import TaskBlockType
from src.compiler.transformer.task_model.task_graph_basics import (
    Attribute, CTaskBlock, Edge, EdgeCluster, EdgeInterface, RearrangeInfo,
    STaskBlock, TaskGraph, Tensor)
from src.simulator.task_rabbit.task_model.precision import Precision


def convert_tensor(tensor_ir: task_ir.Tensor, tensor: Tensor):
    tensor_ir.precision = tensor.get_precision()
    for dim in tensor.get_dims():
        tensor_ir.dims.append(dim)
    if Precision.is_float(tensor_ir.precision):
        for data in tensor.get_data():
            tensor_ir.float_data.append(data)
    elif Precision.is_int32(tensor_ir.precision):
        for data in tensor.get_data():
            tensor_ir.int32_data.append(int(data))
    elif Precision.is_uint32(tensor_ir.precision):
        for data in tensor.get_data():
            tensor_ir.uint32_data.append(data)
    else:
        raise ValueError('Unsupported data type in task graph IR')


def convert_shape(shape_ir: basic_ir.Shape, shape: Dict):
    if shape['y'] != 0:
        shape_ir.ny = shape['y']
    if shape['x'] != 0:
        shape_ir.nx = shape['x']
    if shape['f'] != 0:
        shape_ir.nf = shape['f']
    if shape['r'] != 0:
        shape_ir.nr = shape['r']
    if shape['ky'] != 0:
        shape_ir.nky = shape['ky']
    if shape['kx'] != 0:
        shape_ir.nkx = shape['kx']


def convert_interface(interface_ir: task_ir.EdgeInterface, interface: EdgeInterface):
    interface_ir.edge_id = interface.get_id()
    if interface.position is not None:
        convert_shape(interface_ir.position, interface.get_position())
    if interface.size is not None:
        convert_shape(interface_ir.size, interface.get_size())


def convert_edge_cluster(edge_cluster_ir: task_ir.EdgeCluster, edge_cluster: EdgeCluster):
    shape_ir = edge_cluster_ir.shape
    shape = edge_cluster.get_shape()
    if shape is not None:
        convert_shape(shape_ir, shape)
    for interface in edge_cluster.get_interfaces():
        interface_ir = edge_cluster_ir.interfaces.add()
        convert_interface(interface_ir, interface)


def convert_attr(attr_ir: task_ir.Attribute, attr: Attribute):
    attr_ir.type = attr.get_type()
    attr_ir.precision = attr.get_precision()
    if Precision.is_int(attr_ir.precision):
        attr_ir.int_value = attr.get_value()
    elif Precision.is_float(attr_ir.precision):
        attr_ir.float_value = attr.get_value()
    else:
        raise ValueError('Unsupported precision')


def convert_block(block_ir: task_ir.TaskBlock, block: Union[CTaskBlock, STaskBlock]):
    block_ir.id = block.get_id()
    block_ir.type = block.get_type()
    block_ir.precision = block.get_precision()
    if hasattr(block, 'socket_id'):
        block_ir.socket_id = block.socket_id
    if hasattr(block, 'has_bias'):
        if block.has_bias:
            block_ir.bias_type = BiasType.VECTOR.value
    shape = block.get_shape()
    shape_ir = block_ir.shape
    convert_shape(shape_ir, shape)
    if 'iy' in shape and shape['iy'] != 0:
        shape_ir.niy = shape['iy']
    if 'ix' in shape and shape['ix'] != 0:
        shape_ir.nix = shape['ix']
    if TaskBlockType.is_storage(block.get_type()):
        if block.has_data:
            tensor_ir = block_ir.data
            convert_tensor(tensor_ir, block.get_data())
        if block.has_pipeline_num:
            block_ir.pipeline_num = block.get_pipeline_num()
    if TaskBlockType.is_computation(block.get_type()):
        for attr_name in block.get_attributes():
            attr_ir = block_ir.attributes.add()
            attr = block.get_attribute(attr_name)
            convert_attr(attr_ir, attr)
    for input_cluster in block.get_input_clusters():
        input_cluster_ir = block_ir.input_clusters.add()
        convert_edge_cluster(input_cluster_ir, input_cluster)
    for output_cluster in block.get_output_clusters():
        output_cluster_ir = block_ir.output_clusters.add()
        convert_edge_cluster(output_cluster_ir, output_cluster)


def convert_rearrange_info(rearrange_info_ir: task_ir.RearrangeInfo, rearrange_info: RearrangeInfo):
    rearrange_info_ir.type = rearrange_info.get_type()
    for tensor in rearrange_info.get_matrix():
        tensor_ir = rearrange_info_ir.matrix.add()
        convert_tensor(tensor_ir, tensor)


def convert_edge(edge_ir: task_ir.Edge, edge: Edge):
    edge_ir.id = edge.get_id()
    edge_ir.src_block_id = edge.get_src_block_id()
    edge_ir.dst_block_id = edge.get_dst_block_id()
    for rearrange_info in edge.get_rearrange_info():
        rearrange_info_ir = edge_ir.rearrange_info.add()
        convert_rearrange_info(rearrange_info_ir, rearrange_info)


def ir_generate(task_graph: TaskGraph, task_graph_path: str, readable_file_path: str, task_graph_ir=None):
    '''将TaskGraph对象转换成Task IR

    包括以下几个步骤:
    1. 转换任务组信息
    2. 转换边
    3. 转换任务块
    4. 生成Task IR文件
    5. 可选的, 生成可读IR文件

    Args:
        task_graph: TaskGraph对象
        task_graph_path: 生成的IR保存路径
        readable_file_path: 可读IR文件的保存路径
        task_graph_ir: task_ir.TaskGraph对象

    Returns:
        task_graph_ir: 填充后的task_ir.TaskGraph对象
    '''
    if task_graph_ir is None:
        task_graph_ir = task_ir.TaskGraph()
    group_id = 0
    for c_block_id in task_graph.groups:
        group_ir = task_graph_ir.groups.add()
        group_ir.group_id = group_id
        group_id += 1
        s_blocks_ids = task_graph.get_group(c_block_id)
        blocks_ids = deepcopy(s_blocks_ids)
        blocks_ids.append(c_block_id)
        for block_id in blocks_ids:
            group_ir.block_ids.append(block_id)

    for edge_id in task_graph.edges:
        edge = task_graph.get_edge(edge_id)
        edge_ir = task_graph_ir.edges.add()
        convert_edge(edge_ir, edge)

    for block_id in task_graph.blocks:
        block = task_graph.get_block(block_id)
        block_ir = task_graph_ir.blocks.add()
        convert_block(block_ir, block)

    if task_graph_path is not None:
        with open(task_graph_path, 'wb') as f:
            f.write(task_graph_ir.SerializeToString())

    if readable_file_path is not None:
        with open(readable_file_path, 'w') as f:
            f.write(repr(task_graph_ir))

    return task_graph_ir
