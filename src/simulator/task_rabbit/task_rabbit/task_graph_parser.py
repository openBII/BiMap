# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import logging
from typing import List

import numpy as np
from google.protobuf import text_format

import src.compiler.ir.basic_pb2 as basic_ir
import src.compiler.ir.mapping_pb2 as mapping_ir
from src.compiler.ir.mapping_pb2 import task__pb2 as task_ir
from src.utils.st_draw import STDraw
from src.simulator.task_rabbit.task_checker.checker import TaskChecker
from src.simulator.task_rabbit.task_model import *
from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.edge import (RearrangeInfo,
                                                       RearrangeInfoType)
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from top.global_config import GlobalConfig


class TaskGraphParser:
    """
    将proto协议下的Task IR转换为python代码构建的更易操作的Task Graph模型
    """
    @staticmethod
    def parse(task_ir_path: str, ir_type: str) -> TaskGraph:
        """
        Args:
            task_ir_path: 将要解析的IR的文件路径，IR文件如果以.txt结尾，
            则意味着其为人类可读文件，如果不是，则意味着其为proto二进制文件
            ir_type: 因为Mapping IR含有Task IR，所以本类也支持从Mapping IR中
            解析出Task Graph。该参数为'task'，则表示从Task IR中解析；
            为'map'，则表示从Map IR中解析
        Raises: 
            ValueError: 如果ir类型不是'task'或'map'时，报异常
        """
        if ir_type == 'task':
            logging.info('Load Task IR from {:s}'.format(task_ir_path))
        elif ir_type == 'map':
            logging.info('Load Mapping IR from {:s}'.format(task_ir_path))
        else:
            raise ValueError('Unsupported input file type')

        task_ir = TaskGraphParser.load_task_graph_ir(task_ir_path, ir_type)
        task_graph = TaskGraphParser.convert_ir_to_task_graph(task_ir)
        task_graph.topologize()

        TaskChecker.check_basic(task_graph)
        return task_graph

    @staticmethod
    def load_task_graph_ir(task_ir_path: str, ir_type: str):
        """
        从IR文件中读取proto IR到task_ir上

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 如果给出了除'task'和'map'之外的ir_type，则报错
        """
        if ir_type == 'map':
            the_mapping_ir = mapping_ir.Mapping()
            if (task_ir_path.endswith('.txt')):
                with open(task_ir_path, 'r') as file:
                    text_format.ParseLines(file.readlines(), the_mapping_ir)
            else:
                with open(task_ir_path, 'rb') as file:
                    the_mapping_ir.ParseFromString(file.read())
            task_graph = the_mapping_ir.graph
        elif ir_type == 'task':
            task_graph = task_ir.TaskGraph()
            if (task_ir_path.endswith('.txt')):
                with open(task_ir_path, 'r') as file:
                    text_format.ParseLines(file.readlines(), task_graph)
            else:
                with open(task_ir_path, 'rb') as file:
                    task_graph.ParseFromString(file.read())
        else:
            raise ValueError('Unsupported input file type')

        return task_graph

    @staticmethod
    def convert_ir_to_task_graph(task_ir) -> TaskGraph:
        task_graph = TaskGraph()
        edge_info = {}

        # 构建任务结点
        for _, task_block_ir in enumerate(task_ir.blocks):
            task_block_id = task_block_ir.id
            shape = TaskGraphParser.convert_shape(task_block_ir.shape)
            precision = TaskGraphParser.convert_precision(
                task_block_ir.precision)
            task_block = TaskGraphParser.convert_task_block(
                task_block_id, shape, precision, task_block_ir)
            task_graph.add_node(task_block)

            # 记录结点之间得连接
            for j, output_cluster in enumerate(task_block_ir.output_clusters):
                for _, interface in enumerate(output_cluster.interfaces):
                    one_edge = TaskGraphParser.get_edge_from_id(task_ir.edges,
                                                                interface.edge_id)
                    assert (one_edge.src_block_id == task_block_id)
                    position_shape = TaskGraphParser.convert_shape(
                        interface.position)
                    src_size = TaskGraphParser.convert_shape(interface.size)
                    if src_size.volume == 0:
                        src_size = task_block.output_clusters[j].shape
                    # 记录一条边的信息，包括
                    # 源任务块ID，目的任务块ID，源边簇，目的边簇，源起始位置，目的起始位置，源形状，目的形状，数据重排信息
                    edge_info.update({interface.edge_id: [task_block_id, one_edge.dst_block_id,
                                                          j, None, position_shape, None, src_size, None, None]})

        # 从目的结点角度补全边的信息
        for _, task_block_ir in enumerate(task_ir.blocks):
            for j, input_cluster in enumerate(task_block_ir.input_clusters):
                task_block = task_graph.get_node(task_block_ir.id)
                for _, interface in enumerate(input_cluster.interfaces):
                    _edge = TaskGraphParser.get_edge_from_id(task_ir.edges,
                                                             interface.edge_id)
                    position_shape = TaskGraphParser.convert_shape(
                        interface.position)
                    dst_size = TaskGraphParser.convert_shape(interface.size)
                    if dst_size.volume == 0:
                        dst_size = task_block.input_clusters[j].shape
                    info = edge_info[interface.edge_id]
                    info[3] = j
                    info[5] = position_shape
                    info[7] = dst_size
                    # add_rearrange_info
                    info[8] = TaskGraphParser.get_rearrange_info(
                        _edge.rearrange_info)

        for _, info in edge_info.items():
            task_graph.add_edge(*info)

        IDGenerator.set_base_task_id(max(task_graph.get_all_node_ids()))

        TaskGraphParser.set_groups(task_ir, task_graph)
        return task_graph

    @staticmethod
    def set_groups(task_ir, task_graph: TaskGraph):
        for one_group in task_ir.groups:
            group_id = one_group.group_id
            group_tasks = set(one_group.block_ids)
            task_graph.group(group_tasks, group_id)

    @staticmethod
    def get_edge_from_id(edge_list, edge_id):
        for edge in edge_list:
            if edge.id == edge_id:
                return edge

    @staticmethod
    def get_rearrange_info(rearrange_info_ir_list) -> List[RearrangeInfo]:
        rearrange_info_list = []
        for rearrange_info_ir in rearrange_info_ir_list:
            rearrange_type = RearrangeInfoType(rearrange_info_ir.type)
            rearrange_matrix_list = []
            for tensor_ir in rearrange_info_ir.matrix:
                rearrange_matrix_list.append(tensor_ir.int32_data)
            rearrange_info_list.append(RearrangeInfo(rearrange_type,
                                                     rearrange_matrix_list))
        return rearrange_info_list

    @staticmethod
    def convert_shape(shape_ir) -> Shape:
        shape = Shape(shape_ir.ny, shape_ir.nx, shape_ir.nf, shape_ir.nr,
                      shape_ir.nky, shape_ir.nkx)
        shape.niy, shape.nix = shape_ir.niy, shape_ir.nix
        return shape

    @staticmethod
    def convert_precision(precision) -> Precision:
        if precision == basic_ir.Precision.INT_8:
            return Precision.INT_8
        elif precision == basic_ir.Precision.UINT_8:
            return Precision.UINT_8
        elif precision == basic_ir.Precision.INT_16:
            return Precision.INT_16
        elif precision == basic_ir.Precision.UINT_16:
            return Precision.UINT_16
        elif precision == basic_ir.Precision.INT_32:
            return Precision.INT_32
        elif precision == basic_ir.Precision.UINT_32:
            return Precision.UINT_32
        elif precision == basic_ir.Precision.FLOAT_16:
            return Precision.FLOAT_16
        elif precision == basic_ir.Precision.FLOAT_32:
            return Precision.FLOAT_32
        elif precision == basic_ir.Precision.TERNARY:
            return Precision.TERNARY
        elif precision == basic_ir.Precision.INT_9:
            return Precision.INT_9
        elif precision == basic_ir.Precision.INT_28:
            return Precision.INT_28
        elif precision == basic_ir.Precision.UINT_4:
            return Precision.UINT_4
        else:
            raise TypeError('Unsupported precision of task block!')

    @staticmethod
    def convert_task_block(task_id: int, shape: Shape, precision: Precision, task_block_ir) -> TaskBlock:
        new_block = TaskGraphParser.create_task_block(
            task_id, shape, precision, task_block_ir)
        TaskGraphParser.set_task_block_additional_params(
            new_block, task_block_ir)
        return new_block

    @staticmethod
    def create_task_block(task_id: int, shape: Shape, precision: Precision, task_block_ir) -> TaskBlock:
        data = None
        if task_block_ir.HasField('data'):
            data = TaskGraphParser.convert_task_block_data(task_block_ir)
        bias_type = BiasType.NONE
        if task_block_ir.HasField('bias_type'):
            bias_type = TaskGraphParser.convert_bias_type(task_block_ir)

        if task_block_ir.type == task_ir.TaskBlockType.SI:
            return SITaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SIC2D:
            return SIC2DTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SIFC:
            return SIFCTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SIC:
            return SICTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SW:
            assert data is not None    # 一定要有权重
            return SWTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SB:
            assert data is not None    # 一定要有权重
            return SBTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SWFC:
            assert data is not None    # 一定要有权重
            return SWFCTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.SW2D:
            assert data is not None    # 一定要有权重
            return SW2DTaskBlock(task_id, shape, precision, data)
        elif task_block_ir.type == task_ir.TaskBlockType.CADD:
            n_branch = len(task_block_ir.input_clusters)
            if bias_type == BiasType.VECTOR:
                n_branch -= 1
            return CADDTaskBlock(task_id, shape, n_branch, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CVVH:
            return CVVHTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CVM:
            return CVMTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CC:
            return CCTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CAX: 
            return CAXTaskBlock(task_id, shape, precision)  # CAX bias_type 生成bug
        elif task_block_ir.type == task_ir.TaskBlockType.CC2D:
            return CC2DTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CVS:
            return CVSTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CCMPB:
            return CCMPBTaskBlock(task_id, shape, precision)
        elif task_block_ir.type == task_ir.TaskBlockType.CCMPS:
            return CCMPSTaskBlock(task_id, shape, precision)
        elif task_block_ir.type == task_ir.TaskBlockType.CLUT:
            lut_dw = None   # Look-up Table输入查找索引的位宽
            for attribute in task_block_ir.attributes:
                if attribute.type == task_ir.AttributeType.LUT_DW:
                    lut_dw = attribute.int_value
            assert lut_dw is not None, "LUT_DW must be provided"
            return CLUTTaskBlock(task_id, shape, precision, lut_dw)
        elif task_block_ir.type == task_ir.TaskBlockType.CAVG:
            return CAVGTaskBlock(task_id, shape, precision, bias_type)
        elif task_block_ir.type == task_ir.TaskBlockType.CLIF:
            v_theta_const_en = None
            vm_const_en = None
            for attribute in task_block_ir.attributes:
                if attribute.type == task_ir.AttributeType.VTHETA_CONST_EN:
                    v_theta_const_en = attribute.int_value
                if attribute.type == task_ir.AttributeType.VM_CONST_EN:
                    vm_const_en = attribute.int_value
            return CLIFTaskBlock(task_id, shape, precision, v_theta_const_en, vm_const_en)
        elif task_block_ir.type == task_ir.TaskBlockType.INPUT:
            return InputTaskBlock(task_id, shape, precision, task_block_ir.socket_id)
        elif task_block_ir.type == task_ir.TaskBlockType.OUTPUT:
            return OutputTaskBlock(task_id, shape, precision, task_block_ir.socket_id)
        else:
            raise TypeError('Invalid task block type')

    @staticmethod
    def convert_bias_type(task_block_ir) -> BiasType:
        bias_type = BiasType.NONE
        if task_block_ir.HasField('bias_type'):
            if task_block_ir.bias_type == basic_ir.BiasType.VECTOR:
                bias_type = BiasType.VECTOR
            elif task_block_ir.bias_type == basic_ir.BiasType.CONSTANT:
                bias_type = BiasType.CONSTANT
        return bias_type

    @staticmethod
    def set_task_block_additional_params(task_block: TaskBlock, task_block_ir):
        for attribute in task_block_ir.attributes:
            if attribute.type == task_ir.AttributeType.KERNEL_Y:
                task_block.kernel_y = attribute.int_value
            elif attribute.type == task_ir.AttributeType.KERNEL_X:
                task_block.kernel_x = attribute.int_value
            elif attribute.type == task_ir.AttributeType.STRIDE_Y:
                task_block.stride_y = attribute.int_value
            elif attribute.type == task_ir.AttributeType.STRIDE_X:
                task_block.stride_x = attribute.int_value
            elif attribute.type == task_ir.AttributeType.PAD_UP:
                task_block.pad_up = attribute.int_value
            elif attribute.type == task_ir.AttributeType.PAD_DOWN:
                task_block.pad_down = attribute.int_value
            elif attribute.type == task_ir.AttributeType.PAD_LEFT:
                task_block.pad_left = attribute.int_value
            elif attribute.type == task_ir.AttributeType.PAD_RIGHT:
                task_block.pad_right = attribute.int_value
            elif attribute.type == task_ir.AttributeType.CMP:
                task_block.CMP = attribute.int_value
            elif attribute.type == task_ir.AttributeType.DILATION_Y:
                task_block.dilation_y = attribute.int_value
            elif attribute.type == task_ir.AttributeType.DILATION_X:
                task_block.dilation_x = attribute.int_value
            elif attribute.type == task_ir.AttributeType.BIT_SHIGT_NUM:
                task_block.bit_shift_num = attribute.int_value
            elif attribute.type == task_ir.AttributeType.CONSTANT_A:
                task_block.constant_a = attribute.int_value
            elif attribute.type == task_ir.AttributeType.CONSTANT_B:
                task_block.constant_b = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTH0:
                task_block.v_th_0 = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VLEAKY_ADPT_EN:
                task_block.v_leaky_adpt_en = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VLEAKY_ALPHA:
                task_block.v_leaky_alpha = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VLEAKY_BETA:
                task_block.v_leaky_beta = attribute.int_value
            elif attribute.type == task_ir.AttributeType.TW_EN:
                task_block.tw_en = attribute.int_value
            elif attribute.type == task_ir.AttributeType.TW_LEN:
                task_block.tw_len = attribute.int_value
            elif attribute.type == task_ir.AttributeType.TW_CNT:
                task_block.tw_cnt = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VINIT:
                task_block.v_init = attribute.int_value
            elif attribute.type == task_ir.AttributeType.SEED:
                task_block.seed = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VM_CONST:
                task_block.vm_const = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTHETA_CONST:
                task_block.v_theta_const = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTH_ADPT_EN:
                task_block.v_th_adpt_en = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTH_ALPHA:
                task_block.v_th_alpha = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTH_BETA:
                task_block.v_th_beta = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VTH_INCRE:
                task_block.v_th_incre = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VL:
                task_block.v_l = attribute.int_value
            elif attribute.type == task_ir.AttributeType.REF_LEN:
                task_block.ref_len = attribute.int_value
            elif attribute.type == task_ir.AttributeType.VR:
                task_block.v_reset = attribute.int_value
            elif attribute.type == task_ir.AttributeType.RESET_MODE:
                task_block.reset_mode = attribute.int_value
            elif attribute.type == task_ir.AttributeType.DV:
                task_block.dv = attribute.int_value
            elif attribute.type == task_ir.AttributeType.FIRE_TYPE:
                task_block.fire_type = attribute.int_value

    @staticmethod
    def convert_task_block_data(task_block_ir):
        tensor_ir = task_block_ir.data
        if task_block_ir.precision in [basic_ir.Precision.INT_8, basic_ir.Precision.INT_16, basic_ir.Precision.INT_32,
                                       basic_ir.Precision.TERNARY, basic_ir.Precision.INT_9, basic_ir.Precision.INT_28,
                                       basic_ir.Precision.UINT_4]:
            assert len(tensor_ir.int32_data) != 0
            return np.array(tensor_ir.int32_data, dtype=np.int32).reshape(
                tensor_ir.dims, order='C')
        elif task_block_ir.precision in [basic_ir.Precision.UINT_8, basic_ir.Precision.UINT_16,
                                         basic_ir.Precision.UINT_32]:
            assert len(tensor_ir.uint32_data) != 0
            return np.array(tensor_ir.uint32_data, dtype=np.uint32).reshape(
                tensor_ir.dims, order='C')
        elif task_block_ir.precision in [basic_ir.Precision.FLOAT_16, basic_ir.Precision.FLOAT_32]:
            assert len(tensor_ir.float_data) != 0
            return np.array(tensor_ir.float_data, dtype=np.float32).reshape(
                tensor_ir.dims, order='C')
        else:
            raise TypeError('Invalid data type!')


if __name__ == '__main__':
    case_name = 'cadd'
    case_path = GlobalConfig.Path['test_lib'] + 'mapping_lib/1C1P/cadd.map.txt'
    draw_path = GlobalConfig.Path['temp'] + \
        '{:s}/{:s}.task.html'.format(case_name, case_name)

    task_graph = TaskGraphParser.parse(case_path, 'map')
    STDraw.draw_graph(graph=task_graph, out_path=draw_path,
                      width='1920px', height='1080px')
