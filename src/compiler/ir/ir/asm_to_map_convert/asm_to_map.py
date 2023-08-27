# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum
import numpy as np
import os
import random
import google.protobuf.text_format as text_format
from copy import deepcopy
from typing import Dict, Tuple, Union, List
from top.global_config import GlobalConfig
import src.compiler.ir.asm_pb2 as asm_ir
import src.compiler.ir.task_pb2 as task_ir
import src.compiler.ir.basic_pb2 as basic_ir
import src.compiler.ir.mapping_pb2 as mapping_ir
import src.compiler.ir.data_pb2 as data_ir
from src.compiler.transformer.task_model.task_graph_basics import CTaskBlock, STaskBlock, TaskGraph, Attribute, Edge, Tensor
from src.compiler.transformer.ir_gen.ir_generate_pass import ir_generate


class IOType(Enum):
    INPUT_DATA = data_ir.IOType.INPUT_DATA
    OUTPUT_DATA = data_ir.IOType.OUTPUT_DATA


class TaskBlockType(Enum):
    SI = 0
    SW = 1
    SIC2D = 2
    SIFC = 3
    SIC = 4
    SB = 5
    SWFC = 6
    CADD = 8
    CVVH = 9
    CVM = 10
    CC = 11
    CAX = 12
    CC2D = 13
    CVS = 14
    CCMPB = 15
    CAVG = 16
    CCMPS = 17
    CLUT = 18
    CLIF = 19
    INPUT = 50
    OUTPUT = 51


class AttributeType(Enum):
    KERNEL_X = 0
    KERNEL_Y = 1
    STRIDE_X = 2
    STRIDE_Y = 3
    PAD_UP = 4
    PAD_DOWN = 5
    PAD_LEFT = 6
    PAD_RIGHT = 7
    CMP = 8
    CONSTANT_A = 9
    CONSTANT_B = 10
    DILATION_X = 11
    DILATION_Y = 12
    BIT_SHIGT_NUM = 13


class Precision(Enum):
    INT_8 = 0
    UINT_8 = 1
    INT_16 = 2
    UINT_16 = 3
    INT_32 = 4
    UINT_32 = 5
    FLOAT_16 = 6
    FLOAT_32 = 7
    TERNARY = 8
    INT_9 = 100


class PIIndex(Enum):
    AXON = 1
    SOMA1 = 2
    ROUTER_RECIEVE = 3
    ROUTER_SEND = 4
    SOMA2 = 5
    MEMORY = 6


class BiasType(Enum):
    VECTOR = 1
    CONSTANT = 2


def generate_random_tensor(dims, precision, data_path=None) -> Tensor:
    random_tensor = Tensor()
    random_tensor.dims = dims
    num_elements = 1
    for dim in dims:
        num_elements = num_elements * dim
    random_tensor.precision = precision
    random_tensor.int32_data = list()
    random_tensor.uint32_data = list()

    # if data_path is not None:
    #     input_data = np.fromfile(data_path , dtype=np.int32)
    #     for x in input_data:
    #         random_tensor.int32_data.append(x)
    #     return random_tensor

    if precision == Precision.INT_8.value:
        random_tensor.int32_data = list(
            np.random.randint(-128, 127, num_elements, dtype='int8'))
    elif precision == Precision.TERNARY.value:
        random_tensor.int32_data = list(
            np.random.randint(-1, 2, num_elements, dtype='int8'))
    elif precision == Precision.INT_32.value:
        random_tensor.int32_data = list(
            np.random.randint(-2147483648, 2147483647, num_elements, dtype='int32'))
    elif precision == Precision.UINT_8.value:
        random_tensor.uint32_data = list(
            np.random.randint(0, 255, num_elements, dtype='uint8'))
    else:
        raise ValueError('Unsupported precision type of data')
    return random_tensor


def generate_space_time_coordinate(
        chip_x=0,
        chip_y=0,
        step_group=0,
        phase_group=0,
        core_x=0,
        core_y=0,
        step=0,
        phase=0,
        pi_index=0,
        end_phase=None):
    return SpaceTimeCoordinate(
        chip_array_x=0,
        chip_array_y=0,
        chip_x=chip_x,
        chip_y=chip_y,
        step_group=step_group,
        phase_group=phase_group,
        core_x=core_x,
        core_y=core_y,
        step=step,
        phase=phase,
        pi_index=pi_index,
        end_phase=end_phase
    )


class Axon:
    def __init__(self, axon) -> None:
        self.axon = axon
        self.pic = self.axon.pic
        self.x1_block_name = self.axon.x1_block
        self.output_block_names = list()
        for output_block_name in self.axon.output_block:
            self.output_block_names.append(output_block_name)
        self.reset_x1_addr = self.axon.reset_x1_addr
        self.reset_o_addr = self.axon.reset_o_addr
        self.shape = self.axon.shape
        if self.axon.HasField('bias_block'):
            self.bias_block = self.axon.bias_block
            self.bias_base_addr = self.axon.bias_base_addr
        if self.axon.HasField('p02'):
            self.p02 = self.axon.p02
            self.parse_p02()
        elif self.axon.HasField('p03'):
            self.p03 = self.axon.p03
            self.parse_p03()
        elif self.axon.HasField('p04'):
            self.p04 = self.axon.p04
            self.parse_p04()
        elif self.axon.HasField('p41'):
            self.p41 = self.axon.p41
            self.parse_p41()
        elif self.axon.HasField('p43'):
            self.p43 = self.axon.p43
            self.parse_p43()
        elif self.axon.HasField('p81'):
            self.p81 = self.axon.p81
            self.parse_p81()
        elif self.axon.HasField('p83'):
            self.p83 = self.axon.p83
            self.parse_p83()
        else:
            raise ValueError('Invalid Axon PIC')

    def parse_p02(self):
        self.avg_pooling_en = self.p02.avg_pooling_en
        self.x1_precison = self.p02.x1_precision
        self.bias_type = self.p02.bias_type
        self.has_pad = self.p02.has_pad
        self.constant_b = self.p02.constant_b
        self.a2s2_mode = self.p02.a2s2_mode
        self.stride_x = self.p02.stride_x
        self.stride_y = self.p02.stride_y
        if self.has_pad:
            self.pad_top = self.p02.pad_top
            self.pad_down = self.p02.pad_down
            self.pad_left = self.p02.pad_left
            self.pad_right = self.p02.pad_right
        else:
            self.pad_top = 0
            self.pad_down = 0
            self.pad_left = 0
            self.pad_right = 0

    def parse_p03(self):
        self.tensor_en = self.p03.tensor_en
        self.x1_precison = self.p03.x1_precision
        self.bias_type = self.p03.bias_type
        self.constant_b = self.p03.constant_b
        self.stride_x = self.p03.stride_x
        self.stride_y = self.p03.stride_y
        self.a2s2_mode = self.p03.a2s2_mode

    def parse_p04(self):
        self.x1_precison = self.p04.x1_precision
        self.x2_precision = self.p04.x2_precision
        self.bias_type = self.p04.bias_type
        self.constant_b = self.p04.constant_b
        self.a2s2_mode = self.p04.a2s2_mode

    def parse_p41(self):
        self.x1_precision = self.p41.x1_precision
        self.x2_precision = self.p41.x2_precision
        self.bias_type = self.p41.bias_type
        self.has_pad = self.p41.has_pad
        self.a2s2_mode = self.p41.a2s2_mode
        self.stride_x = self.p41.stride_x
        self.stride_y = self.p41.stride_y
        self.dilate_x = self.p41.dilate_x
        self.dilate_y = self.p41.dilate_y
        if self.has_pad:
            self.pad_top = self.p41.pad_top
            self.pad_down = self.p41.pad_down
            self.pad_left = self.p41.pad_left
            self.pad_right = self.p41.pad_right

    def parse_p43(self):
        self.tensor_en = self.p43.tensor_en
        self.x1_precison = self.p43.x1_precision
        self.x2_precision = self.p43.x2_precision
        self.bias_type = self.p43.bias_type
        self.constant_b = self.p43.constant_b
        self.stride_x = self.p43.stride_x
        self.stride_y = self.p43.stride_y
        self.a2s2_mode = self.p43.a2s2_mode
        self.x2_length = self.p43.x2_length
        self.bias_length = self.p43.bias_length

    def parse_p81(self):
        self.x1_precision = self.p81.x1_precision
        self.x2_precision = self.p81.x2_precision
        self.bias_type = self.p81.bias_type
        self.has_pad = self.p81.has_pad
        self.a2s2_mode = self.p81.a2s2_mode
        self.stride_x = self.p81.stride_x
        self.stride_y = self.p81.stride_y
        self.dilate_x = self.p81.dilate_x
        self.dilate_y = self.p81.dilate_y
        if self.has_pad:
            self.pad_top = self.p81.pad_top
            self.pad_down = self.p81.pad_down
            self.pad_left = self.p81.pad_left
            self.pad_right = self.p81.pad_right

    def parse_p83(self):
        self.tensor_en = self.p83.tensor_en
        self.x1_precison = self.p83.x1_precision
        self.bias_type = self.p83.bias_type
        self.constant_a = self.p83.constant_a
        self.constant_b = self.p83.constant_b
        self.stride_x = self.p83.stride_x
        self.stride_y = self.p83.stride_y
        self.a2s2_mode = self.p83.a2s2_mode
        self.bias_length = self.p83.bias_length


class Soma:
    def __init__(self, soma) -> None:
        self.soma = soma
        self.pic = self.soma.pic
        self.x1_block_name = self.soma.x1_block
        self.output_block_names = list()
        for output_block_name in self.soma.output_block:
            self.output_block_names.append(output_block_name)
        if self.soma.HasField('px5'):
            self.px5 = self.soma.px5
            self.parse_px5()
        elif self.soma.HasField('p06'):
            self.p06 = self.soma.p06
            self.parse_p06()
        elif self.soma.HasField('p26'):
            self.p26 = self.soma.p26
            self.parse_p26()
        elif self.soma.HasField('p07'):
            self.p07 = self.soam.p07
            self.parse_p07()
        elif self.soma.HasField('p08'):
            self.p08 = self.soma.p08
            self.parse_08()
        else:
            raise ValueError('Invalid Soma PIC')

    def parse_px5(self):
        self.pic_mode = self.px5.pic_mode
        self.x1_precision = self.px5.x1_precision
        self.out_precision = self.px5.out_precision
        self.has_pad = self.px5.has_pad
        self.nif = self.px5.nif
        self.nof = self.px5.nof
        self.nix = self.px5.nix
        self.niy = self.px5.niy
        self.nkx = self.px5.nkx
        self.nky = self.px5.nky
        self.stride_x = self.px5.stride_x
        self.stride_y = self.px5.stride_y
        self.pad_top = self.px5.pad_top
        self.pad_down = self.px5.pad_down
        self.pad_left = self.px5.pad_left
        self.pad_right = self.px5.pad_right
        self.bit_shift_num = self.px5.bit_shift_num
        self.x1_base_addr = self.px5.x1_base_addr
        self.o_base_addr = self.px5.o_base_addr
        self.row_pipeline_en = self.px5.row_pipeline_en
        self.row_pipeline_num = self.px5.row_pipeline_num
        self.reset_x1_addr = self.px5.reset_x1_addr
        self.reset_o_addr = self.px5.reset_o_addr

    def parse_p07(self):
        self.lut_block_name = self.soma.lut_block
        self.x1_precision = self.p07.x1_precision
        self.x2_precision = self.p07.x2_precision
        self.neuron_real_num = self.p07.neuron_real_num
        self.group_num = self.p07.group_num
        self.x1_base_addr = self.p07.x1_base_addr
        self.lut_base_addr = self.p07.lut_base_addr
        self.o_base_addr = self.p07.o_base_addr
        self.lut_data_width = self.p07.lut_data_width
        self.bit_shift_num = self.p07.bit_shift_num
        self.row_pipeline_en = self.p07.row_pipeline_en
        self.row_pipeline_num = self.p07.row_pipeline_num
        self.reset_x1_addr = self.p07.reset_x1_addr
        self.reset_o_addr = self.p07.reset_o_addr

    def parse_08(self):
        self.uin_block_name = self.soma.uin_block
        self.s_block_name = self.soma.s_block
        self.v_block_name = self.soma.v_block_name
        self.vm_block_name = self.soma.vm_block_name
        self.vtheta_block_name = self.soma.vtheta_block_name
        self.neuron_num = self.p08.neuron_num
        self.group_num = self.p08.group_num
        self.seed = self.p08.seed
        self.vth0 = self.p08.Vth0
        self.vth_adpt_en = self.p08.Vth_adpt_en
        self.vth_alpha = self.p08.Vth_alpha
        self.vth_beta = self.p08.Vth_beta
        self.vth_incre = self.p08.Vth_Incre
        self.vr = self.p08.VR
        self.vl = self.p08.VL
        self.vleaky_adpt_en = self.p08.Vleaky_adpt_en
        self.vleaky_alpha = self.p08.Vleaky_alpha
        self.vleaky_beta = self.p08.Vleaky_beta
        self.dv = self.p08.dV
        self.ref_len = self.p08.Ref_len
        self.tw_cnt = self.p08.Tw_cnt
        self.vinit = self.p08.Vinit
        self.tw_len = self.p08.Tw_len
        self.tw_en = self.p08.Tw_en
        self.vm_const_en = self.p08.VM_const_en
        self.vm_const = self.p08.Vm_const
        self.vm_len = self.p08.Vm_len
        self.vtheta_const_en = self.p08.Vtheta_const_en
        self.vtheta_const = self.p08.vtheta_const
        self.vtheta_len = self.p08.Vtheta_len
        self.ref_cnt_const_en = self.p08.ref_cnt_const_en
        self.ref_cnt_const = self.p08.ref_cnt_const
        self.reset_mode = self.p08.reset_mode
        self.fire_type = self.p08.fire_type
        self.bit_shift_num = self.p08.bit_shift_num
        self.uin_base_addr = self.p08.uin_base_addr
        self.s_base_addr = self.p08.s_base_addr
        self.v_base_addr = self.p08.v_base_addr
        self.vm_base_addr = self.p08.vm_base_addr
        self.vtheta_base_addr = self.p08.vtheta_base_addr
        self.para_base_addr = self.p08.para_base_addr
        self.row_pipeline_en = self.p08.row_pipeline_en
        self.row_pipeline_num = self.p08.row_pipeline_num
        self.reset_uin_addr = self.p08.reset_uin_addr
        self.reset_o_addr = self.p08.reset_o_addr
        self.reset_s_addr = self.p08.reset_s_addr
        self.reset_vm_addr = self.p08.reset_vm_addr
        self.reset_vtheta_addr = self.p08.reset_vtheta_addr

    def parse_p06(self):
        self.ciso_block_name = self.soma.ciso_block
        self.x1_precision = self.p06.x1_precision
        self.out_precision = self.p06.out_precision
        self.length_in = self.p06.length_in
        self.length_out = self.p06.length_out
        self.length_ciso = self.p06.length_ciso
        self.num_in = self.p06.num_in
        self.num_out = self.p06.num_out
        self.num_ciso = self.p06.num_ciso
        self.x1_base_addr = self.p06.x1_base_addr
        self.ciso_base_addr = self.p06.ciso_base_addr
        self.o_base_addr = self.p06.o_base_addr
        self.bit_shift_num = self.p06.bit_shift_num
        self.row_pipeline_en = self.p06.row_pipeline_en
        self.row_pipeline_num = self.p06.row_pipeline_num
        self.in_ciso_pipe_sel = self.p06.in_ciso_pipe_sel
        self.reset_x1_addr = self.p06.reset_x1_addr
        self.reset_ciso_addr = self.p06.reset_ciso_addr
        self.reset_o_addr = self.p06.reset_o_addr
        self.real_length_in_en = self.p06.real_length_in_en
        self.real_num_in = self.p06.real_num_in

    def parse_p26(self):
        self.ciso_block_name = self.soma.ciso_block
        self.x1_precision = self.p26.x1_precision
        self.out_precision = self.p26.out_precision
        self.length_in = self.p26.length_in
        self.length_out = self.p26.length_out
        self.length_ciso = self.p26.length_ciso
        self.num_in = self.p26.num_in
        self.num_out = self.p26.num_out
        self.num_ciso = self.p26.num_ciso
        self.x1_base_addr = self.p26.x1_base_addr
        self.ciso_base_addr = self.p26.ciso_base_addr
        self.o_base_addr = self.p26.o_base_addr
        self.bit_shift_num = self.p26.bit_shift_num
        self.row_pipeline_en = self.p26.row_pipeline_en
        self.row_pipeline_num = self.p26.row_pipeline_num
        self.in_ciso_pipe_sel = self.p26.in_ciso_pipe_sel
        self.reset_x1_addr = self.p26.reset_x1_addr
        self.reset_ciso_addr = self.p26.reset_ciso_addr
        self.reset_o_addr = self.p26.reset_o_addr
        self.real_length_in_en = self.p26.real_length_in_en
        self.real_num_in = self.p26.real_num_in


class Router:
    def __init__(self, router) -> None:
        self.router = router
        assert self.router.HasField('p09')
        self.send_block_name = self.router.send_block
        self.router_table_block = self.router.router_table_block
        self.receive_block = self.router.receive_block


class PrimList:
    def __init__(self, prim_list):
        self.prim_list = prim_list
        if self.prim_list.HasField('axon'):
            self.axon = self.parse_axon(self.prim_list.axon)
        if self.prim_list.HasField('soma1'):
            self.soma1 = self.parse_soma(self.prim_list.soma1)
        if self.prim_list.HasField('router'):
            self.router = self.parse_router(self.prim_list.router)
        if self.prim_list.HasField('soma2'):
            self.soma2 = self.parse_soma(self.prim_list.soma2)

    def parse_axon(self, axon) -> Axon:
        return Axon(axon)

    def parse_soma(self, soma) -> Soma:
        return Soma(soma)

    def parse_router(self, router) -> Router:
        return Router(router)


class RegisterConfig:
    def __init__(self, register_config) -> None:
        self.register_config = register_config
        self.receive_pi_addr_base = self.register_config.Receive_PI_addr_base
        self.pi_cxy = self.register_config.PI_CXY
        self.pi_nx = self.register_config.PI_Nx
        self.pi_ny = self.register_config.PI_Ny
        self.pi_sign_cxy = self.register_config.PI_sign_CXY
        self.pi_sign_nx = self.register_config.PI_sign_Nx
        self.pi_sign_ny = self.register_config.PI_sign_Ny
        self.instant_pi_en = self.register_config.instant_PI_en
        self.fixed_instant_pi = self.register_config.fixed_instant_PI
        self.instant_pi_number = self.register_config.instant_PI_number
        self.pi_loop_en = self.register_config.PI_loop_en
        self.pi_loop_num = self.register_config.PI_loop_num
        self.start_instant_pi_num = self.register_config.start_instant_PI_num
        self.addr_instant_pi_base = self.register_config.Addr_instant_PI_base


class Core:
    def __init__(self, core_config) -> None:
        self.core_config = core_config
        self.core_x = self.core_config.core_x
        self.core_y = self.core_config.core_y
        self.static_prim_lists, self.instant_prim_lists = self.parse_prim_list()
        self.register_config = RegisterConfig(self.core_config.registers)

    def parse_prim_list(self) -> Tuple[List[PrimList]]:
        static_prim_lists = list()
        instant_prim_lists = list()
        for static_prim_list in self.core_config.static_prim_list:
            static_prim_lists.append(PrimList(static_prim_list))
        for instant_prim_list in self.core_config.instant_prim_list:
            instant_prim_lists.append(PrimList(instant_prim_list))
        return static_prim_lists, instant_prim_lists


class PhaseGroup:
    def __init__(self, phase_group_config) -> None:
        self.phase_group_config = phase_group_config
        self.phase_group_id = self.phase_group_config.phase_group_id
        self.cores = self.parse_core_config()

    def parse_core_config(self) -> List[Core]:
        cores = list()
        for core_config in self.phase_group_config.core_config:
            cores.append(Core(core_config))
        return cores


class Step:
    def __init__(self, step_config) -> None:
        self.step_config = step_config
        self.chip_x = self.step_config.chip_x
        self.chip_y = self.step_config.chip_y
        self.step_group_id = self.step_config.step_group_id
        self.phase_groups = self.parse_phase_group_config()

    def parse_phase_group_config(self) -> List[PhaseGroup]:
        phase_groups = list()
        for phase_group_config in self.step_config.phase_group_config:
            phase_groups.append(PhaseGroup(phase_group_config))
        return phase_groups


class Shape:
    def __init__(self, shape=None) -> None:
        if shape is not None:
            if shape.HasField('nx'):
                self.nx = shape.nx
            if shape.HasField('ny'):
                self.ny = shape.ny
            if shape.HasField('nf'):
                self.nf = shape.nf
            if shape.HasField('nr'):
                self.nr = shape.nr
            if shape.HasField('nkx'):
                self.nkx = shape.nkx
            if shape.HasField('nky'):
                self.nky = shape.nky
            if shape.HasField('nix'):
                self.nix = shape.nix
            if shape.HasField('niy'):
                self.niy = shape.niy


def convert_shape_to_dict(shape) -> Dict:
    shape_dict = {}
    shape_dict.update({'x': shape.nx})
    shape_dict.update({'y': shape.ny})
    shape_dict.update({'f': shape.nf})
    shape_dict.update({'r': shape.nr})
    shape_dict.update({'kx': shape.nkx})
    shape_dict.update({'ky': shape.nky})
    shape_dict.update({'iy': shape.niy})
    shape_dict.update({'ix': shape.nix})
    return shape_dict


class StaticBlock:
    def __init__(self, static_block) -> None:
        self.id = static_block.id
        self.precision = static_block.precision
        self.chip_x = static_block.chip_idx
        self.chip_y = static_block.chip_idy
        self.core_x = static_block.core_idx
        self.core_y = static_block.core_idy
        self.start_addr = static_block.start_addr


class DynamicBlock:
    def __init__(self, dynamic_block) -> None:
        self.io_type = dynamic_block.io_type
        self.id = dynamic_block.id
        self.begin_position = Shape(dynamic_block.begin_position)
        self.shape = Shape(dynamic_block.shape)
        self.chip_x = dynamic_block.chip_idx
        self.chip_y = dynamic_block.chip_idy
        self.core_x = dynamic_block.core_idx
        self.core_y = dynamic_block.core_idy
        self.start_addr = dynamic_block.start_addr
        self.length = dynamic_block.length
        self.step_group = dynamic_block.step_group
        self.phase_group = dynamic_block.phase_group
        self.precision = dynamic_block.precision
        self.phases = [phase for phase in dynamic_block.phases]


class DataConfig:
    def __init__(self, data_config) -> None:
        self.data_config = data_config
        self.static_blocks = self.parse_static_blocks()
        self.dynamic_blocks = self.parse_dynamic_blocks()

    def parse_static_blocks(self) -> List[StaticBlock]:
        static_blocks = list()
        for static_block in self.data_config.static_blocks:
            static_blocks.append(StaticBlock(static_block))
        return static_blocks

    def parse_dynamic_blocks(self) -> List[DynamicBlock]:
        dynamic_blocks = list()
        for dynamic_block in self.data_config.dynamic_blocks:
            dynamic_blocks.append(DynamicBlock(dynamic_block))
        return dynamic_blocks


class Asm:
    def __init__(self, asm_path: str) -> None:
        self.asm_config = asm_ir.Config()
        with open(asm_path, 'r') as f:
            text_format.Parse(f.read(), self.asm_config)
        random.seed(self.asm_config.test_config.random_seed)
        np.random.seed(self.asm_config.test_config.random_seed)
        self.behavior_config = self.asm_config.behavior_config
        self.data_config = DataConfig(self.asm_config.data_config)
        self.test_config = self.asm_config.test_config
        self.steps = self.parse_step_config()

    def parse_step_config(self) -> List[Step]:
        steps = list()
        for step_config in self.behavior_config.step_config:
            steps.append(Step(step_config))
        return steps


class SpaceCoordinate:
    def __init__(self,
                 chip_array_x: int,
                 chip_array_y: int,
                 chip_x: int,
                 chip_y: int,
                 step_group: int,
                 phase_group: int,
                 core_x: int,
                 core_y: int) -> None:
        self.chip_array_x = chip_array_x
        self.chip_array_y = chip_array_y
        self.chip_x = chip_x
        self.chip_y = chip_y
        self.step_group = step_group
        self.phase_group = phase_group
        self.core_x = core_x
        self.core_y = core_y


class TimeCoordinate:
    def __init__(self,
                 step,
                 phase,
                 pi_index,
                 end_phase) -> None:
        self.step = step
        self.phase = phase
        assert self.phase < 32, 'Phase number out of range'
        self.pi_index = pi_index
        self.end_phase = end_phase


class SpaceTimeCoordinate:
    def __init__(self,
                 chip_array_x: int,
                 chip_array_y: int,
                 chip_x: int,
                 chip_y: int,
                 step_group: int,
                 phase_group: int,
                 core_x: int,
                 core_y: int,
                 step: int,
                 phase: int,
                 pi_index: int,
                 end_phase: int) -> None:
        self.space = SpaceCoordinate(
            chip_array_x, chip_array_y, chip_x, chip_y, step_group, phase_group, core_x, core_y)
        self.time = TimeCoordinate(step, phase, pi_index, end_phase)


class Mapping:
    def __init__(self) -> None:
        self.graph = TaskGraph()
        self.space_time_mapping = {}
        # space_time_mapping的每个key为graph中的task_id
        # value为一个列表
        # 列表中的每个元素为一个SpaceTimeCoordinate

        # 测试用例信息记录
        self.random_seed = 0
        self.test_case_name = ''

    def add_block(self, block: Union[STaskBlock, CTaskBlock]):
        self.graph.add_block(block)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge)

    def set_st_coordinate(self, block: Union[STaskBlock, CTaskBlock], st_coordinate: SpaceTimeCoordinate):
        if block.id not in self.graph.blocks:
            self.add_block(block)
        self.space_time_mapping.update({block.id: st_coordinate})

    def get_st_coordinate(self, task_id: int) -> SpaceTimeCoordinate:
        return self.space_time_mapping[task_id]

    def generate_mapping_ir(self, mapping_path=None, readable_result=False):
        mapping = mapping_ir.Mapping()
        ir_generate(self.graph, None, mapping.graph)
        self.convert_st_mapping(mapping.space_time_mapping)

        mapping.graph.test_config.random_seed = self.random_seed
        mapping.graph.test_config.test_case_name = self.test_case_name
        mapping.graph.test_config.test_mode = basic_ir.CaseMode.CASE_OUTPUT

        if mapping_path is not None:
            with open(mapping_path, 'wb') as f:
                f.write(mapping.SerializeToString())
        if readable_result:
            mapping = mapping_ir.Mapping()
            with open(mapping_path, 'rb') as f:
                mapping.ParseFromString(f.read())
            with open(mapping_path + '.txt', 'w') as f:
                f.write(repr(mapping))
        return mapping

    def convert_st_mapping(self, st_mapping=None):
        if st_mapping is None:
            st_mapping = mapping_ir.SpaceTimeMapping()
        for task_id in self.space_time_mapping:
            st_coordinate = self.get_st_coordinate(task_id)
            mapping_dict = st_mapping.node_map_dicts.add()
            mapping_dict.task_id = task_id
            st_coordinate_ir = mapping_dict.space_time_coordinates.add()
            self.convert_st_coordinate(st_coordinate_ir, st_coordinate)
        return st_mapping

    def convert_st_coordinate(self, st_coordinate_ir, st_coordinate: SpaceTimeCoordinate):
        st_coordinate_ir.space.chip_array.x = st_coordinate.space.chip_array_x
        st_coordinate_ir.space.chip_array.y = st_coordinate.space.chip_array_y
        st_coordinate_ir.space.chip.x = st_coordinate.space.chip_x
        st_coordinate_ir.space.chip.y = st_coordinate.space.chip_y
        st_coordinate_ir.space.step_group = st_coordinate.space.step_group
        st_coordinate_ir.space.phase_group = st_coordinate.space.phase_group
        st_coordinate_ir.space.core.x = st_coordinate.space.core_x
        st_coordinate_ir.space.core.y = st_coordinate.space.core_y
        st_coordinate_ir.time.step = st_coordinate.time.step
        st_coordinate_ir.time.phase = st_coordinate.time.phase
        st_coordinate_ir.time.pi_index = st_coordinate.time.pi_index
        if st_coordinate.time.end_phase is not None:
            st_coordinate_ir.time.end_phase = st_coordinate.time.end_phase


class Asm2Mapping:
    def __init__(self, asm_path: str) -> None:
        self.asm_config = Asm(asm_path=asm_path)
        self.mapping = Mapping()
        self.case_name = asm_path.split('/')[-1].split('.')[0]

    def convert_1c1p_asm_to_mapping(self):
        assert len(self.asm_config.steps) == 1
        asm_step = self.asm_config.steps[0]
        self.step = 0
        self.chip_x = asm_step.chip_x
        self.chip_y = asm_step.chip_y
        self.step_group = asm_step.step_group_id
        assert len(asm_step.phase_groups) == 1
        asm_phase_group = asm_step.phase_groups[0]
        self.phase_group = asm_phase_group.phase_group_id
        assert len(asm_phase_group.cores) == 1
        asm_core = asm_phase_group.cores[0]
        self.core_x = asm_core.core_x
        self.core_y = asm_core.core_y
        assert len(asm_core.static_prim_lists) == 1
        asm_static_prim_list = asm_core.static_prim_lists[0]
        self.phase = 0
        if hasattr(asm_static_prim_list, 'axon'):
            self.convert_axon(asm_static_prim_list.axon)
        if hasattr(asm_static_prim_list, 'soma1'):
            self.pi_index = PIIndex.SOMA1.value
        if hasattr(asm_static_prim_list, 'soma2'):
            self.pi_index = PIIndex.SOMA2.value

        self.mapping.random_seed = self.asm_config.test_config.random_seed
        self.mapping.test_case_name = self.asm_config.test_config.test_case_name

    def generate_mapping_ir(self, mapping_path: str = None, readable_result=False):
        return self.mapping.generate_mapping_ir(mapping_path, readable_result)

    def generate_space_time_coordinate(
            self,
            chip_x=0,
            chip_y=0,
            step_group=0,
            phase_group=0,
            core_x=0,
            core_y=0,
            step=0,
            phase=0,
            pi_index=0,
            end_phase=None):
        return SpaceTimeCoordinate(
            chip_array_x=0,
            chip_array_y=0,
            chip_x=chip_x,
            chip_y=chip_y,
            step_group=step_group,
            phase_group=phase_group,
            core_x=core_x,
            core_y=core_y,
            step=step,
            phase=phase,
            pi_index=pi_index,
            end_phase=end_phase
        )

    def convert_axon(self, asm_axon: Axon):
        if hasattr(asm_axon, 'p02'):
            self.convert_p02(asm_axon)

    def convert_p02(self, asm_axon: Axon):
        asm_output = self.asm_config.data_config.dynamic_blocks[0]
        assert asm_output.io_type == IOType.OUTPUT_DATA.value
        asm_input = self.asm_config.data_config.static_blocks[0]
        self.data_path = "/data/{}/{}.dat".format(self.case_name, asm_input.id)
        # 边
        input_edge = Edge(id=0, src_block_id=0, dst_block_id=1)
        output1_edge = Edge(id=1, src_block_id=1, dst_block_id=2)
        output2_edge = Edge(id=100, src_block_id=2, dst_block_id=100)

        if asm_axon.bias_type == BiasType.VECTOR.value:
            bias_edge = Edge(id=2, src_block_id=3, dst_block_id=1)
            self.mapping.graph.add_edge(bias_edge)
        num_branches = asm_axon.shape.nky * asm_axon.shape.nkx
        if not(asm_axon.avg_pooling_en):
            input_edges = list()
            for i in range(num_branches - 1):
                new_edge = Edge(id=3 + i, src_block_id=4 + i, dst_block_id=1)
                input_edges.append(new_edge)
                self.mapping.graph.add_edge(new_edge)

        # 计算任务块
        c_block = CTaskBlock(id=1)
        if asm_axon.avg_pooling_en:
            c_block.set_type(TaskBlockType.CAVG.value)
        else:
            c_block.set_type(TaskBlockType.CADD.value)
        c_block.set_precision(asm_output.precision)
        c_block.set_shape(convert_shape_to_dict(asm_axon.shape))
        attr = Attribute(
            type=AttributeType.CONSTANT_B.value,
            precision=Precision.INT_32.value,
            value=asm_axon.constant_b
        )
        c_block.add_attribute(attr=attr, attr_name='constant_b')
        attr = Attribute(
            type=AttributeType.STRIDE_X.value,
            precision=Precision.INT_32.value,
            value=asm_axon.stride_x
        )
        c_block.add_attribute(attr=attr, attr_name='stride_x')
        attr = Attribute(
            type=AttributeType.STRIDE_Y.value,
            precision=Precision.INT_32.value,
            value=asm_axon.stride_y
        )
        c_block.add_attribute(attr=attr, attr_name='stride_y')
        attr = Attribute(
            type=AttributeType.PAD_UP.value,
            precision=Precision.INT_32.value,
            value=asm_axon.pad_top
        )
        c_block.add_attribute(attr=attr, attr_name='pad_top')
        attr = Attribute(
            type=AttributeType.PAD_DOWN.value,
            precision=Precision.INT_32.value,
            value=asm_axon.pad_down
        )
        c_block.add_attribute(attr=attr, attr_name='pad_down')
        attr = Attribute(
            type=AttributeType.PAD_LEFT.value,
            precision=Precision.INT_32.value,
            value=asm_axon.pad_left
        )
        c_block.add_attribute(attr=attr, attr_name='pad_left')
        attr = Attribute(
            type=AttributeType.PAD_RIGHT.value,
            precision=Precision.INT_32.value,
            value=asm_axon.pad_right
        )
        c_block.add_attribute(attr=attr, attr_name='pad_right')
        input_s_block_shape = {
            'y': asm_axon.shape.niy,
            'x': asm_axon.shape.nix,
            'f': asm_axon.shape.nf,
            'r': -1,
            'kx': -1,
            'ky': -1
        }
        c_block.create_input_cluster(
            edge_id=input_edge.get_id(),
            s_task_block_shape=input_s_block_shape
        )
        if not(asm_axon.avg_pooling_en):
            for i in range(num_branches - 1):
                c_block.create_input_cluster(
                    edge_id=input_edges[i].get_id(),
                    s_task_block_shape=input_s_block_shape
                )
        bias_shape = {
            'y': -1,
            'x': -1,
            'f': asm_axon.shape.nf,
            'r': -1,
            'kx': -1,
            'ky': -1
        }
        if asm_axon.bias_type == BiasType.VECTOR.value:
            c_block.create_input_cluster(
                edge_id=bias_edge.get_id(),
                s_task_block_shape=bias_shape
            )
        c_block.create_empty_output_cluster()
        c_block.add_interface_to_output_cluster(edge_id=output1_edge.get_id())
        st_coordinate = self.generate_space_time_coordinate(
            chip_x=self.chip_x,
            chip_y=self.chip_y,
            step_group=self.step_group,
            phase_group=self.phase_group,
            core_x=self.core_x,
            core_y=self.core_y,
            pi_index=PIIndex.AXON.value
        )
        self.mapping.space_time_mapping.update(
            {c_block.get_id(): st_coordinate})

        # 输入存储任务块
        input_s_block = STaskBlock(id=0)
        input_s_block.set_type(TaskBlockType.SI.value)
        input_s_block.set_precision(asm_input.precision)
        input_s_block.set_shape(input_s_block_shape)
        input_s_block.create_output_cluster(edge_id=input_edge.get_id())
        input_s_block_dims = [input_s_block.get_shape()['y'], input_s_block.get_shape()[
            'x'], input_s_block.get_shape()['f']]

        input_s_block.set_data(generate_random_tensor(
            input_s_block_dims, input_s_block.get_precision(), self.data_path))

        st_coordinate = self.generate_space_time_coordinate(
            chip_x=self.chip_x,
            chip_y=self.chip_y,
            step_group=self.step_group,
            phase_group=self.phase_group,
            core_x=self.core_x,
            core_y=self.core_y,
            pi_index=PIIndex.MEMORY.value
        )
        self.mapping.space_time_mapping.update(
            {input_s_block.get_id(): st_coordinate})

        if not(asm_axon.avg_pooling_en):
            new_input_s_blocks = list()
            for i in range(num_branches - 1):
                new_input_s_block = STaskBlock(id=4 + i)
                new_input_s_block.set_type(TaskBlockType.SI.value)
                new_input_s_block.set_precision(asm_input.precision)
                new_input_s_block.set_shape(input_s_block_shape)
                new_input_s_block.create_output_cluster(
                    edge_id=input_edges[i].get_id())
                new_input_s_block_dims = [new_input_s_block.get_shape(
                )['y'], new_input_s_block.get_shape()['x'], new_input_s_block.get_shape()['f']]
                new_input_s_block.set_data(generate_random_tensor(
                    new_input_s_block_dims, new_input_s_block.get_precision(), self.data_path))
                st_coordinate = self.generate_space_time_coordinate(
                    chip_x=self.chip_x,
                    chip_y=self.chip_y,
                    step_group=self.step_group,
                    phase_group=self.phase_group,
                    core_x=self.core_x,
                    core_y=self.core_y,
                    pi_index=PIIndex.MEMORY.value
                )
                self.mapping.space_time_mapping.update(
                    {new_input_s_block.get_id(): st_coordinate})
                self.mapping.graph.add_block(new_input_s_block)
                new_input_s_blocks.append(new_input_s_block)

        # bias
        if asm_axon.bias_type == BiasType.VECTOR.value:
            bias_block = STaskBlock(id=3)
            bias_block.set_type(TaskBlockType.SB.value)
            bias_block.set_precision(asm_output.precision)
            bias_block.set_shape(bias_shape)
            bias_block.create_output_cluster(edge_id=bias_edge.get_id())
            bias_block_dims = [bias_block.get_shape()['f']]
            bias_block.set_data(generate_random_tensor(
                bias_block_dims, bias_block.get_precision()))
            st_coordinate = self.generate_space_time_coordinate(
                chip_x=self.chip_x,
                chip_y=self.chip_y,
                step_group=self.step_group,
                phase_group=self.phase_group,
                core_x=self.core_x,
                core_y=self.core_y,
                pi_index=PIIndex.MEMORY.value
            )
            self.mapping.space_time_mapping.update(
                {bias_block.get_id(): st_coordinate})
            self.mapping.graph.add_block(bias_block)

        # 输出存储任务块
        output_s_block = STaskBlock(id=2)
        output_s_block.set_type(TaskBlockType.SI.value)
        output_s_block.set_precision(asm_output.precision)
        output_s_block.set_shape(
            {
                'y': asm_axon.shape.ny,
                'x': asm_axon.shape.nx,
                'f': asm_axon.shape.nf,
                'r': -1,
                'kx': -1,
                'ky': -1
            }
        )
        output_s_block.create_input_cluster(
            edge_id=output1_edge.get_id(),
            c_block_output_cluster_shape=c_block.get_output_cluster().get_shape()
        )
        output_s_block.create_output_cluster(edge_id=output2_edge.get_id())
        self.mapping.space_time_mapping.update(
            {output_s_block.get_id(): st_coordinate})

        # 输出结点
        output_block = CTaskBlock(id=100)
        output_block.set_type(TaskBlockType.OUTPUT.value)
        output_block.set_precision(asm_output.precision)
        output_block.socket_id = 0
        output_block.set_shape(
            {
                'y': asm_axon.shape.ny,
                'x': asm_axon.shape.nx,
                'f': asm_axon.shape.nf,
                'r': -1,
                'kx': -1,
                'ky': -1
            }
        )
        output_block.create_input_cluster(
            edge_id=output2_edge.get_id(),
            s_task_block_shape=output_s_block.get_shape()
        )
        st_coordinate = self.generate_space_time_coordinate(
            chip_x=0,
            chip_y=0,
            step_group=0,
            phase_group=0,
            core_x=-1,
            core_y=0,
            pi_index=PIIndex.ROUTER_RECIEVE.value
        )
        self.mapping.space_time_mapping.update(
            {output_block.get_id(): st_coordinate})

        # 任务图
        self.mapping.graph.add_block(c_block)
        self.mapping.graph.add_block(input_s_block)
        self.mapping.graph.add_block(output_s_block)
        self.mapping.graph.add_block(output_block)
        self.mapping.graph.add_edge(input_edge)
        self.mapping.graph.add_edge(output1_edge)
        self.mapping.graph.add_edge(output2_edge)
        self.mapping.graph.add_group(c_block, [input_s_block, output_s_block])
        if not(asm_axon.avg_pooling_en):
            for s_block in new_input_s_blocks:
                self.mapping.graph.get_group(
                    c_block.get_id()).append(s_block.get_id())
        if asm_axon.bias_type == BiasType.VECTOR.value:
            self.mapping.graph.get_group(
                c_block.get_id()).append(bias_block.get_id())

    def convert_soma(self, asm_soma: Soma):
        pass


def execute_asm_to_mapping(behavior_lib_path, mapping_lib_path, readable_result=False):
    os.makedirs(mapping_lib_path, exist_ok=True)

    for asm_case_path in os.listdir(behavior_lib_path):
        asm_path = os.path.join(behavior_lib_path, asm_case_path)
        asm_to_mapping = Asm2Mapping(asm_path=asm_path)

        asm_to_mapping.convert_1c1p_asm_to_mapping()
        mapping_path = os.path.join(
            mapping_lib_path, asm_case_path.split('.')[0] + '.map')
        asm_to_mapping.generate_mapping_ir(mapping_path)
        if readable_result:
            mapping = mapping_ir.Mapping()
            with open(mapping_path, 'rb') as f:
                mapping.ParseFromString(f.read())
            with open(mapping_path + '.txt', 'w') as f:
                f.write(repr(mapping))


if __name__ == '__main__':
    execute_asm_to_mapping(behavior_lib_path=GlobalConfig.Path['test_lib'] + 'behavior_lib/1C1P/P02',
                           mapping_lib_path=GlobalConfig.Path['temp'] + 'P02/',
                           readable_result=True)
