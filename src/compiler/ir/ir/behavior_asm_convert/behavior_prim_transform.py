# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import numpy as np
import src.compiler.ir.basic_pb2 as tj_basic
import src.compiler.ir.data_pb2 as tj_data
import src.compiler.ir.asm_pb2 as tj_asm
from old.primitive import Prim_26_move_split, Prim_06_move_merge


class BehaviorPrimTransform:
    def __init__(self, asm_ir, current_pos) -> None:
        self.data_block_num = 0
        self.data_blocks = {}
        self.axon_output_area = []
        self.soma_output_area = []
        self.router_output_area = []
        self.soma_output_area = []

        self.pos = current_pos
        self.asm_ir = asm_ir

        self.precision_map = {0: tj_basic.Precision.INT_32,
                              1: tj_basic.Precision.INT_8,
                              2: tj_basic.Precision.UINT_8,
                              3: tj_basic.Precision.TERNARY}
        self.bias_type_map = {0: tj_basic.BiasType.CONSTANT,
                              1: tj_basic.BiasType.CONSTANT,
                              2: tj_basic.BiasType.VECTOR,
                              3: tj_basic.BiasType.VECTOR}

    def config_axon_prim(self, axon_config, axon_case):
        axon_config.pic = axon_case.PIC
        if axon_case.PIC == 0x02:
            self.config_p02(axon_config, axon_case)
        elif axon_case.PIC == 0x03:
            self.config_p03(axon_config, axon_case)
        elif axon_case.PIC == 0x04:
            self.config_p04(axon_config, axon_case)
        elif axon_case.PIC == 0x41:
            self.config_p41(axon_config, axon_case)
        elif axon_case.PIC == 0x43:
            self.config_p43(axon_config, axon_case)
        elif axon_case.PIC == 0x81:
            self.config_p81(axon_config, axon_case)
        elif axon_case.PIC == 0x83:
            self.config_p83(axon_config, axon_case)

    def config_soma_prim(self, soma_config, soma_case, soma_type):
        soma_config.pic = soma_case.PIC
        if soma_case.PIC == 0x25 or soma_case.PIC == 0x05:
            self.config_px5(soma_config, soma_case, soma_type)
        elif soma_case.PIC == 0x07:
            self.config_p07(soma_config, soma_case, soma_type)
        elif soma_case.PIC == 0x08:
            self.config_p08(soma_config, soma_case, soma_type)
        elif soma_case.PIC == 0x06 and soma_case.pic_mode == 0:
            self.config_p06(soma_config, soma_case, soma_type)
        elif soma_case.PIC == 0x06 and soma_case.pic_mode == 1:
            soma_config.pic = 0x26  # 注意我们把PIC变成了26
            self.config_p26(soma_config, soma_case, soma_type)

    def config_router_prim(self, router_config, router_case):
        p09_config = router_config.p09
        # 原语参数
        for attr in dir(router_case):
            if attr.startswith('_'):
                continue

            value = getattr(router_case, attr)

            if attr == 'Rhead_mode':
                p09_config.router_head_mode = tj_asm.RouterHeadMode.SINGLE_USE if value == 0 else tj_asm.RouterHeadMode.MULTI_USE
                continue

            if attr == 'T_mode':
                p09_config.packet_size_mode = tj_asm.PacketSizeMode.SINGLE_NEURON if value == 0 else tj_asm.PacketSizeMode.MULTIPLE_NEURON
                continue

            if attr == 'Dout_memory_select':
                p09_config.out_memory = tj_asm.MemoryArea.MEM2 if value == 0 else tj_asm.MemoryArea.MEM3
                continue
            
            if hasattr(p09_config, attr):
                setattr(p09_config, attr, value)

        # 表头
        for head in router_case.RHeadList:
            router_heads = p09_config.router_heads.add()
            router_heads.is_instant_request = head['S']
            router_heads.packet_size_mode = tj_asm.PacketSizeMode.SINGLE_NEURON if head['T'] == 0 else tj_asm.PacketSizeMode.MULTIPLE_NEURON
            router_heads.is_packet_finish = head['P']
            router_heads.relay_type = tj_asm.RelayType.UNI_CAST if head['Q'] == 0 else tj_asm.RelayType.MULTI_RELAY_CAST
            router_heads.dx = head['X']
            router_heads.dy = head['Y']
            router_heads.destination_addr = head['A']

            if 'pack_per_Rhead' in head:
                router_heads.pack_per_router_head = head['pack_per_Rhead']
            if 'A_offset' in head:
                router_heads.destination_offset = head['A_offset']
            if 'Const' in head:
                router_heads.destination_const = head['Const']
            if 'EN' in head:
                router_heads.enable = head['EN']

        # 数据块（在这里如果有的话，就是指要发送的数据，表头数据块不在这里生成）
        data_config = self.asm_ir.data_config
        for block_case in router_case.memory_blocks:
            block_id = 'p09_send_' + str(self.data_block_num)
            router_config.send_block = block_id
            precision = tj_basic.Precision.INT_16
            self.config_static_block(block_id, data_config, block_case, precision)
        
        if not self.router_output_area:
            return
        block_id = 'p09_recieve_' + str(self.data_block_num)
        router_config.receive_block = block_id
        
        assert len(self.router_output_area) == 1
        self.config_o_block(block_id, data_config, self.router_output_area[0], None)

    def general_prim_config(self, prim_config, prim_case):
        for attr in dir(prim_case):
            if attr.startswith('_'):
                continue

            if hasattr(prim_config, attr):
                value = getattr(prim_case, attr)
                # 各种需要转换一下的情况
                if attr.endswith('precision'):
                    setattr(prim_config, attr, self.precision_map[value])
                    continue

                if attr == 'bias_type':
                    setattr(prim_config, attr, self.bias_type_map[value])
                    continue

                if attr == 'a2s2_mode':
                    prim_config.a2s2_mode = False
                    if value:
                        prim_config.a2s2_mode = True
                    continue

                if attr == 'delay_clock':
                    continue

                if attr == 'axon_delay':
                    prim_config.axon_delay = False
                    if value:
                        prim_config.axon_delay = True
                        prim_config.a2s2_mode = True
                        prim_config.delay_clock = prim_case.delay_clock
                    continue

                if attr == 'row_pipeline_en':
                    prim_config.row_pipeline_en = False
                    if value:
                        prim_config.row_pipeline_en = True
                    continue

                if attr.startswith('reset') and attr.endswith('addr'):
                    if not value:
                        setattr(prim_config, attr, False)
                    else:
                        setattr(prim_config, attr, True)
                    continue

                if attr == 'compare_init':
                    cmp_list = self.convert_cmp(
                        prim_case.compare_init, prim_case.out_precision)
                    prim_config.compare_init.extend(cmp_list)
                    continue

                if attr == 'seed':
                    prim_config.seed = prim_case.first_seed
                    continue

                if attr == 'Tw_cnt':
                    prim_config.Tw_cnt = prim_case.first_tw_cnt
                    continue

                setattr(prim_config, attr, value)

    def set_axon_addr(self, axon_config, axon_case):
        axon_config.x1_base_addr = axon_case.x1_base_addr
        if axon_case.PIC != 0x02 and axon_case.PIC != 0x83:
            axon_config.x2_base_addr = axon_case.x2_base_addr
        axon_config.bias_base_addr = axon_case.bias_base_addr
        axon_config.o_base_addr = axon_case.o_base_addr
        axon_config.reset_x1_addr = axon_case.reset_x1_addr
        axon_config.reset_o_addr = axon_case.reset_o_addr

    def convert_cmp(self, cmp, out_precision):
        cmp_uint = np.uint(cmp)
        cmp_bin = bin(cmp_uint)
        cmp_bin = cmp_bin[2:].zfill(32)
        cmp_bin = cmp_bin[-32:]
        cmp_list = list()
        if out_precision == 1 or out_precision == 2:
            for i in range(4):
                readable_cmp_bin = cmp_bin[i * 8: (i + 1) * 8]
                readable_cmp = int(readable_cmp_bin, 2)
                readable_cmp = np.int8(readable_cmp)
                cmp_list.append(readable_cmp)
            return cmp_list
        elif out_precision == 0:
            cmp = int(cmp_bin, 2)
            readable_cmp = np.int32(cmp)
            cmp_list.append(readable_cmp)
            return cmp_list
        elif out_precision == 3:
            for i in range(16):
                readable_cmp_bin = cmp_bin[i * 2: (i + 1) * 2]
                readable_cmp = int(readable_cmp_bin, 2)
                if readable_cmp == 3:
                    readable_cmp = -1
                if readable_cmp == 2:
                    readable_cmp = -2
                cmp_list.append(readable_cmp)
            return cmp_list
        else:
            raise ValueError('Unsupported output type')

    def check_data_block_for_axon(self, prim_case):
        for data_block_case in prim_case.memory_blocks:
            start_addr = data_block_case['start']
            same_addr = False  # 用于检查是都有重复起始地址的数据块
            if hasattr(prim_case, 'x1_base_addr') and \
                    start_addr == prim_case.x1_base_addr:
                same_addr = True
            if hasattr(prim_case, 'x2_base_addr') and \
                    start_addr == prim_case.x2_base_addr:
                assert same_addr == False
                same_addr = True
            if (prim_case.bias_type == 2 or prim_case.bias_type == 3) and \
                    start_addr == prim_case.bias_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'o_base_addr') and \
                    start_addr == prim_case.o_base_addr:
                assert same_addr == False
                same_addr = True
            assert same_addr, '每个任务块都必须有一个对应'

    def config_default_part_of_data(self, data_config, is_dynamic):
        data_config.chip_idx = self.pos.chip_x
        data_config.chip_idy = self.pos.chip_y
        data_config.core_idx = self.pos.core_x
        data_config.core_idy = self.pos.core_y
        data_config.net_id = 0

        if is_dynamic:
            data_config.step_group = self.pos.step_group
            data_config.phase_group = self.pos.phase_group
            data_config.phases.append(self.pos.phase)
            data_config.socket_id = 0

    def config_o_block(self, block_id, data_config, output_area, precison=tj_basic.Precision.INT_32):
        self.data_block_num += 1
        output_config = data_config.dynamic_blocks.add()
        output_config.io_type = tj_data.IOType.OUTPUT_DATA
        output_config.id = block_id

        output_config.start_addr = output_area[0]
        output_config.length = output_area[1]
        if precison is not None:
            output_config.precision = precison

        output_config.direction = tj_basic.Direction.SERDES_SOUTH

        self.config_default_part_of_data(output_config, True)

    def config_static_block(self, block_id, data_config, block_case, precision=tj_basic.Precision.INT_32):
        self.data_block_num += 1
        self.data_blocks[block_id] = block_case['data']
        static_data_config = data_config.static_blocks.add()
        static_data_config.id = block_id
        static_data_config.start_addr = block_case['start']
        static_data_config.precision = precision
        self.config_default_part_of_data(static_data_config, False)

    def config_p02(self, axon_config, p02_case):
        self.general_prim_config(axon_config.p02, p02_case)
        self.set_axon_addr(axon_config, p02_case)
        # 设置形状
        axon_config.shape.nf = p02_case.nif
        axon_config.shape.nix = p02_case.nix
        axon_config.shape.niy = p02_case.niy
        axon_config.shape.nkx = p02_case.nkx
        axon_config.shape.nky = p02_case.nky
        if p02_case.avg_pooling_en:
            axon_config.shape.nx = p02_case.Output_fm_Ox
            axon_config.shape.ny = p02_case.Output_fm_Oy
        else:
            axon_config.shape.nx = p02_case.nix
            axon_config.shape.ny = p02_case.niy          

        axon_config.x1_addr_length = p02_case.Read_X_length
        axon_config.o_addr_length = p02_case.Write_V_length
        if (p02_case.bias_type == 2 or p02_case.bias_type == 3):
            axon_config.bias_addr_length = p02_case.Read_Bias_length
        
        # 设置数据块
        self.check_data_block_for_axon(p02_case)
        data_config = self.asm_ir.data_config
        # P02可能包含数据块：x1, bias, o
        for block_case in p02_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p02_case.x1_base_addr:
                block_id = 'p02_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p02_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif (p02_case.bias_type == 2 or p02_case.bias_type == 3) and \
                    start_addr == p02_case.bias_base_addr:
                block_id = 'p02_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p02_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p02_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p03(self, axon_config, p03_case):
        self.general_prim_config(axon_config.p03, p03_case)
        self.set_axon_addr(axon_config, p03_case)
        # 设置形状
        axon_config.shape.nf = p03_case.nif
        if p03_case.tensor_en:
            axon_config.shape.nix = p03_case.nx
            axon_config.shape.niy = p03_case.ny
            axon_config.shape.nx = p03_case.Ox
            axon_config.shape.ny = p03_case.Oy
        axon_config.shape.n_branch = p03_case.n_branch

        axon_config.x1_addr_length = p03_case.Read_X1_length
        axon_config.x2_addr_length = p03_case.Read_X2_length
        axon_config.o_addr_length = p03_case.Write_V_length
        if (p03_case.bias_type == 2 or p03_case.bias_type == 3):
            axon_config.bias_addr_length = p03_case.Read_Bias_length

        self.check_data_block_for_axon(p03_case)
        data_config = self.asm_ir.data_config
        # P03可能包含数据块：x1, x2, bias, o
        for block_case in p03_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p03_case.x1_base_addr:
                block_id = 'p03_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p03_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p03_case.x2_base_addr:
                block_id = 'p03_x2_' + str(self.data_block_num)
                axon_config.x2_block = block_id
                precision = self.precision_map[p03_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif (p03_case.bias_type == 2 or p03_case.bias_type == 3) and \
                    start_addr == p03_case.bias_base_addr:
                block_id = 'p03_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p03_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p03_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p04(self, axon_config, p04_case):
        self.general_prim_config(axon_config.p04, p04_case)
        self.set_axon_addr(axon_config, p04_case)

        # 设置形状
        axon_config.shape.nf = p04_case.nof
        axon_config.shape.nr = p04_case.nif

        axon_config.x1_addr_length = p04_case.Read_X_length
        axon_config.x2_addr_length = p04_case.Read_weight_length
        axon_config.o_addr_length = p04_case.Write_V_length
        if (p04_case.bias_type == 2 or p04_case.bias_type == 3):
            axon_config.bias_addr_length = p04_case.Read_Bias_length

        self.check_data_block_for_axon(p04_case)
        data_config = self.asm_ir.data_config
        # P04可能包含数据块：x1, x2, bias, o
        for block_case in p04_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p04_case.x1_base_addr:
                block_id = 'p04_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p04_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            if start_addr == p04_case.x2_base_addr:
                block_id = 'p04_x2_' + str(self.data_block_num)
                axon_config.x2_block = block_id
                precision = self.precision_map[p04_case.x2_precision]
                self.config_static_block(
                    block_id, data_config, block_case, precision)
            elif (p04_case.bias_type == 2 or p04_case.bias_type == 3) and \
                    start_addr == p04_case.bias_base_addr:
                block_id = 'p04_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p04_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p04_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p41(self, axon_config, p41_case):
        self.general_prim_config(axon_config.p41, p41_case)
        self.set_axon_addr(axon_config, p41_case)

        # 设置形状
        axon_config.shape.nf = p41_case.nof
        axon_config.shape.nr = p41_case.nif
        axon_config.shape.nkx = p41_case.nkx
        axon_config.shape.nky = p41_case.nky
        axon_config.shape.nix = p41_case.nix
        axon_config.shape.niy = p41_case.niy
        axon_config.shape.nx = p41_case.Output_fm_Ox
        axon_config.shape.ny = p41_case.Output_fm_Oy

        axon_config.x1_addr_length = p41_case.Read_X_length
        axon_config.x2_addr_length = p41_case.Read_weight_length
        axon_config.o_addr_length = p41_case.Write_V_length
        if (p41_case.bias_type == 2 or p41_case.bias_type == 3):
            axon_config.bias_addr_length = p41_case.Read_Bias_length

        self.check_data_block_for_axon(p41_case)
        data_config = self.asm_ir.data_config
        # P41可能包含数据块：x1, x2, bias, o
        for block_case in p41_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p41_case.x1_base_addr:
                block_id = 'p41_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p41_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            if start_addr == p41_case.x2_base_addr:
                block_id = 'p41_x2_' + str(self.data_block_num)
                axon_config.x2_block = block_id
                precision = self.precision_map[p41_case.x2_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif (p41_case.bias_type == 2 or p41_case.bias_type == 3) and \
                    start_addr == p41_case.bias_base_addr:
                block_id = 'p41_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p41_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p41_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p43(self, axon_config, p43_case):
        self.general_prim_config(axon_config.p43, p43_case)
        self.set_axon_addr(axon_config, p43_case)

        # 设置形状
        axon_config.shape.nf = p43_case.nif
        axon_config.shape.n_branch = p43_case.n_branch
        if p43_case.tensor_en:
            axon_config.shape.nix = p43_case.nx
            axon_config.shape.niy = p43_case.ny
            axon_config.shape.nx = p43_case.Ox
            axon_config.shape.ny = p43_case.Oy

        axon_config.x1_addr_length = p43_case.Read_X_length
        axon_config.x2_addr_length = p43_case.Read_A_length
        axon_config.o_addr_length = p43_case.Write_V_length
        if (p43_case.bias_type == 2 or p43_case.bias_type == 3):
            axon_config.bias_addr_length = p43_case.Read_Bias_length

        self.check_data_block_for_axon(p43_case)
        data_config = self.asm_ir.data_config
        # P43可能包含数据块：x1, x2, bias, o
        for block_case in p43_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p43_case.x1_base_addr:
                block_id = 'p43_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p43_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p43_case.x2_base_addr:
                block_id = 'p43_x2_' + str(self.data_block_num)
                axon_config.x2_block = block_id
                precision = self.precision_map[p43_case.x2_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif (p43_case.bias_type == 2 or p43_case.bias_type == 3) and \
                    start_addr == p43_case.bias_base_addr:
                block_id = 'p43_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p43_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p43_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p81(self, axon_config, p81_case):
        self.general_prim_config(axon_config.p81, p81_case)
        self.set_axon_addr(axon_config, p81_case)

        # 设置形状
        axon_config.shape.nf = p81_case.nof
        axon_config.shape.nr = p81_case.nif
        axon_config.shape.nkx = p81_case.nkx
        axon_config.shape.nky = p81_case.nky
        axon_config.shape.nix = p81_case.nix
        axon_config.shape.niy = p81_case.niy
        axon_config.shape.nx = p81_case.Output_fm_Ox
        axon_config.shape.ny = p81_case.Output_fm_Oy

        axon_config.x1_addr_length = p81_case.Read_X_length
        axon_config.x2_addr_length = p81_case.Read_weight_length
        axon_config.o_addr_length = p81_case.Write_V_length
        if (p81_case.bias_type == 2 or p81_case.bias_type == 3):
            axon_config.bias_addr_length = p81_case.Read_Bias_length

        self.check_data_block_for_axon(p81_case)
        data_config = self.asm_ir.data_config
        # P81可能包含数据块：x1, x2, bias, o
        for block_case in p81_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p81_case.x1_base_addr:
                block_id = 'p81_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p81_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            if start_addr == p81_case.x2_base_addr:
                block_id = 'p81_x2_' + str(self.data_block_num)
                axon_config.x2_block = block_id
                precision = self.precision_map[p81_case.x2_precision]
                self.config_static_block(
                    block_id, data_config, block_case, precision)
            elif (p81_case.bias_type == 2 or p81_case.bias_type == 3) and \
                    start_addr == p81_case.bias_base_addr:
                block_id = 'p81_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p81_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p81_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def config_p83(self, axon_config, p83_case):
        self.general_prim_config(axon_config.p83, p83_case)
        self.set_axon_addr(axon_config, p83_case)

        # 设置形状
        axon_config.shape.nf = p83_case.nif
        axon_config.shape.n_branch = p83_case.n_branch

        if p83_case.tensor_en:
            axon_config.shape.nix = p83_case.nx
            axon_config.shape.niy = p83_case.ny
            axon_config.shape.nx = p83_case.Ox
            axon_config.shape.ny = p83_case.Oy

        axon_config.x1_addr_length = p83_case.Read_X_length
        axon_config.o_addr_length = p83_case.Write_V_length
        if (p83_case.bias_type == 2 or p83_case.bias_type == 3):
            axon_config.bias_addr_length = p83_case.Read_Bias_length

        self.check_data_block_for_axon(p83_case)
        data_config = self.asm_ir.data_config
        # P83可能包含数据块：x1, bias, o
        for block_case in p83_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p83_case.x1_base_addr:
                block_id = 'p83_x1_' + str(self.data_block_num)
                axon_config.x1_block = block_id
                precision = self.precision_map[p83_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif (p83_case.bias_type == 2 or p83_case.bias_type == 3) and \
                    start_addr == p83_case.bias_base_addr:
                block_id = 'p83_bias_' + str(self.data_block_num)
                axon_config.bias_block = block_id
                self.config_static_block(block_id, data_config, block_case)
            elif start_addr == p83_case.o_base_addr:
                assert False

        if not self.axon_output_area:
            return
        block_id = 'p83_o_' + str(self.data_block_num)
        axon_config.output_block.append(block_id)
        assert len(self.axon_output_area) == 1
        self.config_o_block(block_id, data_config, self.axon_output_area[0])

    def check_data_block_for_soma(self, prim_case):
        # 不检查P08
        for data_block_case in prim_case.memory_blocks:
            start_addr = data_block_case['start']
            same_addr = False  # 用于检查是都有重复起始地址的数据块
            if hasattr(prim_case, 'x1_base_addr') and \
                    start_addr == prim_case.x1_base_addr:
                same_addr = True
            if hasattr(prim_case, 'lut_base_addr') and \
                    start_addr == prim_case.lut_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'ciso_base_addr') and \
                    start_addr == prim_case.ciso_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'o_base_addr') and \
                    start_addr == prim_case.o_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'uin_base_addr') and \
                    start_addr == prim_case.uin_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'v_base_addr') and \
                    start_addr == prim_case.v_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'vm_base_addr') and \
                    start_addr == prim_case.vm_base_addr:
                assert same_addr == False
                same_addr = True
            if hasattr(prim_case, 'vtheta_base_addr') and \
                    start_addr == prim_case.vtheta_base_addr:
                assert same_addr == False
                same_addr = True
            assert same_addr, '每个任务块都必须有一个对应'

    def config_px5(self, soma_config, px5_case, soma_type):
        self.general_prim_config(soma_config.px5, px5_case)
        # 设置数据块
        self.check_data_block_for_soma(px5_case)
        data_config = self.asm_ir.data_config

        soma_config.px5.x1_addr_length = px5_case.Read_X_length
        soma_config.px5.o_addr_length = px5_case.Write_Y_length

        for block_case in px5_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == px5_case.x1_base_addr:
                block_id = 'px5_x1_' + str(self.data_block_num)
                soma_config.x1_block = block_id
                precision = self.precision_map[px5_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == px5_case.o_base_addr:
                assert False

        output_areas = self.soma1_output_area
        if soma_type == 'soma2':
            output_areas = self.soma2_output_area
        for output_area in output_areas:
            block_id = 'px5_o_' + str(self.data_block_num)
            soma_config.output_block.append(block_id)
            precision = self.precision_map[px5_case.out_precision]
            self.config_o_block(block_id, data_config, output_area, precision)

    def config_p07(self, soma_config, p07_case, soma_type):
        self.general_prim_config(soma_config.p07, p07_case)

        soma_config.p07.x1_addr_length = p07_case.Read_X_length
        soma_config.p07.lut_addr_length = p07_case.Read_LUT_length
        soma_config.p07.o_addr_length = p07_case.Write_Y_length

        # 设置数据块
        self.check_data_block_for_soma(p07_case)
        data_config = self.asm_ir.data_config
        # P02可能包含数据块：x1, bias, o
        for block_case in p07_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p07_case.x1_base_addr:
                block_id = 'p07_x1_' + str(self.data_block_num)
                soma_config.x1_block = block_id
                precision = self.precision_map[p07_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p07_case.lut_base_addr:
                block_id = 'p07_lut_' + str(self.data_block_num)
                soma_config.lut_block = block_id
                precision = self.precision_map[p07_case.x2_precision]
                self.config_static_block(
                    block_id, data_config, block_case, precision)
            elif start_addr == p07_case.o_base_addr:
                assert False

        output_areas = self.soma1_output_area
        if soma_type == 'soma2':
            output_areas = self.soma2_output_area
        for output_area in output_areas:
            block_id = 'p07_o_' + str(self.data_block_num)
            soma_config.output_block.append(block_id)
            precision = self.precision_map[p07_case.x2_precision]
            self.config_o_block(block_id, data_config, output_area, precision)

    def config_p08(self, soma_config, p08_case, soma_type):
        self.general_prim_config(soma_config.p08, p08_case)

        soma_config.p08.uin_addr_length = p08_case.Read_Uin_length
        soma_config.p08.v_addr_length = p08_case.Read_V_length
        soma_config.p08.vm_addr_length = p08_case.Read_VM_length
        soma_config.p08.vtheta_addr_length = p08_case.Read_Vtheta_length
        soma_config.p08.s_addr_length = p08_case.S1_out_length
        soma_config.p08.para_addr_length = p08_case.S1_para_length

        # 设置数据块
        self.check_data_block_for_soma(p08_case)
        data_config = self.asm_ir.data_config
        # P02可能包含数据块：x1, bias, o
        for block_case in p08_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p08_case.uin_base_addr:
                block_id = 'p08_uin_' + str(self.data_block_num)
                soma_config.uin_block = block_id
                precision = tj_basic.Precision.INT_32
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p08_case.v_base_addr:
                block_id = 'p08_v_' + str(self.data_block_num)
                soma_config.v_block = block_id
                precision = tj_basic.Precision.INT_32
                self.config_static_block(
                    block_id, data_config, block_case, precision)
            elif start_addr == p08_case.vm_base_addr:
                block_id = 'p08_vm_' + str(self.data_block_num)
                soma_config.vm_block = block_id
                precision = tj_basic.Precision.INT_32
                self.config_static_block(
                    block_id, data_config, block_case, precision)
            elif start_addr == p08_case.vtheta_base_addr:
                block_id = 'p08_vtheta_' + str(self.data_block_num)
                soma_config.vtheta_block = block_id
                precision = tj_basic.Precision.INT_32
                self.config_static_block(
                    block_id, data_config, block_case, precision)

        output_areas = self.soma1_output_area
        if soma_type == 'soma2':
            output_areas = self.soma2_output_area
        for output_area in output_areas:
            block_id = 'p08_s_' + str(self.data_block_num)
            soma_config.output_block.append(block_id)
            self.fire_map = {0: tj_basic.Precision.INT_32,
                             1: tj_basic.Precision.INT_32,
                             2: tj_basic.Precision.INT_8,
                             3: tj_basic.Precision.INT_8,
                             4: tj_basic.Precision.INT_8,
                             5: tj_basic.Precision.TERNARY,
                             6: tj_basic.Precision.INT_32,
                             7: tj_basic.Precision.INT_8}
            precision = self.fire_map[p08_case.fire_type]
            self.config_o_block(block_id, data_config, output_area, precision)

    def config_p06(self, soma_config, p06_case, soma_type):
        self.general_prim_config(soma_config.p06, p06_case)

        if (hasattr(p06_case, 'Read_in_length')):
            soma_config.p06.x1_addr_length = p06_case.Read_in_length
        if (hasattr(p06_case, 'Read_ciso_length')):
            soma_config.p06.ciso_addr_length = p06_case.Read_ciso_length
        soma_config.p06.o_addr_length = p06_case.Write_Y_length

        # 设置数据块
        # self.check_data_block_for_soma(p06_case)
        data_config = self.asm_ir.data_config
        # P02可能包含数据块：x1, bias, o
        has_x1 = False
        for block_case in p06_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p06_case.x1_base_addr and not has_x1:
                block_id = 'p06_x1_' + str(self.data_block_num)
                soma_config.x1_block = block_id
                precision = self.precision_map[p06_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
                has_x1 = True
            elif start_addr == p06_case.ciso_base_addr:
                block_id = 'p06_ciso_' + str(self.data_block_num)
                soma_config.ciso_block = block_id
                precision = self.precision_map[p06_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p06_case.o_base_addr:
                assert False

        output_areas = self.soma1_output_area
        if soma_type == 'soma2':
            output_areas = self.soma2_output_area
        for output_area in output_areas:
            block_id = 'p06_o_' + str(self.data_block_num)
            soma_config.output_block.append(block_id)
            precision = self.precision_map[p06_case.out_precision]
            self.config_o_block(block_id, data_config, output_area, precision)

    def config_p26(self, soma_config, p26_case, soma_type):
        self.general_prim_config(soma_config.p26, p26_case)

        soma_config.p26.x1_addr_length = p26_case.Read_in_length
        soma_config.p26.ciso_addr_length = p26_case.Write_ciso_length
        soma_config.p26.o_addr_length = p26_case.Write_Y_length

        # 设置数据块
        self.check_data_block_for_soma(p26_case)
        data_config = self.asm_ir.data_config
        # P02可能包含数据块：x1, bias, o
        for block_case in p26_case.memory_blocks:
            start_addr = block_case['start']
            if start_addr == p26_case.x1_base_addr:
                block_id = 'p26_x1_' + str(self.data_block_num)
                soma_config.x1_block = block_id
                precision = self.precision_map[p26_case.x1_precision]
                self.config_static_block(block_id, data_config, block_case, precision)
            elif start_addr == p26_case.o_base_addr:
                assert False

        output_areas = self.soma1_output_area
        if soma_type == 'soma2':
            output_areas = self.soma2_output_area
        for output_area in output_areas:
            block_id = 'p26_o_' + str(self.data_block_num)
            soma_config.output_block.append(block_id)
            precision = self.precision_map[p26_case.out_precision]
            self.config_o_block(block_id, data_config, output_area, precision)

    def add_phase_output_print(self, axon, soma1, router, soma2):
        self.axon_output_area = []
        self.soma1_output_area = []
        self.router_output_area = []
        self.soma2_output_area = []

        line_buffer_ratio = 1
        if soma1 is not None and soma1.row_pipeline_en == 1:
            if hasattr(soma1, 'niy'):
                line_buffer_ratio = soma1.niy / \
                    soma1.row_pipeline_num
            elif hasattr(soma1, 'num_in'):
                if isinstance(soma1, Prim_06_move_merge):
                    if soma1.in_ciso_pipe_sel:
                        line_buffer_ratio = soma1.num_ciso / soma1.row_pipeline_num
                    else:
                        line_buffer_ratio = soma1.num_in / soma1.row_pipeline_num
                else:
                    line_buffer_ratio = soma1.num_in / soma1.row_pipeline_num
            else:  # hasattr(soma1, 'group_num'):
                line_buffer_ratio = soma1.group_num / soma1.row_pipeline_num

        line_buffer_ratio = max(1, line_buffer_ratio)
        if axon is not None:
            if not hasattr(axon, "axon_delay") or not axon.axon_delay:
                self.axon_output_area.append((
                    axon.o_base_addr, int(axon.Write_V_length / line_buffer_ratio)))
        if soma1 is not None:
            if soma1.memory_select == 0:
                length = 0
                if hasattr(soma1, 'S1_out_length'):
                    length = soma1.S1_out_length
                else:
                    if isinstance(soma1, Prim_26_move_split) and soma1.out_ciso_sel == 1:
                        length = soma1.Write_ciso_length
                    else:
                        length = soma1.Write_Y_length

                if router is not None and router.Soma_in_en == 1:
                    length = (router.Addr_Dout_length + 1) * 4
                if soma1.PIC == 0x08:
                    self.soma1_output_area.append((soma1.s_base_addr, length))
                    self.soma1_output_area.append(
                        (soma1.v_base_addr, soma1.Read_V_length))
                    self.soma1_output_area.append(
                        (soma1.vtheta_base_addr, soma1.Read_Vtheta_length))
                    self.soma1_output_area.append(
                        (soma1.para_base_addr, soma1.S1_para_length))
                elif isinstance(soma1, Prim_26_move_split):
                    if soma1.out_ciso_sel == 0:
                        self.soma1_output_area.append(
                            (soma1.ciso_base_addr, soma1.Write_ciso_length))
                        self.soma1_output_area.append(
                            (soma1.o_base_addr, length))
                    else:
                        self.soma1_output_area.append(
                            (soma1.ciso_base_addr, length))
                        self.soma1_output_area.append(
                            (soma1.o_base_addr, soma1.Write_Y_length))
                else:
                    self.soma1_output_area.append((soma1.o_base_addr, length))
            else:
                if isinstance(soma1, Prim_26_move_split):
                    if soma1.out_ciso_sel == 0:
                        self.soma1_output_area.append(
                            (soma1.ciso_base_addr, soma1.Write_ciso_length))
                    else:
                        self.soma1_output_area.append(
                            (soma1.o_base_addr, soma1.Write_Y_length))

                elif soma1.PIC == 0x08:
                    self.soma1_output_area.append(
                        (soma1.v_base_addr, soma1.Read_V_length))
                    self.soma1_output_area.append(
                        (soma1.vtheta_base_addr, soma1.Read_Vtheta_length))
                    self.soma1_output_area.append(
                        (soma1.para_base_addr, soma1.S1_para_length))
        if router is not None:
            if router.Receive_en:
                self.router_output_area.append((
                    0x8000 + router.Addr_Din_base, (router.Addr_Din_length + 1) * 2))
        if soma2 is not None:
            if soma2.PIC == 0x08:
                self.soma2_output_area.append(
                    (soma2.s_base_addr, soma2.S1_out_length))
                self.soma2_output_area.append(
                    (soma2.v_base_addr, soma2.Read_V_length))
                self.soma2_output_area.append(
                    (soma2.vtheta_base_addr, soma2.Read_Vtheta_length))
                self.soma2_output_area.append(
                    (soma2.para_base_addr, soma2.S1_para_length))
            elif isinstance(soma2, Prim_26_move_split):
                self.soma2_output_area.append(
                    (soma2.ciso_base_addr, soma2.Write_ciso_length))
                self.soma2_output_area.append(
                    (soma2.o_base_addr, soma2.Write_Y_length))
            else:
                self.soma2_output_area.append(
                    (soma2.o_base_addr, soma2.Write_Y_length))
