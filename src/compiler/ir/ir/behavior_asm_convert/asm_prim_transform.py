# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import numpy as np
from typing import List
import os
import math
import src.compiler.ir.asm_pb2 as tj_asm
import src.compiler.ir.basic_pb2 as tj_basic
from old.primitive import *


class AsmPrimTransform:
    def __init__(self, case_name, data_config, save_para=False):
        self.premap = {
            tj_basic.Precision.INT_32: 0,
            tj_basic.Precision.INT_8: 1,
            tj_basic.Precision.UINT_8: 2,
            tj_basic.Precision.TERNARY: 3
        }
        self.biasmap = {
            tj_basic.BiasType.VECTOR: 2,
            tj_basic.BiasType.CONSTANT: 0
        }

        self.case_name = case_name
        self.save_para = save_para
        self.data_config_map = {}
        self.record_data_configs(data_config)

    def record_data_configs(self, data_config):
        static_datas = data_config.static_blocks
        dynamic_datas = data_config.dynamic_blocks

        for static_data in static_datas:
            self.data_config_map[static_data.id] = static_data

        for dynamic_data in dynamic_datas:
            self.data_config_map[dynamic_data.id] = dynamic_data

    def transform(self, prim_group_asm):
        prim_group_config = {'axon': None,
                             'soma1': None, 'router': None, 'soma2': None}

        data_blocks = {}
        if prim_group_asm.HasField('axon'):
            axon_asm = prim_group_asm.axon
            prim_group_config['axon'] = self.axon_transform(
                axon_asm, data_blocks)

        if prim_group_asm.HasField('soma1'):
            soma1_asm = prim_group_asm.soma1
            prim_group_config['soma1'] = self.soma_transform(
                soma1_asm, data_blocks)

        if prim_group_asm.HasField('router'):
            rotuer_asm = prim_group_asm.router
            prim_group_config['router'] = self.router_transform(
                rotuer_asm, data_blocks)

        if prim_group_asm.HasField('soma2'):
            soma2_asm = prim_group_asm.soma2
            prim_group_config['soma2'] = self.soma_transform(
                soma2_asm, data_blocks)

        if self.save_para:
            self.save_prim_data(data_blocks)
        return prim_group_config

    def axon_transform(self, axon_asm, data_blocks):
        p = None
        x2_data = None
        bias_data = None
        shape = axon_asm.shape
        if axon_asm.pic == 0x02:
            para = axon_asm.p02
            p = Prim_02_Axon()
            for k, v in para.ListFields():
                if "precision" in k.name:
                    setattr(p, k.name, self.premap[v])
                elif "bias_type" in k.name:
                    setattr(p, k.name, self.biasmap[v])
                else:
                    if isinstance(v, bool):
                        v = int(v)
                    setattr(p, k.name, v)
            p.nif = shape.nf
            p.nkx = shape.nkx
            p.nky = shape.nky
            p.nix = shape.nix
            p.niy = shape.niy

            prim1_in = p.init_data()
            if p.bias_type == 2 or p.bias_type == 3:
                bias_data = prim1_in[1]
        elif axon_asm.pic == 0x03:
            para = axon_asm.p03
            p = Prim_03_Axon()
            for k, v in para.ListFields():
                if "precision" in k.name:
                    setattr(p, k.name, self.premap[v])
                elif "bias_type" in k.name:
                    setattr(p, k.name, self.biasmap[v])
                else:
                    if isinstance(v, bool):
                        v = int(v)
                    setattr(p, k.name, v)
            p.nif = shape.nf
            p.nx = shape.nix
            p.ny = shape.niy
            p.n_branch = shape.n_branch

            prim1_in = p.init_data()
            x2_data = prim1_in[1]
            if p.bias_type == 2 or p.bias_type == 3:
                bias_data = prim1_in[2]
        elif axon_asm.pic == 0x04:
            para = axon_asm.p04
            p = Prim_04_Axon()
            for k, v in para.ListFields():
                if "precision" in k.name:
                    setattr(p, k.name, self.premap[v])
                elif "bias_type" in k.name:
                    setattr(p, k.name, self.biasmap[v])
                else:
                    if isinstance(v, bool):
                        v = int(v)
                    setattr(p, k.name, v)
            p.nof = shape.nf
            p.nif = shape.nr

            prim1_in = p.init_data()
            x2_data = prim1_in[1]
            if p.bias_type == 2 or p.bias_type == 3:
                bias_data = prim1_in[2]
        elif axon_asm.pic in [0x41, 0x81]:
            if axon_asm.pic == 0x41:
                para = axon_asm.p41
                p = Prim_41_Axon()
            else:
                para = axon_asm.p81
                p = Prim_81_Axon()
            for k, v in para.ListFields():
                if "precision" in k.name:
                    setattr(p, k.name, self.premap[v])
                elif "bias_type" in k.name:
                    setattr(p, k.name, self.biasmap[v])
                else:
                    if isinstance(v, bool):
                        v = int(v)
                    setattr(p, k.name, v)
            if axon_asm.pic == 0x41 and p.axon_delay:
                p.loop4_extent = math.sqrt((p.delay_clock - 6)/37) - 1
                p.loop5_extent = math.sqrt((p.delay_clock - 6)/37) - 1
                p.a2s2_mode = True
            p.nof = shape.nf
            p.nif = shape.nr
            p.nkx = shape.nkx
            p.nky = shape.nky
            p.nix = shape.nix
            p.niy = shape.niy

            prim1_in = p.init_data()
            x2_data = prim1_in[1]
            if p.bias_type == 2 or p.bias_type == 3:
                bias_data = prim1_in[2]
        elif axon_asm.pic in [0x43, 0x83]:
            if axon_asm.pic == 0x43:
                para = axon_asm.p43
                p = Prim_43_Axon()
            else:
                para = axon_asm.p83
                p = Prim_83_Axon()

            for k, v in para.ListFields():
                if "precision" in k.name:
                    setattr(p, k.name, self.premap[v])
                elif "bias_type" in k.name:
                    setattr(p, k.name, self.biasmap[v])
                else:
                    if isinstance(v, bool):
                        v = int(v)
                    setattr(p, k.name, v)
            p.nif = shape.nf
            p.nx = shape.nix
            p.ny = shape.niy
            p.n_branch = shape.n_branch

            prim1_in = p.init_data()
            if axon_asm.pic == 0x43:
                x2_data = prim1_in[1]
                if p.bias_type == 2 or p.bias_type == 3:
                    bias_data = prim1_in[2]
            else:
                if p.bias_type == 2 or p.bias_type == 3:
                    bias_data = prim1_in[1]

        p.x1_base_addr = axon_asm.x1_base_addr
        if axon_asm.pic != 0x02 and axon_asm.pic != 0x83:
            p.x2_base_addr = axon_asm.x2_base_addr
        p.bias_base_addr = axon_asm.bias_base_addr
        p.o_base_addr = axon_asm.o_base_addr
        p.reset_x1_addr = axon_asm.reset_x1_addr
        p.reset_o_addr = axon_asm.reset_o_addr

        blocks = []
        if axon_asm.HasField('x1_block'):
            x1_data_config = self.data_config_map[axon_asm.x1_block]
            x1_data = prim1_in[0]
            blocks.append({'name': 'AllInputX',
                           'start': x1_data_config.start_addr,
                           'length': len(x1_data),
                           'data': x1_data,
                           'mode': 0})
            data_blocks[axon_asm.x1_block] = x1_data
        if x2_data is not None and axon_asm.HasField('x2_block'):
            x2_data_config = self.data_config_map[axon_asm.x2_block]
            blocks.append({'name': 'AllWeight',
                           'start': x2_data_config.start_addr,
                           'length': len(x2_data),
                           'data': x2_data,
                           'mode': 0})
            data_blocks[axon_asm.x2_block] = x2_data
        if bias_data is not None and axon_asm.HasField('bias_block'):
            bias_data_config = self.data_config_map[axon_asm.bias_block]
            blocks.append({'name': 'AllBias',
                           'start': bias_data_config.start_addr,
                           'length': len(bias_data),
                           'data': bias_data,
                           'mode': 0})
            data_blocks[axon_asm.bias_block] = bias_data
        p.memory_blocks = blocks
        return p

    def soma_transform(self, soma_asm, data_blocks):
        prim2_in = None
        blocks = []
        if soma_asm.pic == 0x05:
            para = soma_asm.px5
            p = Prim_X5_Soma()
            self.set_soma_para(p, para)
            prim2_in = p.init_data()
        elif soma_asm.pic == 0x07:
            para = soma_asm.p07
            p = Prim_07_LUT()
            self.set_soma_para(p, para)
            prim2_in = p.init_data()
            if soma_asm.HasField('lut_block'):
                lut_data_config = self.data_config_map[soma_asm.lut_block]
                blocks.append({'name': 'lut',
                               'start': lut_data_config.start_addr,
                               'length': len(prim2_in[1]),
                               'data': prim2_in[1],
                               'mode': 0})
                data_blocks[soma_asm.lut_block] = prim2_in[1]
        elif soma_asm.pic == 0x06:
            para = soma_asm.p06
            p = Prim_06_move_merge()
            self.set_soma_para(p, para)
            prim2_in = p.init_data()
            if soma_asm.HasField('ciso_block'):
                ciso_data_config = self.data_config_map[soma_asm.ciso_block]
                blocks.append({'name': 'ciso',
                               'start': ciso_data_config.start_addr,
                               'length': len(prim2_in[1]),
                               'data': prim2_in[1],
                               'mode': 0})
                data_blocks[soma_asm.ciso_block] = prim2_in[1]
        elif soma_asm.pic == 0x26:
            para = soma_asm.p26
            p = Prim_26_move_split()
            self.set_soma_para(p, para)
            prim2_in = p.init_data()
        elif soma_asm.pic == 0x08:
            para = soma_asm.p08
            p = Prim_08_lif()
            self.set_soma_para(p, para)
            prim2_in = p.init_data()

            if soma_asm.HasField('uin_block'):
                uin_data_config = self.data_config_map[soma_asm.uin_block]
                blocks.append({'name': 'uin',
                               'start': uin_data_config.start_addr,
                               'length': len(prim2_in[0]),
                               'data': prim2_in[0],
                               'mode': 0})
                data_blocks[soma_asm.uin_block] = prim2_in[0]

            if soma_asm.HasField('vm_block'):
                vm_data_config = self.data_config_map[soma_asm.vm_block]
                blocks.append({'name': 'vm',
                               'start': vm_data_config.start_addr,
                               'length': len(prim2_in[1]),
                               'data': prim2_in[1],
                               'mode': 0})
                data_blocks[soma_asm.vm_block] = prim2_in[1]

            if soma_asm.HasField('vtheta_block'):
                vtheta_data_config = self.data_config_map[soma_asm.vtheta_block]
                blocks.append({'name': 'vtheta',
                               'start': vtheta_data_config.start_addr,
                               'length': len(prim2_in[2]),
                               'data': prim2_in[2],
                               'mode': 0})
                data_blocks[soma_asm.vtheta_block] = prim2_in[2]

            if soma_asm.HasField('v_block'):
                v_data_config = self.data_config_map[soma_asm.v_block]
                blocks.append({'name': 'v',
                               'start': v_data_config.start_addr,
                               'length': len(prim2_in[3]),
                               'data': prim2_in[3],
                               'mode': 0})
                data_blocks[soma_asm.v_block] = prim2_in[3]

        if soma_asm.pic != 0x08 and soma_asm.HasField('x1_block'):
            x1_data_config = self.data_config_map[soma_asm.x1_block]
            blocks.insert(0, {'name': 'input_x1',
                              'start': x1_data_config.start_addr,
                              'length': len(prim2_in[0]),
                              'data': prim2_in[0],
                              'mode': 0})
            data_blocks[soma_asm.x1_block] = prim2_in[0]

        # if x2_data is not None:
        #     blocks.append({'name': 'input_x2',
        #             'start': p.x2_base_addr,
        #             'length': len(x2_data),
        #             'data': x2_data,
        #             'mode': 0})
        p.memory_blocks = blocks

        return p

    def set_soma_para(self, p, para):
        for k, v in para.ListFields():
            if "precision" in k.name:
                setattr(p, k.name, self.premap[v])
            elif 'compare_init' in k.name:
                p.compare_init = self.convert_to_cmp(
                    list(para.compare_init), para.out_precision)
            else:
                if isinstance(v, bool):
                    v = int(v)
                setattr(p, k.name, v)

    def convert_to_cmp(self, param, out_type=8):
        cmp_bin = '0b'
        if isinstance(param, List):
            if len(param) == 4:
                for p in param:
                    assert (p >= -128 and p <=
                            127), 'Illegal value of parameter CMP'
                    p_uint = np.uint8(p)
                    p_bin = bin(p_uint)
                    p_bin = p_bin[2:].zfill(8)
                    cmp_bin += p_bin
                return int(cmp_bin, 2)  # uint32
            if len(param) == 1:
                assert (param[0] >= -2147483648 and param[0] <=
                        2147483647), 'Illegal value of parameter CMP'
                param_uint = np.uint32(param[0])
                return param_uint
            if len(param) == 16:
                param_dict = {-2: '10', -1: '11', 0: '00', 1: '01'}
                for p in param:
                    assert p in [-2, -1, 0,
                                 1], 'Illegal value of parameter CMP'
                    p_bin = param_dict[p]
                    cmp_bin += p_bin
                return int(cmp_bin, 2)
        else:
            assert isinstance(param, int)
            if out_type == tj_basic.Precision.INT_8 or out_type == tj_basic.Precision.UINT_8:
                assert (param >= -128 and param <=
                        127), 'Illegal value of parameter CMP'
                param_uint = np.uint8(param)
                param_bin = bin(param_uint)
                param_bin = param_bin[2:].zfill(8)
                for i in range(4):
                    cmp_bin += param_bin
                return int(cmp_bin, 2)  # uint32
            if out_type == tj_basic.Precision.INT_32:
                assert (param >= -2147483648 and param <=
                        2147483647), 'Illegal value of parameter CMP'
                param_uint = np.uint32(param)
                return param_uint
            if out_type == tj_basic.Precision.TERNARY:
                assert param in [-1, 0, 1], 'Illegal value of parameter CMP'
                param_dict = {-1: '11', 0: '00', 1: '01'}
                param_bin = param_dict[param]
                for i in range(16):
                    cmp_bin += param_bin
                return int(cmp_bin, 2)

    def router_transform(self, router_asm, data_blocks):
        para = router_asm.p09
        p = Prim_09_Router()
        for k, v in para.ListFields():
            if 'router_head_mode' == k.name:
                if v == tj_asm.RouterHeadMode.SINGLE_USE:
                    p.Rhead_mode = 0
                elif v == tj_asm.RouterHeadMode.MULTI_USE:
                    p.Rhead_mode = 1
            elif 'packet_size_mode' == k.name:
                if v == tj_asm.PacketSizeMode.SINGLE_NEURON:
                    p.T_mode = 0
                if v == tj_asm.PacketSizeMode.MULTIPLE_NEURON:
                    p.T_mode = 1
            elif 'out_memory' == k.name:
                if v == tj_asm.MemoryArea.MEM2:
                    p.Dout_memory_select = 0
                elif v == tj_asm.MemoryArea.MEM3:
                    p.Dout_memory_select = 1
            elif 'router_heads' == k.name:
                for one_head in v:
                    S = int(one_head.is_instant_request)
                    if one_head.packet_size_mode == tj_asm.PacketSizeMode.SINGLE_NEURON:
                        T = 0
                    elif one_head.packet_size_mode == tj_asm.PacketSizeMode.MULTIPLE_NEURON:
                        T = 1
                    PFINISH = int(one_head.is_packet_finish)
                    if one_head.relay_type == tj_asm.RelayType.UNI_CAST:
                        Q = 0
                    elif one_head.relay_type == tj_asm.RelayType.MULTI_RELAY_CAST:
                        Q = 1
                    x = one_head.dx
                    y = one_head.dy
                    A = one_head.destination_addr

                    pack_per_rhead = None
                    if one_head.HasField('pack_per_router_head'):
                        pack_per_rhead = one_head.pack_per_router_head

                    offset = None
                    if one_head.HasField('destination_offset'):
                        offset = one_head.destination_offset

                    const = None
                    if one_head.HasField('destination_const'):
                        const = one_head.destination_const

                    en = None
                    if one_head.HasField('enable'):
                        en = int(one_head.enable)

                    p.addRHead(S, T, PFINISH, Q, x, y, A,
                               pack_per_rhead, offset, const, en)
            else:
                if isinstance(v, bool):
                    v = int(v)
                setattr(p, k.name, v)

        # 计算Addr_Rhead_length
        router_head_num = len(p.RHeadList)
        if para.router_head_mode == tj_asm.RouterHeadMode.SINGLE_USE:
            # 一个表头32 bit
            p.Addr_Rhead_length = max(math.ceil(router_head_num / 4) - 1, 0)
        else:
            # 一个表头64 bit
            p.Addr_Rhead_length = max(math.ceil(router_head_num / 2) - 1, 0)

        if router_asm.HasField('send_block'):
            data_config = self.data_config_map[router_asm.send_block]
            memInit = []
            for i in range((p.Addr_Dout_length + 1) * 4):
                tmp = []
                for j in range(4):
                    tmp.append(np.random.randint(200))
                memInit.append(tmp)
            p.memory_blocks = [
                {'name': 'memInit',
                 'start': data_config.start_addr,
                 'length': len(memInit),
                 'data': memInit,
                 'mode': 0},
            ]
            data_blocks[router_asm.send_block] = memInit

        return p

    def save_prim_data(self, data_blocks):
        out_dir = '/data/' + self.case_name + '/'
        os.makedirs(out_dir, exist_ok=True)

        for block_id, block in data_blocks.items():
            np_block = np.array(block, dtype=np.int32)
            out_path = out_dir + block_id + '.dat'
            np_block.tofile(out_path)
