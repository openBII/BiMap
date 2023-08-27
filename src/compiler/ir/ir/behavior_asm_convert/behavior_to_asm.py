# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import os
import numpy as np
import src.compiler.ir.asm_pb2 as tj_asm
import src.compiler.ir.basic_pb2 as tj_basic
from behavior_asm_convert import behavior_prim_transform as cp
from old import primitive


class CurrentPosition:
    def __init__(self):
        self.chip_x = 0
        self.chip_y = 0
        self.core_x = 0
        self.core_y = 0
        self.step_group = 0
        self.phase_group = 0
        self.phase = 0


class BehaviorConvertor:
    def __init__(self, asm_ir, map_config_case, test_config_case, case_dir):
        self.case_name = test_config_case['tb_name']
        self.case_dir = case_dir
        self.map_config_case = map_config_case
        self.test_config_case = test_config_case
        self.asm_ir = asm_ir

        self.step_to_chip = {}  # {step_id, chip_id}

        self.pos = CurrentPosition()
        self.prim_configer = cp.BehaviorPrimTransform(asm_ir, self.pos)

    def convert(self):
        self.collect_infos()
        self.config_behavior_config()
        self.config_test()
        self.output_file()

        self.output_data_block()

    def collect_infos(self):
        for step_id, step in self.map_config_case.items():
            if isinstance(step_id, int):
                # 这里的group指phase group
                for group_id, group in step.items():
                    if not isinstance(group_id, int):
                        continue

                    for key, _ in group.items():
                        if isinstance(key, tuple):
                            chip_id = key[0]
                            if step_id not in self.step_to_chip:
                                self.step_to_chip[step_id] = chip_id
                            else:
                                # 同一个Step Group中不能出现不同的chip
                                assert chip_id == self.step_to_chip[step_id]

    def config_behavior_config(self):
        behavior_config = self.asm_ir.behavior_config
        behavior_config.sim_clock = self.map_config_case.get(
            'sim_clock', -1)

        for step_id, step_case in self.map_config_case.items():
            if isinstance(step_id, int):
                self.config_step_group(behavior_config, step_id, step_case)
            else:
                assert step_id == 'sim_clock'

    def config_step_group(self, behavior_config, step_id, step_case):
        step_config = behavior_config.step_config.add()
        chip_id = self.step_to_chip[step_id]
        step_config.chip_x = chip_id[0]
        step_config.chip_y = chip_id[1]
        step_config.step_group_id = step_id

        self.pos.chip_x = chip_id[0]
        self.pos.chip_y = chip_id[1]
        self.pos.step_group = step_id
        
        for phase_group_id, phase_group_case in step_case.items():
            if phase_group_id == 'clock':
                if 'sim_clock' not in self.map_config_case and phase_group_case is not None:
                    behavior_config.sim_clock = phase_group_case
                continue
            if 'clock0_in_step' == phase_group_id:
                step_config.clock0_in_step = phase_group_case
                continue
            if 'clock1_in_step' == phase_group_id:
                step_config.clock1_in_step = phase_group_case
                continue
            if 'step_exe_number' == phase_group_id:
                step_config.step_exe_number = phase_group_case
                continue

            # 不知道Step设置里还有没有其他东西
            assert isinstance(phase_group_id, int)

            self.pos.phase_group = phase_group_id
            self.config_phase_group(
                step_config, phase_group_id, phase_group_case)

    def config_phase_group(self, step_config, phase_group_id, phase_group_case):
        phase_group_config = step_config.phase_group_config.add()
        phase_group_config.phase_group_id = phase_group_id

        if 'mode' in phase_group_case:
            if phase_group_case['mode'] == 1:
                phase_group_config.phase_mode = tj_asm.PhaseMode.ADAPTED_CLOCK
            else:
                phase_group_config.phase_mode = tj_asm.PhaseMode.FIXED_CLOCK
                phase_group_config.phase_clock = phase_group_case['clock']
        elif 'clock' in phase_group_case:
            phase_group_config.phase_mode = tj_asm.PhaseMode.FIXED_CLOCK
            phase_group_config.phase_clock = phase_group_case['clock']
        else:
            phase_group_config.phase_mode = tj_asm.PhaseMode.ADAPTED_CLOCK

        for space_id, core_case in phase_group_case.items():
            if isinstance(space_id, tuple):
                core_id = space_id[1]
                assert isinstance(core_id, tuple)
                self.config_core(phase_group_config, core_id, core_case)
            else:
                assert space_id == 'mode' or space_id == 'clock'

    def config_core(self, phase_group_config, core_id, core_case):
        core_config = phase_group_config.core_config.add()
        core_config.core_x = core_id[0]
        core_config.core_y = core_id[1]

        self.pos.core_x = core_id[0]
        self.pos.core_y = core_id[1]

        static_prims = []
        instant_prims = []
        registers = {}

        if 'prims' not in core_case:
            # 处理某些配置格式不一致的情况
            value_list = []
            for a, s1, r, s2 in zip(core_case['axon'], core_case['soma1'],
                                    core_case['router'], core_case['soma2']):
                value_list.append({
                    'axon': a,
                    'soma1': s1,
                    'router': r,
                    'soma2': s2
                })
            static_prims = value_list
        else:
            static_prims = core_case['prims']

        if 'instant_prims' in core_case:
            instant_prims = core_case['instant_prims']

        if 'registers' in core_case:
            registers = core_case['registers']

        self.config_static_prims(core_config, static_prims)
        self.config_instant_prims(core_config, instant_prims)
        self.config_registers(core_config, registers)

    def config_static_prims(self, core_config, static_prims):
        self.pos.phase = 0
        for primitive_group in static_prims:
            self.pos.phase += 1
            prim_config = core_config.static_prim_list.add()
            self.prim_configer.add_phase_output_print(primitive_group['axon'], primitive_group['soma1'],
                                        primitive_group['router'], primitive_group['soma2'])
            if primitive_group['axon'] is not None:
                self.prim_configer.config_axon_prim(
                    prim_config.axon, primitive_group['axon'])
            if primitive_group['soma1'] is not None:
                self.prim_configer.config_soma_prim(
                    prim_config.soma1, primitive_group['soma1'], 'soma1')
            if primitive_group['router'] is not None:
                self.prim_configer.config_router_prim(
                    prim_config.router, primitive_group['router'])
            if primitive_group['soma2'] is not None:
                self.prim_configer.config_soma_prim(
                    prim_config.soma2, primitive_group['soma2'], 'soma2')

    def config_instant_prims(self, core_config, instant_prims):
        for primitive_group in instant_prims:
            prim_config = core_config.instant_prim_list.add()
            self.prim_configer.add_phase_output_print(primitive_group['axon'], primitive_group['soma1'],
                                        primitive_group['router'], primitive_group['soma2'])
            if primitive_group['axon'] is not None:
                self.prim_configer.config_axon_prim(
                    prim_config.axon, primitive_group['axon'])
            if primitive_group['soma1'] is not None:
                self.prim_configer.config_soma_prim(
                    prim_config.soma1, primitive_group['soma1'], 'soma1')
            if primitive_group['router'] is not None:
                self.prim_configer.config_router_prim(
                    prim_config.router, primitive_group['router'])
            if primitive_group['soma2'] is not None:
                self.prim_configer.config_soma_prim(
                    prim_config.soma2, primitive_group['soma2'], 'soma2')

    def config_registers(self, core_config, register_case):
        register_config = core_config.registers
        for key, value in register_case.items():
            if hasattr(register_config, key):
                setattr(register_config, key, value)

    def config_test(self):
        test_config = self.asm_ir.test_config
        test_config.test_case_name = self.case_name
        test_config.random_seed = self.test_config_case['test_seed']
        test_mode = self.test_config_case.get('test_mode', None)
        if test_mode is None or test_mode.value == 0:
            test_config.test_mode = tj_basic.CaseMode.PRIM_OUTPUT
        elif test_mode.value == 1:
            test_config.test_mode = tj_basic.CaseMode.MEMORY_STATE

    def output_file(self):
        if not self.case_dir.endswith('/'):
            self.case_dir += '/'
        os.makedirs(self.case_dir, exist_ok=True)
        out_path = self.case_dir + self.case_name + '.asm.txt'
        with open(out_path, 'w') as out_file:
            out_file.write(str(self.asm_ir))

    def output_data_block(self):
        out_dir = '/data/' + self.case_name + '/'
        os.makedirs(out_dir, exist_ok=True)
    
        for block_id, block in self.prim_configer.data_blocks.items():
            np_block = np.array(block, dtype=np.int32)
            out_path = out_dir + block_id + '.dat'
            np_block.tofile(out_path)


def convert_behavior_to_assembly(map_config, test_config_case, case_dir):
    asm_ir = tj_asm.Config()
    converter = BehaviorConvertor(asm_ir, map_config, test_config_case, case_dir)
    converter.convert()


def smoke_test():
    prim1 = primitive.Prim_04_Axon()
    prim1.x1_precision = 1
    prim1.x2_precision = 1
    prim1.bias_type = 2
    prim1.bias_length = 64
    prim1.nif = 48
    prim1.nof = 64
    # Axon_prim.constant_b = 0

    prim1.reset_x1_addr = 1
    prim1.reset_o_addr = 1
    prim1.x1_base_addr = 0x0000
    prim1.x2_base_addr = 0x4000
    prim1.bias_base_addr = 0x0c00
    prim1.o_base_addr = 0x2000

    a = prim1.init_data()
    blocks = [{
        'name': "input_X",
        'start': prim1.x1_base_addr,
        'data': a[0],
        'mode': 0
    }, {
        'name': "weight",
        'start': prim1.x2_base_addr,
        'data': a[1],
        'mode': 0
    }]
    if prim1.bias_type == 2 or prim1.bias_type == 3:
        blocks.append({
            'name': "bias",
            'start': prim1.bias_base_addr,
            'data': a[2],
            'mode': 0
        })
    prim1.memory_blocks = blocks

    test_config_case = {
        'tb_name': 'caSE1',
    }

    # Case 示例
    map_config_case = {
        'sim_clock': 2006,
        0: {  # Step group ID
            # 'clock0_in_step':
            # 'clock1_in_step':
            # 'step_exe_number': math.floor(map_config.get("sim_clk") / map_config[0].get("clock0_in_step"))
            0: {
                'clock': 2000,  # Phase group ID
                'mode': 1,  # 难道是P_adapt?
                ((0, 0), (0, 0)):
                {  # ((chip ID x, chip ID y), (core ID x, core ID y))
                    'prims': [{
                        'axon': prim1,
                        'soma1': None,
                        'router': None,
                        'soma2': None
                    }],
                    # 'instant_prims': [{'axon': None, 'soma1': None, 'router': None, 'soma2': None}],
                    'registers': {
                        "Receive_PI_addr_base": 0,
                        "PI_CXY": 0,
                        "PI_Nx": 0,
                        "PI_Ny": 0,
                        "PI_sign_CXY": 0,
                        "PI_sign_Nx": 0,
                        "PI_sign_Ny": 0,
                        "instant_PI_en": 0,
                        "fixed_instant_PI": 0,
                        "instant_PI_number": 0,
                        "PI_loop_en": 0,
                        "start_instant_PI_num": 0
                    }
                }
            },
            1: {
                'clock': 1000,
                'mode': 1,
                ((0, 0), (1, 0)): {
                    'prims': [{
                        'axon': None,
                        'soma1': None,
                        'router': None,
                        'soma2': None
                    }],
                    'instant_prims': [{
                        'axon': None,
                        'soma1': None,
                        'router': None,
                        'soma2': None
                    }],
                    'registers': {
                        "Receive_PI_addr_base": 0x7a0 >> 2,
                        "PI_CXY": 0,
                        "PI_Nx": 0,
                        "PI_Ny": 0,
                        "PI_sign_CXY": 0,
                        "PI_sign_Nx": 0,
                        "PI_sign_Ny": 0,
                        "instant_PI_en": 1,
                        "fixed_instant_PI": 1,
                        "instant_PI_number": 0,
                        "PI_loop_en": 0,
                        "start_instant_PI_num": 0
                    }
                }
            }
        }
    }
    convert_behavior_to_assembly(map_config_case, test_config_case)


# smoke_test()
