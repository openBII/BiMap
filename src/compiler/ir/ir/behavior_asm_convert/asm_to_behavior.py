# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import src.compiler.ir.asm_pb2 as tj_asm
from src.compiler.ir.behavior_asm_convert.asm_prim_transform import AsmPrimTransform
from old.generator.map_config_utils import MapConfigGen


class AsmConvertor:
    def __init__(self, case_name, asm, save_para=False):
        self.case_name = case_name
        self.map_config = {}
        self.behavior_asm = asm.behavior_config
        self.data_asm = asm.data_config
        self.save_para = save_para

    def convert(self):
        self.map_config['sim_clock'] = self.behavior_asm.sim_clock
        for step_group_asm in self.behavior_asm.step_config:
            step_group_id = step_group_asm.step_group_id
            step_group_config = self.convert_step_group(step_group_asm)
            self.map_config[step_group_id] = step_group_config

        MapConfigGen.add_router_info(map_config=self.map_config)

    def convert_step_group(self, step_group_asm):
        chip_x = step_group_asm.chip_x
        chip_y = step_group_asm.chip_y
        chip_id = (chip_x, chip_y)

        step_group_config = {}
        if step_group_asm.HasField('clock0_in_step'):
            step_group_config['clock0_in_step'] = step_group_asm.clock0_in_step
        if step_group_asm.HasField('clock1_in_step'):
            step_group_config['clock1_in_step'] = step_group_asm.clock1_in_step
        if step_group_asm.HasField('step_exe_number'):
            step_group_config['step_exe_number'] = step_group_asm.step_exe_number

        for phase_group_asm in step_group_asm.phase_group_config:
            phase_group_id = phase_group_asm.phase_group_id
            phase_group_config = self.convert_phase_group(
                phase_group_asm, chip_id)
            step_group_config[phase_group_id] = phase_group_config
        return step_group_config

    def convert_phase_group(self, phase_group_asm, chip_id):
        phase_group_config = {}
        if phase_group_asm.phase_mode == tj_asm.PhaseMode.FIXED_CLOCK:
            phase_group_config['mode'] = 0
            phase_group_config['clock'] = phase_group_asm.phase_clock
        else:
            phase_group_config['mode'] = 1

        for core_asm in phase_group_asm.core_config:
            core_config = self.convert_core(core_asm)
            core_x = core_asm.core_x
            core_y = core_asm.core_y
            phase_group_config[(chip_id, (core_x, core_y))] = core_config
        return phase_group_config

    def convert_core(self, core_asm):
        core_config = {}
        if len(core_asm.static_prim_list) > 0:
            core_config['prims'] = []
            for prim_group_asm in core_asm.static_prim_list:
                prim_group_config = self.convert_prim(prim_group_asm)
                core_config['prims'].append(prim_group_config)

        if len(core_asm.instant_prim_list) > 0:
            core_config['instant_prims'] = []
            for prim_group_asm in core_asm.instant_prim_list:
                prim_group_config = self.convert_prim(prim_group_asm)
                core_config['instant_prims'].append(prim_group_config)

        return core_config

    def convert_prim(self, prim_group_asm):
        prim_transform = AsmPrimTransform(self.case_name,
                                          self.data_asm, self.save_para)
        return prim_transform.transform(prim_group_asm)
