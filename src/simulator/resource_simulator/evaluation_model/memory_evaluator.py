from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.evaluation_model.evaluator import EvaluationMode
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator
from src.simulator.resource_simulator.evaluation_model.area.cost_model import calc_cache_sram_area_mm2, find_logic_sram_transistor_density, calc_reg_file_area
from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode


class MemoryEvaluator(Evaluator):
    def __init__(self, capacity: int,
                 process_node: ProcessNode,
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(process_node, mode)
        self.capacity = capacity

    def eval_area(self):
        _, sram_bitcell_area_mm2 = find_logic_sram_transistor_density(
            self.process_node.value)
        return calc_cache_sram_area_mm2(self.capacity * 1024, sram_bitcell_area_mm2)
        # return self.capacity / 1024 * 2.5

    def eval_by_model(self, input: TaskBlock):
        # TODO: evaluate memory operations
        pass


class RegisterFileEvaluator(MemoryEvaluator):
    def __init__(self, num_files, num_registers, bitwidth, num_ports,
                 process_node, mode = EvaluationMode.STATIC):
        capacity = num_files * num_registers * bitwidth // (4096)
        super().__init__(capacity, process_node, mode)
        self.num_files = num_files
        self.num_registers = num_registers
        self.bitwidth = bitwidth
        self.num_ports = num_ports

    def eval_area(self):
        transistor_density_mil_mm2, _ = find_logic_sram_transistor_density(
            self.process_node.value
        )
        return calc_reg_file_area(self.num_files, self.num_registers,
                                  self.bitwidth, self.num_ports,
                                  transistor_density_mil_mm2)
