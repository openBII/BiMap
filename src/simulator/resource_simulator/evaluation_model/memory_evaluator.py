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
    


if __name__ == "__main__":
    area = {}
    for i in (262144, 524288, 655360, 786432):
        area[i] = {}
        for j in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            evaluator = RegisterFileEvaluator(num_files=1, num_registers=i, 
                                              bitwidth=32, num_ports=j, 
                                              process_node=ProcessNode.SEVEN)
            area[i][j] = evaluator.eval_area()
    
    ratio = {}
    for i in (262144, 524288, 655360, 786432):
        ratio[i] = []
        for j in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            ratio[i].append(area[i][j] / area[i][1])

    for capacity in (1024, 2048, 2560, 3072):
        sram_evaluator = MemoryEvaluator(capacity=capacity, 
                                         process_node=ProcessNode.SEVEN)
        print(sram_evaluator.eval_area())
