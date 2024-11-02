from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.evaluation_model.evaluator import EvaluationMode
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator
from src.simulator.resource_simulator.evaluation_model.area.cost_model import calc_cache_sram_area_mm2, find_logic_sram_transistor_density
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
