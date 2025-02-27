import math
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator, EvaluationMode
from src.simulator.resource_simulator.config.computation_config import ComputationConfig, ComputationInfo
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.evaluation_model.area.cost_model import calc_systolic_array_area_mm2, find_logic_sram_transistor_density
from src.simulator.resource_simulator.evaluation_model.area.cost_model import calc_vector_area_mm2
from src.simulator.resource_simulator.evaluation_model.area.cost_model import calc_reg_file_area


class ComputationEvaluator(Evaluator):
    def __init__(self, config: ComputationConfig,
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config.process_node, mode)
        self.config = config
    

class MACArrayEvaluator(ComputationEvaluator):
    def __init__(self, config: ComputationConfig,
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        in_precision = task.in_precision
        precision = min(in_precision)
        assert precision in self.config, "Cannot process task in {:s}".format(precision.name)
        computation_info = self.config[precision]
        if task.task_type == TaskBlockType.CVM:
            if task.shape.token == 1:
                num_tiles = math.ceil(
                    task.shape.volume / math.prod(computation_info.parallelism))
                num_data = (computation_info.parallelism[0] + 1) * computation_info.parallelism[1]
                one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency * max(1, math.ceil(num_data * 2 / self.config.local_memory_bandwidth))
                return num_tiles * one_time_latency
            else:
                if computation_info.parallelism[0] == computation_info.parallelism[1]:
                    length = computation_info.parallelism[0]
                    num_tiles = math.ceil(task.shape.volume / length**3)
                    one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency * max(1, math.ceil(2 * length / self.config.local_memory_bandwidth))
                    # one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency * max(1, 2 * math.ceil(length / self.config.local_memory_bandwidth))
                    if len(computation_info.parallelism) == 3:
                        return num_tiles * one_time_latency / computation_info.parallelism[2]
                    else:
                        return num_tiles * one_time_latency
                else:
                    num_tiles = math.ceil(
                        task.shape.volume / 
                        math.prod(computation_info.parallelism))
                    num_data = ((computation_info.parallelism[0] + 1) * 
                                computation_info.parallelism[1])
                    one_time_latency = (
                        2 * self.config.local_memory_latency + 
                        computation_info.latency * 
                        max(1, math.ceil(num_data * 
                                         Precision.get_bytes(precision) / 
                                         self.config.local_memory_bandwidth)))
                    if len(computation_info.parallelism) == 3:
                        return (num_tiles * one_time_latency / 
                                computation_info.parallelism[2])
                    else:
                        return num_tiles * one_time_latency
        else:
            raise NotImplementedError
        
    def eval_area(self):
        # area = 0
        # for _, size in self.config:
        #     area += math.prod(size) * 0.0007
        # return area
        area = 0
        transistor_density_mil_mm2, _ = find_logic_sram_transistor_density(
            self.process_node.value)
        num_registers = 0
        for precision, computation_info in self.config:
            parallelism = computation_info.parallelism
            if precision == Precision.FLOAT_16:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp16', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 3
            elif precision == Precision.FLOAT_32:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp32', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 6
            elif precision == Precision.FLOAT_64:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp64', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 12

        area += calc_reg_file_area(
            1, num_registers, 16, 4,  # 2读2写
            transistor_density_mil_mm2)
        
        return area


class VectorUnitEvaluator(ComputationEvaluator):
    def __init__(self, config: ComputationConfig, 
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        in_precision = task.in_precision
        precision = min(in_precision)
        computation_info = self.config[precision]
        parallelism = computation_info.parallelism
        num_tiles = math.ceil(task.shape.volume / parallelism)
        if task.task_type == TaskBlockType.CADD:
            one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency[task.task_type] + math.ceil(parallelism * 2 / self.config.local_memory_bandwidth)
        else:
            one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency[task.task_type] + math.ceil(parallelism / self.config.local_memory_bandwidth)
        return num_tiles * one_time_latency
        # if task.task_type == TaskBlockType.CRELU:
        #     return task.shape.volume / num_pe
        # elif task.task_type == TaskBlockType.CSSoftMax:
        #     return task.shape.volume / num_pe * 4
        # elif task.task_type == TaskBlockType.CSoftMax:
        #     return task.shape.volume / num_pe * 3
        # elif task.task_type == TaskBlockType.CADD:
        #     volume = 0
        #     for in_task in task.in_tasks:
        #         volume += in_task.shape.volume
        #     branch = volume / task.shape.volume
        #     return task.shape.volume / num_pe * (branch - 1)
        # elif task.task_type == TaskBlockType.CLayerNorm:
        #     return task.shape.volume / num_pe * 5
        # else:
        #     raise NotImplementedError
        
    def eval_area(self):
        # area = 0
        # for _, size in self.config:
        #     area += math.prod(size) * 0.0007
        # return area
        area = 0
        transistor_density_mil_mm2, _ = find_logic_sram_transistor_density(
            self.process_node.value)
        num_registers = 0
        for precision, computation_info in self.config:
            if precision == Precision.FLOAT_16:
                area += calc_vector_area_mm2(
                    0, computation_info.parallelism, 0,
                    0, transistor_density_mil_mm2)
                num_registers += computation_info.parallelism * 3
            elif precision == Precision.FLOAT_32:
                area += calc_vector_area_mm2(
                    0, 0, computation_info.parallelism,
                    0, transistor_density_mil_mm2)
                num_registers += computation_info.parallelism * 6
            elif precision == Precision.FLOAT_64:
                area += calc_vector_area_mm2(
                    0, 0, 0,
                    computation_info.parallelism, transistor_density_mil_mm2)
                num_registers += computation_info.parallelism * 9
            elif precision == Precision.INT_32:
                area += calc_vector_area_mm2(
                    computation_info.parallelism, 0, 0,
                    0, transistor_density_mil_mm2)
                num_registers += computation_info.parallelism * 6
                
        area += calc_reg_file_area(
            1, num_registers, 16, 4, 
            transistor_density_mil_mm2)

        return area
    

class HybridPrecisionMACArrayEvaluator(ComputationEvaluator):
    def __init__(self, config: ComputationConfig,
                 mode: EvaluationMode = EvaluationMode.STATIC) -> None:
        super().__init__(config, mode)

    def eval_by_model(self, task: CTaskBlock):
        
        in_precision = task.in_precision
        precision = Precision.UINT_4 if (Precision.UINT_4 in in_precision) else min(in_precision)
        assert precision in self.config, "[HyPrecMACArrayEvaluator] Config not support {:s}".format(precision.name)
        computation_info = self.config[precision]
        if task.task_type == TaskBlockType.CVM:
            if task.shape.token == 1:
                num_tiles = math.ceil(
                    task.shape.volume / math.prod(computation_info.parallelism))
                num_data = (computation_info.parallelism[0] + 1) * computation_info.parallelism[1]
                one_time_latency = 2 * self.config.local_memory_latency + \
                                    computation_info.latency * max(1, math.ceil(num_data * 2 / self.config.local_memory_bandwidth))
                result = num_tiles * one_time_latency
            
                # print("[Comp-Eval] Task", task.id, "P:", task.precision, "S:", task.shape, "Tiles:", num_tiles, "one_time_latency", one_time_latency, "with", result, "<0>")
                # print("[Comp-Eval] Task", "Load Memory Latency:", 2 * self.config.local_memory_latency, "Execution:", computation_info.latency * max(1, math.ceil(num_data * 2 / self.config.local_memory_bandwidth)))
                return result
            else:
                if computation_info.parallelism[0] == computation_info.parallelism[1]:
                    length = computation_info.parallelism[0]
                    num_tiles = math.ceil(task.shape.volume / length**3)
                    one_time_latency = 2 * self.config.local_memory_latency + computation_info.latency * max(1, math.ceil(2 * length / self.config.local_memory_bandwidth))
                    if len(computation_info.parallelism) == 3:
                        result = num_tiles * one_time_latency / computation_info.parallelism[2]
                    else:
                        result = num_tiles * one_time_latency

                    # print("[Comp-Eval] Task", task.id, "P:", task.precision, "S:", task.shape, "Tiles:", num_tiles, "one_time_latency", one_time_latency, "with", result, "<1>")
                    return result
                else:
                    num_tiles = math.ceil(
                        task.shape.volume / 
                        math.prod(computation_info.parallelism))
                    num_data = ((computation_info.parallelism[0] + 1) * 
                                computation_info.parallelism[1])
                    one_time_latency = (
                        2 * self.config.local_memory_latency + 
                        computation_info.latency * 
                        max(1, math.ceil(num_data * 
                                         Precision.get_bytes(precision) / 
                                         self.config.local_memory_bandwidth)))
                    
                    if len(computation_info.parallelism) == 3:
                        result = (num_tiles * one_time_latency / 
                                computation_info.parallelism[2])
                    else:
                        result = num_tiles * one_time_latency
                    
                    # print("[Comp-Eval] Task", task.id, "P:", task.precision, "S:", task.shape, "Tiles:", num_tiles, "one_time_latency", one_time_latency, "with", result, "<2>")
                    return result
        else:
            raise NotImplementedError
        
    def eval_area(self):
        # area = 0
        # for _, size in self.config:
        #     area += math.prod(size) * 0.0007
        # return area
        area = 0
        transistor_density_mil_mm2, _ = find_logic_sram_transistor_density(
            self.process_node.value)
        num_registers = 0
        for precision, computation_info in self.config:
            parallelism = computation_info.parallelism
            if precision == Precision.FLOAT_16:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp16', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 3
            elif precision == Precision.FLOAT_32:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp32', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 6
            elif precision == Precision.FLOAT_64:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'fp64', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 12
            elif precision == Precision.INT_8:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'int8', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 1
            elif precision == Precision.UINT_4:
                area += calc_systolic_array_area_mm2(
                    parallelism[0], parallelism[1], 'uint4', 
                    transistor_density_mil_mm2)
                num_registers += parallelism[0] * 0.5

        area += calc_reg_file_area(
            1, num_registers, 16, 4,  # 2读2写
            transistor_density_mil_mm2)
        
        return area
    
if __name__ == "__main__":
    from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode
    config = ComputationConfig()
    config.dict[Precision.FLOAT_16] = ComputationInfo((83, 83), 256)
    config.process_node = ProcessNode.SEVEN
    evaluator = MACArrayEvaluator(config)
    area = evaluator.eval_area()
    print(area)