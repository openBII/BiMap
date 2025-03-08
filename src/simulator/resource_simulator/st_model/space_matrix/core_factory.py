from src.simulator.resource_simulator.config.matrix_config import CoreConfig, HybridCoreConfig, CompAirCoreConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.evaluation_model.evaluator import EvaluationMode
from src.simulator.resource_simulator.st_model.space_point.communication_point import CoreCommunicationPoint, HybridCoreCommunicationPoint, CompAirCoreCommunicationPoint
from src.simulator.resource_simulator.st_model.space_point.computation_point import MACArrayPoint, VectorPoint, HybridPrecisionMACArrayPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint, RegisterFilePoint
from src.simulator.resource_simulator.config.computation_config import ComputationConfig, ComputationInfo
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig, HybridCoreCommunicationConfig, CompAirCoreCommunicationConfig
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.resource_simulator.st_model.st_coord import Coord


class CoreFactory(Factory):
    """
    Factory class for creating Core objects
    """

    @staticmethod
    def create_matrix(config: CoreConfig) -> STMatrix:
        core = STMatrix(dim=1, space_level=1)
        # communication_config = CoreCommunicationConfig(
        #     buffer2array_input=config.network['buffer2array_input'],
        #     buffer2array_weight=config.network['buffer2array_weight'],
        #     array2buffer=config.network['array2buffer'],
        #     array2vector=config.network['array2vector'],
        #     buffer2vector_input=config.network['buffer2vector_input'],
        #     buffer2vector_params=config.network['buffer2vector_params'],
        #     vector2buffer=config.network['vector2buffer'],
        #     buffer2router=config.network['buffer2router'],
        #     router2buffer=config.network['router2buffer'],
        #     latency=config.network["latency"]
        # )
        if config.network['bandwidth'] == "inf":
            bandwidth = float("inf")
        else:
            bandwidth = config.network['bandwidth']
        communication_config = CommunicationConfig(
            bandwidth=bandwidth,
            latency=config.network["latency"],
            process_node=config.process_node)
        communication_network = CoreCommunicationPoint(communication_config)
        core.add_communication_network(communication_network)

        mac_array_config = ComputationConfig()
        mac_array_config.process_node = config.process_node
        if 'fp16' in config.mac_array:
            mac_array_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.mac_array['fp16']['parallelism'],
                latency=config.mac_array['fp16']['latency'])
        if 'fp32' in config.mac_array:
            mac_array_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.mac_array['fp32']['parallelism'],
                latency=config.mac_array['fp32']['latency'])
        mac_array_config.local_memory_latency = config.local_memory["latency"]
        mac_array_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        mac_array = MACArrayPoint(mac_array_config)
        core.add_element(coord=Coord(1), element=mac_array)

        vector_unit_config = ComputationConfig()
        vector_unit_config.process_node = config.process_node
        if 'fp16' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.vector_unit['fp16']['parallelism'],
                latency=config.vector_unit['fp16']['latency']
            )
        if 'fp32' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.vector_unit['fp32']['parallelism'],
                latency=config.vector_unit['fp32']['latency']
            )
        vector_unit_config.local_memory_latency = config.local_memory["latency"]
        vector_unit_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        vector_unit = VectorPoint(vector_unit_config)
        core.add_element(coord=Coord(2), element=vector_unit)

        sram_point = MemoryPoint(config.local_memory['capacity'],
                                 config.process_node)
        core.add_element(coord=Coord(0), element=sram_point)

        if hasattr(config, 'register_file'):
            register_file = RegisterFilePoint(
                config.register_file["num_files"],
                config.register_file["num_registers"],
                config.register_file["bitwidth"],
                config.register_file["num_ports"], 
                config.process_node)
            core.add_element(coord=Coord(4), element=register_file)

        return core
    

class HybridCoreFactory(Factory):
    """
    Factory class for creating Core objects
    """

    @staticmethod
    def create_matrix(config: HybridCoreConfig) -> STMatrix:
        core = STMatrix(dim=1, space_level=1)

        # Communication Config
        communication_config = HybridCoreCommunicationConfig(
            config.network["sram_bandwidth"],
            config.network["dram_bandwidth"],
            config.network["flash_bandwidth"],
            config.network["noc_bandwidth"],
            config.network["nop_bandwidth"],
            process_node=config.process_node
        )
        communication_network = HybridCoreCommunicationPoint(communication_config)
        core.add_communication_network(communication_network)

        # MAC Array
        mac_array_config = ComputationConfig()
        mac_array_config.process_node = config.process_node
        if 'fp16' in config.mac_array:
            mac_array_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.mac_array['fp16']['parallelism'],
                latency=config.mac_array['fp16']['latency'])
        if 'fp32' in config.mac_array:
            mac_array_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.mac_array['fp32']['parallelism'],
                latency=config.mac_array['fp32']['latency'])
        if 'int8' in config.mac_array:
            mac_array_config[Precision.INT_8] = ComputationInfo(
                parallelism=config.mac_array['int8']['parallelism'],
                latency=config.mac_array['int8']['latency'])
        if 'uint4' in config.mac_array:
            mac_array_config[Precision.UINT_4] = ComputationInfo(
                parallelism=config.mac_array['uint4']['parallelism'],
                latency=config.mac_array['uint4']['latency'])
        mac_array_config.local_memory_latency = config.local_memory["latency"]
        mac_array_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        mac_array = HybridPrecisionMACArrayPoint(mac_array_config, EvaluationMode.STATIC)
        core.add_element(coord=Coord(1), element=mac_array)

        vector_unit_config = ComputationConfig()
        vector_unit_config.process_node = config.process_node
        if 'fp16' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.vector_unit['fp16']['parallelism'],
                latency=config.vector_unit['fp16']['latency']
            )
        if 'fp32' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.vector_unit['fp32']['parallelism'],
                latency=config.vector_unit['fp32']['latency']
            )
        if 'int8' in config.vector_unit:
            vector_unit_config[Precision.INT_8] = ComputationInfo(
                parallelism=config.vector_unit['int8']['parallelism'],
                latency=config.vector_unit['int8']['latency']
            )
        if 'uint4' in config.vector_unit:
            vector_unit_config[Precision.UINT_4] = ComputationInfo(
                parallelism=config.vector_unit['uint4']['parallelism'],
                latency=config.vector_unit['uint4']['latency']
            )
        
        # Vector
        vector_unit_config.local_memory_latency = config.local_memory["latency"]
        vector_unit_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        vector_unit = VectorPoint(vector_unit_config)
        core.add_element(coord=Coord(2), element=vector_unit)

        # SRAM
        sram_point = MemoryPoint(config.local_memory['capacity'],
                                 config.process_node)
        core.add_element(coord=Coord(0), element=sram_point)

        # DRAM
        dram_point = MemoryPoint(config.dram['capacity'],
                                 config.process_node)
        core.add_element(coord=Coord(3), element=dram_point)

        # Flash
        flash_point = MemoryPoint(config.flash['capacity'],
                                  config.process_node)
        core.add_element(coord=Coord(4), element=flash_point)

        return core


class CompAirCoreFactory(Factory):
    """
    Factory class for creating Core objects
    """
    
    @staticmethod
    def create_matrix(config: CompAirCoreConfig) -> STMatrix:
        core = STMatrix(dim=1, space_level=1)

        # Communication Config
        communication_config = CompAirCoreCommunicationConfig(
            config.network["sram_bandwidth"],
            config.network["dram_bandwidth"],
            config.network["shadow_bandwidth"],
            config.network["noc_bandwidth"],
            config.network["nop_bandwidth"],
            process_node=config.process_node
        )
        
        # FIXME: Change to a correct one
        communication_network = CompAirCoreCommunicationPoint(communication_config)
        core.add_communication_network(communication_network)

        # MAC Array
        mac_array_config = ComputationConfig()
        mac_array_config.process_node = config.process_node
        if 'fp16' in config.mac_array:
            mac_array_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.mac_array['fp16']['parallelism'],
                latency=config.mac_array['fp16']['latency'])
        if 'fp32' in config.mac_array:
            mac_array_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.mac_array['fp32']['parallelism'],
                latency=config.mac_array['fp32']['latency'])
        if 'int8' in config.mac_array:
            mac_array_config[Precision.INT_8] = ComputationInfo(
                parallelism=config.mac_array['int8']['parallelism'],
                latency=config.mac_array['int8']['latency'])
        if 'uint4' in config.mac_array:
            mac_array_config[Precision.UINT_4] = ComputationInfo(
                parallelism=config.mac_array['uint4']['parallelism'],
                latency=config.mac_array['uint4']['latency'])
        mac_array_config.local_memory_latency = config.local_memory["latency"]
        mac_array_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        
        # FIXME: Change to EvaluationMode.Dynamic for PIM
        mac_array = HybridPrecisionMACArrayPoint(mac_array_config, EvaluationMode.STATIC)
        core.add_element(coord=Coord(1), element=mac_array)

        vector_unit_config = ComputationConfig()
        vector_unit_config.process_node = config.process_node
        if 'fp16' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_16] = ComputationInfo(
                parallelism=config.vector_unit['fp16']['parallelism'],
                latency=config.vector_unit['fp16']['latency']
            )
        if 'fp32' in config.vector_unit:
            vector_unit_config[Precision.FLOAT_32] = ComputationInfo(
                parallelism=config.vector_unit['fp32']['parallelism'],
                latency=config.vector_unit['fp32']['latency']
            )
        if 'int8' in config.vector_unit:
            vector_unit_config[Precision.INT_8] = ComputationInfo(
                parallelism=config.vector_unit['int8']['parallelism'],
                latency=config.vector_unit['int8']['latency']
            )
        if 'uint4' in config.vector_unit:
            vector_unit_config[Precision.UINT_4] = ComputationInfo(
                parallelism=config.vector_unit['uint4']['parallelism'],
                latency=config.vector_unit['uint4']['latency']
            )
        
        # Vector (DRAM-PIM)
        vector_unit_config.local_memory_latency = config.local_memory["latency"]
        vector_unit_config.local_memory_bandwidth = config.local_memory["bandwidth"]
        vector_unit = VectorPoint(vector_unit_config)
        core.add_element(coord=Coord(2), element=vector_unit)

        # SRAM (Shadow)
        sram_point = MemoryPoint(config.local_memory['capacity'],
                                 config.process_node)
        core.add_element(coord=Coord(0), element=sram_point)

        # DRAM (Shadow)
        dram_point = MemoryPoint(config.dram['capacity'],
                                 config.process_node)
        core.add_element(coord=Coord(3), element=dram_point)

        return core
    
if __name__ == "__main__":
    import toml

    config = toml.load("top/server.toml")
    config = CoreConfig(config["PCB"]["chiplet"]["core"])
    core = CoreFactory.create_matrix(config)
    print(core)
