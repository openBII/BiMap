from src.simulator.resource_simulator.config.matrix_config import CoreConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import CoreCommunicationPoint
from src.simulator.resource_simulator.st_model.space_point.computation_point import MACArrayPoint, VectorPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.config.computation_config import ComputationConfig
from src.simulator.resource_simulator.config.communication_config import CoreCommunicationConfig
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.resource_simulator.st_model.st_coord import Coord


class CoreFactory(Factory):
    """
    Factory class for creating Core objects
    """

    @staticmethod
    def create_matrix(config: CoreConfig) -> STMatrix:
        core = STMatrix(dim=1, space_level=1)
        communication_config = CoreCommunicationConfig(
            buffer2array_input=config.network['buffer2array_input'],
            buffer2array_weight=config.network['buffer2array_weight'],
            array2buffer=config.network['array2buffer'],
            array2vector=config.network['array2vector'],
            buffer2vector_input=config.network['buffer2vector_input'],
            buffer2vector_params=config.network['buffer2vector_params'],
            vector2buffer=config.network['vector2buffer'],
            buffer2router=config.network['buffer2router'],
            router2buffer=config.network['router2buffer'],
            latency=config.network["latency"]
        )
        communication_network = CoreCommunicationPoint(communication_config)
        core.add_communication_network(communication_network)

        mac_array_config = ComputationConfig()
        mac_array_config[Precision.FLOAT_16] = config.mac_array['fp16']
        mac_array_config[Precision.FLOAT_32] = config.mac_array['fp32']
        mac_array_config.latency = config.mac_array["latency"]
        mac_array = MACArrayPoint(mac_array_config)
        core.add_element(coord=Coord(1), element=mac_array)

        vector_unit_config = ComputationConfig()
        vector_unit_config[Precision.FLOAT_16] = config.vector_unit['fp16']
        vector_unit_config[Precision.FLOAT_32] = config.vector_unit['fp32']
        vector_unit_config.latency = config.vector_unit["latency"]
        vector_unit = VectorPoint(vector_unit_config)
        core.add_element(coord=Coord(2), element=vector_unit)

        sram_point = MemoryPoint(config.local_memory['capacity'])
        core.add_element(coord=Coord(0), element=sram_point)

        return core


if __name__ == "__main__":
    import toml

    config = toml.load("top/server.toml")
    config = CoreConfig(config["PCB"]["chiplet"]["core"])
    core = CoreFactory.create_matrix(config)
    print(core)
