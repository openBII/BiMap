from src.simulator.resource_simulator.config.matrix_config import ComputeChipletConfig
from src.simulator.resource_simulator.st_model.space_matrix.core_factory import CoreFactory
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import SharedMemoryCommunicationPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord


class ComputeChipletFactory(Factory):
    """
    Factory class for creating ComputeChiplet objects
    """

    @staticmethod
    def create_matrix(config: ComputeChipletConfig) -> STMatrix:
        chiplet = STMatrix(dim=2, space_level=2)
        size_x, size_y = config.size
        for i in range(size_x):
            for j in range(size_y):
                core = CoreFactory.create_matrix(config.core)
                chiplet.add_element(coord=Coord((i, j)), element=core)

        shared_memory_coord = Coord((size_x + 1, size_y // 2))
        arbitrator_coord = Coord((size_x, size_y // 2))

        if config.network["topology"] == "star":
            communication_config = CommunicationConfig(
                config.network["bandwidth"], (size_x, size_y),
                config.network["latency"])
            communication_network = SharedMemoryCommunicationPoint(
                communication_config, shared_memory_coord, arbitrator_coord)
        elif config.network["topology"] == "crossbar":
            raise NotImplementedError
        elif config.network["topology"] == "mesh":
            raise NotImplementedError
        else:
            raise NotImplementedError
        chiplet.add_communication_network(communication_network)

        shared_memory = MemoryPoint(config.shared_memory["capacity"])
        chiplet.add_element(coord=Coord((size_x + 1, size_y // 2)), element=shared_memory)

        return chiplet


if __name__ == "__main__":
    import toml

    config = toml.load("top/server.toml")
    config = ComputeChipletConfig(config["PCB"]["chiplet"])
    chiplet = ComputeChipletFactory.create_matrix(config)
    print(chiplet)
