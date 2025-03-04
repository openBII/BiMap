from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.config.matrix_config import HybridPackageConfig, PackageConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import CommunicationPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.st_model.space_matrix.chiplet_factory import ComputeChipletFactory


class PackageFactory(Factory):
    """
    Factory class for creating Package objects
    """
    @staticmethod
    def create_matrix(config: PackageConfig) -> STMatrix:
        package = STMatrix(dim=2, space_level=3)
        
        size_x, size_y = config.size

        for i in range(size_x):
            for j in range(size_y):
                chiplet = ComputeChipletFactory.create_matrix(config.chiplet)
                package.add_element(coord=Coord((i, j)), element=chiplet)

        if config.network["topology"] == "mesh":
            communication_config = CommunicationConfig(
                bandwidth=config.network["bandwidth"], 
                process_node=config.process_node, 
                size=(size_x, size_y),
                latency=config.network["latency"]
            )
            communication_network = CommunicationPoint(communication_config)
        else:
            raise NotImplementedError

        package.add_communication_network(communication_network)

        return package
    

class HybridPackageFactory(Factory):
    """
    Factory class for creating Package objects
    """
    @staticmethod
    def create_matrix(config: HybridPackageConfig) -> STMatrix:
        package = STMatrix(dim=2, space_level=3)
        
        size_x, size_y = config.size

        for i in range(size_x):
            for j in range(size_y):
                chiplet = ComputeChipletFactory.create_matrix(config.chiplet,
                                                              "hybrid")
                package.add_element(coord=Coord((i, j)), element=chiplet)

        if config.network["topology"] == "mesh":
            communication_config = CommunicationConfig(
                bandwidth=config.network["bandwidth"], 
                process_node=config.process_node, 
                size=(size_x, size_y),
                latency=config.network["latency"]
            )
            communication_network = CommunicationPoint(communication_config)
        else:
            raise NotImplementedError

        package.add_communication_network(communication_network)

        return package

class CompAirPackageFactory(Factory):
    """
    Factory class for creating Package objects
    """
    @staticmethod
    def create_matrix(config: HybridPackageConfig) -> STMatrix:
        package = STMatrix(dim=2, space_level=3)
        
        size_x, size_y = config.size

        for i in range(size_x):
            for j in range(size_y):
                chiplet = ComputeChipletFactory.create_matrix(config.chiplet,
                                                              "compair")
                package.add_element(coord=Coord((i, j)), element=chiplet)

        if config.network["topology"] == "mesh":
            communication_config = CommunicationConfig(
                bandwidth=config.network["bandwidth"], 
                process_node=config.process_node, 
                size=(size_x, size_y),
                latency=config.network["latency"]
            )
            communication_network = CommunicationPoint(communication_config)
        else:
            raise NotImplementedError

        package.add_communication_network(communication_network)

        return package
    
if __name__ == "__main__":
    pass
