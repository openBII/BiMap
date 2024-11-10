from src.simulator.resource_simulator.config.matrix_config import ComputeDomainConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import CommunicationPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.st_model.space_matrix.package_factory import PackageFactory


class ComputeDomainFactory(Factory):
    """
    Factory class for creating Package objects
    """
    @staticmethod
    def create_matrix(config: ComputeDomainConfig) -> STMatrix:
        compute_domain = STMatrix(dim=2, space_level=4)
        
        size_x, size_y = config.size

        for i in range(size_x):
            for j in range(size_y):
                package = PackageFactory.create_matrix(config.package)
                compute_domain.add_element(coord=Coord((i, j)), element=package)

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

        compute_domain.add_communication_network(communication_network)

        return compute_domain


if __name__ == "__main__":
    pass
