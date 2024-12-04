from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import CommunicationPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.space_matrix.gpu_server_matrix import GPUServerMatrix


class ServerFactory(Factory):
    """
    Factory class for creating ComputeChiplet objects
    """
    @staticmethod
    def create_matrix(config: ServerConfig, type: BoardType) -> STMatrix:
        server = GPUServerMatrix(dim=2, space_level=4)
        for i in range(config.size[0]):
            for j in range(config.size[1]):
                board = BoardFactory.create_matrix(config.PCB, type)
                server.add_element(coord=Coord((i, j)), element=board)

        if config.network["topology"] == "mesh":
            communication_config = CommunicationConfig(
                config.network["bandwidth"],
                latency=config.network["latency"])
            communication_network = CommunicationPoint(communication_config)
        else:
            raise NotImplementedError
        server.add_communication_network(communication_network)

        return server


if __name__ == "__main__":
    config = ServerConfig("top/server.toml")
    server = ServerFactory.create_matrix(config)
    print(server)
