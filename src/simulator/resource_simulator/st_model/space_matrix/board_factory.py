from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import BoardCommunicationPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.st_model.space_matrix.chiplet_factory import ComputeChipletFactory


class BoardFactory(Factory):
    """
    Factory class for creating ComputeChiplet objects
    """

    @staticmethod
    def create_matrix(config: BoardConfig) -> STMatrix:
        board = STMatrix(dim=1, space_level=3)
        chiplet = ComputeChipletFactory.create_matrix(config.chiplet)
        board.add_element(coord=Coord(0), element=chiplet)

        DRAM = MemoryPoint(config.DRAM["capacity"])
        board.add_element(coord=Coord(1), element=DRAM)

        communication_config = CommunicationConfig(config.network["bandwidth"])
        communication_network = BoardCommunicationPoint(communication_config)
        board.add_communication_network(communication_network)

        return board


if __name__ == "__main__":
    import toml

    config = toml.load("top/server.toml")
    config = BoardConfig(config["PCB"])
    board = BoardFactory.create_matrix(config)
    print(board)
