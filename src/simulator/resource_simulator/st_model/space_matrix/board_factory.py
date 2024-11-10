from enum import Enum
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.factory import Factory
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.space_matrix.board_matrix import ManyCoreBoardMatrix, SharedMemoryBoardMatrix, ChipletBoardMatrix
from src.simulator.resource_simulator.st_model.space_point.communication_point import SharedMemoryCommunicationPoint
from src.simulator.resource_simulator.st_model.space_point.memory_point import DRAMPoint
from src.simulator.resource_simulator.config.communication_config import CommunicationConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.st_model.space_matrix.chiplet_factory import ComputeChipletFactory
from src.simulator.resource_simulator.st_model.space_matrix.package_factory import PackageFactory


class BoardType(Enum):
    SHARED_MEMORY = 0
    DISTRIBUTED_MANY_CORE = 1
    CHIPLET = 2


class BoardFactory(Factory):
    """
    Factory class for creating ComputeChiplet objects
    """

    @staticmethod
    def create_matrix(config: BoardConfig, type: BoardType) -> STMatrix:
        if type == BoardType.DISTRIBUTED_MANY_CORE:
            board = ManyCoreBoardMatrix(dim=1, space_level=3)
        elif type == BoardType.SHARED_MEMORY:
            board = SharedMemoryBoardMatrix(dim=1, space_level=3)
        elif type == BoardType.CHIPLET:
            board = ChipletBoardMatrix(dim=1, space_level=4)
        else:
            board = STMatrix(dim=1, space_level=3)

        if type == BoardType.CHIPLET:
            package = PackageFactory.create_matrix(config.package)
            board.add_element(coord=Coord(0), element=package)
        else:
            chiplet = ComputeChipletFactory.create_matrix(config.chiplet)
            board.add_element(coord=Coord(0), element=chiplet)

        dram = DRAMPoint(config.DRAM["capacity"],
                         config.process_node)
        dram_coord = Coord(2)
        board.add_element(coord=dram_coord, element=dram)

        communication_config = CommunicationConfig(
            config.network["bandwidth"], process_node=config.process_node,
            latency=config.network["latency"])
        communication_network = SharedMemoryCommunicationPoint(
            communication_config, dram_coord, arbitrator_coord=Coord(1))
        board.add_communication_network(communication_network)

        return board


if __name__ == "__main__":
    import toml

    config = toml.load("top/server.toml")
    config = BoardConfig(config["PCB"])
    board = BoardFactory.create_matrix(config)
    print(board)
