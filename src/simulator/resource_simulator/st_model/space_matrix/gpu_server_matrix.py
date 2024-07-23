from typing import List
from copy import deepcopy
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.st_model.space_point.communication_point import CommunicationPoint
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord, create_mlcoord
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix


class GPUServerMatrix(STMatrix):
    def __init__(self, dim: int, space_level: int, 
                 communication_networks: List[CommunicationPoint] = None):
        super().__init__(dim, space_level, communication_networks)

    def generate_path(self, src: MLCoord, dst: MLCoord) -> List[MLCoord]:
        PHY = Coord(3)
        if src[0] != dst[0]:  # 跨PCB
            path = []
            src_phy = create_mlcoord(src[0], PHY)
            path.extend(self._generate_path_on_board(src, src_phy))
            path.append(create_mlcoord(dst[0]))
            dst_phy = create_mlcoord(dst[0], PHY)
            path.extend(self._generate_path_on_board(dst_phy, dst))
            return path
        else:
            return self._generate_path_on_board(src, dst)
            
    def _generate_path_on_board(self, src: MLCoord, 
                                dst: MLCoord) -> List[MLCoord]:
        CHIP = Coord(0)
        DRAM = Coord(2)
        PHY = Coord(3)
        chip_container = self.container[Coord((0, 0))].container[CHIP].container
        for coord in chip_container:
            if isinstance(chip_container[coord], MemoryPoint):
                SHARED_MEMORY = coord
                break
        TENSOR_UNIT = Coord(1)
        ROUTER = Coord(3)
        if ((src.level == 2 and src.bottom_coord == DRAM) or 
            (src.level == 2 and src.bottom_coord == PHY)):
            if dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:  # DRAM -> Shared Memory
                return [src, dst.outer_coord]
            elif dst.level == 4 and dst.bottom_coord == TENSOR_UNIT:  # DRAM -> Tensor Unit
                shared_memory = dst.outer_coord.create_mlcoord_with_different_bottom_coord(SHARED_MEMORY)
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, shared_memory, router, dst]
            elif ((dst.level == 2 and dst.bottom_coord == DRAM) or 
                  (dst.level == 2 and dst.bottom_coord == PHY)):  # DRAM -> DRAM / PHY
                return [src, dst]
            else:
                raise NotImplementedError
        elif src.level == 3 and src.bottom_coord == SHARED_MEMORY:
            if ((dst.level == 2 and dst.bottom_coord == DRAM) or
                (dst.level == 2 and dst.bottom_coord == PHY)):  # Shared Memory -> DRAM / PHY
                chip = src.outer_coord
                return [chip, dst]
            elif dst.level == 4 and dst.bottom_coord == TENSOR_UNIT:  # Shared Memory -> Tensor Unit
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            elif dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:
                return [src, dst]
            else:
                raise NotImplementedError
        elif src.level == 4 and src.bottom_coord == TENSOR_UNIT:
            if dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:  # Tensor Unit -> Shared Memory
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            elif ((dst.level == 2 and dst.bottom_coord == DRAM) or 
                  (dst.level == 2 and dst.bottom_coord == PHY)):  # Tensor Unit -> DRAM / PHY
                shared_memory = src.outer_coord.create_mlcoord_with_different_bottom_coord(SHARED_MEMORY)
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, shared_memory, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
