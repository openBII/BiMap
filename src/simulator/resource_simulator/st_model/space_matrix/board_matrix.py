from typing import List
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.st_model.space_point.communication_point import CommunicationPoint
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord, create_mlcoord
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix


class ManyCoreBoardMatrix(STMatrix):
    def __init__(self, dim: int, space_level: int, 
                 communication_networks: List[CommunicationPoint] = None):
        super().__init__(dim, space_level, communication_networks)

    def generate_path(self, src: MLCoord, dst: MLCoord) -> List[MLCoord]:
        CHIP = Coord(0)
        DRAM = Coord(2)
        TENSOR_UNIT = Coord(1)
        VECTOR_UNIT = Coord(2)
        ROUTER = Coord(3)
        SRAM_BUFFER = Coord(0)
        # Search the boundary cores
        num_row = 0
        chip_container = self.container[CHIP].container
        for coord in chip_container:
            if coord[0] > num_row:
                num_row = coord[0]
        if src.level == 1 and src.bottom_coord == DRAM:
            if dst.level == 3 and dst.bottom_coord == SRAM_BUFFER:  # DRAM -> Local Memory
                if dst[-2][0] == num_row:
                    return [src, router, dst]
                else:
                    boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row, dst[-2][1])))
                    router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                    return [src, boundary_core, router, dst]
            else:
                raise NotImplementedError
        elif src.level == 3 and src.bottom_coord == SRAM_BUFFER:
            if dst.level == 1 and dst.bottom_coord == DRAM:  # Local Memory -> DRAM
                if src[-2][0] == num_row:
                    return [src, router, dst]
                else:
                    boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row, src[-2][1])))
                    router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                    return [src, router, boundary_core, dst]
            elif dst.level == 3 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Local Memory -> Tensor Unit / Vector Unit
                if src.outer_coord == dst.outer_coord:  # In the same core
                    return [src, dst]
                else:
                    src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                    dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                    return [src, src_router, dst.outer_coord, dst_router, dst]
            elif dst.level == 3 and dst.bottom_coord == SRAM_BUFFER: # Local memory -> Another Local Memory
                assert src.outer_coord != dst.outer_coord
                src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, src_router, dst.outer_coord, dst_router, dst]
            else:
                raise NotImplementedError
        elif src.level == 3 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Tensor Unit / Vector Unit -> Local Memory
            if dst.level == 3 and dst.bottom_coord == SRAM_BUFFER:
                assert src.outer_coord == dst.outer_coord
                src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError



class SharedMemoryBoardMatrix(STMatrix):
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
        VECTOR_UNIT = Coord(2)
        ROUTER = Coord(3)
        if ((src.level == 2 and src.bottom_coord == DRAM) or 
            (src.level == 2 and src.bottom_coord == PHY)):
            if dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:  # DRAM -> Shared Memory
                return [src, dst.outer_coord]
            elif dst.level == 4 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # DRAM -> Tensor Unit / Vector Unit
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
            elif dst.level == 4 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Shared Memory -> Tensor Unit / Vector Unit
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            elif dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:
                return [src, dst]
            else:
                raise NotImplementedError
        elif src.level == 4 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):
            if dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:  # Tensor Unit / Vector Unit -> Shared Memory
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            elif ((dst.level == 2 and dst.bottom_coord == DRAM) or 
                  (dst.level == 2 and dst.bottom_coord == PHY)):  # Tensor Unit / Vector Unit -> DRAM / PHY
                shared_memory = src.outer_coord.create_mlcoord_with_different_bottom_coord(SHARED_MEMORY)
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, shared_memory, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    pass