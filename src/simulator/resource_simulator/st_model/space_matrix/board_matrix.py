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
                    router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
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
                return [src, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        

class ChipletBoardMatrix(STMatrix):
    def __init__(self, dim: int, space_level: int, 
                 communication_networks: List[CommunicationPoint] = None):
        super().__init__(dim, space_level, communication_networks)

    def generate_path(self, src: MLCoord, dst: MLCoord) -> List[MLCoord]:
        PACKAGE = Coord(0)
        DRAM = Coord(2)
        TENSOR_UNIT = Coord(1)
        VECTOR_UNIT = Coord(2)
        ROUTER = Coord(3)
        SRAM_BUFFER = Coord(0)
        # Search the boudary chiplets
        num_row_chiplet = 0
        package_container = self.container[PACKAGE].container
        for coord in package_container:
            if coord[0] > num_row_chiplet:
                num_row_chiplet = coord[0]
        # Search the boundary cores
        num_row_core = 0
        num_column_core = 0
        chip_container = self.container[PACKAGE].container[Coord((0, 0))].container
        for coord in chip_container:
            if coord[0] > num_row_core:
                num_row_core = coord[0]
            if coord[1] > num_column_core:
                num_column_core = coord[1]
        if src.level == 1 and src.bottom_coord == DRAM:
            if dst.level == 4 and dst.bottom_coord == SRAM_BUFFER:  # DRAM -> Local Memory
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                if dst[-3][0] == num_row_chiplet:
                    if dst[-2][0] == num_row_core:
                        return [src, router, dst]
                    else:
                        boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                        return [src, boundary_core, router, dst]
                else:
                    boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, dst[-3][1])))
                    if dst[-2][0] == num_row_core:
                        return [src, boundary_chip, router, dst]
                    else:
                        boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                        return [src, boundary_chip, boundary_core, router, dst]
            else:
                raise NotImplementedError
        elif src.level == 4 and src.bottom_coord == SRAM_BUFFER:
            if dst.level == 1 and dst.bottom_coord == DRAM:  # Local Memory -> DRAM
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                if src[-3][0] == num_row_chiplet:
                    if src[-2][0] == num_row_core:
                        return [src, router, dst]
                    else:
                        boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                        return [src, router, boundary_core, dst]
                else:
                    boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                    if src[-2][0] == num_row_core:
                        return [src, router, boundary_chip, dst]
                    else:
                        boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                        return [src, router, boundary_core, boundary_chip, dst]
            elif dst.level == 4 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Local Memory -> Tensor Unit / Vector Unit
                if src.outer_coord == dst.outer_coord and src.outer_coord.outer_coord == dst.outer_coord.outer_coord:  # In the same core
                    return [src, dst]
                else:
                    if src.outer_coord != dst.outer_coord and src.outer_coord.outer_coord == dst.outer_coord.outer_coord:  # different core, same chiplet
                        src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                        dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                        return [src, src_router, dst.outer_coord, dst_router, dst]
                    else:  # different core, different chiplet
                        src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                        dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                        if dst[-3][0] > src[-3][0] and dst[-3][1] > src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                        elif dst[-3][0] > src[-3][0] and dst[-3][1] < src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                        elif dst[-3][0] < src[-3][0] and dst[-3][1] > src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                        elif dst[-3][0] < src[-3][0] and dst[-3][1] < src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                        elif dst[-3][0] > src[-3][0] and dst[-3][1] == src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-2][1])))
                        elif dst[-3][0] < src[-3][0] and dst[-3][1] == src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                        elif dst[-3][0] == src[-3][0] and dst[-3][1] > src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], num_column_core)))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                        elif dst[-3][0] == src[-3][0] and dst[-3][1] < src[-3][1]:
                            src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], 0)))
                            dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                        else:
                            raise NotImplementedError
                        path = [src, src_router, src_boundary_core,  dst.outer_coord.outer_coord, dst_boundary_core, dst_router, dst]
                        if dst_boundary_core == dst.outer_coord:
                            path.remove(dst_boundary_core)
                        if src_boundary_core == src.outer_coord:
                            path.remove(src_boundary_core)
                        return path
            elif dst.level == 4 and dst.bottom_coord == SRAM_BUFFER:  # Local memory -> Another Local Memory
                assert src.outer_coord != dst.outer_coord or src.outer_coord.outer_coord != dst.outer_coord.outer_coord
                if src.outer_coord.outer_coord == dst.outer_coord.outer_coord:  # Same chiplet
                    src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                    dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                    return [src, src_router, dst.outer_coord, dst_router, dst]
                else:
                    src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                    dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                    if dst[-3][0] > src[-3][0] and dst[-3][1] > src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                    elif dst[-3][0] > src[-3][0] and dst[-3][1] < src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                    elif dst[-3][0] < src[-3][0] and dst[-3][1] > src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                    elif dst[-3][0] < src[-3][0] and dst[-3][1] < src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                    elif dst[-3][0] > src[-3][0] and dst[-3][1] == src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-2][1])))
                    elif dst[-3][0] < src[-3][0] and dst[-3][1] == src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                    elif dst[-3][0] == src[-3][0] and dst[-3][1] > src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], num_column_core)))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                    elif dst[-3][0] == src[-3][0] and dst[-3][1] < src[-3][1]:
                        src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], 0)))
                        dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                    else:
                        raise NotImplementedError
                    path = [src, src_router, src_boundary_core, dst.outer_coord.outer_coord, dst_boundary_core, dst_router, dst]
                    if dst_boundary_core == dst.outer_coord:
                        path.remove(dst_boundary_core)
                    if src_boundary_core == src.outer_coord:
                        path.remove(src_boundary_core)
                    return path
            else:
                raise NotImplementedError
        elif src.level == 4 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Tensor Unit / Vector Unit -> Local Memory
            if dst.level == 4 and dst.bottom_coord == SRAM_BUFFER:
                assert src.outer_coord == dst.outer_coord and src.outer_coord.outer_coord == dst.outer_coord.outer_coord
                return [src, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        

class MultiPackageBoardMatrix(STMatrix):
    def __init__(self, dim: int, space_level: int, 
                 communication_networks: List[CommunicationPoint] = None):
        super().__init__(dim, space_level, communication_networks)

    def generate_path(self, src: MLCoord, dst: MLCoord) -> List[MLCoord]:
        COMPUTE = Coord(0)
        DRAM = Coord(2)
        TENSOR_UNIT = Coord(1)
        VECTOR_UNIT = Coord(2)
        ROUTER = Coord(3)
        SRAM_BUFFER = Coord(0)
        # Search the boudary packages
        num_row_package = 0
        compute_container = self.container[COMPUTE].container
        for coord in compute_container:
            if coord[0] > num_row_package:
                num_row_package = coord[0]
        # Search the boundary chiplets
        num_row_chiplet = 0
        num_column_chiplet = 0
        package_container = self.container[COMPUTE].container[Coord((0, 0))].container
        for coord in package_container:
            if coord[0] > num_row_chiplet:
                num_row_chiplet = coord[0]
            if coord[1] > num_column_chiplet:
                num_column_chiplet = coord[1]
        # Search the boundary cores
        num_row_core = 0
        num_column_core = 0
        chip_container = self.container[COMPUTE].container[Coord((0, 0))].container[Coord((0, 0))].container
        for coord in chip_container:
            if coord[0] > num_row_core:
                num_row_core = coord[0]
            if coord[1] > num_column_core:
                num_column_core = coord[1]
        if src.level == 1 and src.bottom_coord == DRAM:
            if dst.level == 5 and dst.bottom_coord == SRAM_BUFFER:  # DRAM -> Local Memory
                if dst[-4][0] == num_row_package:
                    if dst[-3][0] == num_row_chiplet:
                        if dst[-2][0] == num_row_core:
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, router, dst]
                        else:
                            boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_core, router, dst]
                    else:
                        boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, dst[-3][1])))
                        if dst[-2][0] == num_row_core:
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_chip, router, dst]
                        else:
                            boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_chip, boundary_core, router, dst]
                else:
                    boundary_package = dst.outer_coord.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_package, dst[-4][1])))
                    if dst[-3][0] == num_row_chiplet:
                        if dst[-2][0] == num_row_core:
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_package, router, dst]
                        else:
                            boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_package, boundary_core, router, dst]
                    else:
                        boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, dst[-3][1])))
                        if dst[-2][0] == num_row_core:
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_package, boundary_chip, router, dst]
                        else:
                            boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                            router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, boundary_package, boundary_chip, boundary_core, router, dst]
            else:
                raise NotImplementedError
        elif src.level == 4 and src.bottom_coord == SRAM_BUFFER:
            if dst.level == 1 and dst.bottom_coord == DRAM:  # Local Memory -> DRAM
                if src[-4][0] == num_row_package:
                    if src[-3][0] == num_row_chiplet:
                        if src[-2][0] == num_row_core:
                            return [src, router, dst]
                        else:
                            boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, router, boundary_core, dst]
                    else:
                        boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                        if src[-2][0] == num_row_core:
                            return [src, router, boundary_chip, dst]
                        else:
                            boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, router, boundary_core, boundary_chip, dst]
                else:
                    boundary_package = src.outer_coord.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_package, dst[-4][1])))
                    if src[-3][0] == num_row_chiplet:
                        if src[-2][0] == num_row_core:
                            return [src, router, boundary_package, dst]
                        else:
                            boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, router, boundary_core, boundary_package, dst]
                    else:
                        boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                        if src[-2][0] == num_row_core:
                            return [src, router, boundary_chip, boundary_package, dst]
                        else:
                            boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                            router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            return [src, router, boundary_core, boundary_chip, boundary_package, dst]
            elif dst.level == 5 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT, SRAM_BUFFER):  # Local Memory -> Tensor Unit / Vector Unit
                if dst.bottom_coord == SRAM_BUFFER:
                    assert src[-1] != dst[-1] or src[-2] != dst[-2] or src[-3] != dst[-3]
                if src[-1] == dst[-1] and src[-2] == dst[-2] and src[-3] == dst[-3]:  # In the same core
                    return [src, dst]
                else:
                    if src[-1] != dst[-1] and src[-2] == dst[-2] and src[-3] == dst[-3]:  # different core, same chiplet, same package
                        src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                        dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                        return [src, src_router, dst.outer_coord, dst_router, dst]
                    else:
                        if src[-1] != dst[-1] and src[-2] != dst[-2] and src[-3] == dst[-3]:  # different core, different chiplet, same package
                            src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            if dst[-3][0] > src[-3][0] and dst[-3][1] > src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                            elif dst[-3][0] > src[-3][0] and dst[-3][1] < src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] > src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] < src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                            elif dst[-3][0] > src[-3][0] and dst[-3][1] == src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-2][1])))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] == src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                            elif dst[-3][0] == src[-3][0] and dst[-3][1] > src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], num_column_core)))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                            elif dst[-3][0] == src[-3][0] and dst[-3][1] < src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], 0)))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                            else:
                                raise NotImplementedError
                            path = [src, src_router, src_boundary_core, dst.outer_coord.outer_coord, dst_boundary_core, dst_router, dst]
                            if dst_boundary_core == dst.outer_coord:
                                path.remove(dst_boundary_core)
                            if src_boundary_core == src.outer_coord:
                                path.remove(src_boundary_core)
                            return path
                        else:  # different core, different chiplet, different package
                            src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                            dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                            if dst[-4][0] > src[-4][0] and dst[-4][1] > src[-4][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], 0)))
                            elif dst[-4][0] > src[-4][0] and dst[-4][1] < src[-4][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], num_column_chiplet)))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] > src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], 0)))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] < src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], num_column_chiplet)))
                            elif dst[-3][0] > src[-3][0] and dst[-3][1] == src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-2][1])))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-3][1])))
                            elif dst[-3][0] < src[-3][0] and dst[-3][1] == src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-3][1])))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_chiplet, dst[-3][1])))
                            elif dst[-3][0] == src[-3][0] and dst[-3][1] > src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], num_column_core)))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-3][0], num_column_chiplet)))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], 0)))
                            elif dst[-3][0] == src[-3][0] and dst[-3][1] < src[-3][1]:
                                src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], 0)))
                                dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
                                src_boundary_chip = src.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-3][0], 0)))
                                dst_boundary_chip = dst.outer_coord.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-3][0], num_column_chiplet)))
                            else:
                                raise NotImplementedError
                            path = [src, src_router, src_boundary_core, src_boundary_chip, dst.outer_coord.outer_coord.outer_coord, dst_boundary_chip, dst_boundary_core, dst_router, dst]
                            if dst_boundary_core == dst.outer_coord:
                                path.remove(dst_boundary_core)
                            if src_boundary_core == src.outer_coord:
                                path.remove(src_boundary_core)
                            if src_boundary_chip == src.outer_coord.outer_coord:
                                path.remove(src_boundary_chip)
                            if dst_boundary_chip == dst.outer_coord.outer_coord:
                                path.remove(dst_boundary_chip)
                            return path
            # elif dst.level == 5 and dst.bottom_coord == SRAM_BUFFER:  # Local memory -> Another Local Memory
            #     assert src.outer_coord != dst.outer_coord or src.outer_coord.outer_coord != dst.outer_coord.outer_coord
            #     if src.outer_coord.outer_coord == dst.outer_coord.outer_coord:  # Same chiplet
            #         src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
            #         dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
            #         return [src, src_router, dst.outer_coord, dst_router, dst]
            #     else:
            #         src_router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
            #         dst_router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
            #         if dst[-3][0] > src[-3][0] and dst[-3][1] > src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
            #         elif dst[-3][0] > src[-3][0] and dst[-3][1] < src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
            #         elif dst[-3][0] < src[-3][0] and dst[-3][1] > src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
            #         elif dst[-3][0] < src[-3][0] and dst[-3][1] < src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
            #         elif dst[-3][0] > src[-3][0] and dst[-3][1] == src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, dst[-2][1])))
            #         elif dst[-3][0] < src[-3][0] and dst[-3][1] == src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((0, src[-2][1])))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((num_row_core, dst[-2][1])))
            #         elif dst[-3][0] == src[-3][0] and dst[-3][1] > src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], num_column_core)))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], 0)))
            #         elif dst[-3][0] == src[-3][0] and dst[-3][1] < src[-3][1]:
            #             src_boundary_core = src.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((src[-2][0], 0)))
            #             dst_boundary_core = dst.outer_coord.create_mlcoord_with_different_bottom_coord(Coord((dst[-2][0], num_column_core)))
            #         else:
            #             raise NotImplementedError
            #         path = [src, src_router, src_boundary_core, dst.outer_coord.outer_coord, dst_boundary_core, dst_router, dst]
            #         if dst_boundary_core == dst.outer_coord:
            #             path.remove(dst_boundary_core)
            #         if src_boundary_core == src.outer_coord:
            #             path.remove(src_boundary_core)
            #         return path
            else:
                raise NotImplementedError
        elif src.level == 5 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # Tensor Unit / Vector Unit -> Local Memory
            if dst.level == 5 and dst.bottom_coord == SRAM_BUFFER:
                assert src[-1] == dst[-1] and src[-2] == dst[-2] and src[-3] == dst[-3]
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
        CHIP = Coord(0)
        DRAM = Coord(2)
        PHY = Coord(3)
        chip_container = self.container[CHIP].container
        for coord in chip_container:
            if isinstance(chip_container[coord], MemoryPoint):
                SHARED_MEMORY = coord
                break
        TENSOR_UNIT = Coord(1)
        VECTOR_UNIT = Coord(2)
        ROUTER = Coord(3)
        LOCAL_MEMORY = Coord(0)
        if ((src.level == 1 and src.bottom_coord == DRAM) or 
            (src.level == 1 and src.bottom_coord == PHY)):
            if dst.level == 2 and dst.bottom_coord == SHARED_MEMORY:  # DRAM -> Shared Memory
                return [src, dst.outer_coord]
            elif dst.level == 3 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT):  # DRAM -> Tensor Unit / Vector Unit
                shared_memory = dst.outer_coord.create_mlcoord_with_different_bottom_coord(SHARED_MEMORY)
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, shared_memory, router, dst]
            # elif ((dst.level == 2 and dst.bottom_coord == DRAM) or 
            #       (dst.level == 2 and dst.bottom_coord == PHY)):  # DRAM -> DRAM / PHY
            #     return [src, dst]
            else:
                raise NotImplementedError
        elif src.level == 2 and src.bottom_coord == SHARED_MEMORY:
            if ((dst.level == 1 and dst.bottom_coord == DRAM) or
                (dst.level == 1 and dst.bottom_coord == PHY)):  # Shared Memory -> DRAM / PHY
                chip = src.outer_coord
                return [chip, dst]
            elif dst.level == 3 and dst.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT, LOCAL_MEMORY):  # Shared Memory -> Tensor Unit / Vector Unit
                router = dst.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            # elif dst.level == 3 and dst.bottom_coord == SHARED_MEMORY:
            #     return [src, dst]
            else:
                raise NotImplementedError
        elif src.level == 3 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT, LOCAL_MEMORY):
            if dst.level == 2 and dst.bottom_coord == SHARED_MEMORY:  # Tensor Unit / Vector Unit -> Shared Memory
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, dst]
            elif ((dst.level == 1 and dst.bottom_coord == DRAM) or 
                  (dst.level == 1 and dst.bottom_coord == PHY)):  # Tensor Unit / Vector Unit -> DRAM / PHY
                shared_memory = src.outer_coord.create_mlcoord_with_different_bottom_coord(SHARED_MEMORY)
                router = src.create_mlcoord_with_different_bottom_coord(ROUTER)
                return [src, router, shared_memory, dst]
            elif src.level == 3 and src.bottom_coord in (TENSOR_UNIT, VECTOR_UNIT, LOCAL_MEMORY):
                return [src, dst]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    pass