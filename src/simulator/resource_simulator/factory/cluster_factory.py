from top.config import GlobalConfig
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_model.st_point import DDRPoint, ChipPoint
from src.simulator.resource_simulator.st_model.st_coord import Coord
from src.simulator.resource_simulator.evaluation_model.evaluator import CommunicationEvaluator


class ClusterFactory(object):
    """
    Factory class for creating Cluster objects
    """

    @staticmethod
    def create_st_matrix():
        cluster = STMatrix(dim=1, space_level=3)
        cluster.config.bandwidth = GlobalConfig.GroupLevel3['GROUP2_BANDWIDTH']
        for level3_count in range(GlobalConfig.GroupLevel3['GROUP2_NUM']):
            server = STMatrix(dim=3, space_level=2)
            server.config.bandwidth = GlobalConfig.GroupLevel2['GROUP1_BANDWIDTH']
            card_x, card_y, card_z = 0, 0, 0
            for _ in range(GlobalConfig.GroupLevel2['GROUP1_NUM']):
                bandwidth = GlobalConfig.GroupLevel1['CHIP_DDR_BANDWIDTH']
                evaluator = CommunicationEvaluator(bandwidth)
                board = STMatrix(dim=1, space_level=1, evaluator=evaluator)
                board.config.bandwidth = bandwidth
                point_count = 0
                for _ in range(GlobalConfig.GroupLevel1['CHIP_NUM']):
                    chip = ChipPoint()
                    chip.core_num = GlobalConfig.Chip['CORE_NUM']
                    chip.int8_computation = GlobalConfig.Chip['INT8_COMPUTATION']
                    chip.fp16_computation = GlobalConfig.Chip['FP16_COMPUTATION']
                    chip.fp32_computation = GlobalConfig.Chip['FP32_COMPUTATION']
                    chip.memory_capacity = GlobalConfig.Chip['MEMORY_CAPACITY']
                    coord = Coord(point_count)
                    board.add_element(coord, chip)
                    point_count += 1
                for _ in range(GlobalConfig.GroupLevel1['DDR_NUM']):
                    ddr = DDRPoint()
                    ddr.capcity = GlobalConfig.DDR['DDR_CAPACITY']
                    coord = Coord(point_count)
                    board.add_element(coord, ddr)
                    point_count += 1
                
                card_coord = Coord((card_z, card_y, card_x))
                server.add_element(card_coord, board)
                card_x += 1
                if card_x == GlobalConfig.GroupLevel2['GROUP1_X']:
                    card_x = 0
                    card_y += 1
                    if card_y == GlobalConfig.GroupLevel2['GROUP1_Y']:
                        card_y = 0
                        card_z += 1
            cluster.add_element(Coord(level3_count), server)
        return cluster


# ClusterFactory.create_st_matrix()
