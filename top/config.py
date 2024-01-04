import logging
import toml


class GlobalConfig:
    Config = toml.load("top/cluster.toml")
    DDR = Config['DDR']
    Chip = Config['chip']
    GroupLevel1 = Config['group-level-1']
    GroupLevel2 = Config['group-level-2']
    GroupLevel3 = Config['group-level-3']
    
    @staticmethod
    def config():
        # 设置Python日志输出等级
        logging.basicConfig(level=logging.INFO)
        