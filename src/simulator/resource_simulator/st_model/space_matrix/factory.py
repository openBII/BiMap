from src.simulator.resource_simulator.config.matrix_config import Config


class Factory:
    @staticmethod
    def create_matrix(config: Config):
        raise NotImplementedError