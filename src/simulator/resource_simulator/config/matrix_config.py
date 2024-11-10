import toml
from typing import Union
from src.simulator.resource_simulator.evaluation_model.area.process_node import ProcessNode


def convert_process_node(process_node: Union[int, ProcessNode]):
    if type(process_node) is int:
        if process_node == 5:
            return ProcessNode.FIVE
        elif process_node == 6:
            return ProcessNode.SIX
        elif process_node == 7:
            return ProcessNode.SEVEN
        else:
            raise NotImplementedError
    else:
        return process_node


class Config:
    def __init__(self, config, process_node: int = None) -> None:
        if type(config) is str:
            self.config = toml.load(config)
        else:
            self.config = config
        if process_node is not None:
            self.process_node = convert_process_node(process_node)
        else:
            self.process_node = None
        self.handler()
    
    def handler(self):
        pass


class CoreConfig(Config):
    def __init__(self, config, process_node = None):
        super().__init__(config, process_node)

    def handler(self):
        self.network = self.config["network"]
        self.mac_array = self.config["mac_array"]
        self.vector_unit = self.config["vector_unit"]
        self.local_memory = self.config["local_memory"]
        if "register_file" in self.config:
            self.register_file = self.config["register_file"]


class ComputeChipletConfig(Config):
    def __init__(self, config, process_node = None):
        super().__init__(config, process_node)

    def handler(self):
        self.core = CoreConfig(self.config["core"], self.process_node)
        self.size = self.config["size"]
        self.network = self.config["network"]
        if "shared_memory" in self.config:
            self.shared_memory = self.config["shared_memory"]


class PackageConfig(Config):
    def __init__(self, config, process_node = None):
        super().__init__(config, process_node)

    def handler(self):
        self.chiplet = ComputeChipletConfig(self.config["chiplet"],
                                            self.process_node)
        self.size = self.config["size"]
        self.network = self.config["network"]


class BoardConfig(Config):
    def __init__(self, config, process_node=None):
        super().__init__(config, process_node)

    def handler(self):
        if "package" in self.config:
            self.package = PackageConfig(self.config["package"],
                                         self.process_node)
        else:
            self.chiplet = ComputeChipletConfig(self.config["chiplet"], 
                                                self.process_node)
        self.DRAM = self.config["DRAM"]
        self.network = self.config["network"]


class ServerConfig(Config):
    def __init__(self, config, process_node = None):
        super().__init__(config, process_node)

    def handler(self):
        self.PCB = BoardConfig(self.config["PCB"])
        self.size = self.config["size"]
        self.network = self.config["network"]