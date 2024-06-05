import toml


class Config:
    def __init__(self, config) -> None:
        if type(config) is str:
            self.config = toml.load(config)
        else:
            self.config = config
        self.handler()
    
    def handler(self):
        pass


class CoreConfig(Config):
    def __init__(self, config) -> None:
        super().__init__(config)

    def handler(self):
        self.network = self.config["network"]
        self.mac_array = self.config["mac_array"]
        self.vector_unit = self.config["vector_unit"]
        self.local_memory = self.config["local_memory"]


class ComputeChipletConfig(Config):
    def __init__(self, config) -> None:
        super().__init__(config)

    def handler(self):
        self.core = CoreConfig(self.config["core"])
        self.size = self.config["size"]
        self.network = self.config["network"]
        self.shared_memory = self.config["shared_memory"]


class BoardConfig(Config):
    def __init__(self, config) -> None:
        super().__init__(config)

    def handler(self):
        self.chiplet = ComputeChipletConfig(self.config["chiplet"])
        self.DRAM = self.config["DRAM"]
        self.network = self.config["network"]


class ServerConfig(Config):
    def __init__(self, config) -> None:
        super().__init__(config)

    def handler(self):
        self.PCB = BoardConfig(self.config["PCB"])
        self.size = self.config["size"]
        self.network = self.config["network"]