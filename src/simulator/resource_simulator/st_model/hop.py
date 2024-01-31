from src.simulator.resource_simulator.st_model.st_coord import Coord


class Hop():
    def __init__(self, src: Coord, dst: Coord) -> None:
        self.src = src
        self.dst = dst
