from typing import Any
from src.simulator.resource_simulator.st_model.st_coord import Coord


class Hop():
    def __init__(self, src: Coord, dst: Coord) -> None:
        self.src = src
        self.dst = dst


class HopDict(dict):
    def __contains__(self, __hop: Hop) -> bool:
        for hop in self.keys():
            if hop.src == __hop.src and hop.dst == __hop.dst:
                return True
            
    def __setitem__(self, __hop: Hop, __value: Any) -> None:
        for hop in self.keys():
            if hop.src == __hop.src and hop.dst == __hop.dst:
                return super().__setitem__(hop, __value)
        return super().__setitem__(__hop, __value)
            
    def __getitem__(self, __hop: Hop) -> Any:
        for hop in self.keys():
            if hop.src == __hop.src and hop.dst == __hop.dst:
                return super().__getitem__(hop)
            

if __name__ == "__main__":
    hop0 = Hop(Coord(0), Coord(1))
    hop1 = Hop(Coord(0), Coord(1))
    d = {hop0: 0}
    hd = HopDict()
    hd.update({hop0: 0})
    print(hop1 in d)
    print(hop1 in hd)
