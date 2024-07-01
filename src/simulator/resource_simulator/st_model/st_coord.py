#!/usr/bin/env python
# coding: utf-8

"""
MLCoord类表示一个时空坐标，第一维表示空间坐标，第二维表示时间坐标
包含_space_coord和_time_coord
"""
from typing import Tuple, List, Union, Sequence


# 一个层级中的元素坐标，可以为多维坐标
# space_level表示当前这个坐标处在哪个层级中，一般不需要用到
class Coord(tuple):
    def __new__(cls, coord: Tuple, space_level=None):
        if type(coord) is int:
            coord = (coord,)
        return tuple.__new__(Coord, coord)

    def __init__(self, coord: Tuple, space_level=None):
        super().__init__()
        self.space_level = space_level

    @property
    def dim(self):
        return self.__len__()


# Multi-level coordination
class MLCoord(tuple):
    def __new__(cls, *multiple_coords: Tuple[Coord]):
        return tuple.__new__(MLCoord, multiple_coords)

    def __init__(self, *multiple_coords: Tuple[Coord]):
        super().__init__()

    @property
    def level(self):
        return self.__len__()

    @property
    def inner_coord(self):
        # assert(self.__len__() > 1)
        return MLCoord(*self[1:])

    @property
    def outer_coord(self):
        # assert(self.__len__() > 1)
        return MLCoord(*self[:-1])

    @property
    def top_coord(self) -> Coord:
        return self[0]
    
    @property
    def bottom_coord(self) -> Coord:
        return self[-1]
    
    @property
    def empty(self) -> bool:
        return len(self) == 0
    
    @property
    def single_level(self) -> bool:
        return len(self) == 1

    def __str__(self) -> str:
        return super().__str__().replace('), (', ')->(')
    
    def __deepcopy__(self, memo=None):
        return MLCoord(*[self[i] for i in range(self.level)])

    # def down_to_level(self, target_level):
    #     '''
    #     如target_level==2, (1, 2)会变成(1, 2, 0)
    #     '''
    #     assert(target_level <= self.space_level)
    #     if target_level <= self.level:
    #         return self

    #     return self + (0, ) * (target_level - self.level)

    # def up_to_level(self, target_level):
    #     '''
    #     如target_level==2, (1, 2)会变成(1, 2, math.inf)
    #     '''
    #     assert(target_level <= self.space_level)
    #     if target_level <= self.level:
    #         return self

    #     return self + (math.inf, ) * (target_level - self.level)


class EdgeCoord:
    def __init__(self, ml_coord: MLCoord, network_id: int, link_id: int) -> None:
        self.ml_coord = ml_coord
        self.network_id = network_id
        self.link_id = link_id
        

class PathCoord:
    def __init__(self, path: List[Union[Union[MLCoord, Tuple[MLCoord, int], Tuple[MLCoord, int, int]], EdgeCoord]] = None) -> None:
        if path is None:
            self.path: List[EdgeCoord] = []
        else:
            if isinstance(path[0], EdgeCoord):
                self.path = path
            else:
                self.init(path)

    def init(self, path: List[Union[MLCoord, Tuple[MLCoord, int], Tuple[MLCoord, int, int]]]):
        self.path: List[EdgeCoord] = []
        for element in path:
            if isinstance(element, MLCoord):
                ml_coord = element
                network_id = 0
                link_id = 0
            elif len(element) == 2:
                ml_coord, network_id = element
                link_id = 0
            else:
                ml_coord, network_id, link_id = element
            self.path.append(EdgeCoord(ml_coord, network_id, link_id))

    def __iter__(self):
        return iter(self.path)

    def __contains__(self, ml_coord: MLCoord):
        for element in self.path:
            if element.ml_coord == ml_coord:
                return True
        return False

    def __getitem__(self, index: Union[int, MLCoord]):
        if isinstance(index, MLCoord):
            for element in self.path:
                if element.ml_coord == index:
                    return element
        else:
            return self.path[index]
    
    def __len__(self):
        return len(self.path)
    
    def __repr__(self) -> str:
        string = ''
        for element in self.path:
            string += ' -' + repr(element.link_id) + '-> ' + repr(element.ml_coord)
        string = string[5:]
        return string
    
    def extract(self):
        path: List[Tuple[MLCoord, int]] = []
        for element in self.path:
            path.append((element.ml_coord, element.link_id))
        return path
    
    @property
    def level(self):
        return self.path[0].ml_coord.level
    
    @property
    def network_id(self):
        return self.path[0].network_id
    
    @property
    def top_coord(self):
        return self.path[0].ml_coord.top_coord
    
    @property
    def outer_coord(self):
        return self.path[0].ml_coord.outer_coord
    
    @property
    def inner_coord(self):
        inner_path = []
        for element in self.path:
            inner_path.append((element.ml_coord.inner_coord, element.network_id, element.link_id))
        return PathCoord(inner_path)
    
    @property
    def illegal(self):
        return len(self.path) <= 1   

    def append(self, element: EdgeCoord):
        self.path.append(element)

    @property
    def same_level(self):
        for element in self.path:
            if element.ml_coord.level != self.level:
                return False
        return True
    
    @property
    def same_domain(self):
        for element in self.path:
            if element.network_id != self.network_id:
                return False
        return True


if __name__ == "__main__":
    from copy import deepcopy


    coord0 = MLCoord(Coord(0), Coord((0, 0)), Coord(0))
    coord1 = MLCoord(coord0)
    print(coord0)
    print(coord1)
    print(deepcopy(coord0))
    print(Coord(0) == Coord(0))