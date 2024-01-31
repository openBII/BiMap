#!/usr/bin/env python
# coding: utf-8

"""
MLCoord类表示一个时空坐标，第一维表示空间坐标，第二维表示时间坐标
包含_space_coord和_time_coord
"""

import math
from typing import Tuple



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

    def __str__(self) -> str:
        return super().__str__().replace(', ', '->')

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
