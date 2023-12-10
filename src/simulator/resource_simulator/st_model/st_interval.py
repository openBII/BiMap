#!/usr/bin/env python
# coding: utf-8

"""
STInterval类表示一个左闭右闭时空区间

  ————————————————
 |                |
 |    ————        |
 |   |    |       |
 |   |    |       |
 |    ————        |
 |    ST Interval |
 |                |
  ————————————————  -> ST Matrix
"""

from resource_simulator.st_model.st_scope import STScope
from resource_simulator.st_model.st_coord import MLCoord


class STInterval(STScope):
    def __init__(self, left_up: MLCoord, right_down: MLCoord):
        self.left_up = left_up
        self.right_down = right_down
        assert(self.is_harmony())

    def is_harmony(self):
        '''
        判断改区间的时空部分是否和谐
        '''
        # Rule 1
        space_harmony = self.left_up[0].level == \
            self.right_down[0].level
        time_harmony = self.left_up[1].level == \
            self.right_down[1].level

        # Rule 2
        space_harmony &= self.left_up[0] <= self.right_down[0]
        time_harmony &= self.left_up[1] <= self.right_down[1]

        # Rule 3
        if self.left_up[1].level == MLCoord.TIME_LEVEL:
            return False

        # Rule 4
        if time_harmony and space_harmony:
            space_harmony = (self.left_up[0][:-1] == self.right_down[0][:-1])

        return space_harmony and time_harmony

    def __contains__(self, ml_coord: MLCoord):
        for i in range(2):
            left_level = max(len(ml_coord[i]), len(self.left_up[i])) - 1
            right_level = max(len(ml_coord[i]), len(self.right_down[i])) - 1
            if ml_coord[0].down_to_level(left_level) < \
                    self.left_up[i].down_to_level(left_level):
                return False
            if ml_coord[0].up_to_level(right_level) > \
                    self.right_down[i].up_to_level(right_level):
                return False
        return True

    def __iter__(self):
        for i in range(2):
            if self.left_up[i][:-1] != self.right_down[i][:-1]:
                raise ValueError('This Interval is not Iterable')

        space_min, space_max = self.left_up[0][-1], self.right_down[0][-1]
        time_min, time_max = self.left_up[1][-1], self.right_down[1][-1]
        for space in range(space_min, space_max + 1):
            for time in range(time_min, time_max + 1):
                yield MLCoord(self.left_up[0][:-1] + (space,),
                              self.left_up[1][:-1] + (time,))

    def __str__(self) -> str:
        return 'Interval from ' + str(self.left_up) + ' to '\
            + str(self.right_down)


# left = MLCoord((2, 1, 1, 3), (1, 2))
# right = MLCoord((2, 1, 1, 9), (2, 2))
# interval = STInterval(left, right)

# print(left in interval)
# print(MLCoord((2, 1, 1, 2), (1, 2)) in interval)
# print(MLCoord((2, 1, 1), (1, 2)) in interval)
# print(MLCoord((2, 1, 1, 3), (1, 2, 3)) in interval)

# left = MLCoord((2, 1, 1, 3), (1, 2))
# right = MLCoord((2, 1, 1, 9), (1, 4))
# interval = STInterval(left, right)
# print(interval)
# for coord in interval:
#     print(coord)
