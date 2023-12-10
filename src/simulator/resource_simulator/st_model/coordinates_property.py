#!/usr/bin/env python
# coding: utf-8

"""
CoordinatesProperty描述坐标系的属性
这个东西好像没用了
"""


class CoordinatesProperty():
    # chip array -> chip -> phase group -> core
    SPACE_LEVEL = 3
    # step -> phase -> pipeline stage
    TIME_LEVEL = 2

    @staticmethod
    def is_finest_level(ml_coord):
        """
        检查ml_coord指明的位置时空上是否精确到最细粒度
        """
        space = ml_coord.space_coord
        time = ml_coord.time_coord
        return space.level == CoordinatesProperty.SPACE_LEVEL and\
            time.level == CoordinatesProperty.TIME_LEVEL
