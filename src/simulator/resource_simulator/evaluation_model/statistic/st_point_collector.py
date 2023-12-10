#!/usr/bin/env python
# coding: utf-8

"""
写几个Visitor测试一下
STPointCollector用于收集STMatrix里所有的STPoint
"""

from resource_simulator.st_model.matrix_visitor import MatrixVisitor
from resource_simulator.st_model.st_coord import MLCoord


class STPointCollector(MatrixVisitor):
    def __init__(self, st_contest=None):
        super().__init__(st_contest)
        self.all_points = {}

    def visit_point(self, st_point):
        ml_coord = MLCoord(self._space_coord, self._time_coord)
        self.all_points[ml_coord] = st_point

        super().visit_point(st_point)
