#!/usr/bin/env python
# coding: utf-8

"""
MatrixVisitor类为访问STMatrix的访问者
"""

from resource_simulator.st_model.st_coord import Coord


class MatrixVisitor():
    def __init__(self, st_contest=None):
        self._st_contest = st_contest

        self._space_coord = Coord(tuple())
        self._time_coord = Coord(tuple())

    def visit_matrix(self, st_matrix):
        for key, item in st_matrix:
            self._space_coord = Coord(self._space_coord + (key,))
            item.accept(self)
            self._space_coord = self._space_coord.outer_coord

    def visit_column(self, space_column):
        for key, item in space_column:
            self._time_coord = Coord(self._time_coord + (key,))
            item.accept(self)
            self._time_coord = self._time_coord.outer_coord

    def visit_point(self, st_point):
        pass
