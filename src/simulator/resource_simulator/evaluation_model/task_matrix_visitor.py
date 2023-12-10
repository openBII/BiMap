#!/usr/bin/env python
# coding: utf-8

"""
TaskMatrixVisitor 为 TaskVisitor 和 STMatrix 的子类
"""

from resource_simulator.st_model.st_coord import Coord
from resource_simulator.st_model.matrix_visitor import MatrixVisitor
from task_rabbit.task_model.task_visitor import TaskVisitor
from task_rabbit.task_model.task_block import TaskBlock


class TaskMatrixVisitor(TaskVisitor, MatrixVisitor):
    def __init__(self, st_contest=None):
        TaskVisitor.__init__(self)
        MatrixVisitor.__init__(self, st_contest)

    def visit_point(self, st_point):
        for idx in range(len(st_point.index_to_item)):
            item = st_point.get_time(Coord((idx, )), None)
            if isinstance(item, TaskBlock):
                item.accept(self)
            else:
                for sub_item in item.values():
                    sub_item.accept(self)
