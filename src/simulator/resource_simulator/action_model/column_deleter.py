#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, Coord
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph


class ColumnDeleter(object):
    def __init__(self, context):
        self.task_list = []
        self._context = context

    def delete_step_phase(self, space_column, space_coord,
                          time_coord=Coord(())):
        if isinstance(space_column, STPoint):
            self.delete_pi(space_column, space_coord, time_coord)
        else:
            keys = [k for k, v in space_column]
            for k in keys:
                self.delete_step_phase(space_column.container[k], space_coord,
                                       Coord(time_coord + (k,)))

    def delete_pi(self, st_point, space_coord, time_coord):
        keys = [k for k, v in st_point if v]
        for key in keys:
            task = st_point.get_time(Coord((key, )))
            if type(task) is dict:
                for v in task.values():
                    self.task_list.append(v)
                    self._context.take_task_out(v.id, MLCoord(
                        space_coord, Coord(time_coord + (key,))))
            else:
                self.task_list.append(task)
                self._context.take_task_out(task.id, MLCoord(
                    space_coord, Coord(time_coord + (key,))))
