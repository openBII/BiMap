#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from resource_simulator.st_model.st_coord import MLCoord, Coord
from resource_simulator.st_model.st_matrix import STMatrix
from task_rabbit.task_model.task_graph import TaskGraph


class ColumnMerger(object):
    def __init__(self):
        pass

    @staticmethod
    def merge_step(step_column_list, space_coord_list, action,
                   time_coord=Coord(())):
        base_space_coord = space_coord_list[0]
        for column, s_c in zip(step_column_list[1:], space_coord_list[1:]):
            keys = [k for k, i in column]
            for k in keys:
                ColumnMerger.merge_phase(
                    base_space_coord, column.get_time(Coord((k, ))), s_c,
                    action, Coord(time_coord + (k,)))

    @staticmethod
    def merge_phase(base_space_coord, phase_column_2,
                    space_coord, action, time_coord):
        keys = [k for k, i in phase_column_2]
        for k in keys:
            ColumnMerger.merge_pi(
                base_space_coord,
                phase_column_2.get_time(Coord((k,))), space_coord, action,
                Coord(time_coord + (k,)))

    @staticmethod
    def merge_pi(base_space_coord, pi_column_2,
                 space_coord, action, time_coord):
        keys = [k for k, v in pi_column_2 if v]
        for key in keys:
            action.move(MLCoord(space_coord, Coord(time_coord + (key, ))),
                        MLCoord(base_space_coord, Coord(time_coord + (key, ))))
