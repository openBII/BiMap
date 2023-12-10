#!/usr/bin/env python
# coding: utf-8

"""
STContext类记录TaskModel和STMatrix之间的一些关系
比如一个task当前放到了STMatrix中的哪个位置
"""

from resource_simulator.st_model.st_coord import MLCoord


class STContext():
    def __init__(self):
        # task_id : {ml_coord} or task_id: ml_coord
        self.task_to_coord = {}

    def get_ml_coord(self, task_id):
        assert self.is_task_in_matrix(task_id)
        return self.task_to_coord[task_id]

    def is_task_in_matrix(self, task_id) -> bool:
        return task_id in self.task_to_coord

    def put_task_to(self, place, task_id):
        self.task_to_coord[task_id] = place

    def take_task_out(self, task_id):
        item = self.task_to_coord.get(task_id)
        if item is None:
            raise KeyError('No key name {:d} in st_context!'.format(task_id))
        else:
            self.task_to_coord.pop(task_id)
