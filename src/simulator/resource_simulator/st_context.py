#!/usr/bin/env python
# coding: utf-8

"""
STContext类记录TaskModel和STMatrix之间的一些关系
比如一个task当前放到了STMatrix中的哪个位置
"""
from typing import Dict, List, Union
from src.simulator.resource_simulator.st_model.st_coord import MLCoord, PathCoord
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.task_block import TaskBlock


class STContext():
    def __init__(self):
        self.task_to_coord: Dict[int, MLCoord] = {}
        self.edge_to_coords: Dict[Edge, PathCoord] = {}

    def get_ml_coord(self, element: Union[int, TaskBlock, Edge]):
        if type(element) is int:
            return self.get_task_ml_coord(element)
        elif isinstance(element, TaskBlock):
            return self.get_task_ml_coord(element.id)
        elif isinstance(element, Edge):
            return self.get_edge_ml_coord(element)
        else:
            raise TypeError("Unsupported element type")
    
    def get_level(self, edge: Edge):
        return self.get_edge_ml_coord(edge).level
    
    def get_container_coord(self, edge: Edge):
        return self.get_edge_ml_coord(edge).outer_coord
    
    def get_network_id(self, edge: Edge):
        return self.get_edge_ml_coord(edge).network_id

    def get_task_ml_coord(self, task_id: int):
        assert self.is_task_in_matrix(task_id)
        return self.task_to_coord[task_id]
    
    def get_edge_ml_coord(self, edge: Edge):
        assert self.is_edge_in_matrix(edge), 'edge not in matrix'
        return self.edge_to_coords[edge]

    def is_task_in_matrix(self, task_id: int) -> bool:
        return task_id in self.task_to_coord
    
    def is_edge_in_matrix(self, edge: Edge) -> bool:
        return edge in self.edge_to_coords

    def put_task_to(self, place: MLCoord, task_id: int):
        self.task_to_coord[task_id] = place

    def put_edge_to(self, path: PathCoord, edge: Edge):
        self.edge_to_coords[edge] = path

    def take_task_out(self, task_id: int):
        item = self.task_to_coord.get(task_id)
        if item is None:
            raise KeyError('No key named {:d} in st_context!'.format(task_id))
        else:
            self.task_to_coord.pop(task_id)

    def take_edge_out(self, edge: Edge):
        path = self.edge_to_coords.get(edge)
        if path is None:
            raise KeyError('No key named {:d} in st_context!'.format(repr(edge)))
        else:
            return self.edge_to_coords.pop(edge)
