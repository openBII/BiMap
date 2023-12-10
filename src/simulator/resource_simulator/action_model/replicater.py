#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.task_block import TaskBlock


class Replicater(object):
    def __init__(self):
        pass

    @staticmethod
    def copy_node(base_node: TaskBlock):
        new_node = base_node.copy_like()
        for ic_old, ic_new in zip(base_node.input_clusters,
                                  new_node.input_clusters):
            for edge_old in ic_old.all_edges:
                edge_pre_info = deepcopy(
                    edge_old.pre_task.get_output_edge_info(edge_old))
                edge_new = Edge(edge_old.pre_task, new_node)
                edge_old.pre_task.get_output_edge_cluster(edge_old).add_edge(
                    edge_pre_info.position, edge_pre_info.size, edge_new)
                edge_post_info = deepcopy(
                    base_node.get_input_edge_info(edge_old))
                ic_new.add_edge(edge_post_info.position, edge_post_info.size,
                                edge_new)
        for oc_old, oc_new in zip(base_node.output_clusters,
                                  new_node.output_clusters):
            for edge_old in oc_old.all_edges:
                edge_post_info = deepcopy(
                    edge_old.post_task.get_input_edge_info(edge_old))
                edge_new = Edge(new_node, edge_old.post_task)
                edge_old.post_task.get_input_edge_cluster(edge_old).add_edge(
                    edge_post_info.position, edge_post_info.size, edge_new)
                edge_pre_info = deepcopy(
                    base_node.get_output_edge_info(edge_old))
                oc_new.add_edge(edge_pre_info.position, edge_pre_info.size,
                                edge_new)
        return new_node
