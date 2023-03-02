# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import Dict

from src.compiler.transformer.task_model.task_graph_basics import (
    EdgeCluster, EdgeInterface, TaskGraph, init_shape)


class EdgeClusterSimplifier():
    '''对EdgeCluster的信息进行化简

    如果EdgeInterface的position为EdgeCluster的初始位置, size为EdgeCluster的完整形状, 则不记录position和size信息
    由于EdgeCluster的形状可以通过任务块形状推导出来, 所以也可以不记录
    '''

    def __init__(self, task_graph: TaskGraph) -> None:
        EdgeClusterSimplifier.simplify(task_graph)

    @staticmethod
    def simplify(task_graph: TaskGraph):
        for block_id in task_graph.blocks:
            block = task_graph.get_block(block_id)
            for input_cluster in block.input_clusters:
                EdgeClusterSimplifier.simplify_cluster(input_cluster)
            for output_cluster in block.output_clusters:
                EdgeClusterSimplifier.simplify_cluster(output_cluster)

    @staticmethod
    def simplify_cluster(cluster: EdgeCluster):
        for interface in cluster.interfaces:
            if EdgeClusterSimplifier.is_redundant(interface, cluster.shape):
                interface.delete_position()
                interface.delete_size()
        cluster.shape = None

    @staticmethod
    def is_redundant(interface: EdgeInterface, shape):
        position = interface.position
        size = interface.size
        if position == init_shape():
            if size == shape:
                return True
            else:
                return False
        else:
            return False
