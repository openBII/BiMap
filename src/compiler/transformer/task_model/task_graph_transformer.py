# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.check.connection_checker import ConnectionChecker
from src.compiler.transformer.check.mapping_checker import MappingChecker
from src.compiler.transformer.onnx.onnx_basics import ONNXGraph
from src.compiler.transformer.opt.conv_optimizer import ConvOptimizer
from src.compiler.transformer.opt.edge_cluster_simplifier import \
    EdgeClusterSimplifier
from src.compiler.transformer.opt.io_node_inserter import IONodeInserter
from src.compiler.transformer.opt.relu_maxpool_merger import ReLUMaxPoolMerger
from src.compiler.transformer.task_model.group_connector import GroupConnector
from src.compiler.transformer.task_model.null_task_eliminator import \
    NullTaskEliminator
from src.compiler.transformer.task_model.op_transform import OpTransformer
from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class TaskGraphTransformer:
    '''将ONNXGraph转换成TaskGraph

    转换必要的步骤:
    1. 算子-计算存储任务组转换
    2. 生成任务组之间连接
    3. 完成占位任务块的转换
    4. 合法性检查

    可选的优化:
    - relu和maxpool的融合
    - 卷积的优化
    - 插入输入输出任务块
    - 简化EdgeCluster的表示
    '''

    def __init__(self, onnx_graph: ONNXGraph,
                 merge_relu_maxpool: bool = True,
                 optimize_conv: bool = True,
                 insert_io_nodes: bool = True,
                 simplify_edge_clusters: bool = True,
                 readable_file_path: str = None
                 ) -> None:
        self.task_graph = TaskGraph()
        self.onnx_graph = onnx_graph
        self.merge_relu_maxpool = merge_relu_maxpool
        self.optimize_conv = optimize_conv
        self.insert_io_nodes = insert_io_nodes
        self.simplify_edge_clusters = simplify_edge_clusters
        self.readable_file_path = readable_file_path

    def transform(self) -> TaskGraph:
        OpTransformer.op_transform(self.task_graph, self.onnx_graph)
        MappingChecker.check(self.task_graph)
        GroupConnector.group_connect(self.task_graph, self.onnx_graph)
        NullTaskEliminator.null_task_eliminate(self.task_graph)
        if self.merge_relu_maxpool:
            ReLUMaxPoolMerger.ccmpb_blocks_merge(self.task_graph)
        if self.optimize_conv:
            ConvOptimizer.conv2d_replace(self.task_graph)
        if self.insert_io_nodes:
            inserter = IONodeInserter(self.task_graph)
            inserter.insert()
        if self.simplify_edge_clusters:
            EdgeClusterSimplifier.simplify(self.task_graph)
        ConnectionChecker.check(self.task_graph)
        if self.readable_file_path is not None:
            self.task_graph.record(self.readable_file_path)
        return self.task_graph
