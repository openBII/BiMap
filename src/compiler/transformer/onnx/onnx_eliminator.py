# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import List

from src.compiler.transformer.onnx.onnx_basics import ONNXGraph


class ONNXEliminator:
    @staticmethod
    def eliminate(graph: ONNXGraph, eliminated_nodes: List[str], eliminated_data: List[str]):
        '''在ONNXGraph对象中中删除一系列结点和数据

        Args:
            - graph: ONNXGraph对象
            - eliminated_nodes: 待删除结点的name
            - eliminated_data: 待删除数据的name
        '''
        eliminated_nodes = list(set(eliminated_nodes))
        eliminated_data = list(set(eliminated_data))
        for node_name in eliminated_nodes:
            del graph.nodes[node_name]
            del graph.input_connections[node_name]
            del graph.output_connections[node_name]
        for data_name in eliminated_data:
            del graph.data[data_name]
