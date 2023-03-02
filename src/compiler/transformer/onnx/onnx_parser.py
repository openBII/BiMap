# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.onnx.onnx_basics import ONNXGraph

import onnx


class ONNXParser:
    '''将ONNX文件解析为ONNXGraph对象

    Attributes:
        - graph: ONNXGraph对象
        - readable_file_path: 可读文件的保存路径
    '''

    def __init__(self, onnx_model: onnx.ModelProto, readable_file_path: str = None) -> None:
        self.graph = ONNXGraph(onnx_model.graph)
        if readable_file_path is not None:
            self.graph.record(readable_file_path)
