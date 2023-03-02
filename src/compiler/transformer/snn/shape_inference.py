# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import Dict

from onnx.shape_inference import infer_shapes
from src.compiler.transformer.snn.snn_op_type import SNNOpType


class ShapeInference:
    '''SNN的形状推理

    SNN中包含自定义算子无法通过onnx中的infer_shapes函数自动完成形状推理

    Attributes:
        - model: onnx.ModelProto对象
        - snn_op_list: 计算图中包含的属于SNN自定义算子的结点
        - has_shape_dict: 某个数据是否已知形状
    '''

    def __init__(self, onnx_model) -> None:
        self.model = onnx_model
        self.snn_op_list = list()
        self.create_snn_op_list()
        self.has_shape_dict: Dict[str, bool] = {}
        self.infer_shapes()

    def infer_shapes(self):
        '''形状推理

        对当前在snn_op_list中的结点, 
        如果某个结点有输入形状没有输出形状, 就可以完成该结点的形状推理;
        然后调用onnx.infer_shapes完成非自定义算子的形状推理,
        并将已经推理出形状的自定义算子从snn_op_list中删除;
        重复上述过程直到snn_op_list为空

        此方法适用于所有含有自定义算子的ONNX计算图的形状推理
        '''
        if self.has_finished():
            return
        self.model = infer_shapes(self.model)
        while len(self.snn_op_list) != 0:
            self.update_has_shape_dict()
            remove_list = list()
            for node in self.snn_op_list:
                if self.has_shape(node.input[0]) and not self.has_shape(node.output[0]):
                    remove_list.append(node)
                    input = self.get_value_info(node.input[0])
                    output = self.get_value_info(node.output[0])
                    for i, d in enumerate(input.type.tensor_type.shape.dim):
                        if d.HasField('dim_value'):
                            output.type.tensor_type.shape.dim[i].dim_value = d.dim_value
            for node in remove_list:
                self.snn_op_list.remove(node)
            self.model = infer_shapes(self.model)

    def create_snn_op_list(self):
        '''将计算图中所有属于SNN自定义算子的结点加入snn_op_list
        '''
        for node in self.model.graph.node:
            if SNNOpType.is_snn_op(node):
                self.snn_op_list.append(node)

    def update_has_shape_dict(self):
        '''更新当前每个数据是否已知形状的状态
        '''
        for vi in self.model.graph.value_info:
            dims = vi.type.tensor_type.shape.dim
            has_shape = True
            for i, dim in enumerate(dims):
                if not dim.HasField('dim_value'):
                    self.has_shape_dict.update({vi.name: False})
                    has_shape = False
                    break
            if has_shape:
                self.has_shape_dict.update({vi.name: True})
        for vi in self.model.graph.output:
            dims = vi.type.tensor_type.shape.dim
            has_shape = True
            for i, dim in enumerate(dims):
                if not dim.HasField('dim_value'):
                    self.has_shape_dict.update({vi.name: False})
                    has_shape = False
                    break
            if has_shape:
                self.has_shape_dict.update({vi.name: True})

    def has_shape(self, name: str) -> bool:
        '''某个数据是否具有形状

        Args:
            - name: 数据的名字, 计算图中的唯一标识ID
        '''
        return self.has_shape_dict[name]

    def get_value_info(self, name):
        for vi in self.model.graph.value_info:
            if vi.name == name:
                return vi
        for vi in self.model.graph.output:
            if vi.name == name:
                return vi

    def has_finished(self) -> bool:
        '''是否需要进行形状推理

        如果所有数据都已知形状则不需要进行形状推理
        '''
        self.update_has_shape_dict()
        for name in self.has_shape_dict:
            if not self.has_shape_dict[name]:
                return False
        return True
