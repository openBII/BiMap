# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import os
from typing import Tuple, Union

import onnxsim
import torch
from src.compiler.transformer.network_type import NetworkType
from src.compiler.transformer.snn.shape_inference import ShapeInference

import onnx


class ONNXExporter:
    '''将量化后的PyTorch模型导出成ONNX模型

    包括以下几个步骤:
    1. 加载预训练模型
    2. 导出ONNX模型
    3. 对ONNX模型进行优化:
        - ANN: 直接调用onnxsim
        - SNN: 由于存在自定义算子无法调用onnxsim且无法完成形状推理, 会调用ShapeInference来完成形状推理

    Attributes:
        - model: PyTorch模型
        - output_path: ONNX模型的保存路径
        - model_path: 量化后的预训练模型
        - reserve_control_flow: 导出ONNX模型时是否保留控制流信息
        - network_type: PyTorch网络的类型
    '''

    def __init__(
        self, model: torch.nn.Module,
        output_path: str, model_path: str = None,
        reserve_control_flow: bool = False,
        network_type: NetworkType = NetworkType.ANN
    ) -> None:
        self.model = model
        self.output_path = output_path
        self.model_path = model_path
        self.reserve_control_flow = reserve_control_flow
        self.network_type = network_type

    def export(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]]) -> onnx.ModelProto:
        '''
        Args:
            - inputs: 导出ONNX模型时需要PyTorch模型的输入

        Returns:
            - onnx_model: ONNX模型

        Raises:
            - NotImplementedError: 目前还没有支持HNN
        '''
        if self.network_type == NetworkType.ANN:
            return self.export_ann(inputs)
        elif self.network_type == NetworkType.SNN:
            return self.export_snn(inputs)
        else:
            raise NotImplementedError(
                self.network_type.name + 'has not been supported')

    def export_ann(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]]) -> onnx.ModelProto:
        if self.model_path is not None:
            if hasattr(self.model, 'load_quantized_model'):
                self.model.load_quantized_model(
                    checkpoint_path=self.model_path,
                    device=torch.device('cpu')
                )
            else:
                state_dict = torch.load(self.model_path)
                self.model.load_state_dict(state_dict)
        if self.reserve_control_flow:
            self.model = torch.jit.script(self.model)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        torch.onnx.export(model=self.model, args=inputs, f=self.output_path,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True)
        onnx_model = onnx.load(self.output_path)
        onnx_model, _ = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, self.output_path)
        return onnx_model

    def export_snn(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]]) -> onnx.ModelProto:
        if self.model_path is not None:
            if hasattr(self.model, 'load_quantized_model'):
                self.model.load_quantized_model(
                    checkpoint_path=self.model_path,
                    device=torch.device('cpu')
                )
            else:
                state_dict = torch.load(self.model_path)
                self.model.load_state_dict(state_dict)
        if self.reserve_control_flow:
            self.model = torch.jit.script(self.model)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        torch.onnx.export(model=self.model, args=inputs, f=self.output_path,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=False,
                          custom_opsets={'snn': 1})
        onnx_model = onnx.load(self.output_path)
        shape_infer = ShapeInference(onnx_model)
        onnx_model = shape_infer.model
        onnx.save(onnx_model, self.output_path)
        return onnx_model
