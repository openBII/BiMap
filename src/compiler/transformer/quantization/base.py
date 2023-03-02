# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from src.compiler.transformer.quantization.grad import (FakeQuantize,
                                                        FakeQuantizeFloor,
                                                        FakeQuantizeINT32)
from src.compiler.transformer.quantization.utils import (get_int8_tensor,
                                                         setup_random_seed)


class QModule(ABC):
    activation_absmax = 1  # default

    def __init__(self):
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = False
        self.restricted = False
        self.bit_shift_unit = None

    @abstractmethod
    def collect_q_params(self):
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = True

    @abstractmethod
    def quantize(self):
        assert not(self.quantization_mode), 'Model has been quantized'
        self.quantization_mode = True
        self.aware_mode = False
        assert self.q_params_ready, 'Quantization cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def aware(self):
        self.aware_mode = True
        assert self.q_params_ready, 'QAT cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def dequantize(self):
        self.quantization_mode = False

    @abstractmethod
    def restrict(self):
        self.restricted = True
        assert not(self.quantization_mode)


class QModel(QModule, torch.nn.Module):
    def __init__(self, bit_shift_unit=2, activation_absmax=1):
        QModule.__init__(self)
        torch.nn.Module.__init__(self)
        self.bit_shift_unit = bit_shift_unit  # TianjicX1硬件参数
        self.pretrained = False
        QModule.activation_absmax = activation_absmax  # 网络中激活值的范围

    def collect_q_params(self):
        if not(self.pretrained):
            logging.warning(
                'Collecting quantization parameters usually requires a pretrained model')
        QModule.collect_q_params(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.collect_q_params(self.bit_shift_unit)

    def load_model(self, model_path: str, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
        self.pretrained = True

    def save_quantized_model(self, checkpoint_path: str, others: Dict = None):
        q_params_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                q_params_dict[name] = module.bit_shift
        checkpoint = {
            'model': self.state_dict(),
            'q_params': q_params_dict
        }
        if others is not None:
            checkpoint.update(others)
        torch.save(checkpoint, checkpoint_path)

    def load_quantized_model(self, checkpoint_path: str, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict:
        self.q_params_ready = True
        self.quantization_mode = True
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model']
        self.load_state_dict(state_dict)
        q_params_dict = checkpoint['q_params']
        last_module = None
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.bit_shift = q_params_dict[name]
                module.q_params_ready = True
                module.quantization_mode = True
                module.weight_scale = 2 ** module.bit_shift
                last_module = module
        last_module.is_last_node = True
        self.pretrained = True
        return checkpoint

    def quantize(self, keep_output_precision=True):
        QModule.quantize(self)
        last_module = None
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.quantize()
                last_module = module
        if keep_output_precision:
            last_module.is_last_node = True

    def aware(self):
        QModule.activation_absmax = 1
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.aware()

    def dequantize(self):
        QModule.dequantize(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.dequantize()

    def restrict(self):
        QModule.restrict(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.restrict(self.bit_shift_unit)
        QModule.activation_absmax = 1

    # 执行模型的完整流程
    # 是否随机输入
    # 是的话是否固定随机数种子
    # 否的话需要给出输入文件的路径
    # 是否导入预训练模型
    # 是的话给出预训练模型文件路径
    # 是否生成ONNX文件
    # 是的话给出ONNX文件路径
    # 如果相应路径文件为None, 则意味着该选项为False
    def execute(self, is_random_input=True, fix_random_seed=True, input_data_path=None,
                pre_model_path=None, q_params_path=None, result_path=None, export_onnx_path=None):
        # 模型执行的基本准备
        if fix_random_seed:
            random_seed = sum(ord(c) for c in self.model_name)
            setup_random_seed(random_seed)

        # 创建输入数据
        x = None
        if input_data_path is not None:
            pass
        if is_random_input:
            x = get_int8_tensor(self.input_shape)

        # 量化相关设置
        self.collect_q_params()  # 设置量化参数
        self.quantize()   # 将模型置于量化模式

        # 加载预训练的量化模型
        if pre_model_path is not None and q_params_path is not None:
            self.load_quantized_model(
                model_path=pre_model_path,
                q_params_path=q_params_path,
                device=torch.device('cpu')
            )

        # 推理模型产生结果
        y = self._forward(x)
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            y.detach().numpy().tofile(result_path)

        # 导出ONNX模型
        if export_onnx_path is not None:
            os.makedirs(os.path.dirname(export_onnx_path), exist_ok=True)
            from src.compiler.transformer.onnx.onnx_exporter import onnx_export
            onnx_export(model=self, input=x, output_path=export_onnx_path)


class QConv2d(QModule, torch.nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None,
            is_last_node=False):
        torch.nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        QModule.__init__(self)
        self.weight_scale = None
        self.bit_shift = None
        self.is_last_node = is_last_node

    def collect_q_params(self, bit_shift_unit):
        '''卷积中计算量化参数
        weight_absmax * weight_scale = 128
        weight_scale = 2^(bit_shift_unit * n)
        bit_shift = bit_shift_unit * n = log_2 (128 / weight_absmax)
        n = round(log_2 (128 / weight_absmax) / bit_shift_unit)
        这里取整方法可以有很多
        '''
        QModule.collect_q_params(self)
        weight_absmax = self.weight.data.abs().max()
        temp = math.log(128 / weight_absmax, 2) / bit_shift_unit
        if temp - math.floor(temp) >= 0.75:  # 经验公式
            n = math.ceil(temp)
        else:
            n = math.floor(temp)
        self.bit_shift = bit_shift_unit * n
        self.weight_scale = 2 ** self.bit_shift

    def forward(self, x):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
        out = torch.nn.Conv2d.forward(self, x)
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-2147483648,
                            2147483647).div(self.weight_scale).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        self.bias.data = self.bias.data.mul(
            self.weight_scale * 128 / QModule.activation_absmax).round().clamp(-2147483648, 2147483647)  # INT32

    def dequantize(self):
        QModule.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        self.bias.data = self.bias.data.div(
            self.weight_scale * 128 / QModule.activation_absmax)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        self.bias.data = FakeQuantizeINT32.apply(
            self.bias.data, self.weight_scale * 128 / QModule.activation_absmax)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit


class QLinear(QModule, torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, is_last_node=False):
        torch.nn.Linear.__init__(
            self, in_features, out_features, bias, device, dtype)
        QModule.__init__(self)
        self.weight_scale = None
        self.bit_shift = None
        self.is_last_node = is_last_node

    def collect_q_params(self, bit_shift_unit):
        '''卷积中计算量化参数
        weight_absmax * weight_scale = 128
        weight_scale = 2^(bit_shift_unit * n)
        bit_shift = bit_shift_unit * n = log_2 (128 / weight_absmax)
        n = round(log_2 (128 / weight_absmax) / bit_shift_unit)
        这里取整方法可以有很多
        '''
        QModule.collect_q_params(self)
        weight_absmax = self.weight.data.abs().max()
        temp = math.log(128 / weight_absmax, 2) / bit_shift_unit
        if temp - math.floor(temp) >= 0.75:  # 经验公式
            n = math.ceil(temp)
        else:
            n = math.floor(temp)
        self.bit_shift = bit_shift_unit * n
        self.weight_scale = 2 ** self.bit_shift

    def forward(self, x):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
        out = torch.nn.Linear.forward(self, x)
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-2147483648,
                            2147483647).div(self.weight_scale).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        self.bias.data = self.bias.data.mul(
            self.weight_scale * 128 / QModule.activation_absmax).round().clamp(-2147483648, 2147483647)  # INT32

    def dequantize(self):
        QModule.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        self.bias.data = self.bias.data.div(
            self.weight_scale * 128 / QModule.activation_absmax)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        self.bias.data = FakeQuantizeINT32.apply(
            self.bias.data, self.weight_scale * 128 / QModule.activation_absmax)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit


class QAdd(QModule, torch.nn.Module):
    def __init__(self, is_last_node=False):
        torch.nn.Module.__init__(self)
        QModule.__init__(self)
        self.bit_shift = None
        self.is_last_node = is_last_node

    def collect_q_params(self, _):
        QModule.collect_q_params(self)
        self.bit_shift = 0

    def forward(self, x, y):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
            y = y.clamp(-QModule.activation_absmax, QModule.activation_absmax)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
            y = FakeQuantizeFloor.apply(y, 128 / QModule.activation_absmax)
        out = x + y
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-2147483648,
                            2147483647).div(1).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)

    def dequantize(self):
        QModule.dequantize(self)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit


class QAdaptiveAvgPool2d(QModule, torch.nn.AdaptiveAvgPool2d):
    '''
    目前只考虑了整个模型中只出现一个平均池化
    多个平均池化的公式需要仔细推导一下
    '''

    def __init__(self, output_size, input_size, is_last_node=False):
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size)
        QModule.__init__(self)
        self.bit_shift = None
        self.absmax = None
        self.scale = None  # 指的是输出的scale
        self.input_size = input_size
        self.is_last_node = is_last_node

    def collect_q_params(self, bit_shift_unit):
        '''平均池化计算量化参数
        y = (x1 + x2 + ... + x_input_size) / input_size^2
        128y = (128x1 + 128x2 + ... + 128x_input_size) / input_size^2
        n = round((log_2 input_size^2) / 2)
        bit_shift = n * bit_shift_unit
        128 * input_size^2 / 2^bit_shift y = (128x1 + 128x2 + ... + 128x_input_size) / 2^bit_shift
        '''
        QModule.collect_q_params(self)
        self.bit_shift = bit_shift_unit * round(math.log(self.input_size, 2))
        self.scale = 128 * self.input_size ** 2 / 2 ** self.bit_shift
        self.absmax = 2 ** self.bit_shift / self.input_size ** 2

    def forward(self, x):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
            QModule.activation_absmax = self.absmax
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
            QModule.activation_absmax = self.absmax
        out = torch.nn.AdaptiveAvgPool2d.forward(self, x)
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.mul(self.input_size ** 2).clamp(-2147483648, 2147483647).div(2 **
                                                                                   self.bit_shift).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)

    def dequantize(self):
        QModule.dequantize(self)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit
        self.bit_shift = bit_shift_unit * round(math.log(self.input_size, 2))
        self.absmax = 2 ** self.bit_shift / self.input_size ** 2


class Flatten3d(torch.nn.Module):
    def __init__(self) -> None:
        super(Flatten3d, self).__init__()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # [N, H, W, C] -> [N, H * W * C]
        x = x.contiguous().view(x.size(0), -1)
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 3, 3))
    qconv = QConv2d(3, 8, 3)
    conv = torch.nn.Conv2d(3, 8, 3)
    print(conv.weight.data.abs().max())
    qconv.load_state_dict(conv.state_dict())
    qconv.collect_q_params(2)
    print(qconv.weight_scale)
    print(qconv.bit_shift)
    qx = x / x.abs().max() * 128
    qx = qx.round().clamp(-128, 127)
    qconv.quantize()
    qy = qconv(qx)
    print(qy)
    qconv.aware()
    y = qconv(x / x.abs().max())
    print(y * 128)
