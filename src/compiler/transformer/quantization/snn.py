# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, Tuple, Union

import torch
from spikingjelly.clock_driven import encoding
from src.compiler.transformer.quantization.grad import (DifferentiableFloor,
                                                        FakeQuantize,
                                                        FakeQuantizeFloor,
                                                        FakeQuantizeINT28)
from src.compiler.transformer.quantization.surrogate import Rectangle


class Recorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value):
        # FIXME(huanyu): 这里有个pytorch的bug没有修复, 正常应该通过setType()设置形状, 但shape inference还是会missing
        # 这issue好几个月前就提了pytorch还没有修复烦死了😡
        return g.op("snn::Record", input)


class ResetMode(Enum):
    HARD = 0
    SOFT = 1
    SOFT_CONSTANT = 2


class Refractory(torch.nn.Module):
    def __init__(self, ref_len) -> None:
        super(Refractory, self).__init__()
        self.reset = HardUpdateAfterSpike(value=ref_len)

    def forward(self, ref_cnt: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ref_cnt[ref_cnt > 0] = ref_cnt[ref_cnt > 0] - 1
            ref_cnt = self.reset(ref_cnt, spike)
        return ref_cnt


class ExtendedLIF(torch.nn.Module):
    def __init__(self, v_th0,
                 v_leaky_alpha=1, v_leaky_beta=0,
                 v_reset=0, v_init=None,
                 v_th_alpha=1, v_th_beta=0,
                 v_th_incre=0, v_l=None, dv=0,
                 ref_len=0, reset_mode=ResetMode.HARD,
                 window_size=1):
        self.threshold_accumulate = ThresholdAccumulateWithSaturate(
            v_th0=v_th0, v_l=v_l)
        self.lif = LIFWithTensorThresholdAndResetModeAndRefractory(
            v_leaky_alpha=v_leaky_alpha, v_leaky_beta=v_leaky_beta, reset_mode=reset_mode,
            v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.saturate = Saturate(v_l=v_l)
        self.refractory = Refractory(ref_len=ref_len)
        self.threshold_dynamics = ThresholdDynamics(
            v_th_alpha=v_th_alpha, v_th_beta=v_th_beta, v_th_incre=v_th_incre)

    def forward(self, u_in, v_th_adpt, v=None, ref_cnt=None):
        v_th = self.threshold_accumulate.forward(v_th_adpt)
        spike, v = self.lif.forward(u_in, v_th, v, ref_cnt)
        v = self.saturate.forward(v)
        v_th_adpt = self.threshold_dynamics.forward(v_th_adpt, spike)
        ref_cnt = self.refractory.forward(ref_cnt, spike)
        return spike, v, v_th_adpt, ref_cnt


class QModule(ABC):
    def __init__(self):
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = False
        self.pretrained = False

    def collect_q_params(self):
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = True

    def quantize(self):
        assert not(self.quantization_mode), 'Model has been quantized'
        self.quantization_mode = True
        self.aware_mode = False
        if not(self.q_params_ready):
            logging.warning(
                'Quantization cannot be executed unless quantization parameters have been collected')

    @abstractmethod
    def aware(self):
        self.aware_mode = True
        assert self.q_params_ready, 'QAT cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def dequantize(self):
        self.quantization_mode = False


class QModel(QModule, torch.nn.Module):
    def __init__(self, time_window_size: int = None):
        QModule.__init__(self)
        torch.nn.Module.__init__(self)
        self.T = time_window_size

    def collect_q_params(self):
        if not(self.pretrained):
            logging.warning(
                'Collecting quantization parameters usually requires a pretrained model')
        QModule.collect_q_params(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                if hasattr(module, 'collect_q_params'):
                    module.collect_q_params()

    def calculate_q_params(self):
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                if hasattr(module, 'is_encoder'):
                    if module.is_encoder:
                        module.calculate_q_params()

    def load_model(self, model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        state_dict = torch.load(model_path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.pretrained = True

    def save_quantized_model(self, checkpoint_path, others: Dict = None):
        '''
        保存的checkpoint为一个字典, 默认包含模型本身的state_dict和量化参数, key分别为model和q_params
        '''
        q_params_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                q_params_dict[name] = {}
                q_params_dict[name]['weight_scale'] = module.weight_scale
                if isinstance(module, QLIF):
                    q_params_dict[name]['v_th_0'] = module.if_node.fire.v_th
                    q_params_dict[name]['v_reset'] = module.if_node.reset.value
                    q_params_dict[name]['v_init'] = module.if_node.accumulate.v_init
                    q_params_dict[name]['v_leaky_beta'] = module.v_leaky.beta
                    if module.v_leaky.adpt_en:
                        q_params_dict[name]['v_leaky_alpha'] = module.v_leaky.alpha
                else:
                    if module.is_encoder:
                        q_params_dict[name]['input_scale'] = module.input_scale
        checkpoint = {
            'model': self.state_dict(),
            'q_params': q_params_dict
        }
        if others is not None:
            checkpoint.update(others)
        torch.save(checkpoint, checkpoint_path)

    def load_quantized_model(self, checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.q_params_ready = True
        self.quantization_mode = True
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint['model'])
        q_params_dict = checkpoint['q_params']
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.q_params_ready = True
                module.quantization_mode = True
                module.pretrained = True
                module.weight_scale = q_params_dict[name]['weight_scale']
                if isinstance(module, QLIF):
                    module.if_node.fire.v_th = q_params_dict[name]['v_th_0']
                    module.if_node.reset.value = q_params_dict[name]['v_reset']
                    module.if_node.accumulate.v_init = q_params_dict[name]['v_reset']
                    module.v_leaky.beta = q_params_dict[name]['v_leaky_beta']
                    if 'v_leaky_alpha' in q_params_dict[name]:
                        module.v_leaky.alpha = q_params_dict[name]['v_leaky_alpha']
                else:
                    if module.is_encoder:
                        module.input_scale = q_params_dict[name]['input_scale']
        self.pretrained = True
        return checkpoint

    def quantize(self):
        QModule.quantize(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.quantize()

    def aware(self, *dummy_inputs):
        self.forward(*dummy_inputs)
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

    def refresh(self):
        for _, module in self.named_modules():
            if isinstance(module, QModule):
                if hasattr(module, 'first_time'):
                    module.first_time = True
                    module.freeze = True


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
            is_encoder=False):
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
        self.is_encoder = is_encoder
        if is_encoder:
            self.input_scale = None

    def collect_q_params(self):
        QModule.collect_q_params(self)
        if self.is_encoder:
            self.collecting = True
            self.num_inputs = 0
            self.sum_absmax = 0
        weight_absmax = self.weight.data.abs().max()
        self.weight_scale = 128 / weight_absmax

    def calculate_q_params(self):
        self.collecting = False
        self.input_scale = 128 / (self.sum_absmax / self.num_inputs)

    def forward(self, x: torch.Tensor):
        if self.is_encoder and self.q_params_ready:
            if self.collecting:
                self.num_inputs += 1
                self.sum_absmax += x.data.abs().max()
        if self.is_encoder and self.quantization_mode:
            x = x.mul(self.input_scale).round().clamp(-128, 127)
        out = torch.nn.Conv2d.forward(self, x)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-134217728, 134217727)  # INT28
        if self.is_encoder:
            return out, self.weight_scale * self.input_scale if (self.weight_scale is not None and self.input_scale is not None) else None
        else:
            return out, self.weight_scale

    def quantize(self):
        QModule.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale * self.input_scale).round().clamp(-134217728, 134217727)  # INT28
            else:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale).round().clamp(-134217728, 134217727)  # INT28

    def dequantize(self):
        QModule.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.div(
                    self.weight_scale * self.input_scale)
            else:
                self.bias.data = self.bias.data.div(self.weight_scale)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale * self.input_scale)
            else:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale)


class QLinear(QModule, torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, is_encoder=False):
        torch.nn.Linear.__init__(
            self, in_features, out_features, bias, device, dtype)
        QModule.__init__(self)
        self.weight_scale = None
        self.is_encoder = is_encoder
        if is_encoder:
            self.input_scale = None

    def collect_q_params(self):
        QModule.collect_q_params(self)
        if self.is_encoder:
            self.collecting = True
            self.num_inputs = 0
            self.sum_absmax = 0
        weight_absmax = self.weight.data.abs().max()
        self.weight_scale = 128 / weight_absmax

    def calculate_q_params(self):
        self.collecting = False
        self.input_scale = 128 / (self.sum_absmax / self.num_inputs)

    def forward(self, x: torch.Tensor):
        if self.is_encoder and self.q_params_ready:
            if self.collecting:
                self.num_inputs += 1
                self.sum_absmax += x.data.abs().max()
        if self.is_encoder and self.quantization_mode:
            x = x.mul(self.input_scale).round().clamp(-128, 127)
        out = torch.nn.Linear.forward(self, x)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-134217728, 134217727)  # INT28
        if self.is_encoder:
            return out, self.weight_scale * self.input_scale if (self.weight_scale is not None and self.input_scale is not None) else None
        else:
            return out, self.weight_scale

    def quantize(self):
        QModule.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale * self.input_scale).round().clamp(-134217728, 134217727)  # INT28
            else:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale).round().clamp(-134217728, 134217727)  # INT28

    def dequantize(self):
        QModule.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.div(
                    self.weight_scale * self.input_scale)
            else:
                self.bias.data = self.bias.data.div(self.weight_scale)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale * self.input_scale)
            else:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale)


class Accumulate(torch.nn.Module):
    def __init__(self, v_init) -> None:
        super(Accumulate, self).__init__()
        self.v_init = v_init

    def forward(self, u_in, v=None) -> torch.Tensor:
        if v is None:
            v = torch.full_like(u_in, self.v_init)
        return u_in + v


class QAccumulate(QModule, Accumulate):
    def __init__(self, v_init) -> None:
        QModule.__init__(self)
        Accumulate.__init__(self, v_init)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x, weight_scale: torch.Tensor, v=None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                v = FakeQuantizeINT28.apply(v, weight_scale)
        v = Accumulate.forward(self, x, v)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v = v.clamp(-134217728, 134217727)  # INT28
        return v

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_init = round(self.v_init * self.weight_scale.item())
            if v is not None:
                v = v.mul(self.weight_scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.v_init = self.v_init / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_init = round(
            self.v_init * self.weight_scale.item()) / self.weight_scale.item()


class AccumulateWithRefractory(torch.nn.Module):
    def __init__(self, v_init) -> None:
        super(AccumulateWithRefractory, self).__init__()
        self.v_init = v_init

    def forward(self, u_in, v=None, ref_cnt=None) -> torch.Tensor:
        if v is None:
            v = torch.full_like(u_in, self.v_init)
        if ref_cnt is None:
            ref_cnt = torch.zeros_like(u_in)
        ref_mask = (1 - ref_cnt).clamp(min=0)
        return u_in * ref_mask + v


class QAccumulateWithRefractory(QModule, AccumulateWithRefractory):
    def __init__(self, v_init) -> None:
        QModule.__init__(self)
        AccumulateWithRefractory.__init__(self, v_init)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, u_in, weight_scale: torch.Tensor, v=None, ref_cnt=None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                v = FakeQuantizeINT28.apply(v, weight_scale)
        v = AccumulateWithRefractory.forward(self, u_in, v, ref_cnt)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v = v.clamp(-134217728, 134217727)  # INT28
        return v

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_init = round(self.v_init * self.weight_scale.item())
            if v is not None:
                v = v.mul(self.weight_scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.v_init = self.v_init / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_init = round(
            self.v_init * self.weight_scale.item()) / self.weight_scale.item()


class HardUpdateAfterSpike(torch.nn.Module):
    def __init__(self, value) -> None:
        super(HardUpdateAfterSpike, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        out = spike * self.value + (1 - spike) * x  # 避免inplace操作
        return out


class QHardUpdateAfterSpike(QModule, HardUpdateAfterSpike):
    def __init__(self, value) -> None:
        QModule.__init__(self)
        HardUpdateAfterSpike.__init__(self, value)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, spike: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        x = HardUpdateAfterSpike.forward(self, x, spike)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.value = round(self.value * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.value = self.value / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.value = round(self.value * self.weight_scale.item()
                           ) / self.weight_scale.item()


class SoftUpdateAfterSpike(torch.nn.Module):
    def __init__(self, value=None) -> None:
        super(SoftUpdateAfterSpike, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor, spike: torch.Tensor, update: torch.Tensor = None):
        if self.value is None:
            assert update is not None
            out = x + spike * update
        else:
            out = x + spike * self.value
        return out


class QSoftUpdateAfterSpike(QModule, SoftUpdateAfterSpike):
    def __init__(self, value=None) -> None:
        QModule.__init__(self)
        SoftUpdateAfterSpike.__init__(self, value)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, spike: torch.Tensor, weight_scale: torch.Tensor, update: torch.Tensor = None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        x = SoftUpdateAfterSpike.forward(self, x, spike, update)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                if self.value is not None:
                    self.value = round(self.value * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        if self.value is not None:
            self.value = self.value / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        if self.value is not None:
            self.value = round(
                self.value * self.weight_scale.item()) / self.weight_scale.item()


class Saturate(torch.nn.Module):
    def __init__(self, v_l):
        super(Saturate, self).__init__()
        self.v_l = v_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(min=self.v_l)
        return out


class QSaturate(QModule, Saturate):
    def __init__(self, v_l):
        QModule.__init__(self)
        Saturate.__init__(self, v_l)
        self.scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self.scale = scale
        if self.quantization_mode:
            self._quantize()
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                x = FakeQuantizeINT28.apply(x, scale)
        # forward
        x = Saturate.forward(self, x)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_l = round(self.v_l * self.scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.v_l = self.v_l / self.scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_l = round(self.v_l * self.scale.item()) / self.scale.item()


class ThresholdAccumulate(torch.nn.Module):
    def __init__(self, v_th0) -> None:
        super(ThresholdAccumulate, self).__init__()
        self.v_th0 = v_th0

    def forward(self, v_th_adpt) -> torch.Tensor:
        with torch.no_grad():
            v_th_adpt = torch.as_tensor(v_th_adpt)
            v_th = self.v_th0 + v_th_adpt
        return v_th


class QThresholdAccumulate(QModule, ThresholdAccumulate):
    def __init__(self, v_th0):
        QModule.__init__(self)
        Saturate.__init__(self, v_th0)
        self.scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, v_th_adpt, scale: torch.Tensor) -> torch.Tensor:
        self.scale = scale
        if self.quantization_mode:
            self._quantize(v_th_adpt)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v_th_adpt = FakeQuantizeINT28.apply(v_th_adpt, scale)
        # forward
        v_th = ThresholdAccumulate.forward(self, v_th_adpt)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v_th = v_th.clamp(-134217728, 134217727)  # INT28
        return v_th

    def _quantize(self, v) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_th0 = round(self.v_th0 * self.scale.item())
            v = torch.as_tensor(v)
            v = v.mul(self.scale).round().clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.v_th0 = self.v_th0 / self.scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_th0 = round(self.v_th0 * self.scale.item()) / self.scale.item()


class ThresholdAccumulateWithSaturate(torch.nn.Module):
    def __init__(self, v_th0, v_l) -> None:
        super(ThresholdAccumulateWithSaturate, self).__init__()
        self.accumulate = ThresholdAccumulate(v_th0=v_th0)
        self.saturate = Saturate(v_l=v_l)

    def forward(self, v_th_adpt) -> torch.Tensor:
        with torch.no_grad():
            v_th = self.accumulate(v_th_adpt)
            v_th = self.saturate(v_th)
        return v_th


class QThresholdAccumulateWithSaturate(QModel):
    def __init__(self, v_th0, v_l) -> None:
        QModel.__init__(self)
        self.accumulate = QThresholdAccumulate(v_th0=v_th0)
        self.saturate = QSaturate(v_l=v_l)

    def forward(self, v_th_adpt, scale) -> torch.Tensor:
        with torch.no_grad():
            v_th = self.accumulate.forward(v_th_adpt=v_th_adpt, scale=scale)
            v_th = self.saturate.forward(x=v_th_adpt, scale=scale)
        return v_th


class ThresholdDynamics(torch.nn.Module):
    def __init__(self, v_th_alpha, v_th_beta, v_th_incre, v_th_adpt_en=True) -> None:
        super(ThresholdDynamics, self).__init__()
        self.decay = Leaky(alpha=v_th_alpha, beta=v_th_beta,
                           adpt_en=v_th_adpt_en)
        self.update = SoftUpdateAfterSpike(value=v_th_incre)

    def forward(self, v_th_adpt: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        v_th_adpt = self.decay(v_th_adpt)
        v_th_adpt = self.update(v_th_adpt, spike)
        return v_th_adpt


class QThresholdDynamics(QModel):
    def __init__(self, v_th_alpha, v_th_beta, v_th_incre, v_th_adpt_en=True) -> None:
        QModel.__init__(self)
        self.decay = QLeaky(
            alpha=v_th_alpha, beta=v_th_beta, adpt_en=v_th_adpt_en)
        self.update = QSoftUpdateAfterSpike(value=v_th_incre)

    def forward(self, v_th_adpt: torch.Tensor, spike: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        v_th_adpt = self.decay.forward(x=v_th_adpt, weight_scale=scale)
        v_th_adpt = self.update.forward(
            x=v_th_adpt, spike=spike, weight_scale=scale)
        return v_th_adpt


class Fire(torch.nn.Module):
    def __init__(self, surrogate_function) -> None:
        super(Fire, self).__init__()
        self.surrogate_function = surrogate_function

    def forward(self, v, v_th) -> torch.Tensor:
        spike = self.surrogate_function.apply(v, v_th)
        return spike


class FireWithConstantThreshold(torch.nn.Module):
    def __init__(self, surrogate_function, v_th) -> None:
        super(FireWithConstantThreshold, self).__init__()
        self.surrogate_function = surrogate_function
        self.v_th = v_th

    def forward(self, v) -> torch.Tensor:
        spike = self.surrogate_function.apply(v, self.v_th)
        return spike


class QFireWithConstantThreshold(QModule, FireWithConstantThreshold):
    def __init__(self, surrogate_function, v_th) -> None:
        QModule.__init__(self)
        FireWithConstantThreshold.__init__(self, surrogate_function, v_th)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, v: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        spike = FireWithConstantThreshold.forward(self, v)
        return spike

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_th = round(self.v_th * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.v_th = self.v_th / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_th = round(self.v_th * self.weight_scale.item()
                          ) / self.weight_scale.item()


class IF(torch.nn.Module):
    def __init__(self, v_th, v_reset, v_init=None, window_size=1):
        super(IF, self).__init__()
        self.reset = HardUpdateAfterSpike(value=v_reset)
        self.accumulate = Accumulate(
            v_init=self.reset.value if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = FireWithConstantThreshold(
            surrogate_function=Rectangle, v_th=v_th)

    def forward(self, u_in: torch.Tensor, v: torch.Tensor = None):
        # update
        v_update = self.accumulate(u_in, v)
        # fire
        spike = self.fire(v_update)
        v = self.reset(v_update, spike)
        return spike, v


class QIF(QModel):
    def __init__(self, v_th, v_reset, v_init=None, window_size=1):
        QModel.__init__(self)
        self.reset = QHardUpdateAfterSpike(value=v_reset)
        self.accumulate = QAccumulate(
            v_init=self.reset.value if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = QFireWithConstantThreshold(
            surrogate_function=Rectangle, v_th=v_th)

    def forward(self, u_in: torch.Tensor, scale: torch.Tensor, v: torch.Tensor = None):
        # update
        v_update = self.accumulate.forward(u_in, scale, v)
        # fire
        spike = self.fire.forward(v_update, scale)
        v = self.reset.forward(v_update, spike, scale)
        return spike, v


class ResetAfterSpike(torch.nn.Module):
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None) -> None:
        super(ResetAfterSpike, self).__init__()
        self.reset_mode = reset_mode
        if self.reset_mode == ResetMode.HARD:
            assert v_reset is not None
            self.reset = HardUpdateAfterSpike(value=v_reset)
        elif self.reset_mode == ResetMode.SOFT_CONSTANT:
            assert dv is not None
            self.reset = SoftUpdateAfterSpike(value=-dv)
        else:
            assert self.reset_mode == ResetMode.SOFT, "Invalid reset mode"
            self.reset = SoftUpdateAfterSpike()  # 这里要求update必须已经被量化过

    def forward(self, v: torch.Tensor, spike: torch.Tensor, update: torch.Tensor = None):
        if self.reset_mode == ResetMode.SOFT:
            out = self.reset(v, spike, -update)
        else:
            out = self.reset(v, spike)
        return out


class QResetAfterSpike(QModel):
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None) -> None:
        QModel.__init__(self)
        self.reset_mode = reset_mode
        if self.reset_mode == ResetMode.HARD:
            assert v_reset is not None
            self.reset = QHardUpdateAfterSpike(value=v_reset)
        elif self.reset_mode == ResetMode.SOFT_CONSTANT:
            assert dv is not None
            self.reset = QSoftUpdateAfterSpike(value=-dv)
        else:
            assert self.reset_mode == ResetMode.SOFT, "Invalid reset mode"
            self.reset = SoftUpdateAfterSpike()

    def forward(self, v: torch.Tensor, spike: torch.Tensor, scale: torch.Tensor, update: torch.Tensor = None):
        if self.reset_mode == ResetMode.SOFT:
            out = self.reset.forward(v, spike, -update)
        else:
            out = self.reset.forward(v, spike, weight_scale=scale)
        return out


class Leaky(torch.nn.Module):
    def __init__(self, alpha, beta, adpt_en=True):
        super(Leaky, self).__init__()
        self.alpha = alpha
        self.beta = beta
        assert alpha <= 1
        self.adpt_en = adpt_en

    def forward(self, x: torch.Tensor):
        if self.adpt_en:
            out = self.alpha * x + self.beta
        else:
            out = x + self.beta
        return out


class QLeaky(QModule, Leaky):
    def __init__(self, alpha, beta, adpt_en=True):
        QModule.__init__(self)
        Leaky.__init__(self, alpha=alpha, beta=beta, adpt_en=adpt_en)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        # forward
        if self.adpt_en:
            if self.quantization_mode:
                x = torch.floor(self.alpha * x) + self.beta
            elif self.aware_mode:
                assert not(
                    self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
                x = DifferentiableFloor(
                    self.alpha * x * weight_scale) / weight_scale + self.beta
            else:
                x = self.alpha * x + self.beta
        else:
            x = x + self.beta
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.alpha = round(self.alpha * 256) / 256
                self.beta = round(self.beta * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.beta = self.beta / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.alpha = round(self.alpha * 256) / 256
        self.beta = round(self.beta * self.weight_scale.item()
                          ) / self.weight_scale.item()


class LIF(torch.nn.Module):
    def __init__(self, v_th, v_leaky_alpha, v_leaky_beta, v_reset=0, v_leaky_adpt_en=False, v_init=None, window_size=1):
        super(LIF, self).__init__()
        self.if_node = IF(v_th=v_th, v_reset=v_reset,
                          v_init=v_init, window_size=window_size)
        self.v_leaky = Leaky(alpha=v_leaky_alpha,
                             beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v=None):
        spike, v = self.if_node(u_in, v)
        v = self.v_leaky(v)
        return spike, v


class QLIF(QModel):
    def __init__(self, v_th, v_leaky_alpha, v_leaky_beta, v_reset=0, v_leaky_adpt_en=False, v_init=None, window_size=1):
        QModel.__init__(self)
        self.if_node = QIF(v_th=v_th, v_reset=v_reset,
                           v_init=v_init, window_size=window_size)
        self.v_leaky = QLeaky(alpha=v_leaky_alpha,
                              beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, scale: torch.Tensor, v=None):
        spike, v = self.if_node.forward(u_in, scale, v)
        v = self.v_leaky.forward(v, scale)
        return spike, v


class LIFRecorder(Recorder):
    @staticmethod
    def forward(ctx, input, v_th, v_leaky_alpha, v_leaky_beta, v_reset, v_leaky_adpt_en, v_init, time_window_size):
        return input

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, v_th, v_leaky_alpha, v_leaky_beta, v_reset, v_leaky_adpt_en, v_init, time_window_size):
        return g.op("snn::LIFRecorder", input,
                    v_th_f=v_th, v_leaky_alpha_f=v_leaky_alpha,
                    v_leaky_beta_f=v_leaky_beta, v_reset_f=v_reset,
                    v_leaky_adpt_en_i=v_leaky_adpt_en, v_init_f=v_init,
                    time_window_size_f=time_window_size)


class Neuron(QModel):
    def __init__(self, recorder, T):
        super(Neuron, self).__init__()
        self.recorder = recorder
        self.T = T

    def record(self, x: torch.Tensor) -> torch.Tensor:
        return self.recorder.apply(x)

    def _forward(self, *args):
        return args

    def forward(self, *args):
        return self._forward(self.record(args[0]), *args[1:])


class LIFNeuron(Neuron):
    def __init__(self, T, v_th, v_leaky_alpha, v_leaky_beta, v_reset=0, v_leaky_adpt_en=False, v_init=None, window_size=1):
        Neuron.__init__(self, LIFRecorder, T)
        self.if_node = QIF(v_th=v_th, v_reset=v_reset,
                           v_init=v_init, window_size=window_size)
        self.v_leaky = QLeaky(alpha=v_leaky_alpha,
                              beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def record(self, x: torch.Tensor):
        return self.recorder.apply(
            x,
            self.if_node.fire.v_th,
            self.v_leaky.alpha,
            self.v_leaky.beta,
            self.if_node.reset.value,
            self.v_leaky.adpt_en,
            self.if_node.accumulate.v_init,
            self.T
        )

    def _forward(self, u_in: torch.Tensor, scale: torch.Tensor, v=None):
        spike, v = self.if_node.forward(u_in, scale, v)
        v = self.v_leaky.forward(v, scale)
        return spike, v


class IFWithTensorThresholdAndResetModeAndRefractory(torch.nn.Module):
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None, v_init=None, window_size=1):
        super(IFWithTensorThresholdAndResetModeAndRefractory, self).__init__()
        self.reset = ResetAfterSpike(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv)
        self.accumulate = AccumulateWithRefractory(
            v_init=v_reset if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = Fire(surrogate_function=Rectangle)

    def forward(self, u_in: torch.Tensor, v_th: torch.Tensor, v: torch.Tensor = None, ref_cnt: torch.Tensor = None):
        # update
        v_update = self.accumulate(u_in, v, ref_cnt)
        # fire
        spike = self.fire(v_update, v_th)
        v = self.reset(v_update, spike)
        return spike, v


class QIFWithTensorThresholdAndResetModeAndRefractory(QModel):
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None, v_init=None, window_size=1):
        QModel.__init__(self)
        self.reset = QResetAfterSpike(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv)
        self.accumulate = QAccumulateWithRefractory(
            v_init=v_reset if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = Fire(surrogate_function=Rectangle)

    def forward(self, u_in: torch.Tensor, v_th: torch.Tensor, scale: torch.Tensor, v: torch.Tensor = None, ref_cnt: torch.Tensor = None):
        # update
        v_update = self.accumulate.forward(u_in, scale, v, ref_cnt)
        # fire
        spike = self.fire.forward(v_update, v_th)
        v = self.reset.forward(v_update, spike, scale, v_th)
        return spike, v


class LIFWithTensorThresholdAndResetModeAndRefractory(torch.nn.Module):
    def __init__(self, v_leaky_alpha, v_leaky_beta, reset_mode: ResetMode, v_reset=None, dv=None, v_leaky_adpt_en=False, v_init=None, window_size=1):
        super(LIFWithTensorThresholdAndResetModeAndRefractory, self).__init__()
        self.if_node = IFWithTensorThresholdAndResetModeAndRefractory(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.v_leaky = Leaky(alpha=v_leaky_alpha,
                             beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v_th, v=None, ref_cnt=None):
        spike, v = self.if_node(u_in, v_th, v, ref_cnt)
        v = self.v_leaky(v)
        return spike, v


class QLIFWithTensorThresholdAndResetModeAndRefractory(QModel):
    def __init__(self, v_leaky_alpha, v_leaky_beta, reset_mode: ResetMode, v_reset=None, dv=None, v_leaky_adpt_en=False, v_init=None, window_size=1):
        QModel.__init__(self)
        self.if_node = QIFWithTensorThresholdAndResetModeAndRefractory(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.v_leaky = QLeaky(alpha=v_leaky_alpha,
                              beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v_th, scale, v=None, ref_cnt=None):
        spike, v = self.if_node.forward(u_in, v_th, scale, v, ref_cnt)
        v = self.v_leaky.forward(v, scale)
        return spike, v


class QExtendedLIF(QModel):
    def __init__(self, v_th0,
                 v_leaky_alpha=1, v_leaky_beta=0,
                 v_reset=0, v_init=None,
                 v_th_alpha=1, v_th_beta=0,
                 v_th_incre=0, v_l=None, dv=0,
                 ref_len=0, reset_mode=ResetMode.HARD,
                 window_size=1):
        QModel.__init__(self)
        self.threshold_accumulate = QThresholdAccumulateWithSaturate(
            v_th0=v_th0, v_l=v_l)
        self.lif = QLIFWithTensorThresholdAndResetModeAndRefractory(
            v_leaky_alpha=v_leaky_alpha, v_leaky_beta=v_leaky_beta, reset_mode=reset_mode,
            v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.saturate = QSaturate(v_l=v_l)
        self.refractory = Refractory(ref_len=ref_len)
        self.threshold_dynamics = QThresholdDynamics(
            v_th_alpha=v_th_alpha, v_th_beta=v_th_beta, v_th_incre=v_th_incre)

    def forward(self, u_in, v_th_adpt, scale, v=None, ref_cnt=None):
        v_th = self.threshold_accumulate.forward(v_th_adpt, scale)
        spike, v = self.lif.forward(u_in, v_th, scale, v, ref_cnt)
        v = self.saturate.forward(v, scale)
        v_th_adpt = self.threshold_dynamics.forward(v_th_adpt, spike, scale)
        ref_cnt = self.refractory.forward(ref_cnt, spike)
        return spike, v, v_th_adpt, ref_cnt


class OldQLIF(QModule, LIF):
    def __init__(self, v_th_0, v_leaky_alpha=1, v_leaky_beta=0, v_reset=0, v_leaky_adpt_en=False, v_init=None):
        QModule.__init__(self)
        LIF.__init__(self, v_th_0, v_leaky_alpha, v_leaky_beta,
                     v_reset, v_leaky_adpt_en, v_init)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False

    def collect_q_params(self):
        QModule.collect_q_params(self)

    def forward(self, x, weight_scale, v=None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                v = FakeQuantizeINT28.apply(v, weight_scale)
        spike, v = LIF.forward(self, x, v)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v = v.clamp(-134217728, 134217727)  # INT28
        return spike, v

    def quantize(self):
        QModule.quantize(self)

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained:
                self.if_node.fire.v_th = round(
                    self.if_node.fire.v_th * self.weight_scale.item())
                self.if_node.accumulate.v_init = round(
                    self.if_node.accumulate.v_init * self.weight_scale.item())
                self.if_node.reset.value = round(
                    self.if_node.reset.value * self.weight_scale.item())
                self.v_leaky.beta = round(
                    self.v_leaky.beta * self.weight_scale.item())
                if self.v_leaky.adpt_en:
                    self.v_leaky.alpha = round(self.v_leaky.alpha * 256) / 256
            if v is not None:
                v = v.mul(self.weight_scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.if_node.fire.v_th = self.if_node.fire.v_th / self.weight_scale.item()
        self.if_node.reset.value = self.if_node.reset.value / self.weight_scale.item()
        self.v_leaky.beta = self.v_leaky.beta / self.weight_scale.item()
        self.if_node.accumulate.v_init = self.if_node.accumulate.v_init / \
            self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.if_node.fire.v_th = round(
            self.if_node.fire.v_th * self.weight_scale.item()) / self.weight_scale.item()
        self.if_node.reset.value = round(
            self.if_node.reset.value * self.weight_scale.item()) / self.weight_scale.item()
        self.if_node.accumulate.v_init = round(
            self.if_node.accumulate.v_init * self.weight_scale.item()) / self.weight_scale.item()
        self.v_leaky.beta = round(
            self.v_leaky.beta * self.weight_scale.item()) / self.weight_scale.item()
        if self.v_leaky.adpt_en:
            self.v_leaky.alpha = round(self.v_leaky.alpha * 256) / 256


class HU(torch.nn.Module):
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(HU, self).__init__()
        self.window_size = window_size
        self.window_set = None
        self.window_conv = None
        self.sampler = None
        self.non_linear = non_linear
        self.precision_convert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_set is not None:
            x = self.window_set(x)
        if self.window_conv is not None:
            x = self.window_conv(x)
        if self.sampler is not None:
            x = self.sampler(x)
        if self.non_linear is not None:
            x = self.non_linear(x)
        if self.precision_convert is not None:
            x = self.precision_convert(x)
        return x


class WindowSet(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super(WindowSet, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入特征图形状 [N, ..., T] 最后一个维度为时间
        # 输出特征图形状 [N, ..., num, size]
        t = x.size(-1)
        num = t // self.size
        shape = list(spike.size())
        shape[-1] = num
        shape.append(self.size)
        x = x.unsqueeze(-2).reshape(shape)
        return x


class WindowConv(torch.nn.Module):
    def __init__(self) -> None:
        super(WindowConv, self).__init__()

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        # 输入特征图形状 [N, ..., num, size]
        # 输出特征图形状 [N, ..., T = num * size] 最后一个维度为时间
        num = x.size(-2)
        size = x.size(-1)
        t = num * size
        shape = list(x.size())
        shape[-2] = t
        shape.pop()
        x = x.reshape(shape)
        return x


class AverageWindowConv(WindowConv):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super(AverageWindowConv, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入特征图形状 [N, ..., num, size]
        # 输出特征图形状 [N, ..., T]
        x = self.avgpool(x)
        x = self.reshape(x)
        return x


class GlobalAverageWindowConv(AverageWindowConv):
    def __init__(self, window_size: int) -> None:
        super(GlobalAverageWindowConv, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size=window_size)

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        return x


class LearnableWindowConv(WindowConv):
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: Union[int, Tuple[int]]) -> None:
        super(LearnableWindowConv, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.size())
        batch_size = 1
        for i in range(0, len(shape) - 2):
            batch_size *= shape[i]
        num = x.size(-2)
        size = x.size(-1)
        x = x.reshape(batch_size, num, size)
        x = self.conv(x)
        shape[-1] = x.size(-1)
        x = x.reshape(shape)
        x = self.reshape(x)
        return x


class PrecisionConvert(torch.nn.Module):
    def __init__(self, converter: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(PrecisionConvert, self).__init__()
        self.converter = converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.converter(x)
        return x


class A2SPrecisionConvert(PrecisionConvert):
    def __init__(self, converter: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(A2SPrecisionConvert, self).__init__(converter=converter)


class SignPrecisionConvert(A2SPrecisionConvert):
    def __init__(self) -> None:
        super(SignPrecisionConvert, self).__init__(converter=torch.sign)


class Sampler(torch.nn.Module):
    def __init__(self, window_size) -> None:
        super(Sampler, self).__init__()
        self.window_size = window_size


class LearnableSampler(Sampler):
    def __init__(self, window_size: int) -> None:
        super(LearnableSampler, self).__init__(window_size=window_size)
        self.linear = torch.nn.Linear(1, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.linear(x)
        return x


class RateCodingSampler(Sampler):
    def __init__(self, window_size: int, encoder: torch.nn.Module) -> None:
        super(RateCodingSampler, self).__init__(window_size=window_size)
        self.encoder = encoder

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.size())
        shape.append(self.window_size)
        out = torch.zeros(shape)
        x = self.normalize(x)
        for i in range(self.window_size):
            out[..., i] = self.encoder(x)
        return out


class PoissonSampler(RateCodingSampler):
    def __init__(self, window_size: int) -> None:
        super().__init__(window_size, encoding.PoissonEncoder())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min())


class A2SHU(HU):
    def __init__(self, window_size: int, converter: Callable[[torch.Tensor], torch.Tensor],
                 non_linear: torch.nn.Module = None) -> None:
        super(A2SHU, self).__init__(window_size, non_linear)
        self.precision_convert = A2SPrecisionConvert(converter=converter)

    def check(self):
        assert (self.window_set is None and self.window_conv is None and
                self.sampler is not None and self.precision_convert is not None)


class A2SLearnableCoding(A2SHU):
    def __init__(self, window_size: int, converter: Callable[[torch.Tensor], torch.Tensor],
                 non_linear: torch.nn.Module = None) -> None:
        super(A2SLearnableCoding, self).__init__(
            window_size, converter, non_linear)
        self.sampler = LearnableSampler(window_size=self.window_size)
        self.check()


class A2SLearnableCodingSignConvert(A2SLearnableCoding):
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(A2SLearnableCodingSignConvert, self).__init__(
            window_size, torch.sign, non_linear)


class A2SRateCoding(A2SHU):
    def __init__(self, window_size: int, encoder: torch.nn.Module,
                 converter: Callable[[torch.Tensor], torch.Tensor], non_linear: torch.nn.Module = None) -> None:
        super(A2SRateCoding, self).__init__(window_size, converter, non_linear)
        self.sampler = RateCodingSampler(
            window_size=self.window_size, encoder=encoder)
        self.check()


class A2SPoissonCodingSignConvert(A2SRateCoding):
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(A2SPoissonCodingSignConvert, self).__init__(
            window_size, encoding.PoissonEncoder(), torch.sign, non_linear)


class S2AHU(HU):
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(S2AHU, self).__init__(window_size, non_linear)
        self.window_set = WindowSet(size=window_size)

    def check(self):
        assert (self.window_set is not None and self.window_conv is not None and
                self.sampler is None and self.precision_convert is None)


class S2ARateCoding(S2AHU):
    def __init__(self, window_size: int, kernel_size: int, stride: int, non_linear: torch.nn.Module = None) -> None:
        super(S2ARateCoding, self).__init__(window_size, non_linear)
        self.window_conv = AverageWindowConv(
            kernel_size=kernel_size, stride=stride)
        self.check()


class S2AGlobalRateCoding(S2AHU):
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(S2AGlobalRateCoding, self).__init__(window_size, non_linear)
        self.window_conv = GlobalAverageWindowConv(window_size=window_size)
        self.check()


class S2ALearnableRateCoding(S2AHU):
    def __init__(self, window_size: int, num_windows: int, kernel_size: int,
                 stride: int, padding: Union[int, Tuple[int]], non_linear: torch.nn.Module = None) -> None:
        super(S2ALearnableRateCoding, self).__init__(window_size, non_linear)
        self.window_conv = LearnableWindowConv(
            in_channels=num_windows, kernel_size=kernel_size, stride=stride, padding=padding)
        self.check()


if __name__ == '__main__':
    from spikingjelly.clock_driven import encoding

    class SNN(QModel):
        def __init__(self):
            super(SNN, self).__init__()
            self.linear = QLinear(28 * 28, 10, bias=False)
            self.lif = QLIF(v_th=1, v_leaky_alpha=0.5,
                            v_leaky_beta=0, v_reset=0)

        def forward(self, x: torch.Tensor, v: torch.Tensor = None):
            x = x.view(x.size(0), -1)
            x, q_param = self.linear(x)
            out, v = self.lif(x, q_param, v)
            return out, v

    x = torch.rand(1, 1, 28, 28)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    encoder = encoding.PoissonEncoder().to(device)
    snn = SNN().to(device)
    snn.collect_q_params()
    snn.quantize()
    snn.aware(x)
    length = 2
    v = None
    for _ in range(length):
        x = encoder(x)
        spike, v = snn(x, v)

    x = (x > 0).float()
    lif = OldQLIF(v_th_0=1, v_leaky_alpha=0.9, v_leaky_beta=0.5,
                  v_reset=0.2, v_leaky_adpt_en=True, v_init=0.1)
    new_lif = QLIF(v_th=1, v_leaky_alpha=0.9, v_leaky_beta=0.5,
                   v_reset=0.2, v_leaky_adpt_en=True, v_init=0.1)
    lif.quantize()
    new_lif.quantize()
    scale = torch.as_tensor(100)
    _, v_ref = lif.forward(x, scale)
    _, v = new_lif.forward(x, scale)
    #assert (v_ref - v).abs().mean() < 1

    conv = LearnableWindowConv(
        in_channels=5, kernel_size=3, stride=1, padding=1)
    x = torch.randn(3, 4, 4, 4, 5, 9)
    y = conv(x)
    print(y)
    sampler = PoissonSampler(window_size=2)
    z = sampler(x)
    print(z.shape)
