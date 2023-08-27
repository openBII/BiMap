# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

"""
一些算子的前向推导过程
"""


def lif_forward(v1: torch.Tensor, v2: Union[torch.Tensor, float],
                v_th_0, v_leaky_alpha, v_leaky_beta,
                v_leaky_adpt_en=True, v_reset=0):
    v1 = torch.from_numpy(v1).type(torch.float64)
    VECTOR = True if v1.ndim == 1 else False
    if VECTOR:
        v1 = v1.unsqueeze(0)
    else:
        v1 = v1.permute(2, 0, 1).unsqueeze(0)
    v2 = torch.from_numpy(v2).type(torch.float64)
    if VECTOR:
        v2 = v2.unsqueeze(0)
    else:
        v2 = v2.permute(2, 0, 1).unsqueeze(0)

    if v2 is None:
        v2 = v_reset
    # update
    v3 = v1 + v2
    v3 = v3.clamp(-134217728, 134217727)
    # fire
    spike = (v3 > v_th_0).float()
    v3 = spike * v_reset + (1 - spike) * v3  # 避免inplace操作
    # leaky
    if v_leaky_adpt_en:
        v2 = torch.floor(v_leaky_alpha / 256 * v3) + v_leaky_beta
    else:
        v2 = v3 + v_leaky_beta
    v2 = v2.clamp(-134217728, 134217727)

    if VECTOR:
        return (spike.squeeze(0).detach().numpy().astype(np.int32),
                v2.squeeze(0).detach().numpy().astype(np.int32))
    else:
        return (spike.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.int32),
                v2.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.int32))


def conv_forward(x_with_pad: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: Tuple[int], dilation: Tuple[int]):
    input = torch.from_numpy(x_with_pad)
    input = input.permute(2, 0, 1).unsqueeze(0)
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    output = F.conv2d(input.double(), weight.double(), bias.double(),
                      stride=stride, dilation=dilation)
    return output.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.int32)


def avgpool_forward(x_with_pad, kernel_size, stride):
    input = torch.from_numpy(x_with_pad).type(torch.float64)
    input = input.permute(2, 0, 1).unsqueeze(0)
    output = F.avg_pool2d(
        input, kernel_size=kernel_size, stride=stride)
    output = output * kernel_size[0] * kernel_size[1]
    return output.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.int32)


def maxpool_forward(x_with_pad, kernel_size, stride):
    input = torch.from_numpy(x_with_pad).type(torch.float64)
    input = input.permute(2, 0, 1).unsqueeze(0)
    assert kernel_size[0] == kernel_size[1] and stride[0] == stride[1]
    output = F.max_pool2d(
        input, kernel_size=kernel_size[0], stride=stride[0])
    return output.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.int32)
