# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class Rectangle(torch.autograd.Function):

    window_size = 1

    @staticmethod
    def forward(ctx, v3: torch.Tensor, v_th) -> torch.Tensor:
        ctx.save_for_backward(v3, torch.as_tensor(v_th, device=v3.device))
        out = (v3 > v_th).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v3, v_th = ctx.saved_tensors
        mask = torch.abs(v3 - v_th) < Rectangle.window_size / 2
        return grad_output * mask.float() * 1 / Rectangle.window_size, None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, v_th0) -> torch._C.Value:
        # return g.op("snn::LIF", input, v_th0_f=v_th0).setType(input.type().with_sizes(_get_tensor_sizes(input)))
        return g.op("snn::RectangleFire", input, v_th0_f=v_th0)
