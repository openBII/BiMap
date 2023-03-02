# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import math
from inspect import isclass

import torch


class LUT:
    '''查找表生成器

    这个类的功能目前还没有被整合到Transformer中, 待整合后补充说明
    '''

    def __init__(self, function, input_width=8, output_width=8, fp_input_absmax=1, fp_output_absmax=1):
        self.__function = function
        self.__function_instance = None
        if isclass(self.__function):
            if issubclass(self.__function, torch.nn.Module):
                self.__function_instance = self.__function()
        self.__input_max = 2 ** (input_width - 1) - 1
        self.__input_min = -2 ** (input_width - 1)
        self.__output_max = 2 ** (output_width - 1) - 1
        self.__output_min = -2 ** (output_width - 1)
        self.__input_scale = fp_input_absmax / self.__input_max
        self.__output_scale = fp_output_absmax / self.__output_max
        self.__lut_size = 2 ** input_width

    def __call__(self, x) -> int:
        if self.__function_instance is not None:
            x = torch.tensor(x)
            x = x.clamp(self.__input_min, self.__input_max)
            y = 1 / self.__output_scale * \
                self.__function_instance(self.__input_scale * x)
            return math.floor(y.floor().clamp(self.__output_min, self.__output_max).item())
        else:
            if x > self.__input_max:
                x = self.__input_max
            elif x < self.__input_min:
                x = self.__input_min
            y = 1 / self.__output_scale * \
                self.__function(self.__input_scale * x)
            y = math.floor(y)
            if y > self.__output_max:
                return self.__output_max
            elif y < self.__output_min:
                return self.__output_min
            else:
                return y

    def generate(self):
        lut = [0] * self.__lut_size
        for i in range(self.__input_max + 1):
            lut[i] = self.__call__(i)
        for i in range(self.__input_max + 1, self.__lut_size):
            lut[i] = self.__call__(i - self.__lut_size)
        return lut


if __name__ == "__main__":
    import math

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    sigmoid_lut = LUT(function=sigmoid, input_width=4)
    x = -128
    print(sigmoid_lut(x))

    torch_sigmoid_lut = LUT(function=torch.nn.Sigmoid, input_width=4)
    print(torch_sigmoid_lut(x))

    lut_list = sigmoid_lut.generate()
    print(lut_list)
    for i in range(-8, 8):
        print(sigmoid_lut(i))
        print(torch_sigmoid_lut(i))
