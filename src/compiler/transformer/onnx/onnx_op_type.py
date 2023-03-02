# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class ONNXOpType(Enum):
    '''ONNX算子类型枚举类
    '''
    Clip = 'Clip'
    Div = 'Div'
    Floor = 'Floor'
    Reshape = 'Reshape'
    Relu = 'Relu'
    MaxPool = 'MaxPool'
    Flatten = 'Flatten'
    Concat = 'Concat'
    Conv = 'Conv'
    Gemm = 'Gemm'
    Add = 'Add'
    GlobalAveragePool = 'GlobalAveragePool'
    Output = 'output'
    AveragePool = 'AveragePool'
    Cut = 'Cut'
    Transpose = 'Transpose'
    LIFRecorder = 'LIFRecorder'
    RectangleFire = 'RectangleFire'
    Mul = 'Mul'
    LIF = 'LIF'
    Sub = 'Sub'
    Constant = 'Constant'

    @staticmethod
    def get_op_type(name: str):
        for op in ONNXOpType:
            if op.value == name:
                return op

    @staticmethod
    def op_change_dtype(name: str):
        if name in (ONNXOpType.Conv.value, ONNXOpType.Gemm.value, ONNXOpType.Add.value,
                    ONNXOpType.GlobalAveragePool.value):
            return True
        else:
            return False

    @staticmethod
    def is_fire(name: str):
        if name in (ONNXOpType.RectangleFire.value, ):
            return True
        else:
            return False

    @staticmethod
    def op_not_change_dtype(name: str):
        if name in (ONNXOpType.Reshape.value, ONNXOpType.Relu.value, ONNXOpType.MaxPool.value, ONNXOpType.Clip.value,
                    ONNXOpType.Div.value, ONNXOpType.Floor.value, ONNXOpType.Flatten.value, ONNXOpType.Concat.value,
                    ONNXOpType.Transpose.value):
            return True
        else:
            return False

    @staticmethod
    def is_spiking_neuron(name: str):
        if name in (ONNXOpType.LIF.value, ):
            return True
        else:
            return False


if __name__ == '__main__':
    print(ONNXOpType.Clip.value)
