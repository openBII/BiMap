# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class ONNXAttributeDataType(Enum):
    '''ONNX中的属性的数据类型枚举类
    '''
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    SPARSE_TENSOR = 6
    TYPE_PROTO = 7
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSORS = 12
    TYPE_PROTOS = 14

    @staticmethod
    def get_dtype(value: int):
        for dtype in ONNXAttributeDataType:
            if dtype.value == value:
                return dtype
            
    @staticmethod
    def get_dtype_name(value: int):
        return str(ONNXAttributeDataType.get_dtype(value))