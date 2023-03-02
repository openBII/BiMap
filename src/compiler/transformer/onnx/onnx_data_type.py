# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class ONNXDataType(Enum):
    '''ONNX数据类型枚举类

    包含自定义的数据类型
    '''
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16
    TERNERY = 17  # self-defined onnx data type
    INT28 = 18  # self-defined onnx data type

    @staticmethod
    def get_dtype(value: int):
        for dtype in ONNXDataType:
            if dtype.value == value:
                return dtype

    @staticmethod
    def get_dtype_name(value: int):
        return str(ONNXDataType.get_dtype(value))

    @staticmethod
    def is_float(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype in (ONNXDataType.FLOAT, ONNXDataType.COMPLEX64):
            return True
        else:
            return False

    @staticmethod
    def is_int32(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype in (ONNXDataType.UINT8, ONNXDataType.INT8, ONNXDataType.UINT16, ONNXDataType.INT16,
                           ONNXDataType.INT32, ONNXDataType.BOOL, ONNXDataType.FLOAT16, ONNXDataType.BFLOAT16):
            return True
        else:
            return False

    @staticmethod
    def is_string(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype == ONNXDataType.STRING:
            return True
        else:
            return False

    @staticmethod
    def is_int64(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype == ONNXDataType.INT64:
            return True
        else:
            return False

    @staticmethod
    def is_double(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype in (ONNXDataType.DOUBLE, ONNXDataType.COMPLEX128):
            return True
        else:
            return False

    @staticmethod
    def is_uint64(value: int):
        input_dtype = ONNXDataType.get_dtype(value)
        if input_dtype in (ONNXDataType.UINT32, ONNXDataType.UINT64):
            return True
        else:
            return False


if __name__ == '__main__':
    print('Raw data of {:s} cannot be parsed'.format(
        str(ONNXDataType.get_dtype(1))))
