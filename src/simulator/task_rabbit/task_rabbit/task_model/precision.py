# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from enum import Enum


class Precision(Enum):
    """
    Precision 类负责描述精度
    """
    INT_8 = 0
    UINT_8 = 1
    INT_16 = 2
    UINT_16 = 3
    INT_32 = 4
    UINT_32 = 5
    FLOAT_16 = 6
    FLOAT_32 = 7
    TERNARY = 8
    INT_28 = 9
    UINT_4 = 10

    INT_9 = 100

    @staticmethod
    def get_precision(value: int):
        for precision in Precision:
            if precision.value == value:
                return precision

    @staticmethod
    def get_precision_name(value: int):
        precision = Precision.get_precision(value)
        return str(precision)

    @staticmethod
    def is_float(value: int):
        precision = Precision.get_precision(value)
        if precision in (Precision.FLOAT_16, Precision.FLOAT_32):
            return True
        else:
            return False

    @staticmethod
    def is_int(value: int):
        if Precision.is_int32(value) or Precision.is_uint32(value):
            return True
        else:
            return False

    @staticmethod
    def is_int32(value: int):
        precision = Precision.get_precision(value)
        if precision in (Precision.INT_8, Precision.INT_16, Precision.INT_32, Precision.TERNARY, Precision.INT_28, Precision.INT_9):
            return True
        else:
            return False

    @staticmethod
    def is_uint32(value: int):
        precision = Precision.get_precision(value)
        if precision in (Precision.UINT_8, Precision.UINT_16, Precision.UINT_32, Precision.UINT_4):
            return True
        else:
            return False
