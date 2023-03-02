# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class Constant(Enum):
    '''一些常数的枚举类
    '''
    INT32_MIN = -2147483648
    INT32_MAX = 2147483647
    INT8_MIN = -128
    INT8_MAX = 127
    TERNARY_MAX = 1
    TERNARY_MIN = -1
