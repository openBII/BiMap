# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class SocketType(Enum):
    '''Socket类型枚举类
    '''
    INPUT = 1
    OUTPUT = 2