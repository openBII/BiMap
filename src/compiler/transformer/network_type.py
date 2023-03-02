# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class NetworkType(Enum):
    '''神经网络类型枚举类
    '''
    ANN = 1
    SNN = 2
    HNN = 3
