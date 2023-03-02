# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class BiasType(Enum):
    '''计算任务块的计算需要bias时的bias类型
    '''
    VECTOR = 1
    CONSTANT = 2
