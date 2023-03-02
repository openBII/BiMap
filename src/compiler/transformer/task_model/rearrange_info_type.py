# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class RearrangeInfoType(Enum):
    '''数据重排信息的枚举类
    '''
    IDENTITY = 0
    RESHAPE = 1
    PERMUTE = 2
    ROTATE = 3
    SHIFT = 4
    SCALE = 5
    SHEAR = 6
    REFLECT = 7
    AFFINE = 8
    PROJECT = 9
    SHUFFLE = 10

    @staticmethod
    def get_rearrange_info_type(value: int):
        for rtype in RearrangeInfoType:
            if rtype.value == value:
                return rtype

    @staticmethod
    def get_rearrange_info_name(value: int):
        rtype = RearrangeInfoType.get_rearrange_info_type(value)
        return str(rtype)
