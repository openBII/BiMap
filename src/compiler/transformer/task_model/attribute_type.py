# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class AttributeType(Enum):
    '''Task Graph中的Task Block的属性枚举类
    '''
    KERNEL_X = 0
    KERNEL_Y = 1
    STRIDE_X = 2
    STRIDE_Y = 3
    PAD_TOP = 4
    PAD_DOWN = 5
    PAD_LEFT = 6
    PAD_RIGHT = 7
    CMP = 8
    CONSTANT_A = 9
    CONSTANT_B = 10
    DILATION_X = 11
    DILATION_Y = 12
    BIT_SHIGT_NUM = 13
    VTH0 = 14  # LIF+
    VTH_ALPHA = 15
    VTH_BETA = 16
    VTH_INCRE = 17
    VR = 18
    VL = 19
    VLEAKY_ALPHA = 20
    VLEAKY_BETA = 21
    DV = 22
    REF_LEN = 23
    TW_CNT = 24
    VINIT = 25
    TW_LEN = 26
    SEED = 27
    LUT_DW = 28  # LUT
    VTH_ADPT_EN = 29  # LIF+
    VLEAKY_ADPT_EN = 30
    TW_EN = 31
    RESET_MODE = 32
    FIRE_TYPE = 33
    VM_CONST_EN = 34
    VM_CONST = 35
    VTHETA_CONST_EN = 36
    VTHETA_CONST = 37

    @staticmethod
    def get_attribute_type(value: int):
        for atype in AttributeType:
            if atype.value == value:
                return atype

    @staticmethod
    def get_attribute_name(value: int):
        atype = AttributeType.get_attribute_type(value)
        return str(atype)
