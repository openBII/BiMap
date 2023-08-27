// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "patch.h"
#include <iostream>

device_t DEVICE = CPU;
int64_t sign_cast_64_32(int64_t sum) {
    if (sum > 0x7fffffff) {
        sum = 0x7fffffff;
    } else if (sum <= -2147483648) {
        sum = -2147483648;
    }
    return sum;
}
int32_t int2_t(int32_t x) {
    x &= 0b11;
    if (x == 3)
        x = -1;
    return x;
}
int32_t transType(int32_t &data, const int32_t &type, const int32_t &j) {
    switch (type) {
        case 1:  // int8
            data = (data >> (j * 8)) & 0xff;
            data = int8_t(data);
            break;
        case 2:  // uint8
            data = (data >> (j * 8)) & 0xff;
            data = uint8_t(data);
            break;
        case 3:  // tenery
            data = (data >> (j * 2)) & 0b11;
            data = int2_t(data);
            break;
        default:  // int32
            break;
    }
    return data;
}
