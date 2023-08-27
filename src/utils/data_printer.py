# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import numpy as np
import os
import struct
def p(path, dtype=np.int32):
    with open(path, "rb") as f:
        a = np.fromfile(f, dtype=dtype)
    print(len(a))
    # for i in range(3):
    #     for j in range(3):
    #         print(a[32*(i*3+j)], end=" ")
    #     print()
    a = [struct.pack(">i", x).hex() for x in a]
    print(a)
    # a = np.array([(x>>24) for x in a])
    # print(a.reshape((32,-1)))




if __name__ == "__main__":
    dir = "/data/case2_1phase_move"
    for x in os.listdir(dir):
        try:
            print(x)
            p(dir + '/' + x, dtype=np.int32)
        except:
            pass
