# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


class Computation():
    """
    Computation 类负责评估TaskBlock的计算
    """

    def __init__(self, total_computation=0):
        self.total_computation = total_computation
