# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum

import onnx


class SNNOpType(Enum):
    '''ONNX中自定义的SNN算子的枚举类
    '''
    RectangleFire = 'RectangleFire'
    LIFRecorder = 'LIFRecorder'

    @staticmethod
    def is_snn_op(node: onnx.NodeProto):
        for op in SNNOpType:
            if node.op_type == op.value:
                return True
        return False
