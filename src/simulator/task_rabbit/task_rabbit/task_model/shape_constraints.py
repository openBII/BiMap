# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from enum import Enum

from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class DimType(Enum):
    ADAPTED = 0
    REQUIRED = 1
    ABSENT = -1


class ShapeConstraints:
    ShapeConstraint = {
        TaskBlockType.SI: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SIC: (DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SIC2D: (DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SW: (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),
        TaskBlockType.SIFC: (DimType.ADAPTED, DimType.ADAPTED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SWFC: (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SB: (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.SW2D: (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),

        TaskBlockType.CADD: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CAVG: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CVVH: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CVM: (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.CC: (DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),
        TaskBlockType.CC2D: (DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),
        TaskBlockType.CAX: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CVS: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CCMPB: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CCMPS: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED),
        TaskBlockType.CLUT: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.CLIF: (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED,
                             DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),

        TaskBlockType.INPUT: (DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
        TaskBlockType.OUTPUT: (DimType.ADAPTED, DimType.ADAPTED,
                               DimType.ADAPTED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)
    }

    InputConstraint = {
        TaskBlockType.SI: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SIC: ((DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SIC2D: ((DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SW: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED), ),
        TaskBlockType.SIFC: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SWFC: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SB: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SW2D: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED), ),

        TaskBlockType.CADD: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CAVG: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CVVH: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CVM: ((DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
                            (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED,
                             DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
                            (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CAX: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                            (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED,
                             DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                            (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CC: ((DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
                           (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED,
                            DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),
                           (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CC2D: ((DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED,
                              DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CVS: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                            (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CCMPB: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CCMPS: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CLUT: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.CLIF: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED,
                              DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED,
                              DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED,
                              DimType.ABSENT, DimType.ABSENT, DimType.ABSENT),
                             (DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT)),
        TaskBlockType.OUTPUT: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
    }

    OutputConstraint = {
        TaskBlockType.SI: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SIC: ((DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SIC2D: ((DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SW: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED), ),
        TaskBlockType.SIFC: ((DimType.ABSENT, DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SWFC: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SB: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.SW2D: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED), ),

        TaskBlockType.CADD: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CAVG: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CVVH: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CVM: ((DimType.ABSENT, DimType.ABSENT, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CAX: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CC: ((DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CC2D: ((DimType.REQUIRED, DimType.REQUIRED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CVS: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CCMPB: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CCMPS: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CLUT: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
        TaskBlockType.CLIF: ((DimType.ADAPTED, DimType.ADAPTED, DimType.REQUIRED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),

        TaskBlockType.INPUT: ((DimType.ADAPTED, DimType.ADAPTED, DimType.ADAPTED, DimType.ABSENT, DimType.ABSENT, DimType.ABSENT), ),
    }
