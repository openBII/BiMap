# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from copy import copy

from src.simulator.task_rabbit.task_model.computation import Computation
from src.simulator.task_rabbit.task_model.edge_cluster import EdgeCluster
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.shape_constraints import \
    ShapeConstraints
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class InputTaskBlock(TaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision, socket_id: int):
        super().__init__(task_id, shape, TaskBlockType.INPUT, precision)
        self.socket_id = socket_id

    def _construct_clusters(self) -> None:
        # 0进1出
        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx,
                  self._shape.nf, self._shape.nr)))

    def _construct_computation(self) -> None:
        self._computation = Computation(0)

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 32, 0, 1)
        self._storage._local_storage = 0

    def _check_clusters_shape(self) -> None:
        assert len(self._output_clusters) == 1
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_INPUT(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = InputTaskBlock(copy(self.shape),
                                        IDGenerator.get_next_task_id(),
                                        self.precision)
        new_task_block.socket_id = self.socket_id
        return new_task_block
