# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from copy import copy, deepcopy

from numpy import ndarray

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


class STaskBlock(TaskBlock):
    """
    STaskBlock类描述一个存储任务块
    """

    def __init__(self, task_id: int, shape: Shape, task_type: TaskBlockType, precision: Precision, data: ndarray = None):
        # 存储任务块存储的数据
        self._data = data       # type: ndarray
        # 行流水行数
        self._pipeline_num = shape.ny      # tpye: int
        super().__init__(task_id, shape, task_type, precision)

    @property
    def pipeline_num(self) -> int:
        return self._pipeline_num

    @pipeline_num.setter
    def pipeline_num(self, value: int):
        '''
        设置行流水参数，流水数更新后，需要重新构建存储信息
        raises:
            ValueError: 当设置的行流水参数大于任务块y方向大小时，抛出异常
        '''
        if value > self._shape.ny:
            raise ValueError('The pipeline_num {:d} should not be greater than ny {:d}'.format(
                value, self._shape.ny))

        self._pipeline_num = value
        self._construct_storage()

    def _construct_computation(self) -> None:
        self._computation = Computation()

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 16, 2,
                                self._pipeline_num)

    def _construct_clusters(self) -> None:
        '''
        构建存储任务块的边簇结构
        存储任务块部分为一进一出，如SI，部分没有输入边簇有1个输出边簇，如SW

        raises:
            ValueError: 当设置的行流水参数大于任务块y方向大小时，抛出异常
        '''
        self._input_clusters.append(EdgeCluster(self._shape))
        self._output_clusters.append(EdgeCluster(self._shape))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, other):
        assert type(other) is ndarray
        self._data = other


class SICTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SIC, precision, data)

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 16, 3,
                                self._pipeline_num)

    def _construct_clusters(self) -> None:
        # 1进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nr)))
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, 0, self._shape.nr)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SIC(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SICTaskBlock(IDGenerator.get_next_task_id(),
                                      copy(self.shape),
                                      self.precision, data)
        return new_task_block


class SIC2DTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SIC2D, precision, data)

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 16, 1,
                                self._pipeline_num)

    def _construct_clusters(self) -> None:
        # 1进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nr)))
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, 0, self._shape.nr)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SIC2D(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SIC2DTaskBlock(IDGenerator.get_next_task_id(),
                                        copy(self.shape),
                                        self.precision, data)
        return new_task_block


class SIFCTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SIFC, Precision.INT_8, data)
        self._type = TaskBlockType.SIFC

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 16, 3,
                                self._pipeline_num)

    def _construct_clusters(self, type=None):
        # 1进1出
        if type == 1:
            self._input_clusters.append(
                EdgeCluster(Shape(1, 1, self._shape.nr)))
        else:
            self._input_clusters.append(
                EdgeCluster(Shape(0, 0, self._shape.nr)))
        self._output_clusters.append(
            EdgeCluster(Shape(0, 0, 0, self._shape.nr)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SIFC(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SIFCTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision, data)
        return new_task_block


class SITaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SI, precision, data)

    def _construct_storage(self) -> None:
        if self.precision == Precision.INT_32:
            # Axon的输出128B对齐
            self._storage = Storage(self._precision, self._shape, 128, 2,
                                self._pipeline_num)
        else:
            self._storage = Storage(self._precision, self._shape, 16, 2,
                                self._pipeline_num)
        
    def _construct_clusters(self) -> None:
        # 1进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

        # 先不支持二维的情况
        assert (self._shape.ny == 0 and self._shape.nx == 0) or \
               (self._shape.ny != 0 and self._shape.nx != 0)

    def accept(self, visitor):
        visitor.visit_SI(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SITaskBlock(IDGenerator.get_next_task_id(),
                                     copy(self.shape),
                                     self.precision, data)
        return new_task_block


class SWTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SW, precision)
        self._data = []
        if data is not None:
            self._data = data

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 32, 2,
                                self._pipeline_num)

    def _construct_clusters(self) -> None:
        # 0进1出
        self._output_clusters.append(
            EdgeCluster(Shape(0, 0, self._shape.nf, self._shape.nr,
                              self._shape.nky, self._shape.nkx)))

    def _check_clusters_shape(self) -> None:
        assert len(self._output_clusters) == 1
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SW(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SWTaskBlock(IDGenerator.get_next_task_id(),
                                     copy(self.shape),
                                     self.precision, data)
        return new_task_block


class SBTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_32, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SB, precision)
        self._data = data

    def _construct_clusters(self) -> None:
        # 0进1出
        self._output_clusters.append(
            EdgeCluster(Shape(0, 0, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        assert len(self._output_clusters) == 1
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SB(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SBTaskBlock(IDGenerator.get_next_task_id(),
                                     copy(self.shape),
                                     self.precision, data)
        return new_task_block


class SW2DTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SW2D, precision)
        self._data = []
        if data is not None:
            self._data = data

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 32, 2,
                                self._pipeline_num)

    def _construct_clusters(self) -> None:
        # 0进1出
        self._output_clusters.append(
            EdgeCluster(Shape(0, 0, self._shape.nf, self._shape.nr,
                              self._shape.nky, self._shape.nkx)))

    def _check_clusters_shape(self) -> None:
        assert len(self._output_clusters) == 1
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SW2D(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SW2DTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision, data)
        return new_task_block


class SWFCTaskBlock(STaskBlock):
    def __init__(self, task_id: int, shape: Shape, precision: Precision = Precision.INT_8, data: ndarray = None):
        super().__init__(task_id, shape, TaskBlockType.SWFC, precision)
        self._data = data

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 32, 2,
                                self._pipeline_num)  # TODO: 改不了不知道是啥
        # Ref: storage.py

    def _construct_clusters(self) -> None:
        # 0进1出
        self._output_clusters.append(
            EdgeCluster(Shape(0, 0, self._shape.nf, self._shape.nr)))

    def _check_clusters_shape(self) -> None:
        assert len(self._output_clusters) == 1
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_SWFC(self)

    def copy_like(self) -> TaskBlock:
        data = deepcopy(self._data)
        new_task_block = SWFCTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision, data)
        return new_task_block
