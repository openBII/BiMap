# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import math
from copy import copy

from src.simulator.resource_simulator.hardware_rule import \
    HardwareRule
from src.simulator.task_rabbit.task_model.bias_type import BiasType
from src.simulator.task_rabbit.task_model.computation import Computation
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.task_rabbit.task_model.edge_cluster import EdgeCluster
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.shape_constraints import \
    ShapeConstraints
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


def ceil(num):
    return math.ceil(num / 32)


class CTaskBlock(TaskBlock):
    """
    CTaskBlock类描述一个计算任务块
    """

    def __init__(self, task_id: int, shape: Shape,
                 task_type: TaskBlockType, precision: Precision,
                 bias_type: BiasType):
        assert precision is not None
        assert bias_type is not None
        self.bias_type = bias_type
        super().__init__(task_id, shape, task_type, precision)

    def _construct_clusters(self) -> None:
        # 大部分都是一进一出
        self._input_clusters.append(EdgeCluster(self._shape))
        self._output_clusters.append(EdgeCluster(self._shape))

    def _construct_storage(self) -> None:
        self._storage = Storage(self._precision, self._shape, 32, 0, 1)
        self._storage._local_storage = 0


class CADDTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape, n_branch: int,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        self.n_branch = n_branch
        super().__init__(task_id, shape, TaskBlockType.CADD, precision, bias_type)

    def _construct_computation(self) -> None:
        # clk = self.n_branch * self._shape.nx * self._shape.ny * \
        #     ceil(self._shape.nf) * 32  # 做平均池化时n_branches是池化窗宽高乘积
        if self.bias_type == BiasType.VECTOR:
            load_bias_clk = 4
        else:
            load_bias_clk = 1
        load_inputx_clk = 1
        mac_cmp_clk = self.n_branch
        if self._precision == Precision.INT_32:
            nf_group_num = ceil(self._shape.nf / 8)
            write_v_clk = 1
        elif self._precision == Precision.INT_8 or self._precision == Precision.UINT_8:
            nf_group_num = ceil(self._shape.nf / 32)
            write_v_clk = 4
        elif self._precision == Precision.TERNARY:
            nf_group_num = ceil(self._shape.nf / 128)
            write_v_clk = 16
        hardware_loop3_clk = load_bias_clk + load_inputx_clk + mac_cmp_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num * self._shape.ny * self._shape.nx

        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # n+1进1出
        for _ in range(self.n_branch):
            self._input_clusters.append(
                EdgeCluster(Shape(self._shape.ny, self._shape.nx,
                                  self._shape.nf)))  # input

        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(EdgeCluster(
                Shape(0, 0, self._shape.nf)))  # bias

        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx,
                  self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == self.n_branch + 1
        else:
            assert len(self._input_clusters) == self.n_branch
        assert len(self._output_clusters) == 1
        for idx in range(self.n_branch):             # input
            self._input_clusters[idx].shape.check(
                ShapeConstraints.InputConstraint[self._type][0])
        if self.bias_type == BiasType.VECTOR:
            # bias
            self._input_clusters[-1].shape.check(
                ShapeConstraints.InputConstraint[self._type][1])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CADD(self)

    def copy_like(self) -> TaskBlock:
        new_task = CADDTaskBlock(IDGenerator.get_next_task_id(),
                                 copy(self.shape),
                                 self.n_branch,
                                 self.precision,
                                 self.bias_type)
        return new_task


class CAVGTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.NONE):
        super().__init__(task_id, shape, TaskBlockType.CAVG, precision, bias_type)
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.stride_x = 0
        self.stride_y = 0

    def _construct_computation(self) -> None:
        # clk = self.n_branch * self._shape.nx * self._shape.ny * \
        #     ceil(self._shape.nf) * 32  # 做平均池化时n_branches是池化窗宽高乘积
        load_bias_clk = 1
        load_x_clk = 1
        mac_cmp_clk = self._shape.nkx * self._shape.nky
        if self._precision == Precision.INT_32:
            nf_group_num = ceil(self._shape.nf / 8)
            write_v_clk = 1
        elif self._precision == Precision.INT_8 or self._precision == Precision.UINT_8:
            nf_group_num = ceil(self._shape.nf / 32)
            write_v_clk = 4
        elif self._precision == Precision.TERNARY:
            nf_group_num = ceil(self._shape.nf / 128)
            write_v_clk = 16
        hardware_loop3_clk = load_bias_clk + load_x_clk + mac_cmp_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num * self._shape.nx * self._shape.ny
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # Vector: 2进1出
        # Else: 1进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.niy, self._shape.nix, self._shape.nf)))  # input
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(
                EdgeCluster(Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 2
        else:
            assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[1].shape.check(
                ShapeConstraints.InputConstraint[self._type][1])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CAVG(self)

    def copy_like(self) -> TaskBlock:
        new_task = CAVGTaskBlock(IDGenerator.get_next_task_id(),
                                 copy(self.shape),
                                 self.precision,
                                 self.bias_type)
        new_task.pad_up = self.pad_up
        new_task.pad_down = self.pad_down
        new_task.pad_left = self.pad_left
        new_task.pad_right = self.pad_right
        new_task.stride_x = self.stride_x
        new_task.stride_y = self.stride_y
        return new_task


class CVVHTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CVVH, precision, bias_type)

    def _construct_computation(self) -> None:
        # clk = self._shape.nx * self._shape.ny * ceil(self._shape.nf) * 32
        if self.bias_type == BiasType.VECTOR:
            load_bias_clk = 4
        else:
            load_bias_clk = 1
        load_x_clk = 1
        mac_cmp_clk = 1
        write_v_clk = 16
        nf_group_num = ceil(self._shape.nf / 32)
        hardware_loop3_clk = load_bias_clk + load_x_clk + mac_cmp_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num * self._shape.ny * self._shape.nx

        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 3进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))  # X1
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))  # X2
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(
                EdgeCluster(Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 3
        else:
            assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[2].shape.check(
                ShapeConstraints.InputConstraint[self._type][1])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CVVH(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CVVHTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision,
                                       self.bias_type)
        return new_task_block


class CVMTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CVM, precision, bias_type)

    def _construct_computation(self) -> None:
        # clk = self._shape.nr * 2 * ceil(self._shape.nf) * 32
        # 指的是InputX的计算精度
        if self._precision == Precision.INT_8 or self._precision == Precision.UINT_8:
            data_in_16B = 16
        elif self._precision == Precision.TERNARY:
            data_in_16B = 64
        elif self._precision == Precision.INT_32:
            data_in_16B = 4
        nr_group_num = ceil(self._shape.nr / data_in_16B)
        weight_precision = Precision.INT_8
        if weight_precision == Precision.INT_8:
            load_bias_clk = 4
            write_v_clk = 4
            nf_group_num = ceil(self._shape.nf / (256 // 8))
        elif weight_precision == Precision.TERNARY:
            load_bias_clk = 16
            write_v_clk = 16
            nf_group_num = ceil(self._shape.nf / (256 // 2))
        load_inputx_clk = 1
        mac_cmp_load_reuse_clk = data_in_16B * \
            nr_group_num  # MLP模式，L0_num_in_last_row在RTL级没做
        hardware_loop3_clk = load_bias_clk + load_inputx_clk * \
            nr_group_num + mac_cmp_load_reuse_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 3进1出
        self._input_clusters.append(EdgeCluster(
            Shape(0, 0, 0, self._shape.nr)))  # input
        self._input_clusters.append(EdgeCluster(
            Shape(0, 0, self._shape.nf, self._shape.nr)))  # weight
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(EdgeCluster(
                Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(EdgeCluster(
            Shape(0, 0, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 3
        else:
            assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[2].shape.check(
                ShapeConstraints.InputConstraint[self._type][2])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CVM(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CVMTaskBlock(IDGenerator.get_next_task_id(),
                                      copy(self.shape),
                                      self.precision,
                                      self.bias_type)
        return new_task_block


class CCTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CC, precision, bias_type)
        self.dilation_x = 0
        self.dilation_y = 0
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.stride_x = 0
        self.stride_y = 0

    def _construct_computation(self) -> None:
        # clk0 = self._shape.nkx * self._shape.nky * self._shape.nr * 2
        # # 乘法加法各算一次
        # clk = self._shape.nx * self._shape.ny * ceil(
        #     self._shape.nf) * 32 * clk0
        # 指的是InputX的计算精度
        if self._precision == Precision.INT_8 or self._precision == Precision.UINT_8:
            data_in_16B = 16
        elif self._precision == Precision.TERNARY:
            data_in_16B = 64
        elif self._precision == Precision.INT_32:
            data_in_16B = 4
        mac_num_per_group = 32
        mac_group_num = 4
        nr_group_num = ceil(self._shape.nr / data_in_16B)
        nf_group_num = ceil(self._shape.nf / mac_num_per_group)
        load_bias_clk = 4
        load_inputx_reg_num = 4
        if nr_group_num <= load_inputx_reg_num:
            load_inputx_clk = nr_group_num
        else:
            load_inputx_clk = load_inputx_reg_num
        mac_cmp_load_reuse_clk = self._shape.nkx * self._shape.nky * self._shape.nr
        write_v_clk = 16
        hardware_loop3_clk = load_bias_clk + load_inputx_clk + \
            load_inputx_clk * nr_group_num + mac_cmp_load_reuse_clk + write_v_clk
        hardware_loop4_num = ceil(self._shape.nx / mac_group_num)
        clk = hardware_loop3_clk * nf_group_num * hardware_loop4_num * self._shape.ny

        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 3进1出
        self._input_clusters.append(EdgeCluster(
            Shape(self._shape.niy, self.shape.nix, 0, self.shape.nr)))  # input
        self._input_clusters.append(EdgeCluster(
            Shape(0, 0, self._shape.nf, self._shape.nr, self._shape.nky,
                  self._shape.nkx)))  # weight
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(EdgeCluster(
                Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 3
        else:
            assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[2].shape.check(
                ShapeConstraints.InputConstraint[self._type][2])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CC(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CCTaskBlock(IDGenerator.get_next_task_id(),
                                     copy(self.shape),
                                     self.precision,
                                     self.bias_type)
        new_task_block.pad_up = self.pad_up
        new_task_block.pad_down = self.pad_down
        new_task_block.pad_left = self.pad_left
        new_task_block.pad_right = self.pad_right
        new_task_block.stride_x = self.stride_x
        new_task_block.stride_y = self.stride_y
        new_task_block.dilation_x = self.dilation_x
        new_task_block.dilation_y = self.dilation_y
        return new_task_block


class CAXTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CAX, precision, bias_type)

    def _construct_computation(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            load_bias_clk = 4
        else:
            load_bias_clk = 1
        load_x_clk = 1
        mac_cmp_clk = 1
        write_v_clk = 16
        nf_group_num = ceil(self._shape.nf / 32)
        hardware_loop3_clk = load_bias_clk + load_x_clk + mac_cmp_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num * self._shape.ny * self._shape.nx
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 3进1出 A*X+Bias
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))  # input
        self._input_clusters.append(
            EdgeCluster(Shape(0, 0, self._shape.nf)))  # alpha
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(
                EdgeCluster(Shape(0, 0, self._shape.nf)))  # Bias
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 3
        else:
            assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[2].shape.check(
                ShapeConstraints.InputConstraint[self._type][2])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CAX(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CAXTaskBlock(IDGenerator.get_next_task_id(),
                                      copy(self.shape),
                                      self.precision,
                                      self.bias_type)
        return new_task_block


class CC2DTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CC2D, precision, bias_type)
        self.dilation_x = 0
        self.dilation_y = 0
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.stride_x = 0
        self.stride_y = 0

    def _construct_computation(self) -> None:
        # clk0 = self._shape.nkx * self._shape.nky * self._shape.nr * 2
        # # 乘法加法各算一次
        # clk = self._shape.nx * self._shape.ny * ceil(
        #     self._shape.nf) * 32 * clk0
        mac_num_per_group = HardwareRule.PARALLEL_Y
        mac_group_num = HardwareRule.PARALLEL_X
        nf_group_num = ceil(self._shape.nf / mac_num_per_group)
        load_bias_clk = 4
        load_inputx_reg_size = 16
        if self._shape.nx < load_inputx_reg_size:
            load_inputx_clk = 1
        elif self._shape.nx < load_inputx_reg_size * 2:
            load_inputx_clk = 2
        else:
            load_inputx_clk = 3
        mac_cmp_clk = self._shape.nkx * self._shape.nky * self._shape.nr
        write_v_clk = 16
        hardware_loop3_clk = load_bias_clk + load_inputx_clk * \
            self._shape.nky * self._shape.nr + mac_cmp_clk + write_v_clk
        hardware_loop4_num = ceil(self._shape.nx / mac_group_num)
        clk = hardware_loop3_clk * nf_group_num * hardware_loop4_num * self._shape.ny
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 3进1出
        self._input_clusters.append(EdgeCluster(
            Shape(self._shape.niy, self.shape.nix, 0, self.shape.nr)))  # input
        self._input_clusters.append(EdgeCluster(
            Shape(0, 0, self._shape.nf, self._shape.nr, self._shape.nky,
                  self._shape.nkx)))  # weight
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(EdgeCluster(
                Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 3
        else:
            assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[2].shape.check(
                ShapeConstraints.InputConstraint[self._type][2])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CC2D(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CC2DTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision,
                                       self.bias_type)
        new_task_block.pad_up = self.pad_up
        new_task_block.pad_down = self.pad_down
        new_task_block.pad_left = self.pad_left
        new_task_block.pad_right = self.pad_right
        new_task_block.stride_x = self.stride_x
        new_task_block.stride_y = self.stride_y
        new_task_block.dilation_x = self.dilation_x
        new_task_block.dilation_y = self.dilation_y
        return new_task_block


class CVSTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bias_type: BiasType = BiasType.VECTOR):
        super().__init__(task_id, shape, TaskBlockType.CVS, precision, bias_type)
        self.constant_a = 0

    def _construct_computation(self) -> None:
        # clk = self._shape.nx * self._shape.ny * ceil(self._shape.nf) * 32
        if self.bias_type == BiasType.VECTOR:
            load_bias_clk = 4
        else:
            load_bias_clk = 1
        load_x_clk = 1
        mac_cmp_clk = 1
        write_v_clk = 16
        nf_group_num = ceil(self._shape.nf / 32)
        hardware_loop3_clk = load_bias_clk + load_x_clk + mac_cmp_clk + write_v_clk
        clk = hardware_loop3_clk * nf_group_num * self._shape.ny * self._shape.nx
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # a*X+Bias
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))  # input X
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters.append(EdgeCluster(
                Shape(0, 0, self._shape.nf)))  # bias
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        if self.bias_type == BiasType.VECTOR:
            assert len(self._input_clusters) == 2
        else:
            assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        if self.bias_type == BiasType.VECTOR:
            self._input_clusters[1].shape.check(
                ShapeConstraints.InputConstraint[self._type][1])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CVS(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CVSTaskBlock(IDGenerator.get_next_task_id(),
                                      copy(self.shape),
                                      self.precision,
                                      self.bias_type)
        new_task_block.constant_a = self.constant_a
        return new_task_block


class CCMPBTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 bit_shift_num: int = 0):
        self._bit_shift_num = bit_shift_num
        super().__init__(task_id, shape, TaskBlockType.CCMPB, precision, BiasType.NONE)
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.stride_x = 0
        self.stride_y = 0
        self.CMP = 0

        if shape.niy == 0:
            shape.niy = shape.ny
        if shape.nix == 0:
            shape.niy = shape.nx

    @property
    def bit_shift_num(self):
        return self._bit_shift_num

    @bit_shift_num.setter
    def bit_shift_num(self, value):
        self._bit_shift_num = value
        self._construct_computation

    def _construct_computation(self) -> None:
        # f方向多少个16B
        f_align = self._shape.nf
        if self._precision == Precision.TERNARY:
            f_align *= 2
        elif self._precision == Precision.INT_32:
            f_align *= 32
        elif (self._precision == Precision.INT_8) or \
                (self._precision == Precision.UINT_8):
            f_align *= 8
        f_align = math.ceil(math.ceil(f_align / 8) / 16)

        # 共输出多少个16B
        out_num = self.shape.volume * f_align // self.shape.nf

        # 不存在数据精度转换时，每得到16B输出平均需要1+1 = 2clk
        # int32-int8/int8-int2, 每得到16B输出平均需要4+1 = 5clk
        # int32-int2，每得到16B输出平均需要平均需要1+4 = 5clk
        each_num_clock = 2
        if self.bit_shift_num > 0:
            each_num_clock = 5

        self._computation = Computation(each_num_clock*out_num)

    def _construct_clusters(self) -> None:
        # 1进1出
        self._input_clusters.append(EdgeCluster(
            Shape(self._shape.niy, self.shape.nix, self._shape.nf)))  # input
        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CCMPB(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CCMPBTaskBlock(IDGenerator.get_next_task_id(), copy(self.shape),
                                        self.precision)
        new_task_block.pad_up = self.pad_up
        new_task_block.pad_down = self.pad_down
        new_task_block.pad_left = self.pad_left
        new_task_block.pad_right = self.pad_right
        new_task_block.stride_x = self.stride_x
        new_task_block.stride_y = self.stride_y
        new_task_block.CMP = self.CMP
        new_task_block.bit_shift_num = self.bit_shift_num
        return new_task_block


class CCMPSTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8):
        super().__init__(task_id, shape, TaskBlockType.CCMPS, precision, BiasType.NONE)
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.stride_x = 0
        self.stride_y = 0
        self.CMP = 0
        self.bit_shift_num = 0

        if shape.niy == 0:
            shape.niy = shape.ny
        if shape.nix == 0:
            shape.niy = shape.nx

    def _construct_computation(self) -> None:
        self._computation = Computation(0)

    def _construct_clusters(self) -> None:
        # 1进1出
        self._input_clusters.append(EdgeCluster(
            Shape(self._shape.niy, self.shape.nix, self._shape.nf)))  # input
        self._output_clusters.append(EdgeCluster(
            Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 1
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CCMPS(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CCMPSTaskBlock(IDGenerator.get_next_task_id(),
                                        copy(self.shape),
                                        self.precision)
        new_task_block.pad_up = self.pad_up
        new_task_block.pad_down = self.pad_down
        new_task_block.pad_left = self.pad_left
        new_task_block.pad_right = self.pad_right
        new_task_block.stride_x = self.stride_x
        new_task_block.stride_y = self.stride_y
        new_task_block.CMP = self.CMP
        new_task_block.bit_shift_num = self.bit_shift_num
        return new_task_block


class CLUTTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8, input_data_width=8):
        self.input_data_width = input_data_width
        super().__init__(task_id, shape, TaskBlockType.CLUT, precision, BiasType.NONE)
        self.bit_shift_num = 0  # 是 bit_shift_num 对输入数据先截取的

    @property
    def lut_len(self):
        return 2 ** self.input_data_width

    def _construct_computation(self) -> None:
        # 平均查找一个输入得到输出就是1clk，多余的时钟在读写，但是是和数据精度相关
        # 默认：输入int32 输出int8，每得到16B输出，需要21个clk（查找输出16个数，21clk）
        # 输入int32，输出int32，每得到16B输出，需要6clk（查找输出4个数，6clk）
        # 输入int8，输出int8，每得到16B输出，需要18clk（查找输出16个数，18clk）
        # 输入int8，输出int32，每得到64B输出，需要21clk（查找输出16个数，21clk）
        clk = self._shape.ny * self._shape.nx * self._shape.nf * 1.5
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 2进1出
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))  # input
        self._input_clusters.append(
            EdgeCluster(Shape(0, 0, self.lut_len)))  # lut
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        assert len(self._input_clusters) == 2
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CLUT(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CLUTTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision,
                                       self.input_data_width)
        new_task_block.bit_shift_num = self.bit_shift_num
        return new_task_block


class CLIFTaskBlock(CTaskBlock):
    def __init__(self, task_id: int, shape: Shape,
                 precision: Precision = Precision.INT_8,
                 v_theta_const_en=1, vm_const_en=1):
        # 这两个参数会影响input clusters的数量, 只能放在调用父类构造函数之前
        self.vm_const_en = vm_const_en
        self.v_theta_const_en = v_theta_const_en
        super().__init__(task_id, shape, TaskBlockType.CLIF, precision, BiasType.NONE)
        self.bit_shift_num = 0  # 是 bit_shift_num 对输入数据先截取的
        # 基本参数
        self.v_th_0 = None
        self.v_leaky_alpha = 1
        self.v_leaky_beta = 0
        self.v_leaky_adpt_en = False
        # 时间窗相关
        self.tw_en = False
        self.tw_len = 0
        self.tw_cnt = 0
        self.v_init = 0
        # lfsr相关
        self.seed = 0
        self.vm_const = 0
        # v_theta相关
        self.v_theta_const = 0
        self.v_th_adpt_en = 0
        self.v_th_alpha = 1
        self.v_th_beta = 0
        self.v_th_incre = 0
        # 下限饱和
        self.v_l = self.v_th_0
        # 不应期长度
        self.ref_len = 0
        # 复位相关
        self.v_reset = 0
        self.reset_mode = 0
        self.dv = 0
        # 输出类型
        self.fire_type = 5

    def _construct_computation(self) -> None:
        # clk = self._shape.ny * self._shape.nx * \
        #     math.ceil(self._shape.nf / 4) * 7  # 平均每7个时钟处理16B
        read_para_clk = 2
        read_vm_clk = 1
        read_vtheta_clk = 1
        read_v_clk = 1
        read_uin_clk = 1
        cmp_vth = 1
        write_vtheta_clk = 1
        write_v_clk = 1
        process_16B_neuron = read_vm_clk + read_vtheta_clk + read_v_clk + \
            read_uin_clk + cmp_vth + write_vtheta_clk + write_v_clk
        write_s_clk_reuse = 4
        clk = read_para_clk + process_16B_neuron * \
            math.ceil(self._shape.nf / 4) + write_s_clk_reuse
        self._computation = Computation(clk)

    def _construct_clusters(self) -> None:
        # 至少3进1出
        # Integrate
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        # 膜电位
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        # ref_cnt
        self._input_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        if not self.v_theta_const_en:
            self._input_clusters.append(
                EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        if not self.vm_const_en:
            self._input_clusters.append(
                EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))
        # 输出, 一般为spike
        self._output_clusters.append(
            EdgeCluster(Shape(self._shape.ny, self._shape.nx, self._shape.nf)))

    def _check_clusters_shape(self) -> None:
        num_input_clusters = 3
        if not self.v_theta_const_en:
            num_input_clusters += 1
        if not self.vm_const_en:
            num_input_clusters += 1
        assert len(self._input_clusters) == num_input_clusters
        assert len(self._output_clusters) == 1
        self._input_clusters[0].shape.check(
            ShapeConstraints.InputConstraint[self._type][0])
        self._input_clusters[1].shape.check(
            ShapeConstraints.InputConstraint[self._type][1])
        self._input_clusters[2].shape.check(
            ShapeConstraints.InputConstraint[self._type][2])
        if not self.v_theta_const_en:
            self._input_clusters[3].shape.check(
                ShapeConstraints.InputConstraint[self._type][3])
        if not self.vm_const_en:
            self._input_clusters[4].shape.check(
                ShapeConstraints.InputConstraint[self._type][4])
        self._output_clusters[0].shape.check(
            ShapeConstraints.OutputConstraint[self._type][0])

    def accept(self, visitor):
        visitor.visit_CLIF(self)

    def copy_like(self) -> TaskBlock:
        new_task_block = CLIFTaskBlock(IDGenerator.get_next_task_id(),
                                       copy(self.shape),
                                       self.precision,
                                       self.v_theta_const_en,
                                       self.vm_const_en)
        new_task_block.bit_shift_num = self.bit_shift_num
        new_task_block.v_th_0 = self.v_th_0
        new_task_block.v_leaky_alpha = self.v_leaky_alpha
        new_task_block.v_leaky_beta = self.v_leaky_beta
        new_task_block.v_leaky_adpt_en = self.v_leaky_adpt_en
        new_task_block.tw_en = self.tw_en
        new_task_block.tw_len = self.tw_len
        new_task_block.tw_cnt = self.tw_cnt
        new_task_block.v_init = self.v_init
        new_task_block.seed = self.seed
        new_task_block.vm_const = self.vm_const
        new_task_block.v_theta_const = self.v_theta_const
        new_task_block.v_th_adpt_en = self.v_th_adpt_en
        new_task_block.v_th_alpha = self.v_th_alpha
        new_task_block.v_th_beta = self.v_th_beta
        new_task_block.v_th_incre = self.v_th_incre
        new_task_block.v_l = self.v_l
        new_task_block.ref_len = self.ref_len
        new_task_block.v_reset = self.v_reset
        new_task_block.reset_mode = self.reset_mode
        new_task_block.dv = self.dv
        new_task_block.fire_type = self.fire_type
        return new_task_block
