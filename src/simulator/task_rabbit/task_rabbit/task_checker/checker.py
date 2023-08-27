# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


from .data_checker import DataChecker
from .edge_size_checker import EdgeSizeChecker
from .io_shape_checker import IOShapeChecker
from .scs_checker import SCSChecker
from .shape_checker import ShapeChecker
from .unique_id_checker import UniqueIDChecker


class TaskChecker():
    """
    TaskChecker类对TaskGraph的一系列合法性进行检查
    """
    @staticmethod
    def check_basic(graph):
        # data_checker = DataChecker()
        # graph.accept(data_checker)
        # assert data_checker.passed

        edge_size_checker = EdgeSizeChecker()
        graph.accept(edge_size_checker)
        assert edge_size_checker.passed

        IO_shape_checker = IOShapeChecker()
        graph.accept(IO_shape_checker)
        assert IO_shape_checker.passed

        scs_checker = SCSChecker()
        graph.accept(scs_checker)
        assert scs_checker.passed

        unique_id_checker = UniqueIDChecker()
        graph.accept(unique_id_checker)
        assert unique_id_checker.passed

        task_shape_checker = ShapeChecker()
        graph.accept(task_shape_checker)
