# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from collections import OrderedDict
from typing import List

import xlwt

from task_rabbit.task_graph_parser import TaskGraphParser
from task_rabbit.task_model import Shape, TaskBlock
from src.compiler.transformer.onnx.onnx_basics import ONNXData
from top.config import GlobalConfig


class NodeInfo:
    def __init__(self):
        self.node_name: str = ''
        self.node_type: str = ''
        self.shape = Shape()
        self.computation: int = 0
        # prop for proportion
        self.computation_prop: float = 0
        self.back_computation: int = 0
        # prop for proportion
        self.back_computation_prop: float = 0
        self.storage: int = 0
        self.storage_prop: float = 0
        # 输入输出数据
        self.input_data_list: List[ONNXData] = []
        self.output_data_list: List[ONNXData] = []
        # 将每个算子的输出认为是动态数据
        # 整个网络的输入也是动态数据，但该数据不处在任何算子的统计之中
        # 有些情况，如Conv+Relu+bn，动态数据只算一份
        # 所以设置Relu与BN的cal_dynamic为False
        # 多输出认为动态数据存两份
        self.cal_dynamic = True

        # 属性
        # 这里一些常见的属性直接作为成员变量
        self.strides = ''
        self.pads = ''
        self.dilations = ''
        self.alpha = ''
        self.beta = ''
        self.other_attributes = {}


class TaskGraphStatistic():
    def __init__(self, task_graph: TaskBlock):
        # 网络模型整体的统计
        self.total_computation = 0
        self.graph = task_graph
        self.total_back_computation = 0
        self.total_dynamic_storage = 0
        self.total_static_storage = 0
        self.total_number = 0
        self.type_number = {}
        self.type_computation = {}
        self.type_storage = {}
        self.shape_max = {}
        # self.attribute_max = {}

    def statis_nodes(self):
        total_computation = 0
        total_storage = 0
        # 逐结点的统计
        self.node_statistic = OrderedDict()
        for task_id, task_block in self.graph:
            node_info = NodeInfo()
            node_info.node_type = task_block.task_type
            node_info.shape = task_block.shape
            node_info.computation = task_block.get_computation()  # 正常存储任务节点是不需要
            node_info.storage = task_block.get_storage()
            self.node_statistic[task_id] = node_info
            total_computation += node_info.computation
            total_storage += node_info.storage

        self.total_computation = total_computation
        self.total_storage = total_storage

        for _, node_info in self.node_statistic.items():
            node_info.computation_prop = node_info.computation / self.total_computation
            node_info.storage_prop = node_info.storage / self.total_storage

    def statis_model(self):
        for node_info in self.node_statistic.values():
            if node_info.node_type not in self.type_number:
                self.type_number[node_info.node_type] = 0
            self.type_number[node_info.node_type] += 1

            if node_info.node_type not in self.type_computation:
                self.type_computation[node_info.node_type] = 0
            self.type_computation[node_info.node_type] += node_info.computation

            if node_info.node_type not in self.type_storage:
                self.type_storage[node_info.node_type] = 0
            self.type_storage[node_info.node_type] += node_info.storage

            # if node_info.node_type not in self.shape_max:
            #     self.shape_max[node_info.node_type] = node_info.shape
            # else:
            #     self.shape_max[node_info.node_type] = Shape.max_shape(node_info.shape,
            #                                                           self.shape_max[node_info.node_type])

        self.total_number = len(self.node_statistic)


class ExcelGenerator:
    def __init__(self, task_statis: TaskGraphStatistic):
        self.statis = task_statis

    def output_operator_statistic(self, sheet: xlwt.Worksheet):
        first_row = ('算子名', '算子类型', '前向计算量', '前向中计算量占比',
                     '数据量', '数据量占比', 'nix',
                     'niy', 'nr', 'nx', 'ny', 'nf', 'nkx', 'nky')
        for col, value in enumerate(first_row):
            sheet.write(0, col, value)

        row = 1
        for node_name, node_info in self.statis.node_statistic.items():
            sheet.write(row, 0, node_name)
            sheet.write(row, 1, str(node_info.node_type))
            sheet.write(row, 2, node_info.computation)
            sheet.write(row, 3, "{:.3f}".format(
                node_info.computation_prop * 100) + '%')
            sheet.write(row, 4, node_info.storage)
            sheet.write(row, 5, "{:.3f}".format(
                node_info.storage_prop * 100) + '%')
            # sheet.write(row, 6, node_info.static_storage)
            # sheet.write(row, 7, "{:.3f}".format(
            #     node_info.static_storage_prop * 100) + '%')
            # sheet.write(row, 8, node_info.total_storage)
            # sheet.write(row, 9, "{:.3f}".format(
            #     node_info.total_storage_prop * 100) + '%')
            # 如果输入大于3个或输出大于2个会报错
            # data_col = 10
            # data_num = 0
            # for data_info in node_info.input_data_list:
            #     sheet.write(row, data_col, str(data_info.shape))
            #     static = '静态' if data_info.is_static else '动态'
            #     sheet.write(row, data_col + 1, data_info.precision + ' ' + static)
            #     data_col += 2
            #     data_num += 1
            #     if data_num == 4:
            #         break
            # data_col = 18
            # data_num = 0
            # for data_info in node_info.output_data_list:
            #     sheet.write(row, data_col, str(data_info.shape))
            #     # 输出只能是动态，所以不输出动静态信息
            #     sheet.write(row, data_col + 1, data_info.precision)
            #     data_col += 2
            #     data_num += 1
            #     if data_num == 2:
            #         break
            # sheet.write(row, 22, node_info.shape.nbatch)
            sheet.write(row, 6, node_info.shape.nix)
            sheet.write(row, 7, node_info.shape.niy)
            sheet.write(row, 8, node_info.shape.nr)
            sheet.write(row, 9, node_info.shape.nx)
            sheet.write(row, 10, node_info.shape.ny)
            sheet.write(row, 11, node_info.shape.nf)
            sheet.write(row, 12, node_info.shape.nkx)
            sheet.write(row, 13, node_info.shape.nky)
            # sheet.write(row, 31, node_info.shape.n1)
            # sheet.write(row, 32, node_info.shape.n2)
            # sheet.write(row, 33, node_info.shape.n3)
            # sheet.write(row, 34, str(node_info.strides))
            # sheet.write(row, 35, str(node_info.pads))
            # sheet.write(row, 36, str(node_info.dilations))
            # sheet.write(row, 37, str(node_info.alpha))
            # sheet.write(row, 38, str(node_info.beta))
            # other_attr_string = ''
            # for key, value in node_info.other_attributes.items():
            #     other_attr_string += key + ': ' + str(value) + "; "
            # sheet.write(row, 39, other_attr_string)
            row += 1

    def output_operator_back_statistic(self, sheet: xlwt.Worksheet):
        first_row = ('算子名', '算子类型', '前向计算量', '前向中计算量占比')
        for col, value in enumerate(first_row):
            sheet.write(0, col, value)

        row = 1
        for node_name, node_info in self.statis.node_statistic.items():
            row += 1
            sheet.write(row, 0, node_name)
            sheet.write(row, 1, str(node_info.node_type))
            sheet.write(row, 2, node_info.computation)
            sheet.write(row, 3, "{:.3f}%".format(
                node_info.computation_prop * 100))
            # sheet.write(row, 4, node_info.back_computation)
            # sheet.write(row, 5, "{:.3f}%".format(
            #     node_info.back_computation_prop * 100))
            # if node_info.back_computation > 0:
            #     sheet.write(row, 6, "{:.3f}%".format(
            #         node_info.computation / node_info.back_computation))
            # else:
            #     sheet.write(row, 6, "--")

    def output_model_statistic(self, sheet: xlwt.Worksheet):
        first_row = ('算子类型', '算子数', '算子数占比', '前向计算量总量', '前向中计算量占比',
                     '数据总量', '数据量占比')  # , 'stride', 'pad', 'dialation', 'alpha', 'beta')

        for col, value in enumerate(first_row):
            sheet.write(0, col, value)
        row = 1
        for type_name, type_number in self.statis.type_number.items():
            sheet.write(row, 0, str(type_name))
            sheet.write(row, 1, type_number)
            number_prop = "{:.3f}".format(
                100 * type_number / self.statis.total_number) + '%'
            sheet.write(row, 2, number_prop)
            type_computation = self.statis.type_computation[type_name]
            sheet.write(row, 3, type_computation)
            computation_prop = "{:.3f}".format(
                100 * type_computation / self.statis.total_computation) + '%'
            sheet.write(row, 4, computation_prop)

            storage = self.statis.type_storage[type_name]
            sheet.write(row, 5, storage)
            storage_prop = "{:.3f}".format(
                100 * storage / self.statis.total_storage) + '%'
            sheet.write(row, 6, storage_prop)

            # max_shape = self.statis.shape_max[type_name]
            # sheet.write(row, 13, max_shape.nbatch)
            # sheet.write(row, 14, max_shape.nix)
            # sheet.write(row, 15, max_shape.niy)
            # sheet.write(row, 16, max_shape.nr)
            # sheet.write(row, 17, max_shape.nx)
            # sheet.write(row, 18, max_shape.ny)
            # sheet.write(row, 19, max_shape.nf)
            # sheet.write(row, 20, max_shape.nkx)
            # sheet.write(row, 21, max_shape.nky)
            # sheet.write(row, 22, max_shape.n1)
            # sheet.write(row, 23, max_shape.n2)
            # sheet.write(row, 24, max_shape.n3)

            row += 1

        row += 1
        sheet.write(row, 0, '模型汇总')
        sheet.write(row, 1, '总算子数')
        sheet.write(row, 2, '总前向计算')
        # sheet.write(row, 3, '总反向计算')
        sheet.write(row, 3, '总存储')
        row += 1
        sheet.write(row, 1, self.statis.total_number)
        sheet.write(row, 2, self.statis.total_computation)
        # sheet.write(row, 3, self.statis.total_back_computation)
        sheet.write(row, 3, self.statis.total_storage)

    def output_to_excel(self, output_file_path):
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet1 = book.add_sheet('逐算子前向统计')
        # sheet2 = book.add_sheet('逐算子反向统计')
        sheet2 = book.add_sheet('模型整体统计')

        self.output_operator_statistic(sheet1)
        # self.output_operator_back_statistic(sheet2)
        self.output_model_statistic(sheet2)

        book.save(output_file_path)


if __name__ == '__main__':
    # 从task IR读入Task Graph
    task_path = GlobalConfig.Path['test_lib'] + 'task_lib/1N/lenet.task.txt'
    case_name = 'lenet'
    init = TaskGraphParser(
        case_path=task_path, case_name=case_name, ir_type='task')

    # 统计Task Graph信息
    statistic = TaskGraphStatistic(init.task_graph)
    statistic.statis_nodes()
    statistic.statis_model()
    print(statistic.total_computation)
    print(statistic.total_storage)

    # 导出excel表格
    ExcelGenerator(statistic).output_to_excel('temp/test.xls')
