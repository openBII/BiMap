# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from copy import deepcopy
from typing import List

from src.compiler.transformer.task_model.attribute_type import AttributeType
from src.compiler.transformer.task_model.task_graph_basics import (Attribute,
                                                                   CTaskBlock,
                                                                   TaskGraph)
from src.compiler.transformer.task_model.task_graph_eliminator import \
    TaskGraphEliminator
from src.simulator.task_rabbit.task_model.precision import Precision


class ReLUMaxPoolMerger:
    @staticmethod
    def ccmpb_blocks_merge(task_graph: TaskGraph):
        '''将图中ReLU和MaxPool融合'''
        eliminated_blocks_ids = list()
        eliminated_groups_ids = list()
        eliminated_edges_ids = list()
        for c_block_id, _ in task_graph.groups.items():
            c_block = task_graph.get_block(c_block_id)
            op_type = c_block.get_original_node().get_op_type()
            next_c_blocks_ids = task_graph.get_next_c_blocks_ids_of_c_block(
                c_block_id)
            if next_c_blocks_ids is None:
                continue
            if len(next_c_blocks_ids) == 1:
                next_c_block_id = next_c_blocks_ids[0]
                next_c_block = task_graph.get_block(next_c_block_id)
                output_s_block_id = task_graph.get_next_blocks_ids(c_block_id)[
                    0]
                output_s_block = task_graph.get_block(output_s_block_id)
                if output_s_block.get_input_cluster(0).get_num_interfaces() > 1:
                    continue  # 说明发生了Concat操作, 暂不做这种融合
                next_op_type = next_c_block.get_original_node().get_op_type()
                if (op_type == 'Relu' and next_op_type == 'MaxPool'):
                    ReLUMaxPoolMerger.relu_maxpool_merge(
                        task_graph, c_block, next_c_block, eliminated_blocks_ids, eliminated_groups_ids, eliminated_edges_ids)
                if (op_type == 'MaxPool' and next_op_type == 'Relu'):
                    ReLUMaxPoolMerger.maxpool_relu_merge(
                        task_graph, c_block, next_c_block, eliminated_blocks_ids, eliminated_groups_ids, eliminated_edges_ids)
        TaskGraphEliminator.eliminate(task_graph, eliminated_blocks_ids,
                                      eliminated_groups_ids, eliminated_edges_ids)

    @staticmethod
    def relu_maxpool_merge(task_graph: TaskGraph, block: CTaskBlock, next_block: CTaskBlock,
                           eliminated_blocks_ids: List, eliminated_groups_ids: List, eliminated_edges_ids: List):
        cmp = block.get_attribute('cmp').get_value()
        next_cmp = next_block.get_attribute('cmp').get_value()
        next_block.get_attribute('cmp').set_value(max(cmp, next_cmp))
        bit_shift_num = 0
        if 'bit_shift_num' in block.get_attributes().keys():
            bit_shift_num = block.get_attribute('bit_shift_num').get_value()
        else:
            bit_shift_num = next_block.get_attribute(
                'bit_shift_num').get_value()
        if 'bit_shift_num' in next_block.get_attributes().keys():
            next_block.get_attribute('bit_shift_num').set_value(bit_shift_num)
        else:
            new_attr = Attribute(type=AttributeType.BIT_SHIGT_NUM.value,
                                 precision=Precision.INT_32.value, value=bit_shift_num)
            next_block.add_attribute(new_attr, 'bit_shift_num')
        block_input_edge_id = block.get_input_cluster(
            0).get_interface(0).get_id()
        next_block.get_input_cluster(0).get_interface(
            0).set_id(block_input_edge_id)
        edge = task_graph.get_edge(block_input_edge_id)
        edge.set_dst_block_id(next_block.get_id())
        # 修改组信息
        task_graph.groups[next_block.get_id()] = deepcopy(
            task_graph.groups[block.get_id()])
        # 需要删除的对象
        s_block_id = task_graph.get_next_blocks_ids(block.get_id())[0]
        s_block = task_graph.get_block(s_block_id)
        input_edge_id = s_block.get_input_cluster(0).get_interface(0).get_id()
        output_edge_id = s_block.get_output_cluster().get_interface(0).get_id()
        eliminated_blocks_ids.append(s_block_id)
        eliminated_blocks_ids.append(block.get_id())
        eliminated_edges_ids.append(input_edge_id)
        eliminated_edges_ids.append(output_edge_id)
        eliminated_groups_ids.append(block.get_id())

    @staticmethod
    def maxpool_relu_merge(task_graph: TaskGraph, block: CTaskBlock, next_block: CTaskBlock,
                           eliminated_blocks_ids: List, eliminated_groups_ids: List, eliminated_edges_ids: List):
        cmp = block.get_attribute('cmp').get_value()
        next_cmp = next_block.get_attribute('cmp').get_value()
        block.get_attribute('cmp').set_value(max(cmp, next_cmp))
        bit_shift_num = 0
        if 'bit_shift_num' in block.get_attributes().keys():
            bit_shift_num = block.get_attribute('bit_shift_num').get_value()
        else:
            bit_shift_num = next_block.get_attribute(
                'bit_shift_num').get_value()
        if 'bit_shift_num' in block.get_attributes().keys():
            block.get_attribute('bit_shift_num').set_value(bit_shift_num)
        else:
            new_attr = Attribute(type=AttributeType.BIT_SHIGT_NUM.value,
                                 precision=Precision.INT_32.value, value=bit_shift_num)
            block.add_attribute(new_attr, 'bit_shift_num')
        next_block_output_edge_id = next_block.get_output_cluster().get_interface(0).get_id()
        block.get_output_cluster().get_interface(0).set_id(next_block_output_edge_id)
        edge = task_graph.get_edge(next_block_output_edge_id)
        edge.set_src_block_id(block.get_id())
        # 需要删除的对象
        s_block_id = task_graph.get_next_blocks_ids(block.get_id())[0]
        s_block = task_graph.get_block(s_block_id)
        input_edge_id = s_block.get_input_cluster(0).get_interface(0).get_id()
        output_edge_id = s_block.get_output_cluster().get_interface(0).get_id()
        eliminated_blocks_ids.append(s_block_id)
        eliminated_blocks_ids.append(next_block.get_id())
        eliminated_edges_ids.append(input_edge_id)
        eliminated_edges_ids.append(output_edge_id)
        eliminated_groups_ids.append(next_block.get_id())
