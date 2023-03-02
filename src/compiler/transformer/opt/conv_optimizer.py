# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from src.compiler.transformer.task_model.task_block_type import TaskBlockType
from src.compiler.transformer.task_model.task_graph_basics import TaskGraph


class ConvOptimizer:
    @staticmethod
    def conv2d_replace(task_graph: TaskGraph):
        '''将图中符合要求的CC任务块替换成CC2D任务块

        限制条件:
        - CC的输入通道数小于8
        - CC为输入计算任务块, 即CC的输入数据不由其他计算产生

        会将CC所在的任务组进行替换, 包括CC->CC2D, SIC->SIC2D, SW->SW2D
        '''
        for c_block_id, _ in task_graph.groups.items():
            c_block = task_graph.get_block(c_block_id)
            if c_block.get_type() == TaskBlockType.CC.value and c_block.get_shape()['r'] < 8 and task_graph.is_input(c_block_id):
                c_block.set_type(TaskBlockType.CC2D.value)
                pad_top = c_block.get_attribute('pad_top').get_value()
                pad_down = c_block.get_attribute('pad_down').get_value()
                pad_left = c_block.get_attribute('pad_left').get_value()
                pad_right = c_block.get_attribute('pad_right').get_value()
                c_block.get_attribute('pad_top').set_value(0)
                c_block.get_attribute('pad_down').set_value(0)
                c_block.get_attribute('pad_left').set_value(0)
                c_block.get_attribute('pad_right').set_value(0)
                c_block.shape['iy'] = c_block.shape['iy'] + pad_top + pad_down
                c_block.shape['ix'] = c_block.shape['ix'] + pad_top + pad_down
                last_blocks_ids = task_graph.get_last_blocks_ids(c_block_id)
                for last_block_id in last_blocks_ids:
                    last_block = task_graph.get_block(last_block_id)
                    if last_block.get_type() == TaskBlockType.SIC.value:
                        last_block.set_type(TaskBlockType.SIC2D.value)
                        last_block.shape['y'] = last_block.shape['y'] + \
                            pad_top + pad_down
                        last_block.shape['x'] = last_block.shape['x'] + \
                            pad_left + pad_right
                    if last_block.get_type() == TaskBlockType.SW.value:
                        last_block.set_type(TaskBlockType.SW2D.value)
