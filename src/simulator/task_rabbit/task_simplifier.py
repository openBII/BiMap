# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import copy
import logging
import os

from google.protobuf import text_format

import src.compiler.ir.mapping_pb2 as mapping_ir
from src.compiler.mapper.map_ir_generator import MapIRGenerator
from resource_simulator.st_env import STEnv, STMatrix
from task_rabbit.task_graph_parser import TaskGraphParser
from task_rabbit.task_model import *
from task_rabbit.task_model import TaskBlockType
from task_rabbit.task_model.id_generator import IDGenerator
from top.config import GlobalConfig


class TaskSimplifier:
    """
    简化掉Task IR里的一些冗余信息
    也会为没有OUTPUT结点的IR添加OUTPUT结点
    """
    @staticmethod
    def simplify(ir_path: str, new_ir_path: str, ir_type: str, readable_result: bool = True) -> TaskGraph:
        """
        Args:
            ir_path: 将要简化的IR的文件路径
            new_ir_path: 简化结果的存放路径
            ir_type: 该参数为'task'，则表示简化Task IR；为'map'，则表示简化Map IR
        Raises: 
            ValueError: 如果ir类型不是'task'或'map'时，报异常
        """

        task_graph = TaskGraphParser.parse(ir_path, ir_type)
        st_env = STEnv(task_graph, STMatrix())

        task_graph = None
        if ir_type == 'task':
            new_ir, _ = TaskSimplifier.simplify_task(st_env)
        elif ir_type == 'map':
            new_ir = TaskSimplifier.simplify_map(ir_path, st_env)
        else:
            raise ValueError('Unsupported input file type')

        TaskSimplifier._save_ir(new_ir, new_ir_path, readable_result)

    @staticmethod
    def simplify_task(st_env: STEnv):
        task_graph = st_env.task_graph
        # 增加OUTPUT结点
        node_keys = set(task_graph.get_all_node_ids())
        new_outputs = set()
        for node_id in node_keys:
            node = task_graph.get_node(node_id)
            socket_id = 0
            if TaskBlockType.is_storage_task(node.task_type) and \
                    len(node.out_tasks) == 0:

                id_output = IDGenerator.get_next_task_id()
                node_output = OutputTaskBlock(id_output, copy.copy(
                    node.shape), node.precision, socket_id)
                task_graph.add_node(node_output)
                task_graph.connect(node_id, id_output)

                socket_id += 1
                new_outputs.add(id_output)

        generator = MapIRGenerator(st_env=st_env)
        generator.convert_task_graph()
        simplified_task_ir = generator.tj_graph

        return simplified_task_ir, new_outputs

    @staticmethod
    def simplify_map(ir_path, st_env: STEnv):
        simplified_graph, new_outputs = TaskSimplifier.simplify_task(st_env)

        old_mapping = mapping_ir.Mapping()
        if (ir_path.endswith('.txt')):
            with open(ir_path, 'r') as file:
                text_format.ParseLines(file.readlines(), old_mapping)
        else:
            with open(ir_path, 'rb') as file:
                old_mapping.ParseFromString(file.read())
        new_mapping = mapping_ir.Mapping()
        new_mapping.graph.CopyFrom(simplified_graph)
        new_mapping.space_time_mapping.CopyFrom(old_mapping.space_time_mapping)

        # 映射新加入的output结点
        for task_id in new_outputs:
            map_ir_dict = new_mapping.space_time_mapping.node_map_dicts.add()
            map_ir_dict.task_id = task_id
            tj_space_time_coordinate = map_ir_dict.space_time_coordinates.add()
            tj_space = tj_space_time_coordinate.space
            tj_time = tj_space_time_coordinate.time
            tj_space.chip_array.x, tj_space.chip_array.y = 0, 0
            tj_space.chip.x, tj_space.chip.y = 0, 0
            tj_space.step_group, tj_space.phase_group = 0, 0
            tj_space.core.x, tj_space.core.y = -1, 0
            tj_time.step, tj_time.phase = 0, 0
            tj_time.pi_index = mapping_ir.PIIndex.ROUTER_RECIEVE

        return new_mapping

    @staticmethod
    def _save_ir(simplified_ir, new_ir_path: str, readable_result: bool = True):
        os.makedirs(os.path.dirname(new_ir_path), exist_ok=True)
        if readable_result:
            if not new_ir_path.endswith('.txt'):
                new_ir_path += '.txt'
            with open(new_ir_path, 'w') as f:
                f.write(repr(simplified_ir))
        else:
            with open(new_ir_path, 'wb') as out_file:
                out_file.write(simplified_ir.SerializeToString())


if __name__ == '__main__':
    for root, _, files in os.walk(GlobalConfig.Path['test_lib']):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.task.txt'):
                TaskSimplifier.simplify(full_path, full_path, 'task')
                logging.info('Simplified ' + full_path)
            if file.endswith('.map.txt'):
                TaskSimplifier.simplify(full_path, full_path, 'map')
                logging.info('Simplified ' + full_path)
