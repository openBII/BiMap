#!/usr/bin/env python
# coding: utf-8


"""
STEnv类描述性能级仿真环境
"""

from copy import copy, deepcopy
from typing import List, Union, Dict, Tuple, Iterable
from top.config import GlobalConfig
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.state.history import History
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.resource_simulator.evaluation_model.evaluation_model import EvaluationModel
from src.simulator.resource_simulator.action_model import ActionModel
from src.simulator.resource_simulator.action_model.splitter import SplitType
from src.simulator.resource_simulator.evaluation_model.evaluation import MemoryEvaluation
from src.simulator.task_rabbit.task_checker.checker import TaskChecker
from src.simulator.resource_simulator.st_model.st_coord import MLCoord
from src.simulator.resource_simulator.scheduler import Scheduler
from src.simulator.task_rabbit.task_model.input_type import InputType
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.hop import Hop
from src.simulator.resource_simulator.evaluation_model.recorder import CommunicationRecord
from src.simulator.task_rabbit.task_model.vtask_block import VTaskBlock
from src.simulator.resource_simulator.sync.sync_table import SyncTable
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint
from src.simulator.resource_simulator.state.call import Call


class STEnv():
    def __init__(self, task_graph: TaskGraph, st_matrix: STMatrix):
        self._task_graph = task_graph
        self._st_matrix = st_matrix

        self._context = STContext()
        self._sync_table = SyncTable()

        self._evaluator = EvaluationModel(task_graph, st_matrix, self._context)
        self._actor = ActionModel(task_graph, st_matrix, self._context, self._sync_table)

        self._history = History()  # Memento

        IDGenerator.set_base_task_id(self._task_graph)
        # self._history.new_state(self.context)

        # self._statistic_cache = {}  # type: Dict[str, CoreStatistic]

    # ST Env
    @property
    def context(self):
        return self._context

    # Task graph data
    @property
    def task_graph(self):
        return self._task_graph
    
    def get_sync_id(self):
        return self._sync_table.get_sync_id()

    def get_task(self, task_id):
        return self._task_graph.get_node(task_id)

    def get_all_task_id(self):
        return self._task_graph.get_all_node_id()

    def get_task_num(self):
        return len(self._task_graph)

    def get_in_tasks(self, task_id):
        return self.get_task(task_id).in_tasks

    def get_out_tasks(self, task_id):
        return self.get_task(task_id).out_tasks

    # St matrix data
    @property
    def st_matrix(self):
        return self._st_matrix

    def get_ml_coord(self, task_id):
        return self._context.get_ml_coord(task_id)

    def get_space_point(self, ml_coord: MLCoord):
        return self._st_matrix.get_element(ml_coord)

    def get_space(self, space_coord):
        return self._st_matrix.get_space(space_coord)

    def get_all_space(self):
        pass

    def is_empty(self, ml_coord):
        # ml_coord 需要精细到时空最细粒度
        return self._st_matrix.is_empty(ml_coord)

    # Evaluations
    def eva_task_storage(self, task_id):
        task = self._task_graph.get_node(task_id)
        if task is None or not task.is_activate:
            raise ValueError('The task is not enable')
        return task.get_storage()

    def eva_task_computation(self, task_id):
        task = self._task_graph.get_node(task_id)
        if task is None or not task.is_activate:
            raise ValueError('The task is not enable')
        return task.get_computation()

    def eva_data_transmission(self, src_task_id, dst_task_id):
        pass

    def eva_st_point(self, ml_coord):
        self._evaluator.eva_st_point(ml_coord)

    def eva_space_column(self, space_coord):
        pass

    def eva_st_matrix(self, ml_coord=None):
        pass

    def eva_from_task_to_task(self, start_task_id, end_task_id):
        pass

    # Non-empty Space number
    def get_space_num(self, top_space_coord=None, time_coord=None):
        pass

    def get_max_time(self, top_space_coord=None):
        pass

    def get_task_time(self, task: TaskBlock, iteration: int):
        if self._context.is_task_in_matrix(task.id):
            ml_coord = self._context.get_ml_coord(task)
            space_point = self.st_matrix.get_element(ml_coord)
            start, end = space_point.recorder.get_record(task.id, iteration)
            return start, end
        else:
            return None, None
    
    def get_edge_time(self, edge: Edge, iteration: int):
        container_coord = self._context.get_container_coord(edge)
        network_id = self._context.get_network_id(edge)
        if container_coord.empty:
            space_matrix = self.st_matrix
        else:
            space_matrix = self.st_matrix.get_element(container_coord)
        recorder = space_matrix.communication_networks[network_id].evaluator.recorder
        time_dict: Dict[Hop, CommunicationRecord] = {}
        for key, record in recorder:
            if key[0] == edge and key[1] == iteration:
                time_dict[key[2]] = record
        return container_coord, network_id, time_dict
    
    def show_overall_time(self, tick_num: int = 1):
        for iteration in range(tick_num):
            for task_id, task in self._task_graph:
                if not task.is_enable():
                    continue
                start, end = self.get_task_time(task, iteration)
                if not (start is None or end is None):
                    print("Task {:d}: [{:2f}, {:2f}]".format(task_id, start, end))
                    for edge in task.output_edges:
                        if not edge.is_enable():
                            continue
                        while True:
                            container_coord, network_id, time_dict = self.get_edge_time(edge, iteration)
                            for hop in time_dict:
                                assert time_dict[hop].percent == 1, "Unfinished edge"
                                print("From Task {:d} to {:d} Network {:s} Hop {:s}: [{:2f}, {:2f}]".format(edge.in_task.id, edge.out_task.id, repr(container_coord) + '.' + str(network_id), repr(hop), time_dict[hop].start_time, time_dict[hop].end_time))
                            if isinstance(edge.out_task, VTaskBlock):
                                edge = edge.out_task.output_edges[0]
                            else:
                                break

    def get_computation(self, ml_coord):
        pass

    def get_route_size(self, src_ml_coord, dst_ml_coord):
        pass

    def get_memory_overflow_space(self):
        pass

    def get_memory(self, ml_coord: MLCoord, tasks: Iterable[STaskBlock] = None):
        memory_point = self.get_space_point(ml_coord)
        storage = 0
        if tasks is None:
            for task in memory_point._tasks:
                storage += task.get_storage()
        else:
            for task in tasks:
                storage += task.get_storage()
        return storage

    def does_memory_overflow(self, memory_coord: MLCoord, tasks: Iterable[STaskBlock] = None):
        memory_point: MemoryPoint = self.get_space_point(memory_coord)
        return self.get_memory(memory_coord, tasks) > memory_point.capacity
    
    def does_mlp_memory_overflow(self, input: STaskBlock, weight: StaticTaskBlock, output: Union[STaskBlock, OutputTaskBlock], memory_coord: MLCoord, split_vector: SplitVector):
        split_input = self.split_task(input.id, SplitVector(nr=split_vector.nr), record=False)
        split_weight = self.split_task(weight.id, split_vector, record=False)
        split_output = self.split_task(output.id, SplitVector(nf=split_vector.nf), record=False)
        if self.does_memory_overflow(memory_coord, [split_input[0], split_weight[0], split_output[0]]):
            return True
        if split_vector.nr > 1:
            if self.does_memory_overflow(memory_coord, ([split_output[0]] * split_vector.nr) + [split_output[0]]):
                return True
        return False

    def does_pointwise_memory_overflow(self, input: STaskBlock, memory_coord: MLCoord, split_vector: SplitVector):
        split_input = self.split_task(input.id, SplitVector(nf=split_vector.nf), record=False)
        if self.does_memory_overflow(memory_coord, [split_input[0]] * 2):
            return True
        else:
            return False
    
    def get_computation_bottleneck(self):
        pass

    def get_route_bottleneck(self):
        pass

    def get_phase_group_overflow(self):
        pass

    def get_core_overflow(self):
        pass

    def get_phase_overflow(self):
        pass

    def get_state(self):
        pass

    def is_mapped(self, task_id):
        pass

    def is_all_mapped(self):
        pass

    def mapped_in(self, task_id):
        pass

    def get_mapped_tasks(self):
        pass

    def get_unmapped_tasks(self):
        pass

    # Check
    def check_graph(self):
        TaskChecker.check_graph(self._task_graph)

    def check_st_matrix(self):
        pass

    def check_mapping(self):
        pass

    # Action
    # def split_task(self, task_id: int, split_vector: Shape,
    #                split_funcs: Union[SplitType, List[SplitType]] = SplitType.Average):  # [SplitType]
    #     if isinstance(split_funcs, SplitType):
    #         split_funcs = [deepcopy(split_funcs)] * 6
    #     return self._actor.split_task(task_id, split_vector, split_funcs)
    def split_FFN(self, split_vector_up: SplitVector, split_vector_act: SplitVector, split_vector_down: SplitVector,
                  up_input: STaskBlock, split_up_input: List[STaskBlock],
                  up_weight: StaticTaskBlock, up_mlp: CTaskBlock, up_output: STaskBlock,
                  activation: CTaskBlock, activation_output: STaskBlock,
                  down_weight: StaticTaskBlock, down_mlp: CTaskBlock, down_output: STaskBlock):
        task_dict: Dict[str, Dict[str, List[TaskBlock]]] = dict()
        task_dict["up"] = dict()
        task_dict["activation"] = dict()
        task_dict["down"] = dict()
        if split_vector_up.nr > 1:
            new_tasks = self.split_mlp(up_input, split_up_input, up_weight, up_mlp, up_output, split_vector_up)
            if len(new_tasks) == 6:
                split_up_input, split_up_weight, split_up_mlp, split_up_mlp_output, split_up_add, split_up_add_output = new_tasks
            else:
                split_up_weight, split_up_mlp, split_up_mlp_output, split_up_add, split_up_add_output = new_tasks
            task_dict["up"]["add"] = split_up_add
            task_dict["up"]["add_output"] = split_up_add_output
        else:
            new_tasks = self.split_mlp(up_input, split_up_input, up_weight, up_mlp, up_output, split_vector_up)
            if len(new_tasks) == 4:
                split_up_input, split_up_weight, split_up_mlp, split_up_mlp_output = new_tasks
            else:
                split_up_weight, split_up_mlp, split_up_mlp_output = new_tasks
        task_dict["up"]["input"] = split_up_input
        task_dict["up"]["weight"] = split_up_weight
        task_dict["up"]["mlp"] = split_up_mlp
        task_dict["up"]["mlp_output"] = split_up_mlp_output
        if split_vector_up.nr > 1:
            new_tasks = self.split_pointwise(up_output, split_up_add_output, activation, activation_output, split_vector_act)
        else:
            new_tasks = self.split_pointwise(up_output, split_up_mlp_output, activation, activation_output, split_vector_act)
        if len(new_tasks) == 3:
            split_activation_input, split_activation, split_activation_output = new_tasks
            task_dict["activation"]["input"] = split_activation_input
        else:
            split_activation, split_activation_output = new_tasks
            if split_vector_up.nr > 1:
                task_dict["activation"]["input"] = split_up_add_output
            else:
                task_dict["activation"]["input"] = split_up_mlp_output
        task_dict["activation"]["compute"] = split_activation
        task_dict["up"]["output"] = split_activation_output
        if split_vector_up.nr > 1:
            new_tasks = self.split_mlp(activation_output, split_activation_output, down_weight, down_mlp, down_output, split_vector_down)
            if len(new_tasks) == 6:
                split_down_input, split_down_weight, split_down_mlp, split_down_mlp_output, split_down_add, split_down_add_output = new_tasks
                task_dict["down"]["input"] = split_down_input
            else:
                split_down_weight, split_down_mlp, split_down_mlp_output, split_down_add, split_down_add_output = new_tasks
                task_dict["down"]["input"] = split_activation_output
            task_dict["down"]["add"] = split_down_add
            task_dict["down"]["add_output"] = split_down_add_output
        else:
            new_tasks = self.split_mlp(activation_output, split_activation_output, down_weight, down_mlp, down_output, split_vector_down)
            if len(new_tasks) == 4:
                split_down_input, split_down_weight, split_down_mlp, split_down_mlp_output = new_tasks
                task_dict["down"]["input"] = split_down_input
            else:
                split_down_weight, split_down_mlp, split_down_mlp_output = new_tasks
                task_dict["down"]["input"] = split_activation_output
        task_dict["down"]["weight"] = split_down_weight
        task_dict["down"]["mlp"] = split_down_mlp
        task_dict["down"]["mlp_output"] = split_down_mlp_output
        return task_dict

    def split_attention(self):
        pass

    def split_reduction(self, input: STaskBlock, split_inputs: List[STaskBlock], compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector, precision: Precision = None):
        return self._actor.split_reduction(input, split_inputs, compute, output, split_vector, precision)

    def split_mlp(self, input: STaskBlock, split_inputs: List[STaskBlock], weight: StaticTaskBlock, compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector, precision: Precision = None):
        new_tasks = self._actor.split_mlp(input, split_inputs, weight, compute, output, split_vector, precision)
        
        reverse_call = Call(self._actor.reverse_split, new_tasks, [compute, output] + split_inputs)
        self._history.push_state(reverse_call)
        return new_tasks
    
    def split_pointwise(self, input: STaskBlock, split_inputs: List[STaskBlock], compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector):
        new_tasks = self._actor.split_pointwise(input, split_inputs, compute, output, split_vector)

        reverse_call = Call(self._actor.reverse_split, new_tasks, [compute, output] + split_inputs)
        self._history.push_state(reverse_call)
        return new_tasks

    def split_task(self, task_id: int, split_vector: SplitVector, is_static: bool = False, record: bool = True):
        return self._actor.split_task(task_id, split_vector, is_static, record)

    def connect_tasks(self, in_tasks: Iterable[TaskBlock], out_tasks: Iterable[TaskBlock]):
        self._actor.connect_tasks(in_tasks, out_tasks)

    def copy_task(self, task_id: int, num: int = 1, is_static: bool = False):
        return self._actor.copy_task(task_id, num, is_static)
    
    def split_and_copy_task(self, task_id: int, split_vector: SplitVector, num: int = 1, is_static: bool = False):
        return self._actor.split_and_copy_task(task_id, split_vector, num, is_static)

    def split_group(self, task_id_list: List[int], split_vector: Union[Shape, List[Shape]],
                    split_funcs: Union[SplitType, List[SplitType], List[List[SplitType]]] = SplitType.Average) -> List[int]:
        new_task_id_list = []
        num_tasks = len(task_id_list)
        if isinstance(split_vector, Shape):
            split_vector_list = [deepcopy(split_vector)
                                 for _ in range(num_tasks)]
        else:
            split_vector_list = split_vector
        if isinstance(split_funcs, SplitType):
            split_funcs_list = [deepcopy(split_funcs)
                                for _ in range(num_tasks)]
        else:
            if isinstance(split_funcs[0], SplitType):
                split_funcs_list = [deepcopy(split_funcs)
                                    for _ in range(num_tasks)]
            else:
                split_funcs_list = split_funcs
        for i in range(num_tasks):
            new_task_id_list.extend(self.split_task(
                task_id_list[i], split_vector_list[i], split_funcs_list[i]))
        return new_task_id_list

    def delete_task(self, task_id):
        return self._actor.delete_task(task_id)
    
    def delete_tasks(self, tasks: Iterable[TaskBlock]):
        self._actor.delete_tasks(tasks)

    def replicate_task(self, task_id):
        return self._actor.replicate_task(task_id)

    def replicate_group(self, task_id_list) -> List[int]:
        new_task_id_list = []
        for task_id in task_id_list:
            new_task_id_list.append(self.replicate_task(task_id))
        return new_task_id_list

    def fuse_task(self, task_id_list):
        return self._actor.fuse_task(task_id_list)

    def set_pipeline_num(self, task_id, pipeline_num):
        self._actor.set_pipeline_num(task_id, pipeline_num)

    def enable_task(self, task_id):
        self._actor.enable_task(task_id)

    def disable_task(self, task_id):
        self._actor.disable_task(task_id)

    def merge_column(self, space_coord_list):
        self._actor.merge_column(space_coord_list)

    def delete_column(self, space_coord):
        return self._actor.delete_column(space_coord)

    def put_in(self, ml_coord: MLCoord, task_id):
        self._actor.put_in(ml_coord, task_id)
        reverse_call = Call(self._actor.take_out, ml_coord, task_id)
        self._history.push_state(reverse_call)

    def put_tasks_in(self, ml_coord: MLCoord, tasks: Iterable[TaskBlock]):
        self._actor.put_tasks_in(ml_coord, tasks)
        reverse_call = Call(self._actor.take_tasks_out, ml_coord, tasks)
        self._history.push_state(reverse_call)

    def sync(self, ml_coord: MLCoord, sync_id: int):
        self._actor.sync(ml_coord, sync_id)

    def put_group_in(self, ml_coord: MLCoord, task_id):
        """将计算任务块和相应的输入存储任务块放到一个核的一个phase内
        task_id必须对应一个计算任务块
        ml_coord指定计算任务块的坐标
        """
        c_task = self.get_task(task_id)
        c_task_type = c_task.task_type
        assert TaskBlockType.is_compute_task(
            c_task_type), "This method can only be applied to computational TaskBlocks"
        self.put_in(ml_coord, task_id)
        new_ml_coord = MLCoord(
            ml_coord.space_coord, (ml_coord.time_coord[0], ml_coord.time_coord[1], PIIndex.MEMORY.value))
        for in_task in self.get_in_tasks(task_id):
            self.put_in(ml_coord=new_ml_coord, task_id=in_task.id)

    def take_out(self, ml_coord: MLCoord, task_id: int):
        self._actor.take_out(ml_coord, task_id)
        reverse_call = Call(self._actor.put_in, ml_coord, task_id)
        self._history.push_state(reverse_call)

    def task_tasks_out(self, ml_coord: MLCoord, tasks: Iterable[TaskBlock]):
        self._actor.take_tasks_out(ml_coord, tasks)
        reverse_call = Call(self._actor.put_tasks_in, ml_coord, tasks)
        self._history.push_state(reverse_call)

    def take_group_out(self, ml_coord: MLCoord, task_id: int = None) -> List[TaskBlock]:
        """
        将某个核的某个phase的计算任务块和相应的输入存储任务块取出
        """
        task_list = list()
        assert PIIndex.is_compute_task(
            ml_coord), "This method can only be applied to computational TaskBlocks"
        c_task = self.take_out(ml_coord, task_id)
        assert TaskBlockType.is_compute_task(c_task.task_type)
        task_list.append(c_task)
        new_ml_coord = MLCoord(
            ml_coord.space_coord, (ml_coord.time_coord[0], ml_coord.time_coord[1], PIIndex.MEMORY.value))
        for in_task in self.get_in_tasks(c_task.id):
            task_list.append(self.take_out(
                ml_coord=new_ml_coord, task_id=in_task.id))
        return task_list

    # def take_out_task(self, ml_coord, task_id):
    #     self._actor.take_out_task(task_id, ml_coord)

    def move(self, src_coord, dml_coord, task_id=None):
        assert PIIndex.is_same_task_type(
            src_coord, dml_coord), "Cannot move task because of inconsistent pipeline stage"
        return self._actor.move(src_coord, dml_coord, task_id)

    def move_group(self, src_coord, dml_coord, task_id=None) -> List[TaskBlock]:
        """
        将某个核的某个phase的计算任务块和相应的输入存储任务块移到某个核的某个phase
        """
        assert PIIndex.is_same_task_type(
            src_coord, dml_coord), "Cannot move task because of inconsistent pipeline stage"
        assert PIIndex.is_compute_task(
            src_coord), "This method can only be applied to computational TaskBlocks"
        task_list = list()
        c_task = self.move(src_coord, dml_coord, task_id)
        assert TaskBlockType.is_compute_task(c_task.task_type)
        task_list.append(c_task)
        memory_src_coord = MLCoord(
            src_coord.space_coord, (src_coord.time_coord[0], src_coord.time_coord[1], PIIndex.MEMORY.value))
        memory_dml_coord = MLCoord(
            dml_coord.space_coord, (dml_coord.time_coord[0], dml_coord.time_coord[1], PIIndex.MEMORY.value))
        for in_task in self.get_in_tasks(c_task.id):
            task_list.append(self.move(
                memory_src_coord, memory_dml_coord, in_task.id))
        return task_list

    def connect(self, src_task, src_index, src_info,
                dst_task, dst_index, dst_info):
        self._actor.connect(src_task, src_index, src_info, dst_task,
                            dst_index, dst_info)

    def map_edge(self, edge: Edge, path: List[Union[MLCoord, Tuple[MLCoord, int], Tuple[MLCoord, int, int]]]):
        new_tasks, new_edges = self._actor.map_edge(edge, path)
        reverse_call = Call(self._actor.reverse_map_edge, edge, new_tasks, new_edges)
        self._history.push_state(reverse_call)

    def map_edges(self, edges: Iterable[Edge], path: List[Union[MLCoord, Tuple[MLCoord, int], Tuple[MLCoord, int, int]]]):
        new_tasks = []
        new_edges = []
        for edge in copy(edges):
            new_task, new_edge = self._actor.map_edge(edge, path)
            new_tasks.append(new_task)
            new_edges.append(new_edge)
        reverse_call = Call(self._actor.reverse_map_edges, edges, new_tasks, new_edges)
        self._history.push_state(reverse_call)

    def simulate(self, tick_num: int = 1, input_type: InputType = InputType.BATCH):
        if tick_num > 1 and input_type == InputType.PIPELINE:
            raise NotImplementedError
        activated_tasks = self._task_graph.input(tick_num, input_type)
        for node in self._task_graph.static_nodes:
            node.init_ticks(tick_num)
        activated_tasks = activated_tasks | self._task_graph.static_nodes
        # 初始化scheduler，传入activated_tasks
        scheduler = Scheduler(self._st_matrix, self._context, self._task_graph, self._sync_table, activated_tasks)
        scheduler.schedule()

    # State control
    def undo(self):
        self._history.pop_state()

    def lock(self):
        self._history.lock()

    def reset(self):
        pass

    def mark(self, label: str):
        self._history.mark(label)

    def undo_mark(self, label: str):
        self._history.pop_mark(label)

    def get_compute_task_id(self, task_id_list: List[int]) -> List:
        compute_task_id_list = list()
        for task_id in task_id_list:
            if TaskBlockType.is_compute_task(self.get_task(task_id).task_type):
                compute_task_id_list.append(task_id)
        return compute_task_id_list

    # Cache
    # def clear_cache(self):
    #     self._statistic_cache.clear()
    #     self.split_dt.clear()


def create_st_env(task_path, case_name) -> STEnv:
    from src.simulator.task_rabbit.initial_pass import execute_initial_pass
    init = execute_initial_pass(
        case_path=task_path, case_name=case_name, input_type='task')
    st_matrix = STMatrix()
    st_env = STEnv(init.task_graph, st_matrix)
    # st_env.check_graph()
    return st_env


if __name__ == '__main__':
    from task_rabbit.initial_pass import InitialPass
    from src.compiler.mapper.passes.final_pass import FinalPass
    from resource_simulator.st_draw import STDraw
    import numpy as np
    from flow.execute import exe_task_rabbit_with_task, exe_task_rabbit_with_map

    def compare(task_path, map_path, task_name, map_name, task_id):
        exe_task_rabbit_with_task(
            case_path=task_path, case_name=task_name)
        exe_task_rabbit_with_map(
            case_path=map_path, case_name=map_name)
        ref_result = np.fromfile(
            GlobalConfig.Path["temp"] + task_name + '/task_out/task_block' + str(task_id) + '.dat', dtype=np.int32)
        split_result = np.fromfile(
            GlobalConfig.Path["temp"] + map_name + '/map_out/task_block' + str(task_id) + '.dat', dtype=np.int32)
        assert (ref_result == split_result).all(), 'Comparison failed!'
        print('Comparison successful!')

    task_path = GlobalConfig.Path["test_lib"] + 'task_lib/1P/ccmpb.task'
    init = InitialPass(path=task_path, input_type='task')
    task_graph = init.task_graph
    st_matrix = STMatrix()
    st_env = STEnv(task_graph, st_matrix)
    st_env.check_graph()

    st_env.split_task(task_id=1, split_vector=Shape(
        ny=2, nx=2, nf=2, nr=1, nky=1, nkx=1))

    st_env.check_graph()

    STDraw.draw_graph(task_graph, out_path=GlobalConfig.Path["temp"] + 'ut_ccmpb/ut_ccmpb.task.html',
                      width='1920px', height='1080px')

    map_path = GlobalConfig.Path["test_lib"] + 'mapping_lib/1P/ccmpb.map'
    FinalPass(st_env=st_env, out_path=map_path)

    compare(task_path, map_path, task_name='ut_ccmpb',
            map_name='ut_ccmpb_split', task_id=2)
