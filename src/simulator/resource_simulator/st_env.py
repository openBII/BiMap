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
from src.simulator.resource_simulator.st_model.space_point.memory_point import MemoryPoint, DRAMPoint
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
    
    def get_communication_network(self, ml_coord: MLCoord, network_id: int = 0):
        space_matrix = self.get_space_point(ml_coord)
        return space_matrix.communication_networks[network_id]

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

    def get_latency(self, iteration: int = 0):
        latency = 0
        for output_id in self._task_graph._outputs:
            output = self._task_graph[output_id]
            _, end = self.get_task_time(output, iteration)
            latency = max(end, latency)
        return latency 

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
        if isinstance(memory_point, DRAMPoint):
            return self.get_memory(memory_coord, tasks) > memory_point.capacity * 1024 * 1024 * 1024
        else:
            return self.get_memory(memory_coord, tasks) > memory_point.capacity * 1024
    
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

    def split_ffn(self, task_dict: Dict, 
                  up_input: STaskBlock, split_up_input: List[STaskBlock],
                  split_vector_up: SplitVector, 
                  split_vector_act: SplitVector, 
                  split_vector_down: SplitVector):
        split_task_dict: Dict[str, Dict[str, List[TaskBlock]]] = dict()
        split_task_dict["up"] = dict()
        split_task_dict["activation"] = dict()
        split_task_dict["down"] = dict()
        up_weight = task_dict["up"]["weight"]
        up_mlp = task_dict["up"]["compute"]
        up_output = task_dict["up"]["output"]
        activation = task_dict["activation"]["compute"]
        activation_output = task_dict["activation"]["output"]
        down_weight = task_dict["down"]["weight"]
        down_mlp = task_dict["down"]["compute"]
        down_output = task_dict["down"]["output"]

        self.split_mlp(split_up_input, up_weight, up_mlp, up_output, 
                       split_vector_up, up_input, split_task_dict["up"])

        split_up_output = split_task_dict["up"]["output"]
        new_tasks = self.split_pointwise(up_output, split_up_output, activation, 
                                         activation_output, split_vector_act,
                                         split_task_dict["activation"])
 
        split_activation_output = split_task_dict["activation"]["output"]
        self.split_mlp(split_activation_output, down_weight, 
                       down_mlp, down_output, split_vector_down,
                       activation_output,
                       split_task_dict["down"])

        return split_task_dict
    
    def split_ffn_block(self, task_dict: Dict, 
                        up_input: STaskBlock, split_up_input: List[STaskBlock],
                        split_vector_up: SplitVector, 
                        split_vector_act: SplitVector, 
                        split_vector_down: SplitVector,
                        add_split_vector: SplitVector,
                        layer_norm_split_vector: SplitVector):
        split_task_dict = {}
        assert add_split_vector.nf == split_vector_up.nr
        split_task_dict["ffn"] = self.split_ffn(
            task_dict["ffn"],
            up_input,
            split_up_input,
            split_vector_up,
            split_vector_act,
            split_vector_down
        )
        split_task_dict["add"] = {}
        self.split_elementwise(
            [task_dict["ffn"]["down"]["output"], up_input],
            [split_task_dict["ffn"]["down"]["output"], split_up_input if "input" not in split_task_dict["ffn"]["up"] else split_task_dict["ffn"]["up"]["input"]],
            task_dict["add"]["compute"], task_dict["add"]["output"],
            add_split_vector, split_task_dict["add"])

        add_output = task_dict["add"]["output"]
        split_add_output = split_task_dict["add"]["output"]
        layer_norm = task_dict["layer_norm"]["compute"]
        layer_norm_output = task_dict["layer_norm"]["output"]
        split_task_dict["layer_norm"] = {}
        self.split_pointwise(
            add_output, split_add_output,
            layer_norm, layer_norm_output, 
            layer_norm_split_vector, split_task_dict["layer_norm"]
        )       
        if "input" in split_task_dict["layer_norm"]:
            split_task_dict["add"]["output"] = split_task_dict["layer_norm"]["input"]
        # split_task_dict["layer_norm"] = self.split_layer_norm(
        #     task_dict["add"]["output"], split_add_output,
        #     task_dict["layer_norm"], layer_norm_add_split_vector,
        #     layer_norm_product_split_vector, layer_norm_div_split_vector
        # )    
        # if "input" in split_task_dict["layer_norm"]["add"]:
        #     split_task_dict["add"]["output"] = split_task_dict["layer_norm"]["add"]["input"]
        return split_task_dict
    
    def split_transformer_layer(self, task_dict: Dict,
                                split_embedding: List[TaskBlock], head: int,
                                query_split_vectors: List[SplitVector],
                                key_split_vectors: List[SplitVector],
                                value_split_vectors: List[SplitVector],
                                dot_product_split_vectors: List[SplitVector],
                                softmax_split_vectors: List[SplitVector],
                                attention_split_vectors: List[SplitVector],
                                mlp_split_vector: SplitVector,
                                add_split_vector: SplitVector,
                                layer_norm_split_vector: SplitVector,
                                split_vector_up: SplitVector, 
                                split_vector_act: SplitVector, 
                                split_vector_down: SplitVector,
                                ffn_add_split_vector: SplitVector,
                                ffn_layer_norm_split_vector: SplitVector,
                                embedding: TaskBlock = None):
        split_task_dict = {}
        split_task_dict["attention"] = self.split_attention_block(
            task_dict["attention"],
            split_embedding,
            head,
            query_split_vectors,
            key_split_vectors,
            value_split_vectors,
            dot_product_split_vectors,
            softmax_split_vectors,
            attention_split_vectors,
            mlp_split_vector,
            add_split_vector,
            layer_norm_split_vector,
            embedding
        )
        split_task_dict["ffn"] = self.split_ffn_block(
            task_dict["ffn"],
            task_dict["attention"]["layer_norm"]["output"],
            split_task_dict["attention"]["layer_norm"]["output"],
            split_vector_up,
            split_vector_act,
            split_vector_down,
            ffn_add_split_vector,
            ffn_layer_norm_split_vector
        )
        return split_task_dict
    
    def split_attention_block(self, task_dict: Dict, 
                              split_embedding: List[TaskBlock], head: int,
                              query_split_vectors: List[SplitVector],
                              key_split_vectors: List[SplitVector],
                              value_split_vectors: List[SplitVector],
                              dot_product_split_vectors: List[SplitVector],
                              softmax_split_vectors: List[SplitVector],
                              attention_split_vectors: List[SplitVector],
                              mlp_split_vector: SplitVector,
                              add_split_vector: SplitVector,
                              layer_norm_split_vector: SplitVector,
                              embedding: TaskBlock = None):
        split_task_dict = {}
        assert add_split_vector.nf == query_split_vectors[0].nr
        split_task_dict["multi_head_attention"] = self.split_multi_head_attention(
            task_dict["multi_head_attention"],
            split_embedding,
            head, 
            query_split_vectors,
            key_split_vectors,
            value_split_vectors,
            dot_product_split_vectors,
            softmax_split_vectors,
            attention_split_vectors,
            mlp_split_vector,
            embedding)
        split_task_dict["add"] = {}
        self.split_elementwise(
            [task_dict["multi_head_attention"]["mlp"]["output"], embedding],
            [split_task_dict["multi_head_attention"]["mlp"]["output"], split_embedding if "input" not in split_task_dict["multi_head_attention"][0]["query"] else split_task_dict["multi_head_attention"][0]["query"]["input"]],
            task_dict["add"]["compute"], task_dict["add"]["output"],
            add_split_vector, split_task_dict["add"])
        
        split_add_output = split_task_dict["add"]["output"]
        layer_norm = task_dict["layer_norm"]["compute"]
        layer_norm_output = task_dict["layer_norm"]["output"]
        split_task_dict["layer_norm"] = {}
        self.split_pointwise(
            task_dict["add"]["output"], split_add_output,
            layer_norm, layer_norm_output, 
            layer_norm_split_vector, split_task_dict["layer_norm"]
        )       
        if "input" in split_task_dict["layer_norm"]:
            split_task_dict["add"]["output"] = split_task_dict["layer_norm"]["input"]
            
        return split_task_dict  

    def split_layer_norm(self, input: STaskBlock, split_input: List[STaskBlock],
                         task_dict: Dict,
                         add_split_vector: SplitVector,
                         product_split_vector: SplitVector,
                         div_split_vector: SplitVector):
        split_task_dict = {}
        average_output = task_dict["average"]["output"]
        add_mean = task_dict["add_mean"]["compute"]
        add_mean_output = task_dict["add_mean"]["output"]
        split_task_dict["add"] = {}
        self.split_scale(input, split_input, add_mean, average_output, 
                         add_mean_output, add_split_vector,
                         split_task_dict["add"])
        if "input" in split_task_dict["add"]:           
            split_add_input = split_task_dict["add"]["input"]
            reduce_sum_mean = task_dict["reduce_sum_mean"]["compute"]
            self.connect_tasks(split_add_input, [reduce_sum_mean])

        product = task_dict["product"]["compute"]
        product_output = task_dict["product"]["output"]
        split_add_output = split_task_dict["add"]["output"]
        split_task_dict["product"] = {}
        self.split_pointwise(add_mean_output,
                             split_add_output,
                             product,
                             product_output,
                             product_split_vector,
                             split_task_dict["product"])

        div = task_dict["div"]["compute"]
        div_output = task_dict["div"]["output"]
        var = task_dict["sqrt"]["output"]
        split_product_output = split_task_dict["product"]["output"]
        split_task_dict["div"] = {}
        self.split_scale(product_output,
                         split_product_output,
                         div, var, div_output,
                         div_split_vector,
                         split_task_dict["div"])

        return split_task_dict
    
    # def split_multi_head_attention(self, task_dict: Dict, 
    #                                split_embedding: List[TaskBlock], head: int,
    #                                query_split_vectors: List[SplitVector],
    #                                key_split_vectors: List[SplitVector],
    #                                value_split_vectors: List[SplitVector],
    #                                dot_product_split_vectors: List[SplitVector],
    #                                scale_split_vectors: List[SplitVector],
    #                                softmax_exp_split_vectors: List[SplitVector],
    #                                softmax_div_split_vectors: List[SplitVector],
    #                                attention_split_vectors: List[SplitVector],
    #                                mlp_split_vector: SplitVector,
    #                                embedding: TaskBlock = None):
    #     split_task_dict = {}
    #     nr = query_split_vectors[0].nr
    #     for i in range(head):
    #         assert query_split_vectors[i].nr == nr
    #         split_task_dict[i] = self.split_attention(
    #             task_dict[i],
    #             split_embedding,
    #             query_split_vectors[i],
    #             key_split_vectors[i],
    #             value_split_vectors[i],
    #             dot_product_split_vectors[i],
    #             scale_split_vectors[i],
    #             softmax_exp_split_vectors[i],
    #             softmax_div_split_vectors[i],
    #             attention_split_vectors[i],
    #             embedding)
            
    #     concat = task_dict["concat"]["move"]
    #     concat_output = task_dict["concat"]["output"]
    #     split_concat_output = self.split_task(concat_output.id, SplitVector())
    #     self.connect_tasks([concat], split_concat_output)
    #     split_task_dict["concat"] = {}
    #     split_task_dict["concat"]["output"] = split_concat_output
    #     mlp_weight = task_dict["mlp"]["weight"]
    #     mlp = task_dict["mlp"]["compute"]
    #     mlp_output = task_dict["mlp"]["output"]
    #     split_task_dict["mlp"] = {}
    #     self.split_mlp(
    #         split_concat_output, mlp_weight, mlp, mlp_output,
    #         mlp_split_vector, input=concat_output,
    #         task_dict=split_task_dict["mlp"])
    #     if "input" in split_task_dict["mlp"]:
    #         split_task_dict["concat"]["output"] = split_task_dict["mlp"]["input"]

    #     return split_task_dict

    def split_multi_head_attention(self, task_dict: Dict, 
                                   split_embedding: List[TaskBlock], head: int,
                                   query_split_vectors: List[SplitVector],
                                   key_split_vectors: List[SplitVector],
                                   value_split_vectors: List[SplitVector],
                                   dot_product_split_vectors: List[SplitVector],
                                   softmax_split_vectors: List[SplitVector],
                                   attention_split_vectors: List[SplitVector],
                                   mlp_split_vector: SplitVector,
                                   embedding: TaskBlock = None):
        split_task_dict = {}
        nr = query_split_vectors[0].nr
        for i in range(head):
            assert query_split_vectors[i].nr == nr
            split_task_dict[i] = self.split_attention(
                task_dict[i],
                split_embedding,
                query_split_vectors[i],
                key_split_vectors[i],
                value_split_vectors[i],
                dot_product_split_vectors[i],
                softmax_split_vectors[i],
                attention_split_vectors[i],
                embedding)
            
        concat = task_dict["concat"]["move"]
        concat_output = task_dict["concat"]["output"]
        split_concat_output = self.split_task(concat_output.id, SplitVector())
        self.connect_tasks([concat], split_concat_output)
        split_task_dict["concat"] = {}
        split_task_dict["concat"]["output"] = split_concat_output
        mlp_weight = task_dict["mlp"]["weight"]
        mlp = task_dict["mlp"]["compute"]
        mlp_output = task_dict["mlp"]["output"]
        split_task_dict["mlp"] = {}
        self.split_mlp(
            split_concat_output, mlp_weight, mlp, mlp_output,
            mlp_split_vector, input=concat_output,
            task_dict=split_task_dict["mlp"])
        if "input" in split_task_dict["mlp"]:
            split_task_dict["concat"]["output"] = split_task_dict["mlp"]["input"]

        return split_task_dict

    # def split_attention(self, task_dict: Dict,
    #                     split_embedding: List[TaskBlock],
    #                     query_split_vector: SplitVector,
    #                     key_split_vector: SplitVector,
    #                     value_split_vector: SplitVector,
    #                     dot_product_split_vector: SplitVector,
    #                     scale_split_vector: SplitVector,
    #                     softmax_exp_split_vector: SplitVector,
    #                     softmax_div_split_vector: SplitVector,
    #                     attention_split_vector: SplitVector,
    #                     embedding: TaskBlock = None):
    #     split_task_dict = {}

    #     assert (query_split_vector.nr == key_split_vector.nr == 
    #             value_split_vector.nr)

    #     # MLP of query
    #     query_mlp = task_dict["query"]["compute"]
    #     query_weight = task_dict["query"]["weight"]
    #     query = task_dict["query"]["output"]
    #     split_task_dict["query"] = {}
    #     self.split_mlp(split_embedding, query_weight, 
    #                    query_mlp, query, query_split_vector,
    #                    input=embedding, 
    #                    task_dict=split_task_dict["query"])

    #     # MLP of key
    #     key_mlp = task_dict["key"]["compute"]
    #     key_weight = task_dict["key"]["weight"]
    #     key = task_dict["key"]["output"]
    #     split_task_dict["key"] = {}
    #     self.split_mlp(split_embedding, key_weight, 
    #                    key_mlp, key, key_split_vector,
    #                    task_dict=split_task_dict["key"])

    #     # MLP of value
    #     value_mlp = task_dict["value"]["compute"]
    #     value_weight = task_dict["value"]["weight"]
    #     value = task_dict["value"]["output"]
    #     split_task_dict["value"] = {}
    #     self.split_mlp(split_embedding, value_weight, 
    #                    value_mlp, value, value_split_vector,
    #                    task_dict=split_task_dict["value"])

    #     # Dot product between query and key cache
    #     dot_product = task_dict["dot_product"]["compute"]
    #     dot_product_output = task_dict["dot_product"]["output"]
    #     new_key_cache = task_dict["concat_key"]["output"]
    #     split_query_output = split_task_dict["query"]["output"]
    #     split_task_dict["dot_product"] = {}
    #     self.split_dot_product(query, split_query_output, new_key_cache, 
    #                            dot_product, dot_product_output, 
    #                            dot_product_split_vector,
    #                            task_dict=split_task_dict["dot_product"])

    #     # Scale
    #     scale = task_dict["scale"]["compute"]
    #     scale_output = task_dict["scale"]["output"]
    #     split_dot_product_output = split_task_dict["dot_product"]["output"]
    #     split_task_dict["scale"] = {}
    #     self.split_pointwise(dot_product_output, split_dot_product_output, 
    #                          scale, scale_output, scale_split_vector, 
    #                          task_dict=split_task_dict["scale"])

    #     # SoftMax
    #     softmax_exp = task_dict["softmax"]["exp"]["compute"]
    #     softmax_exp_output = task_dict["softmax"]["exp"]["output"]
    #     split_scale_output = split_task_dict["scale"]["output"]
    #     split_task_dict["softmax"] = {}
    #     split_task_dict["softmax"]["exp"] = {}
    #     self.split_pointwise(scale_output, split_scale_output, softmax_exp,
    #                          softmax_exp_output, softmax_exp_split_vector,
    #                          task_dict=split_task_dict["softmax"]["exp"])

    #     softmax_reduce_output = task_dict["softmax"]["reduction"]["output"]
    #     softmax_div = task_dict["softmax"]["div"]["compute"]
    #     softmax_div_output = task_dict["softmax"]["div"]["output"]
    #     split_softmax_exp_output = split_task_dict["softmax"]["exp"]["output"]
    #     split_task_dict["softmax"]["div"] = {}
    #     self.split_scale(softmax_exp_output, split_softmax_exp_output, 
    #                      softmax_div, softmax_reduce_output, softmax_div_output, 
    #                      softmax_div_split_vector,
    #                      task_dict=split_task_dict["softmax"]["div"])

    #     # Attention
    #     new_value_cache = task_dict["concat_value"]["output"]
    #     attention = task_dict["attention"]["compute"]
    #     attention_output = task_dict["attention"]["output"]
    #     split_softmax_div_output = split_task_dict["softmax"]["div"]["output"]
    #     split_task_dict["attention"] = {}
    #     self.split_dot_product(softmax_div_output, split_softmax_div_output, 
    #                            new_value_cache, attention, attention_output, 
    #                            attention_split_vector,
    #                            task_dict=split_task_dict["attention"])
        
    #     return split_task_dict

    def split_attention(self, task_dict: Dict,
                        split_embedding: List[TaskBlock],
                        query_split_vector: SplitVector,
                        key_split_vector: SplitVector,
                        value_split_vector: SplitVector,
                        dot_product_split_vector: SplitVector,
                        softmax_split_vector: SplitVector,
                        attention_split_vector: SplitVector,
                        embedding: TaskBlock = None):
        split_task_dict = {}

        assert (query_split_vector.nr == key_split_vector.nr == 
                value_split_vector.nr)

        # MLP of query
        query_mlp = task_dict["query"]["compute"]
        query_weight = task_dict["query"]["weight"]
        query = task_dict["query"]["output"]
        split_task_dict["query"] = {}
        self.split_mlp(split_embedding, query_weight, 
                       query_mlp, query, query_split_vector,
                       input=embedding, 
                       task_dict=split_task_dict["query"])

        # MLP of key
        key_mlp = task_dict["key"]["compute"]
        key_weight = task_dict["key"]["weight"]
        key = task_dict["key"]["output"]
        split_task_dict["key"] = {}
        self.split_mlp(split_embedding, key_weight, 
                       key_mlp, key, key_split_vector,
                       task_dict=split_task_dict["key"])

        # MLP of value
        value_mlp = task_dict["value"]["compute"]
        value_weight = task_dict["value"]["weight"]
        value = task_dict["value"]["output"]
        split_task_dict["value"] = {}
        self.split_mlp(split_embedding, value_weight, 
                       value_mlp, value, value_split_vector,
                       task_dict=split_task_dict["value"])

        # Dot product between query and key cache
        dot_product = task_dict["dot_product"]["compute"]
        dot_product_output = task_dict["dot_product"]["output"]
        new_key_cache = task_dict["concat_key"]["output"]
        split_query_output = split_task_dict["query"]["output"]
        split_task_dict["dot_product"] = {}
        self.split_dot_product(query, split_query_output, new_key_cache, 
                               dot_product, dot_product_output, 
                               dot_product_split_vector,
                               task_dict=split_task_dict["dot_product"])

        # SoftMax
        softmax = task_dict["softmax"]["compute"]
        softmax_output = task_dict["softmax"]["output"]
        split_dot_product_output = split_task_dict["dot_product"]["output"]
        split_task_dict["softmax"] = {}
        self.split_pointwise(dot_product_output, split_dot_product_output, 
                             softmax, softmax_output, softmax_split_vector, 
                             task_dict=split_task_dict["softmax"])

        # Attention
        new_value_cache = task_dict["concat_value"]["output"]
        attention = task_dict["attention"]["compute"]
        attention_output = task_dict["attention"]["output"]
        split_softmax_output = split_task_dict["softmax"]["output"]
        split_task_dict["attention"] = {}
        self.split_dot_product(softmax_output, split_softmax_output, 
                               new_value_cache, attention, attention_output, 
                               attention_split_vector,
                               task_dict=split_task_dict["attention"])
        
        return split_task_dict

    def split_reduction(self, input: STaskBlock, split_inputs: List[STaskBlock], compute: CTaskBlock, output: Union[STaskBlock, OutputTaskBlock], split_vector: SplitVector, precision: Precision = None):
        return self._actor.split_reduction(input, split_inputs, compute, output, split_vector, precision)
    
    def split_dot_product(self, input: STaskBlock, 
                          split_inputs: List[STaskBlock], 
                          weight: STaskBlock, 
                          compute: CTaskBlock, 
                          output: Union[STaskBlock, OutputTaskBlock], 
                          split_vector: SplitVector,
                          task_dict: Dict = None):
        task_dict = self._actor.split_dot_product(input, split_inputs, weight,
                                                  compute, output,
                                                  split_vector, task_dict)
        new_tasks = []
        for name in task_dict:
            if name != "output":
                new_tasks.append(task_dict[name])
        reverse_call = Call(self._actor.reverse_split, new_tasks, 
                            [compute, weight, output] + split_inputs)
        self._history.push_state(reverse_call)
        return new_tasks

    def split_mlp(self, split_inputs: List[STaskBlock], weight: StaticTaskBlock, 
                  compute: CTaskBlock, 
                  output: Union[STaskBlock, OutputTaskBlock], 
                  split_vector: SplitVector,
                  input: STaskBlock = None,
                  task_dict: Dict = None):
        task_dict = self._actor.split_mlp(input, split_inputs, weight, compute, 
                                          output, split_vector, task_dict)
        new_tasks = []
        for name in task_dict:
            if name != "output":
                new_tasks.append(task_dict[name])
        reverse_call = Call(self._actor.reverse_split, new_tasks, 
                            [compute, output] + split_inputs)
        self._history.push_state(reverse_call)
        return task_dict
    
    def split_pointwise(self, input: STaskBlock, split_inputs: List[STaskBlock], 
                        compute: CTaskBlock, 
                        output: Union[STaskBlock, OutputTaskBlock], 
                        split_vector: SplitVector,
                        task_dict: Dict = None):
        task_dict = self._actor.split_pointwise(input, split_inputs, compute, 
                                                output, split_vector,
                                                task_dict)
        new_tasks = []
        for name in task_dict:
            new_tasks.append(task_dict[name])
        reverse_call = Call(self._actor.reverse_split, new_tasks, 
                            [compute, output] + split_inputs)
        self._history.push_state(reverse_call)
        return new_tasks
    
    def split_elementwise(self, inputs: List[STaskBlock],
                          split_inputs: List[List[STaskBlock]],
                          compute: CTaskBlock,
                          output: Union[STaskBlock, OutputTaskBlock],
                          split_vector: SplitVector,
                          task_dict: Dict = None):
        task_dict = self._actor.split_elementwise(inputs, split_inputs,
                                                  compute, output,
                                                  split_vector, task_dict)
        new_tasks = []
        for name in task_dict:
            new_tasks.append(task_dict[name])
        old_tasks = [compute, output]
        for split_input in split_inputs:
            old_tasks += split_input
        reverse_call = Call(self._actor.reverse_split, new_tasks, old_tasks)
        self._history.push_state(reverse_call)
        return new_tasks
    
    def split_scale(self, input: STaskBlock, split_inputs: List[STaskBlock], 
                    compute: CTaskBlock, scale: STaskBlock,
                    output: Union[STaskBlock, OutputTaskBlock], 
                    split_vector: SplitVector, task_dict: Dict = None):
        task_dict = self._actor.split_scale(input, split_inputs, compute,
                                            scale, output, split_vector,
                                            task_dict)
        new_tasks = []
        for name in task_dict:
            new_tasks.append(task_dict[name])
        reverse_call = Call(self._actor.reverse_split, new_tasks, 
                            [compute, output] + split_inputs)
        self._history.push_state(reverse_call)
        return new_tasks

    def split_task(self, task_id: int, split_vector: SplitVector, 
                   is_static: bool = False, record: bool = True):
        return self._actor.split_task(task_id, split_vector, is_static, record)

    def connect_tasks(self, in_tasks: Iterable[TaskBlock], 
                      out_tasks: Iterable[TaskBlock]):
        self._actor.connect_tasks(in_tasks, out_tasks)

    def add_nodes_between(self, source: TaskBlock, destination: TaskBlock, 
                          nodes: List[TaskBlock]):
        self._actor.add_nodes_between(source, destination, nodes)

    def copy_task(self, task_id: int, num: int = 1, is_static: bool = False,
                  record: bool = True):
        return self._actor.copy_task(task_id, num, is_static, record)
    
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

    def auto_edge_map(self):
        tasks = copy(self._task_graph.get_all_node_ids())
        for task_id in tasks:
            if task_id in self._task_graph._inputs:
                continue
            else:
                task = self._task_graph[task_id]
                for output_edge in task.enabled_output_edges:
                    src: TaskBlock = output_edge.in_task
                    dst: TaskBlock = output_edge.out_task
                    if not dst.is_enable():
                        continue
                    src_mlcoord = self.get_ml_coord(src.id)
                    dst_mlcoord = self.get_ml_coord(dst.id)
                    self.map_edge(output_edge, 
                                  self._st_matrix.generate_path(src_mlcoord,
                                                                dst_mlcoord))

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
