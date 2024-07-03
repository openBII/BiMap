#!/usr/bin/env python
# coding: utf-8

"""
STPoint类描述STMatrix中的一个点
具体的说，在tianjicX架构中，表示一个Core model的一个Phase
"""

from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.st_model.tick import Tick
from typing import List
from src.simulator.resource_simulator.evaluation_model.evaluator import Evaluator
from src.simulator.resource_simulator.evaluation_model.recorder import Recorder
from src.simulator.resource_simulator.sync.sync_task import SyncTask
from src.simulator.resource_simulator.sync.sync_table import SyncTable


class STPoint():
    def __init__(self):
        #TODO: Slots
        self._tasks: List[TaskBlock] = []
        self._pc = 0
        self.evaluator = Evaluator()
        self.recorder = Recorder()
        
    def add_task(self, task: TaskBlock):
        self._tasks.append(task)

    def get_tasks(self):
        return self._tasks
    
    def get_task_ids(self):
        return [task.id for task in self._tasks]
    
    def get_last_task(self):
        if self.empty:
            return None
        else:
            return self._tasks[-1]

    def process(self, sync_table: SyncTable):
        sync_task = self._tasks[self._pc]
        if isinstance(sync_task, SyncTask):
            sync_task.time = self.recorder.max_time
            if sync_task.task_info[0] is None:
                assert sync_task.time == 0
            else:
                sync_table.update(sync_task)
            if sync_table.synchronized(sync_task.sync_id):
                self.increment_pc()
                self.recorder.max_time = sync_table.get_time(sync_task.sync_id)
                return True
            else:
                return False
        else:
            return True
    
    def increment_pc(self):
        self._pc += 1
        if self._pc >= len(self._tasks):
            self._pc = 0

    def decrement_pc(self):
        self._pc -= 1

    def __getitem__(self, item):
        return self._tasks[item]

    def __setitem__(self, key, value):
        self._tasks[key] = value

    def __len__(self):
        return len(self._tasks)
    
    def __iter__(self):
        return iter(self._tasks)

    def __str__(self):
        return str(self._tasks)

    def __repr__(self):
        return str(self._tasks)
    
    @property
    def empty(self):
        return len(self._tasks) == 0
    
    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator
    

# class MemoryPoint(STPoint):
#     def __init__(self, capacity):
#         super().__init__()
#         self.capacity = capacity
#         self.memory_recorder = MemoryRecorder()

#     def process(self, task_id: int):
#         for task in self._tasks:
#             if task_id == task.id:
#                 if isinstance(task, OutputTaskBlock):
#                     start_time, duration, iteration = task.consume()
#                     self.memory_recorder.record(task_id, iteration, start_time, duration)
#                     return True, task
#                 else:
#                     start_time, available_time, iteration, input_flag = task.consume()
#                     self.memory_recorder.record(task_id, iteration, start_time, 0)
#                     if input_flag:
#                         task.fire(iteration, available_time, self.memory_recorder.update, self.memory_recorder.update_start_time)
#                     else:
#                         task.fire(iteration, available_time, self.memory_recorder.update)
#                     return True, task
#         return False, None
    

# class ComputationPoint(STPoint):
#     def __init__(self, pe_info: Dict[Precision, Tuple[int]]):
#         super().__init__()
#         self.pe_info: Dict[Precision, Tuple[int]] = pe_info
#         self.compute_recorder = ComputationRecorder()
#         self.computation_evaluator = ComputationEvaluator(self.pe_info)
        
#     def process(self, task_id: int):
#         if self._tasks[self._pc].id == task_id:
#             task = self._tasks[self._pc]
#             duration = self.computation_evaluator(task)
#             # The start time is obtained considering data dependencies
#             start_time, iteration, consumed_ticks = task.consume()
#             self.compute_recorder.record(task_id, iteration, start_time, duration)
#             self.increment_pc()
#             # TODO: 计算如果可以和访存流水如何计算存储任务块的存活时间
#             task.callback(self.compute_recorder.max_time, consumed_ticks, duration)
#             task.fire(iteration, self.compute_recorder.max_time)
#             return True, task
#         else:
#             return False, None
        


    

# class CommunicationPoint(STPoint):
#     def __init__(self, bandwidth):
#         super().__init__()
#         self.bandwidth = bandwidth
#         self.communication_evaluator = CommunicationEvaluator(self.bandwidth)

#     def process(self, edges: List[Edge]) -> List[Edge]:
#         start_time_heap = []
#         tick_dict: Dict[Edge, List[Tick]] = {}
#         for edge in edges:
#             tick = edge._consume()  # same edge may appear N times for N different iterations
#             if edge in tick_dict:
#                 tick_dict[edge].append(tick)
#             else:
#                 tick_dict.update({edge: [tick]})
#             heapq.heappush(start_time_heap, (tick.time, (edge, tick.iteration)))
#         for edge in tick_dict:
#             ticks = tick_dict[edge]
#             for tick in ticks:
#                 if self.communication_evaluator.edge_not_mapped((edge, tick.iteration)):
#                     self.communication_evaluator.create_edge_map((edge, tick.iteration), self.edge_map[edge])
#         finished_edges, finish_time = self.communication_evaluator.eval(start_time_heap)
#         for edge in edges:
#             ticks = tick_dict[edge]
#             tick = ticks.pop(0)
#             if (edge, tick.iteration) in finished_edges:
#                 edge._fire(tick, finish_time, self.communication_evaluator.recorder.correct_time)
#             else:
#                 edge._put_back(tick, finish_time)
#         for edge in finished_edges:
#             edges.remove(edge[0])
#         return edges


# class ProcessorPoint(STPoint):
#     def __init__(self):
#         super().__init__()
#         # Configuration Info
#         self.memory_capacity: float = 0
#         self.pe_array_info: Dict[Precision, Tuple[int]] = {}
#         # Storage Tasks
#         self._memory: Set[STaskBlock] = {}
#         # Recorder
#         self.compute_recorder = ComputationRecorder()
#         self.memory_recorder = MemoryRecorder()
#         # Evaluator
#         self.computation_evaluator = ComputationEvaluator(self.pe_array_info)

#     @property
#     def memory(self):
#         return self._memory

#     @memory.setter
#     def memory(self, item: STaskBlock):
#         if item.id in self._memory:
#             raise ValueError('Task ' + str(id) + ' already exists')
#         assert TaskBlockType.is_storage_task(item.task_type), '{:s} is not a Storage task'.format(type(item).__name__)
#         self._memory[item.id] = item

#     @memory.deleter
#     def memory(self):
#         self._memory = {}

#     def get_memory_ids(self):
#         return [memory.id for memory in self._memory]

#     def process_computation(self, task_id: int):
#         if self._tasks[self._pc].id == task_id:
#             task = self._tasks[self._pc]
#             duration = self.computation_evaluator(task)
#             # The start time is obtained considering data dependencies
#             start_time, iteration, consumed_ticks = task.consume()
#             self.compute_recorder.record(task_id, iteration, start_time, duration)
#             self.increment_pc()
#             # TODO: 计算如果可以和访存流水如何计算存储任务块的存活时间
#             task.callback(self.compute_recorder.max_time, consumed_ticks, duration)
#             task.fire(iteration, self.compute_recorder.max_time)
#             return True, task
#         else:
#             return False, None
    
#     def process_memory(self, task_id: int):
#         for memory in self._memory:
#             if task_id == memory.id:
#                 start_time, available_time, iteration = memory.consume()
#                 self.memory_recorder.record(task_id, iteration, start_time, 0)
#                 memory.fire(iteration, available_time, self.memory_recorder.update)
#                 return True, memory
        
#     def process(self, task_id: int):
#         if task_id in self.get_task_ids():
#             self.process_computation(task_id)
#         elif task_id in self.get_memory_ids():
#             self.process_memory(task_id)
#         else:
#             raise TypeError("Task cannot be processed by " + self.__class__.__name__)
    

# class CorePoint(ProcessorPoint):
#     def __init__(self):
#         super().__init__()


# class ChipPoint(ProcessorPoint):
#     def __init__(self, core_x, core_y):
#         super().__init__()
        
        # 定义一些芯片的基本参数，如计算量、存储量等
        # self.int8_computation = 0
        # self.fp16_computation = 0
        # self.fp32_computation = 0
        # self.memory_capacity = 0
        # self.num_core = []

        # 路由任务，也可以考虑删掉
        # self._router_send = {}
        # self._router_receive = {}
        
        # 定义Recorder
        # 双工，分开记
        # self.router_recieve_recorder = RouterRecorder()
        # self.router_send_recorder = RouterRecorder()
    
    # @property
    # def memory(self):
    #     return self._memory

    # @memory.setter
    # def memory(self, item: TaskBlock):
    #     if item.id in self._memory:
    #         raise ValueError('There has been this memory task')
    #     assert TaskBlockType.is_storage_task(item.task_type), '{:s} is not a Storage task'.format(type(item).__name__)
    #     if hasattr(item, 'id'):
    #         self._memory[item.id] = item
    #     else:
    #         self._memory[item] = item

    # @memory.deleter
    # def memory(self):
    #     self._memory = {}

    # @property
    # def router_receive(self):
    #     return self._router_receive

    # @router_receive.setter
    # def router_receive(self, item):
    #     if item in self._router_receive:
    #         raise ValueError('There has been this router_receive task')
    #     if hasattr(item, 'id'):
    #         self._router_receive[item.id] = item
    #     else:
    #         self._router_receive[item] = item

    # @router_receive.deleter
    # def router_receive(self):
    #     self._router_receive = {}

    # @property
    # def router_send(self):
    #     return self._router_send

    # @router_send.setter
    # def router_send(self, item):
    #     if item in self._router_send:
    #         raise ValueError('There has been this router_send task')
    #     if hasattr(item, 'id'):
    #         self._router_send[item.id] = item
    #     else:
    #         self._router_send[item] = item

    # @router_send.deleter
    # def router_send(self):
    #     self._router_send = {}
        
    # def process(self, task_id: int):
    #     # Computational task
    #     if task_id in [task.id for task in self._tasks]:
    #         if self._tasks[self._pc].id == task_id:
    #             task = self._tasks[self._pc]
    #             # TODO: evaluate
    #             duration = 10
    #             # The start time is obtained considering data dependencies
    #             start_time, iteration, consumed_ticks = task.consume()
    #             self.compute_recorder.record(task_id, iteration, start_time, duration)
    #             self._pc += 1
    #             if self._pc >= len(self._tasks):
    #                 self._pc = 0
    #             # TODO: 计算如果可以和访存流水如何计算存储任务块的存活时间
    #             task.callback(self.compute_recorder.max_time, consumed_ticks, duration)
    #             task.fire(iteration, self.compute_recorder.max_time)
    #             return True, task
    #         else:
    #             return False, None
            
    #     # Storage task
    #     for memory in self._memory:
    #         if task_id == memory.id:
    #             start_time, available_time, iteration = memory.consume()
    #             self.memory_recorder.record(task_id, iteration, start_time, 0)
    #             memory.fire(iteration, available_time, self.memory_recorder.update)
    #             return True, memory
        
    #     return False, None


# class DDRPoint(MemoryPoint):
#     def __init__(self):
#         super().__init__()
#         # 定义一些DDR的基本参数
#         self.bank_num = 0

#     def process(self, task_id: int):
#         for task in self._tasks:
#             if task_id == task.id:
#                 if isinstance(task, OutputTaskBlock):
#                     start_time, duration, iteration = task.consume()
#                     self.memory_recorder.record(task_id, iteration, start_time, duration)
#                     return True, task
#                 else:
#                     start_time, available_time, iteration, input_flag = task.consume()
#                     self.memory_recorder.record(task_id, iteration, start_time, 0)
#                     if input_flag:
#                         task.fire(iteration, available_time, self.memory_recorder.update, self.memory_recorder.update_start_time)
#                     else:
#                         task.fire(iteration, available_time, self.memory_recorder.update)
#                     return True, task
#         return False, None
        

# st_point = STPoint()
# st_point[0] = 1
# st_point[2] = 'haha'
# st_point[2] = 'haha1'
# st_point.add(Coord((1, )), 32)
# # st_point.add(Coord((1, 2)), 3)
# print(st_point)
# a = st_point.pop(Coord((2, )), 'haha')
# print(a)
# print(st_point)
# st_point.add(Coord((5, )), 'ufidui')
# print(st_point.get_time(Coord((5, )), 'ufidui'))
