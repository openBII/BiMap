#!/usr/bin/env python
# coding: utf-8

"""
STPoint类描述STMatrix中的一个点
具体的说，在tianjicX架构中，表示一个Core model的一个Phase
"""

from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.evaluation_model.recorder import ComputationRecorder, MemoryRecorder, RouterRecorder
from src.simulator.resource_simulator.st_model.tick import Tick

from typing import List


class STPoint():
    def __init__(self):
        #TODO: Slots
        self._tasks: List[TaskBlock] = []
        self._pc = 0
        
        
        
    def add_task(self, task: TaskBlock):
        self._tasks.append(task)

    def get_tasks(self):
        return self._tasks

    def proceed(self, tick: Tick):
        self._ticks.append(tick)
        
        # 如果第一个task没有触发
        # return 
    
        new_tick = Tick(0)
        return new_tick

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


class ChipPoint(STPoint):
    def __init__(self):
        super().__init__()
        
        # 定义一些芯片的基本参数，如计算量、存储量等
        self.int8_computation = 0
        self.fp16_computation = 0
        self.fp32_computation = 0
        self.memory_capacity = 0
        self.core_num = 0
        
        # 计算任务放入到self._tasks中，其他任务另单独创造容器
        # 存储任务
        self._memory = {}

        # 路由任务，也可以考虑删掉
        self._router_send = {}
        self._router_receive = {}
        
        # 定义Recorder
        self.compute_recorder = ComputationRecorder()
        # 双工，分开记
        self.router_recieve_recorder = RouterRecorder()
        self.router_send_recorder = RouterRecorder()
    
    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, item: TaskBlock):
        if item.id in self._memory:
            raise ValueError('There has been this memory task')
        assert TaskBlockType.is_storage_task(item.task_type), '{:s} is not a Storage task'.format(type(item).__name__)
        if hasattr(item, 'id'):
            self._memory[item.id] = item
        else:
            self._memory[item] = item

    @memory.deleter
    def memory(self):
        self._memory = {}

    @property
    def router_receive(self):
        return self._router_receive

    @router_receive.setter
    def router_receive(self, item):
        if item in self._router_receive:
            raise ValueError('There has been this router_receive task')
        if hasattr(item, 'id'):
            self._router_receive[item.id] = item
        else:
            self._router_receive[item] = item

    @router_receive.deleter
    def router_receive(self):
        self._router_receive = {}

    @property
    def router_send(self):
        return self._router_send

    @router_send.setter
    def router_send(self, item):
        if item in self._router_send:
            raise ValueError('There has been this router_send task')
        if hasattr(item, 'id'):
            self._router_send[item.id] = item
        else:
            self._router_send[item] = item

    @router_send.deleter
    def router_send(self):
        self._router_send = {}
        
    def proceed(self, task_id: int, timer):
        
        if task_id in self._tasks:
            if self._tasks[self._pc].id == task_id:
                task = self._tasks[self._pc]
                self.compute_recorder.record(task_id, 0, 10, timer)
                self._pc += 1
                if self._pc >= len(self._tasks):
                    self._pc = 0
                task.fire(1)
                activated_tasks = set()
                for activated_task in task.out_tasks:
                    if activated_task.activated:
                        activated_tasks.add(activated_task.id)
                return True, activated_tasks
            else:
                return False, None
    
        if task_id in self._memory:
            pass
        
        return False, None


class DDRPoint(STPoint):
    def __init__(self):
        super().__init__()
        
        # 定义一些DDR的基本参数
        self.capcity = 0
        self.bank_num = 0
        
        # 定义Recorder
        self.compute_recorder = ComputationRecorder()
        self.router_recieve_recorder = RouterRecorder()
        self.router_send_recorder = RouterRecorder()
        
    
        

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