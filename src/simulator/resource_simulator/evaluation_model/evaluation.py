#!/usr/bin/env python
# coding: utf-8

"""
Evaluation 类负责记录评估结果
分为MemoryEvaluation、ComputationEvaluation和RouteEvaluation
"""



class MemoryEvaluation():
    def __init__(self, total_memory, details=None, access_count=None):
        self.total_memory = total_memory
        self.overflow = 0 # (total_memory > HardwareRule.MEMORY_SIZE)
        self.details = details
        self.access_count = access_count


class ComputationEvaluation():
    def __init__(self):
        self.max_computation = None
        self.phase_computation = None


class RouteEvaluation():
    def __init__(self):
        pass


class Evaluation():
    def __init__(self):
        self.memory_evaluation = None
        self.computation_evaluation = None
        self.route_evaluation = None
