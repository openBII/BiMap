#!/usr/bin/env python
# coding: utf-8

"""
EvaluationModel 类负责各种评估的响应,
"""

from src.simulator.resource_simulator.evaluation_model.evaluation import MemoryEvaluation

# TODO: pipeline加在哪


class MemoryEvaluator():
    # @staticmethod
    # def evaluate_memory(core: SpaceColumn):
    #     st_points = core.get_all_st_point()
    #     total_memory = {}
    #     access_count = {}

    #     for phase, st_point in st_points.items():
    #         one_phase_memory = 0
    #         access_count[phase] = 0
    #         for storage_task in st_point.memory.values():
    #             one_phase_memory += storage_task.get_storage().local_storage
    #             access_count[phase] += storage_task.get_storage().access_count
    #         total_memory[phase] = one_phase_memory
    #     return MemoryEvaluation(max(total_memory.values()), total_memory, access_count)

    @staticmethod
    def evaluate_memory(core):
        st_points = core.get_all_st_point()
        total_memory = {}
        for phase, st_point in st_points.items():   # 没有考虑pipeline和边两端rearrange问题
            one_phase_memory = 0
            total_buffer = set()
            for storage_task in st_point.memory.values():
                total_buffer.add(storage_task)
            for in_task in st_point.axon.in_tasks:
                total_buffer.add(in_task)
            for out_task in st_point.axon.out_tasks:
                total_buffer.add(out_task)
            # for in_task in st_point.soma1.in_tasks:
            #     total_buffer.add(in_task)
            # for out_task in st_point.soma1.out_tasks:
            #     total_buffer.add(out_task)
            # for in_task in st_point.soma2.in_tasks:
            #     total_buffer.add(in_task)
            # for out_task in st_point.soma2.out_tasks:
            #     total_buffer.add(out_task)

            for storage_task in total_buffer:
                one_phase_memory += storage_task.get_storage().local_storage
            total_memory[phase] = one_phase_memory
        return MemoryEvaluation(max(total_memory.values()), total_memory)
