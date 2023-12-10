#!/usr/bin/env python
# coding: utf-8

"""
EvaluationModel 类负责各种评估的响应
"""

from resource_simulator.evaluation_model.evaluation import MemoryEvaluation
from resource_simulator.st_model.st_matrix import STMatrix
from task_rabbit.task_model.task_graph import TaskGraph
from resource_simulator.st_context import STContext
from resource_simulator.evaluation_model.memory_evaluator import MemoryEvaluator


class EvaluationModel():
    def __init__(self, task_graph, st_matrix, st_context):
        self._task_graph = task_graph  # type: TaskGraph
        self._st_matrix = st_matrix    # type: STMatrix
        self._context = st_context     # type: STContext

    def eva_data_transmission(self, src_task_id, dst_task_id):
        '''
        判断从src_task_id到dst_task_id的数据传输量
        先判断src_task_id到dst_task_id是否直接相连，否则数据传输量为0
        一般情况下，数据从传输量就是src_task或dst_task中相应边EdgeInfo中的size大小
        也可以带一个数据类型信息，表示传递的数据是什么类型的。
        '''
        pass

    def eva_space_column(self, space_coord):
        '''
        评估一个column, 当前阶段可以表示一个Core
        返回综合评估信息Evaluation
        '''
        pass

    def eva_st_point(self, ml_coord):
        '''
        评估一个point, 具体表现为评估一个Core的一个Phase
        返回综合评估信息Evaluation
        '''
        pass

    def eva_st_matrix(self, ml_coord=None):
        '''
        评估一个matrix, 可以为几个Core，1个Phase Group， 几个Phase Group, 直至这个Matrix
        时间上，可以是一个Phase, 或几个Phase, 或一个Step
        返回综合评估信息Evaluation
        '''
        pass

    def eva_from_task_to_task(self, start_task_id, end_task_id):
        '''
        评估任务图图上，从start_task到end_task所涉及的STMatrix区域的评估
        暂时先不做
        返回综合评估信息Evaluation
        '''
        pass

    # 这类的评估可以优先实现
    # Non-empty Space number
    def get_space_num(self, top_space_coord=None, time_coord=None) -> int:
        '''
        获得已有的Space的数目
        top_space_coord 表示在该坐标之下的一层space的数目
        如果为None则返回core的数目
        time_coord暂时忽略
        '''
        pass

    def get_max_time(self, top_space_coord=None) -> int:
        '''
        获得在top_space_coord下的最大Phase数
        '''
        pass

    def get_memory(self, ml_coord) -> MemoryEvaluation:
        '''
        获得ml_coord的memory评估
        ml_coord表示一个space column
        返回MemoryEvaluation
        '''
        core = self._st_matrix.get_space(ml_coord)
        return MemoryEvaluator.evaluate_memory(core)

    def get_clock(self, ml_coord) -> int:
        '''
        获得ml_coord的clock数评估
        '''
        pass

    def get_computation(self, ml_coord):
        '''
        获得ml_coord的compution评估
        返回ComputationEvaluation
        '''
        pass

    def get_route_size(self, src_ml_coord, dst_ml_coord) -> int:
        '''
        获得从src_ml_coord到dst_ml_coord的路由量
        目前，coord应表示core或point
        '''
        pass

    def get_memory_overflow_space(self):
        '''
        返回超内存的Core, 以及相应的内存量
        '''
        pass

    def get_computation_bottleneck(self):
        '''
        返回计算时间最长的那个Point, 以及相应的计算时间
        '''
        pass

    def get_route_bottleneck(self):
        '''
        返回路由最大的两端，以及相应的路由量
        '''
        pass

    def get_phase_group_overflow(self):
        '''
        返回超phase group数量限制的Chip, 以及相应的phase group数量
        '''
        pass

    def get_core_overflow(self):
        '''
        返回超所core数量限制的Chip, 以及相应的core数量
        '''
        pass

    def get_phase_overflow(self):
        '''
        返回超所phase数量限制的step(及core), 以及相应的phase数量
        '''
        pass

    def get_state(self):
        '''
        返回当前状态下的执行状态
        如果成功返回SUCCESS
        如果失败返回具体的失败类型，如超内存、不满足数据依赖等。
        '''
        pass

    def is_all_mapped(self):
        '''
        是否任务图上的所有结点都map到了matrix上
        '''
        pass

    def is_mapped(self, task_id):
        '''
        task_id代表的任务块是否map到了matrix上
        '''
        pass

    def mapped_in(self, task_id):
        '''
        返回task_id代表的任务块map到了哪
        '''
        pass

    def get_mapped_tasks(self):
        '''
        获得已映射的task_id
        '''
        pass

    def get_unmapped_tasks(self):
        '''
        获得没映射的task_id
        '''
        pass

    # Check
    def check_graph(self):
        '''
        检查任务图是否合法
        '''
        pass

    def check_st_matrix(self):
        '''
        检查st_matrix是否合法
        '''
        pass

    def check_mapping(self):
        '''
        检查任务图在st_matrix的映射是否合法
        '''
        pass
