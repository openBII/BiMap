from typing import Dict, List, Tuple
from src.simulator.task_rabbit.task_model.edge import Edge


class Recorder():
    def __init__(self, slot=1):
        # {(task ID, iteration): [start time, end time]}
        self.recorder_time: Dict[Tuple[int, int], List[int, int]] = {}
        # 当前计算负载的最晚结束时间
        # 后续加入的任务的起始时间需要晚于该时间
        self.max_time = 0
        # TODO: 加入多slot
        self.slot = slot
    
    # Input the time duration of certain recorder
    def recorder(self, time_duration):
        pass

    def reset(self):
        self.recorder_time = []
        self.max_time = 0
        

class ComputationRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def record(self, id: int, iteration: int, min_start: int, time_duration: int):
        allow_time = max(min_start, self.max_time)
        self.max_time = allow_time + time_duration
        time_record = {(id, iteration): [allow_time, self.max_time]}
        self.recorder_time.update(time_record)


class MemoryRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def record(self, id: int, iteration: int, min_start: int, time_duration: int):
        time_record = {(id, iteration): [min_start, min_start + time_duration]}
        self.recorder_time.update(time_record)

    def update(self, id: int, iteration: int, end_time: int):
        if end_time > self.recorder_time[(id, iteration)][1]:
            self.recorder_time[(id, iteration)][1] = end_time

    def update_start_time(self, id: int, iteration: int, start_time: int):
        # 如果当前存储块为直接从输入块获取数据，则开始时间为使用该存储的计算块开始计算的时间
        if self.recorder_time[(id, iteration)][0] == 0:
            self.recorder_time[(id, iteration)][0] = start_time


class CommunicationRecord:
    def __init__(self, start_time: float = 0, end_time: float = 0, percent: float = 0) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.percent = percent


class CommunicationRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)
        # recorder: {(Edge, iteration, Hop): CommunicationRecord}
        self.recorder_time: Dict[Tuple, CommunicationRecord] = {}

    def __contains__(self, key: Tuple):
        return key in self.recorder_time
    
    def __getitem__(self, key: Tuple):
        return self.recorder_time[key]
    
    def update(self, key: Tuple, value: CommunicationRecord):
        self.recorder_time.update({key: value})
        

class RouterRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def recorder(self, id, min_start, time_duration):
        # 不重叠策略
        allow_time = max(min_start, self.max_time)
        time_tuple = (id, allow_time, allow_time + time_duration)
        self.recorder_time.append(time_tuple)
        
        self.max_time = allow_time + time_duration
        
