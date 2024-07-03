from typing import Dict, List, Tuple
from src.simulator.resource_simulator.st_model.tick import Tick
from src.simulator.task_rabbit.task_model.edge import Edge
from src.simulator.resource_simulator.st_model.hop import Hop


class Recorder():
    def __init__(self, slot=1):
        # {(task ID, iteration): [start time, end time]}
        self.recorder_time: Dict[Tuple[int, int], List[float, float]] = {}
        # 当前计算负载的最晚结束时间
        # 后续加入的任务的起始时间需要晚于该时间
        self.max_time: float = 0
        # TODO: 加入多slot
        self.slot = slot
    
    # Input the time duration of certain recorder
    def record(self, time_duration):
        pass

    def get_record(self, task_id: int, iteration: int):
        return self.recorder_time[(task_id, iteration)]

    def reset(self):
        self.recorder_time = []
        self.max_time = 0
        

class ComputationRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def __contains__(self, task_id: int):
        for task_info in self.recorder_time:
            if task_info[0] == task_id:
                return True
        return False

    def record(self, id: int, iteration: int, min_start: float, time_duration: float):
        allow_time = max(min_start, self.max_time)
        self.max_time = allow_time + time_duration
        time_record = {(id, iteration): [float(allow_time), float(self.max_time)]}
        self.recorder_time.update(time_record)

    def remove(self, id: int, iteration: int):
        del self.recorder_time[(id, iteration)]

    def reset(self, time: float):
        self.max_time = time


class MemoryRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def record(self, id: int, iteration: int, min_start: float, time_duration: float):
        allow_time = max(min_start, self.max_time)
        time_record = {(id, iteration): [allow_time, allow_time + time_duration]}
        self.recorder_time.update(time_record)

    def update(self, id: int, iteration: int, end_time: int):
        if end_time > self.recorder_time[(id, iteration)][1]:
            self.recorder_time[(id, iteration)][1] = end_time

    def update_start_time(self, id: int, iteration: int, start_time: int):
        # 如果当前存储块为直接从输入块获取数据，则开始时间为使用该存储的计算块开始计算的时间
        if self.recorder_time[(id, iteration)][0] < start_time:
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
        self.recorder_time: Dict[Tuple[Edge, int, Hop], CommunicationRecord] = {}

    def __iter__(self):
        self._iter_keys = iter(self.recorder_time.keys())
        return self

    def __next__(self):
        try:
            key = next(self._iter_keys)
            return key, self.recorder_time[key]
        except StopIteration:
            raise StopIteration

    def __contains__(self, __key: Tuple[Edge, int, Hop]):
        for key in self.recorder_time:
            if __key[0] == key[0] and __key[1] == key[1] and __key[2].src == key[2].src and __key[2].dst == key[2].dst and __key[2].link_id == key[2].link_id:
                return True
        return False
    
    def __getitem__(self, __key: Tuple[Edge, int, Hop]):
        for key in self.recorder_time:
            if __key[0] == key[0] and __key[1] == key[1] and __key[2].src == key[2].src and __key[2].dst == key[2].dst and __key[2].link_id == key[2].link_id:
                return self.recorder_time[key]
        return self.recorder_time[__key]
    
    def update(self, __key: Tuple[Edge, int, Hop], value: CommunicationRecord):
        for key in self.recorder_time:
            if __key[0] == key[0] and __key[1] == key[1] and __key[2].src == key[2].src and __key[2].dst == key[2].dst and __key[2].link_id == key[2].link_id:
                self.recorder_time.update({key: value})
                return
        self.recorder_time.update({__key: value})

    def correct_time(self, tick: Tick, time: float):
        assert time >= 0
        start_time = float("inf")
        for key in self.recorder_time:
            if key[0] == tick.edge and key[1] == tick.iteration:
                self.recorder_time[key].start_time += time
                self.recorder_time[key].end_time += time
                if start_time > self.recorder_time[key].start_time:
                    start_time = self.recorder_time[key].start_time
        tick.start_callback(tick.task_id, tick.iteration, start_time)


class RouterRecorder(Recorder):
    def __init__(self, slot=1):
        super().__init__(slot)

    def recorder(self, id, min_start, time_duration):
        # 不重叠策略
        allow_time = max(min_start, self.max_time)
        time_tuple = (id, allow_time, allow_time + time_duration)
        self.recorder_time.append(time_tuple)
        
        self.max_time = allow_time + time_duration
        
