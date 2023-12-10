class Recorder():
    def __init__(self, slot = 1):
        # List中存储三元组(ID, start, end)
        # ID为计算任务快的ID, start为该计算任务块的开始时间，end为结束时间
        self.recorder_time = []
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
    def __init__(self, slot = 1):
        super().__init__(slot)

    def recorder(self, id, min_start, time_duration):
        allow_time = max(min_start, self.max_time)
        time_tuple = (id, allow_time, allow_time + time_duration)
        self.recorder_time.append(time_tuple)
        
        self.max_time = allow_time + time_duration


class MemoryRecorder(Recorder):
    def __init__(self, slot = 1):
        super().__init__(slot)

    def recorder(self, id, min_start, time_duration):
        time_tuple = (id, min_start, min_start + time_duration)
        self.recorder_time.append(time_tuple)
        
        self.max_time = max(self.max_time, min_start + time_duration)

        

class RouterRecorder(Recorder):
    def __init__(self, slot = 1):
        super().__init__(slot)

    def recorder(self, id, min_start, time_duration):
        # 不重叠策略
        allow_time = max(min_start, self.max_time)
        time_tuple = (id, allow_time, allow_time + time_duration)
        self.recorder_time.append(time_tuple)
        
        self.max_time = allow_time + time_duration
        
