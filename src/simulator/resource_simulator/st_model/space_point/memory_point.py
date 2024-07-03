from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.recorder import MemoryRecorder
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.resource_simulator.sync.sync_table import SyncTable
from src.simulator.resource_simulator.sync.sync_task import SyncTask


class MemoryPoint(STPoint):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.recorder = MemoryRecorder()

    def __repr__(self):
        return 'capacity: ' + repr(self.capacity) + '\n'

    def process(self, task_id: int, sync_table: SyncTable):
        if not super().process(sync_table):
            return False
        index = -1
        for i, task in enumerate(self._tasks):
            if not isinstance(task, SyncTask):
                if task_id == task.id:
                    index = i
                    break
        if index == -1:
            raise ValueError("Task {:d} is not in this MemoryPoint".format(task_id))
        task = self._tasks[index]
        # 如果存储任务在同步barrier之后, 不能执行
        for i in range(self._pc, index):
            if isinstance(self._tasks[i], SyncTask):
                return False
        head = self._tasks[self._pc]
        self._tasks[self._pc] = task
        self._tasks[index] = head
        self.increment_pc()

        if isinstance(task, OutputTaskBlock):
            start_time, duration, iteration = task.consume()
            self.recorder.record(task_id, iteration, start_time, duration)
            return True
        elif isinstance(task, StaticTaskBlock):
            start_time, iteration = task.consume()
            self.recorder.record(task_id, iteration, start_time, time_duration=0)
            task.fire(iteration, start_time)
            return True
        elif isinstance(task, STaskBlock):
            start_time, available_time, iteration, input_flag = task.consume()
            self.recorder.record(task_id, iteration, start_time, 0)
            available_time = max(available_time, self.recorder.max_time)
            if input_flag:
                task.fire(iteration, available_time, self.recorder.update, self.recorder.update_start_time)
            else:
                task.fire(iteration, available_time, self.recorder.update)
            return True
        else:
            raise TypeError("Unsupported task type")