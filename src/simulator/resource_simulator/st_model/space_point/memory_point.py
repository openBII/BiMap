from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.recorder import MemoryRecorder
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock

class MemoryPoint(STPoint):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.memory_recorder = MemoryRecorder()

    def __repr__(self):
        return 'capacity: ' + repr(self.capacity) + '\n'

    def process(self, task_id: int):
        for task in self._tasks:
            if task_id == task.id:
                if isinstance(task, OutputTaskBlock):
                    start_time, duration, iteration = task.consume()
                    self.memory_recorder.record(task_id, iteration, start_time, duration)
                    return True, task
                else:
                    start_time, available_time, iteration, input_flag = task.consume()
                    self.memory_recorder.record(task_id, iteration, start_time, 0)
                    if input_flag:
                        task.fire(iteration, available_time, self.memory_recorder.update, self.memory_recorder.update_start_time)
                    else:
                        task.fire(iteration, available_time, self.memory_recorder.update)
                    return True, task
        return False, None