from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.resource_simulator.evaluation_model.recorder import MemoryRecorder
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock


class MemoryPoint(STPoint):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.recorder = MemoryRecorder()

    def __repr__(self):
        return 'capacity: ' + repr(self.capacity) + '\n'

    def process(self, task_id: int):
        task: STaskBlock
        for task in self._tasks:
            if task_id == task.id:
                if isinstance(task, OutputTaskBlock):
                    start_time, duration, iteration = task.consume()
                    self.recorder.record(task_id, iteration, start_time, duration)
                    return True
                elif isinstance(task, StaticTaskBlock):
                    start_time, iteration = task.consume()
                    self.recorder.record(task_id, iteration, start_time, time_duration=0)
                    task.fire(iteration, start_time)
                    return True
                else:
                    start_time, available_time, iteration, input_flag = task.consume()
                    self.recorder.record(task_id, iteration, start_time, 0)
                    if input_flag:
                        task.fire(iteration, available_time, self.recorder.update, self.recorder.update_start_time)
                    else:
                        task.fire(iteration, available_time, self.recorder.update)
                    return True
        return False