from typing import Set, Union
from ordered_set import OrderedSet
from src.simulator.resource_simulator.st_model.st_matrix import STMatrix
from src.simulator.resource_simulator.st_context import STContext
from src.simulator.task_rabbit.task_model.task_block import TaskBlock


class Scheduler():
    def __init__(self, st_matrix: STMatrix, st_context: STContext, initial_tasks: Set[TaskBlock] = None) -> None:
        self._activated_task_id = OrderedSet([task.id for task in initial_tasks])
        self._st_matrix = st_matrix
        self._st_context = st_context
        
    def add_activated_tasks(self, task: Union[TaskBlock, int]):
        if type(task) is int:
            self._activated_task_id.add(task)
        else:
            if task.activated:
                self._activated_task_id.add(task.id)
            for out_task in task.out_tasks:
                if out_task.activated:
                    self._activated_task_id.add(out_task.id)

    def get_next_activated_task(self) -> int:
        return self._activated_task_id.pop()

    def completed(self) -> bool:
        if len(self._activated_task_id):
            return False
        else:
            return True

    def schedule(self):
        task_id = self.get_next_activated_task()
        ml_coord = self._st_context.get_ml_coord(task_id)
        st_point = self._st_matrix.get_element(ml_coord)
        if st_point is None:
            raise Exception("st_point is None")
        
        status, task = st_point.proceed(task_id)
        if status:
            self.add_activated_tasks(task)
        else:
            self.add_activated_tasks(task_id)
        