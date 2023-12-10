from sortedcontainers import SortedSet
from queue import Queue
from resource_simulator.st_model.st_matrix import STMatrix
from resource_simulator.st_context import STContext


class Scheduler():
    def __init__(self, st_matrix: STMatrix, st_context: STContext) -> None:
        self._activated_task_id = SortedSet()
        self._st_matrix = st_matrix
        self._st_context = st_context
        
    def add_activated_task(self, task_id):
        self._activated_task_id.add(task_id)

    def get_next_activated_task(self) -> int:
        # if self._activated_task_id.empty():
        #     return None
        return self._activated_task_id.pop()

    def schedule(self):
        next_task_id = self.get_next_activated_task()
        ml_coord = self._st_context.get_ml_coord(next_task_id)
        st_point = self._st_matrix.get_element(ml_coord)
        if st_point is None:
            raise Exception("st_point is None")
        
        status, new_enabled_id = st_point.proceed(next_task_id)

        if status is False:
            self.add_activated_task(next_task_id)
        else:
            if new_enabled_id:
                self.add_activated_task(new_enabled_id)
        
        