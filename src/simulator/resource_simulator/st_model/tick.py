from enum import Enum
from typing import Callable


class CallbackType(Enum):
    STORAGE = 0
    STATIC = 1


class Tick():
    def __init__(self, task_id: int, iteration: int = 0, time: int = 0, callback: Callable = None, start_callback: Callable = None) -> None:
        self.task_id = task_id
        self.iteration = iteration
        self.time = time
        # TODO(huanyu): 改成字典 Dict[CallbackType, Callable]
        self.callback = callback
        self.start_callback = start_callback
        self.edge_callback: Callable = None
        self.edge = None
        
    def __str__(self):
        return 'tick' + str(self.task_id) + '.' + str(self.iteration)

    def __eq__(self, other):
        if other is None:
            return False
        return self.task_id == other.task_id and self.iteration == other.iteration
    
    def __lt__(self, other):
        # 为了优先级队列
        return  self.iteration < other.iteration

    def __hash__(self):
        return hash((self.task_id, self.iteration))
