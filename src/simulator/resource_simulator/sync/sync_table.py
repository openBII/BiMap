from typing import Dict, Tuple
from src.simulator.resource_simulator.sync.sync_task import SyncTask


class SyncEntry:
    def __init__(self) -> None:
        self.time: float = 0
        self.sync_state: Dict[Tuple[int, int], bool] = {}

    def __iter__(self):
        return iter(self.sync_state)
    
    def __setitem__(self, task_info: Tuple[int, int], state: bool):
        self.sync_state[task_info] = state

    def __getitem__(self, task_info: Tuple[int, int]):
        return self.sync_state[task_info]


class SyncTable:
    def __init__(self) -> None:
        self.counter = 0
        self.table: Dict[int, SyncEntry] = {}

    def __contains__(self, task_id: int):
        for sync_id in self.table:
            entry = self.table[sync_id]
            for task_info in entry:
                if task_info[0] == task_id:
                    return True
        return False
    
    def __getitem__(self, sync_id: int):
        return self.table[sync_id]

    def get_sync_id(self):
        self.counter += 1
        return self.counter - 1

    def synchronized(self, sync_id: int):
        entry = self.table[sync_id]
        for task in entry:
            if not entry[task]:
                return False
        return True

    def get_time(self, sync_id: int):
        return self.table[sync_id].time
    
    def update(self, sync_task: SyncTask):
        self.table[sync_task.sync_id][sync_task.task_info] = True
        self.table[sync_task.sync_id].time = max(sync_task.time, self.table[sync_task.sync_id].time)

    def add(self, sync_task: SyncTask, task_id: int = None, iteration: int = None):
        if sync_task.sync_id not in self.table:
            self.table[sync_task.sync_id] = SyncEntry()
        if task_id is None or iteration is None:
            self.table[sync_task.sync_id][sync_task.task_info] = False
        else:
            self.table[sync_task.sync_id][(task_id, iteration)] = False
