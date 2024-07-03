from src.simulator.task_rabbit.task_model.task_block import TaskBlock


class SyncTask():
    def __init__(self, sync_id: int, task: TaskBlock, iteration: int = 0) -> None:
        self.sync_id = sync_id
        if task is None:
            self.task_info = (None, iteration)
        else:
            self.task_info = (task.id, iteration)
        self.time: float = 0

    def __repr__(self) -> str:
        return "Sync ID: " + str(self.sync_id) 