class Tick():
    def __init__(self, task_id, iteration = 0) -> None:
        self.task_id = task_id
        self.iteration = iteration
        self.time = 0
        
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
