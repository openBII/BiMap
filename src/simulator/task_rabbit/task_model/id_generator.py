class IDGenerator():
    """
    IDGenerator 类负责产生不重复的ID
    """
    task_num = 0

    @staticmethod
    def get_next_task_id() -> int:
        IDGenerator.task_num += 1
        return IDGenerator.task_num - 1

    @staticmethod
    def set_base_task_id(base):
        if type(base) is int:
            IDGenerator.task_num = base
        else:
            if len(base.get_all_node_ids()) > 0:
                IDGenerator.task_num = max(base.get_all_node_ids())

    @staticmethod
    def get_connection_id(in_task, out_task) -> str:
        '''
        构造一个连接的ID
        现在连接还暂时没有ID
        '''
        pass
