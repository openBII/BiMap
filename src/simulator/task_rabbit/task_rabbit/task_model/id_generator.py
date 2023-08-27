# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


class IDGenerator():
    """
    IDGenerator 类负责产生不重复的ID
    """
    task_num = 0

    @staticmethod
    def get_next_task_id() -> int:
        IDGenerator.task_num += 1
        return IDGenerator.task_num

    @staticmethod
    def set_base_task_id(base_id):
        IDGenerator.task_num = base_id

    @staticmethod
    def get_connection_id(in_task, out_task) -> str:
        '''
        构造一个连接的ID
        现在连接还暂时没有ID
        '''
        pass
