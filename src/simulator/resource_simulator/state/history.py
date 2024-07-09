import logging
from typing import List
from src.simulator.resource_simulator.state.call import Call


class HistoryInfo():
    def __init__(self, call: Call = None):
        self.call: Call = call
        self.lock = False  # whether this is a lock history


class History():
    """
    Saves historical states
    """
    def __init__(self):
        self._state: List[HistoryInfo] = []
        # 接受的所有请求列表，方便生成直接的MiL文件
        self._request_stack: List[Call] = []

    def add_request(self, order):
        self._request_stack.append(order)

    def pop_request(self):
        if not self._state:
            logging.warning("Can't pop the order any more")
            return
        self._request_stack.pop()

    def push_state(self, reverse_call: Call = None) -> None:
        self._state.append(HistoryInfo(reverse_call))

    def pop_state(self) -> HistoryInfo:
        if not self._state:
            raise ValueError('There is no history')

        old_info = self._state.pop()

        reverse_call = old_info.call
        if reverse_call is not None:
            reverse_call.call()

        return old_info

    def lock(self):
        if not self._state:
            logging.warning("The state shouldn't be empty")
            return

        current_state = self._state.pop()
        current_state.lock = True
        self._state.clear()
        self._state.append(current_state)

    def reset(self):
        '''
        回到上一个lock的state
        '''
        state_num = len(self._state)
        for i in range(state_num - 1, 0, -1):
            if self._state[i].lock:
                break
            else:
                self.pop_state()

        return self._state[len(self._state) - 1]

    def mark(self, label: str):
        self._state.append(label)

    def pop_mark(self, label: str):
        while self._state[-1] != label:
            self.pop_state()
        self._state.pop()

    def get_code(self):
        codes = ''
        for code in self._request_stack:
            codes += 'M.'
            codes += code.to_string()
            codes += '\n'
        return codes
