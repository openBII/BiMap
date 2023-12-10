"""
History 类保存历史状态

2019-04-10, weihao.zhang
"""

import copy
import logging
from typing import List


class HistoryInfo():
    def __init__(self, content, call):
        self.content = content
        self.call = call
        self.lock = False  # whether this is a lock history


class History():
    def __init__(self):
        self._state = []  # type: List[HistoryInfo]
        self._order_stack = []

    def add_order(self, order):
        self._order_stack.append(order)

    def pop_order(self):
        if not self._state:
            logging.warn("Can't pop the order any more")
            return
        self._order_stack.pop()

    def new_state(self, value, reverse_call=None):
        self._state.append(HistoryInfo(copy.deepcopy(value), reverse_call))
        # for item in self.state:
        #   print(item._core_group)
        # print()

    def old_state(self):
        if not self._state:
            raise ValueError('There is no history')

        if len(self._state) == 1:
            raise ValueError('Can\'t pop anymore')

        old_info = self._state.pop()
        info = self._state[len(self._state) - 1]

        state = info.content
        reverse_call = old_info.call
        if reverse_call is not None:
            reverse_call.call()

        return copy.deepcopy(state)

    def lock(self):
        if not self._state:
            logging.warn("The state shouldn't be empty")
            return

        current_state = self._state.pop()
        current_state.lock = True
        self._state.clear()
        self._state.append(current_state)

    def reset(self):
        state_num = len(self._state)
        for i in range(state_num - 1, 0, -1):
            if self._state[i].lock:
                break
            else:
                self.old_state()

        info = self._state[len(self._state) - 1]
        state = info.content
        return copy.deepcopy(state)

    def get_code(self):
        codes = ''
        for code in self._order_stack:
            codes += 'M.'
            codes += code.to_string()
            codes += '\n'
        return codes
