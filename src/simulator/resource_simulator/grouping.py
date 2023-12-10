"""
Grouping 类记录网络层于核模型的分组信息

2019-04-10, weihao.zhang
"""

from typing import Dict, List
from collections import OrderedDict


class Grouping():
    def __init__(self):

        self.group_num = 0
        # bi-direction mapping
        # layer grouping, not update
        self._group_to_layers = OrderedDict()  # type: Dict[str, List[str]]
        self._layer_to_groups = OrderedDict()  # type: Dict[str, str]

        # core grouping, update
        self._group_to_cores = OrderedDict()  # type: Dict[str, List[str]]
        self._core_to_groups = OrderedDict()  # type: Dict[str, List[str]]

    # grouping interface
    def set_group(self, group_key: str, layer_keys: List[str], core_keys: List[str]):
        if group_key in self._group_to_cores:
            self._refresh_group(group_key, layer_keys, core_keys)
            return

        self.group_num += 1

        self._group_to_layers[group_key] = layer_keys
        self._refresh_core_to_groups(group_key, core_keys)
        self._group_to_cores[group_key] = core_keys

    def _refresh_group(self, group_key, layer_keys, core_keys):
        self._group_to_layers[group_key] = layer_keys    # can be removed?
        original_cores = set(self._group_to_cores[group_key])
        self._group_to_cores[group_key] = core_keys
        self._refresh_core_to_groups(group_key, core_keys, original_cores)

    def _refresh_core_to_groups(self, group_key, core_keys, original_cores=None):
        if original_cores is None:
            original_cores = set()

        for core_key in core_keys:
            if core_key in original_cores:
                original_cores.remove(core_key)
                continue
            if core_key not in self._core_to_groups:
                self._core_to_groups[core_key] = [group_key]
            else:
                self._core_to_groups[core_key].append(group_key)

        for core_key in original_cores:
            if core_key in self._core_to_groups:
                self._core_to_groups[core_key].remove(group_key)
                if not self._core_to_groups[core_key]:
                    self._core_to_groups.pop(core_key)

    def remove_core(self, core_key):
        groups_contain_the_core = self._core_to_groups.get(core_key, [])
        for group_key in groups_contain_the_core:
            self._group_to_cores[group_key].remove(core_key)

    def add_cores(self, groups_key, core_keys):
        pass

    def get_group_to_layers(self) -> Dict[str, List[str]]:
        return self._group_to_layers

    def get_layer_to_group(self) -> Dict[str, str]:
        return self._layer_to_groups

    def get_core_to_groups(self) -> Dict[str, List[str]]:
        return self._core_to_groups

    def get_group_to_cores(self) -> Dict[str, List[str]]:
        return self._group_to_cores

    def get_layer_keys_by_group_key(self, group_key: str) -> List[str]:
        return self._group_to_layers.get(group_key, [])

    def get_core_keys_by_group_key(self, group_key: str) -> List[str]:
        return self._group_to_cores.get(group_key, [])

    def get_group_key_by_layer_key(self, layer_key: str) -> str:
        return self._layer_to_groups.get(layer_key, None)

    def get_group_keys_by_core_key(self, core_key: str) -> List[str]:
        return self._core_to_groups.get(core_key, None)

    def __contains__(self, item):
        return item in self._group_to_cores
