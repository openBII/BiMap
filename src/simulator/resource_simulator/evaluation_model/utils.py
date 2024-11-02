from collections import defaultdict
import heapq
from typing import List


# 检查两条路径的重合线段
def paths_overlap_directed(pathA, pathB):
    for i in range(len(pathA)):
        for j in range(len(pathB)):
            if pathA[i] == pathB[j]:
                return True
    return False

def create_overlap_groups(keys, paths):
    overlap_groups = defaultdict(set)
    for key1 in keys:
        for key2 in keys:
            if paths_overlap_directed(paths[key1], paths[key2]):
                overlap_groups[key1].add(key2)
    return overlap_groups