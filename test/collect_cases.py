# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from os import listdir
from os.path import isfile, join


def collect_model_cases(cases_path):
    file_list = []
    for f in listdir(cases_path):
        if isfile(join(cases_path, f)) and f.endswith('.onnx'):
            file_list.append(f.split('.')[0])
    return file_list


def collect_task_cases(cases_path):
    file_list = []
    for f in listdir(cases_path):
        if isfile(join(cases_path, f)) and f.endswith('.task.txt'):
            file_list.append(f.split('.')[0])
    return file_list


def collect_map_cases(cases_path):
    file_list = []
    for f in listdir(cases_path):
        if isfile(join(cases_path, f)) and f.endswith('.map.txt'):
            file_list.append(f.split('.')[0])
    return file_list


def collect_asm_cases(cases_path):
    file_list = []
    for f in listdir(cases_path):
        if isfile(join(cases_path, f)) and f.endswith('.asm.txt'):
            file_list.append(f.split('.')[0])
    return file_list
