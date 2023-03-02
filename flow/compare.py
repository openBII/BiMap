# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
import os
import pickle
from typing import Dict

import numpy as np
from top.global_config import GlobalConfig

from flow import execute


class Compare:
    @staticmethod
    def compare_two_files(file1: str, file2: str):
        not_match = os.system("diff " + file1 + " " + file2 + "  >> /dev/null")
        return False if not_match else True

    # 输入文件名list, 表示要比较的文件名
    # 以及两个目录，表示两个存放比较文件的文件夹
    # 只要有一个不一样的，就返回False
    @staticmethod
    def compare_files(golden_dir, compared_dir, file_list = None):
        if file_list is None:
            file_list = os.listdir(golden_dir)

        compared_list = os.listdir(compared_dir)
        if len(compared_list) != len(file_list):
            return False

        result = True
        for file_name in file_list:
            golden_file = golden_dir + file_name
            compared_file = compared_dir + file_name
            result = Compare.compare_two_files(
                golden_file, compared_file) and result
        return result

    @staticmethod
    def compare_checkpoints(case_name, check_list):
        if len(check_list) <= 1:
            return True

        # Check
        for check_point in check_list:
            if check_point not in execute.Execution.OUTPUT_MAP:
                logging.error("Nothing to be compared")
                raise ValueError()

        # 这个不能乱比较
        result_file = GlobalConfig.Path['temp'] + case_name + '/'
        os.makedirs(result_file, exist_ok=True)
        result_file += 'compare.txt'

        golden_dir = execute.Execution.OUTPUT_MAP[check_list[0]].replace(
            'CASE_NAME', case_name)
        golden_list = os.listdir(golden_dir)
        total_result = False
        with open(result_file, 'w') as f:
            total_result = True
            for i, check_point in enumerate(check_list):
                if i == 0:
                    result = True  # golden
                else:
                    compared_dir = execute.Execution.OUTPUT_MAP[check_point].replace(
                        'CASE_NAME', case_name)
                    result = Compare.compare_files(
                        golden_dir, compared_dir, golden_list)
                total_result = total_result and result
                f.write(check_point + ': ' + str(result) + '\n')

        if total_result:
            logging.warning('Compare Pass!')
        else:
            logging.warning('Compare Failed!')
        return total_result

    @staticmethod
    def compare_numpy_arrays(golden_data_path: str, compared_data: Dict[int, np.ndarray]):
        golden_data: Dict[int, np.ndarray] = {}
        with open(golden_data_path, 'rb') as f:
            golden_data = pickle.load(f)

        passed = True
        for task_id, golden in golden_data.items():
            if task_id not in compared_data:
                logging.error(
                    'can not find task {:d} in out_data'.format(task_id))
                passed = False
                continue
            out_data = compared_data[task_id]
            if golden.shape != out_data.shape:
                logging.error(
                    'error in shape of task block {:d}'.format(task_id))
                passed = False
            elif not (golden == out_data).all():
                passed = False
                logging.error(
                    'error in data of task block {:d}'.format(task_id))
            else:
                if (out_data == np.zeros_like(out_data)).all():
                    logging.warning(
                        'All elements of task block {:d} are zeros, which means the checking result may be inaccurate'.format(task_id))
                passed = True
        return passed
