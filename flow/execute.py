# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
import os

from top.global_config import GlobalConfig

from flow.executors import exe_onnx_transform


class Execution:
    FLOW_MAP = {
        'C1': exe_onnx_transform,
        'MOVE': lambda source_dir, destination_dir: os.system('rm -rf ' + destination_dir + ' && mv ' + source_dir + ' ' + destination_dir)
    }

    INPUT_MAP = {
        'C1': [GlobalConfig.Path['temp'] + 'CASE_NAME/CASE_NAME.onnx', 'CASE_NAME'],
    }

    OUTPUT_MAP = {
    }

    @staticmethod
    def execute(flow_id: str, *args):
        if flow_id not in Execution.FLOW_MAP:
            logging.warning('Nothing to be done')
            return False
        else:
            return Execution.FLOW_MAP[flow_id](*args)

    @staticmethod
    def auto_execute(flow_id: str, *args, **kwargs):
        '''
        自动执行FLOW_MAP中有的或没有的流程
        如果flow在FLOW_MAP中 则执行FLOW_MAP指定的函数 参数为args
        如果没有 如S6_1-O5-C3-S5 则按照默认的输入输出路径 依次执行S6_1 O5 C3 S5
        其中输入按照模板替换得到 模板根据kwargs中的键值对替换 
        这是如果设置了args 则args是第一个组件的输入
        '''
        execution_state = True

        if not args and flow_id in Execution.FLOW_MAP and flow_id in Execution.INPUT_MAP:
            args = Execution.INPUT_MAP[flow_id]

        # 输入参数模板替换
        args_specific = []
        if args:
            for arg in args:
                if not isinstance(arg, str):
                    args_specific.append(arg)
                    continue
                for key, value in kwargs.items():
                    arg = arg.replace(key, value)
                args_specific.append(arg)

        if flow_id in Execution.FLOW_MAP and flow_id in Execution.INPUT_MAP:
            execution_state = Execution.FLOW_MAP[flow_id](*args_specific)

            # for move
            execution_state = True if execution_state == 0 else execution_state
        else:
            flows = flow_id.split('-')
            for i, flow in enumerate(flows):
                if flow not in Execution.FLOW_MAP:
                    raise ValueError(flow + ' not valid')

                # 第一个flow的输入可以指定
                if i == 0 and args:
                    Execution.FLOW_MAP[flow](*args_specific)
                    continue

                # 输入参数模板替换
                args_specific = []
                args_template = Execution.INPUT_MAP[flow]
                for arg in args_template:
                    for key, value in kwargs.items():
                        if isinstance(arg, str):
                            arg = arg.replace(key, value)
                    args_specific.append(arg)

                execution_state = Execution.FLOW_MAP[flow](
                    *args_specific) and execution_state
                # for move
                execution_state = True if execution_state == 0 else execution_state
                pass

        return execution_state
