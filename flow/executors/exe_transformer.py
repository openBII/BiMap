# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging

from src.compiler.transformer.opt.optimize_config import OptimizeConfig
from src.compiler.transformer.transformer import ONNXTransformer
from top.global_config import GlobalConfig

GlobalConfig.config()


def exe_onnx_transform(onnx_path: str, case_name: str, task_graph_path: str = None,
                       optimize_config: OptimizeConfig = None, readable_result: bool = True):
    logging.info('Transformer Begin.')
    transformer = ONNXTransformer(
        onnx_model_path=onnx_path,
        task_graph_path=task_graph_path,
        case_name=case_name,
        optimize_config=optimize_config,
        readable=readable_result
    )
    transformer.transform()
    logging.info('Transformer Finish.')
    return True


if __name__ == '__main__':
    onnx_model_path = 'test/unit_tests/cases/lenet/lenet.onnx'
    task_graph_path = 'temp/lenet/lenet.task'
    case_name = 'lenet'
    optimizer = OptimizeConfig()
    optimizer.simplify_edge_clusters = True
    optimizer.insert_io_nodes = True

    exe_onnx_transform(onnx_model_path, case_name, task_graph_path, optimizer)
