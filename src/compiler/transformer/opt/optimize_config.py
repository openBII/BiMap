# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

class OptimizeConfig:
    '''转换器的优化配置

    Attributes:
        merge_relu_maxpool: 是否进行relu和maxpool的任务块融合
        optimize_conv_storage: 是否对卷积的存储进行优化
        insert_io_nodes: 是否插入输出输出任务块
        simplify_edge_clusters: 是否对EdgeCluster的表示进行简化
        reserve_control_flow: 是否在导出ONNX文件时保留控制流信息
    '''
    merge_relu_maxpool = True
    optimize_conv_storage = True
    insert_io_nodes = False
    simplify_edge_clusters = False
    reserve_control_flow = False
