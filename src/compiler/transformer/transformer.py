# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import abc
import logging
import os
from typing import Tuple, Union

import onnxsim
import torch
from src.compiler.transformer.ir_gen.ir_generate_pass import ir_generate
from src.compiler.transformer.network_type import NetworkType
from src.compiler.transformer.onnx.onnx_converter import ONNXConverter
from src.compiler.transformer.onnx.onnx_exporter import ONNXExporter
from src.compiler.transformer.onnx.onnx_parser import ONNXParser
from src.compiler.transformer.opt.optimize_config import OptimizeConfig
from src.compiler.transformer.snn.shape_inference import ShapeInference
from src.compiler.transformer.snn.snn_op_type import SNNOpType
from src.compiler.transformer.task_model.task_graph_transformer import \
    TaskGraphTransformer
from top.global_config import GlobalConfig

import onnx


class Transformer:
    '''Transformer基类

    抽象类

    Attributes:
        - input_path: 输入文件的路径
        - task_graph_path: 输出任务图IR的路径
        - case_name: 当前转换的任务名称, 如不提供则从task_graph_path中提取
        - optimize_config: 优化的配置
        - readable: 是否生成可读文件, 包括以下几个文件:
            - readable_onnx: 后缀为.onnx.txt, 可读的onnx文件
            - readable_task_graph: 后缀为.task.debug.txt, 可读的task IR
            - readable_ir: 后缀为.task.txt, .task文件的txt版本
        - cache_dir: 可读文件等中间文件会生成到此文件夹下, 如果文件夹不存在则自动创建
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_path: str,
                 task_graph_path: str, case_name: str = None,
                 optimize_config: OptimizeConfig = None, readable: bool = True,
                 cache_dir: str = None) -> None:
        '''Transformer构造函数

        生成一系列文件的路径, 解析optimize_config, 设置Python日志输出等级, 创建cache文件夹
        '''
        self.input_path = input_path
        self.task_graph_path = task_graph_path
        self.case_name = task_graph_path.split(
            '/')[-1].split('.')[0] if case_name is None else case_name
        self.optimize_config = optimize_config
        self.parse_optimize_config()
        self.readable = readable
        self.cache_dir = os.path.join(
            GlobalConfig.Path['temp'], self.case_name) if cache_dir is None else cache_dir
        if self.readable:
            self.readable_onnx = os.path.join(
                self.cache_dir, self.case_name + '.onnx.txt')
            self.readable_task_graph = os.path.join(
                self.cache_dir, self.case_name + '.task.debug.txt')
            self.readable_ir = os.path.join(
                self.cache_dir, self.case_name + '.task.txt')
        else:
            self.readable_onnx = None
            self.readable_task_graph = None
            self.readable_ir = None
        GlobalConfig.config()
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse_optimize_config(self):
        '''解析optimize_config

        Attributes:
            - merge_relu_maxpool: 是否融合relu和maxpool
            - optimize_conv: 是否对卷积进行优化
            - insert_io_nodes: 是否插入输入输出任务块
            - simplify_edge_clusters: 是否对IR中EdgeCluster的表示进行化简
            - reserve_control_flow: 是否在导出ONNX模型时保留控制流, 只在PyTorchTransformer中使用
        '''
        if self.optimize_config is None:
            self.merge_relu_maxpool = OptimizeConfig.merge_relu_maxpool
            self.optimize_conv = OptimizeConfig.optimize_conv_storage
            self.insert_io_nodes = OptimizeConfig.insert_io_nodes
            self.simplify_edge_clusters = OptimizeConfig.simplify_edge_clusters
            self.reserve_control_flow = OptimizeConfig.reserve_control_flow
        else:
            self.merge_relu_maxpool = self.optimize_config.merge_relu_maxpool
            self.optimize_conv = self.optimize_config.optimize_conv_storage
            self.insert_io_nodes = self.optimize_config.insert_io_nodes
            self.simplify_edge_clusters = self.optimize_config.simplify_edge_clusters
            self.reserve_control_flow = self.optimize_config.reserve_control_flow

    def is_snn(self, onnx_model: onnx.ModelProto) -> bool:
        '''判断输入ONNX模型是否为SNN网络
        '''
        graph = onnx_model.graph
        for n in graph.node:
            if SNNOpType.is_snn_op(n):
                return True
        return False

    def transform(self, input: Union[torch.Tensor, Tuple[torch.Tensor]] = None):
        '''转换方法

        1. onnx_handler导出或处理ONNX模型
        2. ONNXParser将ONNX模型转换成可操作的类
        3. ONNXConverter在ONNX计算图上进行一系列特殊处理, 例如化简、转换等, 为ONNX到Task Graph的转换做准备
        3. TaskGraphTransformer将ONNX模型转换成任务图
        5. 通过任务图生成最终的IR文件
        '''
        onnx_model, snn = self.onnx_handler(input)

        parser = ONNXParser(onnx_model, self.readable_onnx)
        converter = ONNXConverter(
            graph=parser.graph, is_snn=snn, readable_file_path=self.readable_onnx)
        onnx_graph = converter.convert()

        transformer = TaskGraphTransformer(
            onnx_graph=onnx_graph,
            merge_relu_maxpool=self.merge_relu_maxpool,
            optimize_conv=self.optimize_conv,
            insert_io_nodes=self.insert_io_nodes,
            simplify_edge_clusters=self.simplify_edge_clusters,
            readable_file_path=self.readable_task_graph
        )
        task_graph = transformer.transform()

        return ir_generate(task_graph, self.task_graph_path, self.readable_ir)

    @abc.abstractmethod
    def onnx_handler(self) -> Tuple[onnx.ModelProto, bool]:
        '''生成或处理ONNX模型

        Returns:
            - 第一个返回值为ONNX模型
            - 第二个返回值为模型是否为SNN
        '''
        pass


class ONNXTransformer(Transformer):
    '''对输入ONNX文件进行转换, 转换得到对应的任务图IR

    Args:
        - onnx_model_path: 输入ONNX文件的路径
    '''

    def __init__(self, onnx_model_path: str,
                 task_graph_path: str, case_name: str = None,
                 optimize_config: OptimizeConfig = None, readable: bool = True,
                 cache_dir: str = None) -> None:
        super().__init__(input_path=onnx_model_path, task_graph_path=task_graph_path,
                         case_name=case_name, optimize_config=optimize_config,
                         readable=readable, cache_dir=cache_dir)

    def onnx_handler(self, _) -> Tuple[onnx.ModelProto, bool]:
        '''处理ONNX模型

        Returns:
            - 第一个返回值为ONNX模型
            - 第二个返回值为模型是否为SNN
        '''
        onnx_model = onnx.load(self.input_path)
        snn = self.is_snn(onnx_model)
        if not snn:
            onnx_model, _ = onnxsim.simplify(onnx_model)
        else:
            shape_infer = ShapeInference(onnx_model)
            onnx_model = shape_infer.model
            onnx.save(onnx_model, self.input_path)
        return onnx_model, snn


class PyTorchTransformer(Transformer):
    '''将PyTorch模型转换成任务图IR

    Attributes:
        - input_path: 这里指含量化信息的预训练模型路径, 如不提供会出现warning
        - pytorch_model: PyTorch模型对象
        - onnx_model_path: ONNX模型保存路径
        - network_type: 输入网络的类型

    Args:
        - pretrained_model_path: 包含量化信息的预训练模型路径
    '''

    def __init__(self, pytorch_model: torch.nn.Module,
                 task_graph_path: str,
                 case_name: str = None,
                 pretrained_model_path: str = None,
                 optimize_config: OptimizeConfig = None,
                 readable=True,
                 cache_dir: str = None) -> None:
        super().__init__(pretrained_model_path, task_graph_path,
                         case_name, optimize_config, readable, cache_dir)
        if self.input_path is None:
            logging.warning('Pretrained model should be provided')
        self.pytorch_model = pytorch_model
        self.onnx_model_path = os.path.join(
            self.cache_dir, self.case_name + '.onnx')
        self.network_type = self.get_network_type()

    def get_network_type(self):
        '''获取PyTorch模型的网络类型
        '''
        if hasattr(self.pytorch_model, 'T'):
            return NetworkType.SNN
        elif hasattr(self.pytorch_model, 'restrict'):
            return NetworkType.ANN
        else:
            raise NotImplementedError

    def onnx_handler(self, input: Union[torch.Tensor, Tuple[torch.Tensor]]):
        '''导出ONNX模型

        Returns:
            - 第一个返回值为ONNX模型
            - 第二个返回值为模型是否为SNN
        '''
        onnx_exporter = ONNXExporter(
            model=self.pytorch_model,
            output_path=self.onnx_model_path,
            model_path=self.input_path,
            reserve_control_flow=self.reserve_control_flow,
            network_type=self.network_type
        )
        onnx_model = onnx_exporter.export(input)
        snn = True if self.network_type == NetworkType.SNN else False
        return onnx_model, snn
