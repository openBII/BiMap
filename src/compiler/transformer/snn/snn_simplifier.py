# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from copy import deepcopy
from queue import Queue
from typing import List

from src.compiler.transformer.onnx.onnx_basics import ONNXGraph, ONNXNode
from src.compiler.transformer.onnx.onnx_eliminator import ONNXEliminator
from src.compiler.transformer.onnx.onnx_op_type import ONNXOpType


class SNNSimplifier:
    '''对表示SNN的ONNX计算图进行简化

    Attributes:
        graph: ONNXGraph对象
        counter: 表示脉冲神经元的计算任务块的计数
    '''

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph
        self.counter: int = -1

    def simplify(self):
        '''简化方法

        首先通过Recorder结点定位脉冲神经元,
        提取出代表膜电位的数据,
        找到代表Fire的结点并获取该结点的输出数据,
        通过Recorder和输入输出构建脉冲神经元结点,
        更改脉冲神经元结点前后结点的的连接,
        删除不需要的结点和子图
        '''
        eliminated_data = list()
        eliminated_nodes = list()
        new_nodes = list()
        for node_name in self.graph.nodes:
            node = self.graph.nodes[node_name]
            if not self.graph.is_output_node(node.name):
                recorder = self.graph.get_next_node(node_name)
                if node.op_type == ONNXOpType.Clip.value:
                    if recorder.op_type == ONNXOpType.LIFRecorder.value:
                        inputs = node.inputs
                        next_node = self.graph.get_next_node(recorder.name)
                        eliminated_nodes.append(next_node.name)
                        eliminated_data.extend(next_node.outputs)
                        v_mem_input = self.eliminate_constant_node_and_save_data(
                            next_node, eliminated_nodes)  # Accumulate
                        if v_mem_input is None:
                            for input in next_node.inputs:
                                if self.graph.is_input_data(input):
                                    v_mem_input = [input]
                        while not ONNXOpType.is_fire(next_node.op_type):
                            output_nodes = self.graph.get_output_nodes(
                                next_node.name)
                            for output_node in output_nodes:
                                if ONNXOpType.is_fire(output_node.op_type):
                                    eliminated_nodes.append(output_node.name)
                                    self.eliminate_constant_node(
                                        output_node, eliminated_nodes, eliminated_data)
                                    next_node = output_node
                                    break
                                else:
                                    eliminated_nodes.append(output_node.name)
                                    eliminated_data.extend(output_node.outputs)
                                    self.eliminate_constant_node(
                                        output_node, eliminated_nodes, eliminated_data)
                            if ONNXOpType.is_fire(next_node.op_type):
                                break
                            next_node = output_nodes[0]
                        inputs.extend(v_mem_input)
                        outputs = next_node.outputs
                        lif = self.create_lif(recorder, inputs, outputs)
                        new_nodes.append(lif)
                        # 更改LIF前结点的输出
                        last_node = self.graph.get_last_node(node_name)
                        for i, output_node_name in enumerate(self.graph.output_connections[last_node.name]):
                            if output_node_name == node_name:
                                self.graph.output_connections[last_node.name][i] = lif.name
                                self.graph.input_connections[lif.name] = [
                                    last_node.name]
                        # 更改LIF后结点的输入
                        self.graph.output_connections[lif.name] = []
                        for output_node_name in self.graph.output_connections[next_node.name]:
                            output_node = self.graph.nodes[output_node_name]
                            # 删除不需要的子图
                            # XXX(huanyu): 这里加入神经元结束标志会更好一点
                            if output_node.op_type == ONNXOpType.Sub.value or output_node.op_type == ONNXOpType.Mul.value:
                                self.eliminate_sub_graph(
                                    output_node, eliminated_nodes, eliminated_data)
                            else:
                                # 真正需要的输出
                                for i, input_node_name_of_output_node in enumerate(self.graph.input_connections[output_node_name]):
                                    if input_node_name_of_output_node == next_node.name:
                                        self.graph.input_connections[output_node_name][i] = lif.name
                                        self.graph.output_connections[lif.name].append(
                                            output_node_name)
                        eliminated_nodes.append(node_name)
                        eliminated_data.extend(node.outputs)
                        eliminated_nodes.append(recorder.name)
                        eliminated_data.extend(recorder.outputs)
        ONNXEliminator.eliminate(self.graph, eliminated_nodes, eliminated_data)
        self.graph.add_nodes(new_nodes)

    def create_lif(self, recorder: ONNXNode, inputs: List[str], outputs: List[str]):
        '''创建LIF神经元结点

        Args:
            - recorder: 记录脉冲神经元信息的结点
            - inputs: 结点输入数据
            - outputs: 结点输出数据
        '''
        self.counter += 1
        return LIF(recorder, self.counter, inputs, outputs)

    def eliminate_sub_graph(self, node: ONNXNode, eliminated_nodes: List[str], eliminated_data: List[str]):
        '''以某个结点为起始, 删除后续的所有结点

        通过N叉树的层序遍历实现, 要求子图中不能成环
        '''
        q: Queue[ONNXNode] = Queue()
        q.put(node)
        while not q.empty():
            current = q.get()
            self.eliminate_constant_node(
                current, eliminated_nodes, eliminated_data)
            eliminated_nodes.append(current.name)
            eliminated_data.extend(current.outputs)
            for output_node_name in self.graph.output_connections[current.name]:
                q.put(self.graph.nodes[output_node_name])

    def eliminate_constant_node(self, node: ONNXNode, eliminated_nodes: List[str], eliminated_data: List[str]):
        '''删除产生某个结点输入数据的常量结点
        '''
        for input_node_name in self.graph.input_connections[node.name]:
            input_node = self.graph.nodes[input_node_name]
            if input_node.op_type == ONNXOpType.Constant.value:
                eliminated_nodes.append(input_node_name)
                eliminated_data.extend(input_node.outputs)

    def eliminate_constant_node_and_save_data(self, node: ONNXNode, eliminated_nodes: List[str]):
        '''在Accumulate的步骤删除掉无用的constant结点, 保留constant结点的输出数据作为输入膜电位的占位数据
        '''
        for input_node_name in self.graph.input_connections[node.name]:
            input_node = self.graph.nodes[input_node_name]
            if input_node.op_type == ONNXOpType.Constant.value:
                eliminated_nodes.append(input_node_name)
                return input_node.outputs


class LIF(ONNXNode):
    '''在ONNX计算图中代表LIF神经元
    '''

    def __init__(self, recorder: ONNXNode, counter: int, inputs: List[str], outputs: List[str]):
        super(LIF, self).__init__()
        self.name = 'LIF_' + str(counter)
        self.op_type = ONNXOpType.LIF.value
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = deepcopy(recorder.attributes)
