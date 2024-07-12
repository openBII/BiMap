from typing import Dict, Union
from math import sqrt
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType

def create_data(task_graph: TaskGraph, shape: Shape, precision: Precision, is_output: bool = False):
    IDGenerator.set_base_task_id(task_graph)
    if is_output:
        data = OutputTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    else:
        data = STaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(data)
    return data

def create_static(task_graph: TaskGraph, shape: Shape, precision: Precision):
    IDGenerator.set_base_task_id(task_graph)
    data = StaticTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(data)
    return data

def create_input(task_graph: TaskGraph, shape: Shape, precision: Precision):
    IDGenerator.set_base_task_id(task_graph)
    input = InputTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(input)
    return input

def create_compute(task_graph: TaskGraph, shape: Shape, type: TaskBlockType, precision: Precision, constant: float = None):
    IDGenerator.set_base_task_id(task_graph)
    compute = CTaskBlock(IDGenerator.get_next_task_id(), shape, type, precision, constant=constant)
    task_graph.add_node(compute)
    return compute

def create_FFN():
    pass

def create_mlp(task_graph: TaskGraph, shape: Shape, precision: Precision, 
               is_output: bool = False, task_dict: Dict = None):
    IDGenerator.set_base_task_id(task_graph)
    weight = create_static(task_graph, shape, precision)
    mlp = create_compute(task_graph, shape, TaskBlockType.CVM, precision)
    output = create_data(task_graph, Shape(nf=shape.nf), precision, is_output)
    task_graph.connect(weight.id, mlp.id)
    task_graph.connect(mlp.id, output.id)
    if task_dict is not None:
        task_dict["compute"] = mlp
        task_dict["weight"] = weight
        task_dict["output"] = output
    return weight, mlp, output

def create_softmax(task_graph: TaskGraph, length: int, precision: Precision, task_dict: Dict):
    IDGenerator.set_base_task_id(task_graph)
    exp = create_compute(task_graph, Shape(nf=length), TaskBlockType.CEXP, precision)
    exp_output = create_data(task_graph, Shape(nr=length), precision)
    reduction = create_compute(task_graph, Shape(nr=length, nf=1), TaskBlockType.CAVG, precision)
    reduction_output = create_data(task_graph, Shape(nf=1), precision)
    scale = create_compute(task_graph, Shape(nf=length), TaskBlockType.CVS, precision)
    output = create_data(task_graph, Shape(nf=length), precision)
    task_graph.connect(exp.id, exp_output.id)
    task_graph.connect(exp_output.id, reduction.id)
    task_graph.connect(reduction.id, reduction_output.id)
    task_graph.connect(exp_output.id, scale.id)
    task_graph.connect(reduction_output.id, scale.id)
    task_graph.connect(scale.id, output.id)
    task_dict["exp"] = {}
    task_dict["exp"]["compute"] = exp
    task_dict["exp"]["output"] = exp_output
    task_dict["reduction"] = {}
    task_dict["reduction"]["compute"] = reduction
    task_dict["reduction"]["output"] = reduction_output
    task_dict["scale"] = {}
    task_dict["scale"]["compute"] = scale
    task_dict["scale"]["output"] = output
    return exp, output


def create_attention(task_graph: TaskGraph, d_model: int, d_encode: int, d_value: int,
                     seq_len: int, precision: Precision, is_output: bool = False):
    task_dict: Dict[str, Union[Dict[str, TaskBlock], TaskBlock]] = dict()
    IDGenerator.set_base_task_id(task_graph)
    embedding = create_data(task_graph, Shape(nr=d_model), precision)
    query_weight = create_data(task_graph, Shape(nr=d_model, nf=d_encode), precision)
    key_weight = create_data(task_graph, Shape(nr=d_model, nf=d_encode), precision)
    value_weight = create_data(task_graph, Shape(nr=d_model, nf=d_value), precision)
    query_mlp = create_compute(task_graph, Shape(nr=d_model, nf=d_encode), TaskBlockType.CVM, precision)
    key_mlp = create_compute(task_graph, Shape(nr=d_model, nf=d_encode), TaskBlockType.CVM, precision)
    value_mlp = create_compute(task_graph, Shape(nr=d_model, nf=d_value), TaskBlockType.CVM, precision)
    query = create_data(task_graph, Shape(nf=d_encode), precision)
    key = create_data(task_graph, Shape(nf=d_encode), precision)
    value = create_data(task_graph, Shape(nf=d_value), precision)
    key_cache = create_data(task_graph, Shape(nr=d_encode, nf=seq_len - 1), precision)
    new_key_cache = create_data(task_graph, Shape(nr=d_encode, nf=seq_len), precision)
    dot_product = create_compute(task_graph, Shape(nr=d_encode, nf=seq_len), TaskBlockType.CVM, precision)
    dot_product_output = create_data(task_graph, Shape(nf=seq_len), precision)
    scale = create_compute(task_graph, Shape(nf=seq_len), TaskBlockType.CVS, precision, constant=1 / sqrt(d_encode))
    scale_output = create_data(task_graph, Shape(nf=seq_len), precision)
    task_dict["softmax"] = {}
    exp, softmax_output = create_softmax(task_graph, seq_len, precision, task_dict["softmax"])
    value_cache = create_data(task_graph, Shape(nr=seq_len - 1, nf=d_value), precision)
    new_value_cache = create_data(task_graph, Shape(nr=seq_len, nf=d_value), precision)
    attention_mlp = create_compute(task_graph, Shape(nr=seq_len, nf=d_value), TaskBlockType.CVM, precision)
    output = create_data(task_graph, Shape(nf=d_value), precision, is_output)
    task_graph.connect(embedding.id, query_mlp.id)
    task_graph.connect(embedding.id, key_mlp.id)
    task_graph.connect(embedding.id, value_mlp.id)
    task_graph.connect(query_weight.id, query_mlp.id)
    task_graph.connect(key_weight.id, key_mlp.id)
    task_graph.connect(value_weight.id, value_mlp.id)
    task_graph.connect(query_mlp.id, query.id)
    task_graph.connect(key_mlp.id, key.id)
    task_graph.connect(value_mlp.id, value.id)
    task_graph.connect(query.id, dot_product.id)
    task_graph.connect(key.id, new_key_cache.id)
    task_graph.connect(key_cache.id, new_key_cache.id)
    task_graph.connect(new_key_cache.id, dot_product.id)
    task_graph.connect(dot_product.id, dot_product_output.id)
    task_graph.connect(dot_product_output.id, scale.id)
    task_graph.connect(scale.id, scale_output.id)
    task_graph.connect(scale_output.id, exp.id)
    task_graph.connect(softmax_output.id, attention_mlp.id)
    task_graph.connect(value.id, new_value_cache.id)
    task_graph.connect(value_cache.id, new_value_cache.id)
    task_graph.connect(new_value_cache.id, attention_mlp.id)
    task_graph.connect(attention_mlp.id, output.id)
    task_dict["embedding"] = embedding
    task_dict["query"] = {}
    task_dict["query"]["weight"] = query_weight
    task_dict["query"]["mlp"] = query_mlp
    task_dict["query"]["output"] = query
    task_dict["key"] = {}
    task_dict["key"]["weight"] = key_weight
    task_dict["key"]["mlp"] = key_mlp
    task_dict["key"]["output"] = key
    task_dict["value"] = {}
    task_dict["value"]["weight"] = value_weight
    task_dict["value"]["mlp"] = value_mlp
    task_dict["value"]["output"] = value
    task_dict["cache"] = {}
    task_dict["cache"]["key"] = {}
    task_dict["cache"]["key"]["old"] = key_cache
    task_dict["cache"]["key"]["new"] = new_key_cache
    task_dict["cache"]["value"] = {}
    task_dict["cache"]["value"]["old"] = value_cache
    task_dict["cache"]["value"]["new"] = new_value_cache
    task_dict["dot_product"] = {}
    task_dict["dot_product"]["mlp"] = dot_product
    task_dict["dot_product"]["output"] = dot_product_output
    task_dict["scale"] = {}
    task_dict["scale"]["compute"] = scale
    task_dict["scale"]["output"] = scale_output
    task_dict["softmax"] = {}
    task_dict["attention"] = {}
    task_dict["attention"]["mlp"] = attention_mlp
    task_dict["output"] = output
    return task_dict