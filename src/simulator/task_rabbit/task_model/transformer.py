from copy import copy
from enum import Enum
from typing import Dict, Union, Iterable, Sequence
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


class AttentionType(Enum):
    PREFILL = 0
    DECODE = 1


def create_data(task_graph: TaskGraph, shape: Shape, 
                precision: Precision, is_output: bool = False):
    if is_output:
        data = OutputTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    else:
        data = STaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(data)
    return data

def create_static(task_graph: TaskGraph, shape: Shape, precision: Precision):
    data = StaticTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(data)
    return data

def create_input(task_graph: TaskGraph, shape: Shape, precision: Precision, 
                 generate_data: bool = True):
    input = InputTaskBlock(IDGenerator.get_next_task_id(), shape, precision)
    task_graph.add_node(input)
    if generate_data:
        data = create_data(task_graph, shape, precision)
        task_graph.connect(input.id, data.id)
        return input, data
    return input

def create_compute(task_graph: TaskGraph, shape: Shape, type: TaskBlockType, 
                   precision: Precision, constant: float = None):
    compute = CTaskBlock(IDGenerator.get_next_task_id(), shape, type, precision, 
                         constant=constant)
    task_graph.add_node(compute)
    return compute

def create_ffn(task_graph: TaskGraph, input: TaskBlock, inner_dim: int,
               precision: Precision, activation_type: TaskBlockType,
               is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["up"] = {}
    up_output, _ = create_mlp(task_graph, input, 
                              Shape(nr=input.shape.nf, nf=inner_dim),
                              precision,
                              task_dict=task_dict["up"])
    task_dict["activation"] = {}
    act_output, _ = create_pointwise(task_graph, up_output, precision, 
                                     activation_type, 
                                     task_dict=task_dict["activation"])
    task_dict["down"] = {}
    output, _ = create_mlp(task_graph, act_output, 
                           Shape(nf=input.shape.nf, nr=inner_dim),
                           precision, is_output,
                           task_dict=task_dict["down"])
    return output, task_dict

def create_mlp(task_graph: TaskGraph, input: TaskBlock, shape: Shape, 
               precision: Precision, is_output: bool = False, 
               task_dict: Dict = None, weight: TaskBlock = None):
    assert (input.shape.nf == shape.nr or input.shape.nr == shape.nr)
    task_dict = {} if task_dict is None else task_dict
    if weight is None:
        weight = create_static(task_graph, Shape(nr=shape.nr, nf=shape.nf), 
                               precision)
    compute_shape = Shape(batch=input.shape.batch, token=input.shape.token,
                          nr=shape.nr, nf=shape.nf)
    mlp = create_compute(task_graph, compute_shape, TaskBlockType.CVM, 
                         precision)
    output_shape = Shape(batch=input.shape.batch, token=input.shape.token,
                         nf=shape.nf)
    output = create_data(task_graph, output_shape, precision, is_output)
    task_graph.connect(weight.id, mlp.id)
    task_graph.connect_tasks_in_sequence([input, mlp, output])
    task_dict["compute"] = mlp
    task_dict["weight"] = weight
    task_dict["output"] = output
    return output, task_dict

def create_tiled_mlp_cyclic_weight(split_vector: SplitVector,
                                   task_graph: TaskGraph, input: TaskBlock, 
                                   shape: Shape, precision: Precision,
                                   is_output: bool = False,
                                   task_dict: Dict = None,
                                   static_weight: bool = True,
                                   output_offchip: bool = False,
                                   is_input_on_chip: bool = True,
                                   weight_offchip: bool = True):
    """
    映射策略为: 拆分batch和token维度, 循环权重完成计算
    """
    assert (input.shape.nf == shape.nr or input.shape.nr == shape.nr)
    task_dict = {} if task_dict is None else task_dict
    batch = input.shape.batch
    token = input.shape.token
    nf = shape.nf // split_vector.nf
    nr = shape.nr // split_vector.nr
    if static_weight:
        weight_shape = Shape(nr=nr, nf=nf)
    else:
        weight_shape = Shape(batch=batch, token=nf, nf=nr)
    if not is_input_on_chip:
        input_on_chip = create_data(task_graph, input.shape, precision)
    weight = create_static(task_graph, weight_shape, precision)
    if weight_offchip:
        weight_on_chip = create_data(task_graph, weight_shape, precision)
    compute_shape = Shape(batch=batch, token=token, nr=nr, nf=nf)
    mlp = create_compute(task_graph, compute_shape, TaskBlockType.CVM, 
                         precision)
    output_shape = Shape(batch=batch, token=token, nf=shape.nf)
    if output_offchip:
        output = create_data(task_graph, output_shape, precision)
    else:
        output = create_data(task_graph, output_shape, precision, is_output)
    if weight_offchip:
        task_graph.connect_tasks_in_sequence([weight, weight_on_chip, mlp])
    else:
        task_graph.connect_tasks_in_sequence([weight, mlp])
    if not is_input_on_chip:
        task_graph.connect_tasks_in_sequence(
            [input, input_on_chip, mlp, output])
    else:
        task_graph.connect_tasks_in_sequence(
            [input, mlp, output])
    task_dict["compute"] = mlp
    if weight_offchip:
        task_dict["weight_offchip"] = weight
        task_dict["weight_on_chip"] = weight_on_chip
    else:
        task_dict["weight_on_chip"] = weight
    task_dict["output_on_chip"] = output
    if not is_input_on_chip:
        task_dict["input_on_chip"] = input_on_chip
    # 循环权重
    cyclic_weight = create_static(task_graph, weight_shape,
                                  precision)
    cyclic_mlp = create_compute(task_graph, compute_shape, TaskBlockType.CVM, 
                                precision)
    if output_offchip:
        cyclic_output = create_data(task_graph, output_shape, precision, 
                                    is_output)
    task_graph.connect_tasks_in_sequence([cyclic_weight, cyclic_mlp])
    if output_offchip:
        if not is_input_on_chip:
            task_graph.connect_tasks_in_sequence(
                [input_on_chip, cyclic_mlp, output, cyclic_output])
        else:
            task_graph.connect_tasks_in_sequence(
                [input, cyclic_mlp, output, cyclic_output])
    else:
        if not is_input_on_chip:
            task_graph.connect_tasks_in_sequence(
                [input_on_chip, cyclic_mlp, output])
        else:
            task_graph.connect_tasks_in_sequence(
                [input, cyclic_mlp, output])
    task_dict["cyclic_compute"] = cyclic_mlp
    task_dict["cyclic_weight_on_chip"] = cyclic_weight
    if output_offchip:
        task_dict["output_offchip"] = cyclic_output
    if output_offchip:
        return cyclic_output, task_dict
    else:
        return output, task_dict

def create_tiled_elementwise(task_graph: TaskGraph, input: TaskBlock,
                             precision: Precision, type: TaskBlockType, 
                             is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    mask = create_static(task_graph, input.shape, precision)
    mask_on_chip = create_data(task_graph, input.shape, precision)
    special_function = create_compute(task_graph, input.shape, type, precision)
    output = create_data(task_graph, input.shape, precision, is_output)
    task_graph.connect(input.id, special_function.id)
    task_graph.connect(special_function.id, output.id)
    task_graph.connect_tasks_in_sequence([mask, mask_on_chip, special_function])
    task_dict["compute"] = special_function
    task_dict["output"] = output
    task_dict["mask_offchip"] = mask
    task_dict["mask_on_chip"] = mask_on_chip
    return output, task_dict

# def create_mm(task_graph: TaskGraph, input: TaskBlock, shape: Shape,
#               precision: Precision, is_output: bool = False,
#               task_dict: Dict = None, weight: TaskBlock = None):
#     task_dict = {} if task_dict is None else task_dict
#     weight = create_static(task_graph, Shape(nr=shape.nr, nf=shape.nf))
#     mm = create_compute(task_graph, shape, TaskBlockType.CMM, precision)
#     output = create_data(task_graph, 
#                          Shape(token=shape.token, nf=shape.nf), 
#                          precision, is_output)
#     task_graph.connect(weight.id, mm.id)
#     task_graph.connect_tasks_in_sequence([input, mm, output])
#     task_dict["compute"] = mm
#     task_dict["input"] = input
#     task_dict["weight"] = weight
#     task_dict["output"] = output
#     return output, task_dict

def create_pointwise(task_graph: TaskGraph, input: TaskBlock,
                     precision: Precision, type: TaskBlockType, 
                     is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    special_function = create_compute(task_graph, input.shape, type, precision)
    output = create_data(task_graph, input.shape, precision, is_output)
    task_graph.connect(input.id, special_function.id)
    task_graph.connect(special_function.id, output.id)
    task_dict["compute"] = special_function
    task_dict["output"] = output
    return output, task_dict

def create_softmax(task_graph: TaskGraph, input: TaskBlock,
                   precision: Precision, is_output: bool = False,
                   task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["exp"] = {}
    exp_output, _ = create_pointwise(task_graph, input, precision, 
                                     TaskBlockType.CEXP, 
                                     task_dict=task_dict["exp"])
    task_dict["reduction"] = {}
    reduction_output, _ = create_reduce_sum(
        task_graph, exp_output, precision, task_dict=task_dict["reduction"])
    task_dict["div"] = {}
    output, _ = create_div(task_graph, [exp_output, reduction_output],
                           exp_output.shape, precision,
                           task_dict=task_dict["div"])
    return output, task_dict

def create_scale(task_graph: TaskGraph, input: TaskBlock, precision: Precision,
                 is_output: bool = False, task_dict: Dict = None, 
                 value: Union[float, TaskBlock] = None):
    task_dict = {} if task_dict is None else task_dict
    if isinstance(value, TaskBlock):
        scale = create_compute(task_graph, input.shape, TaskBlockType.CVS, precision)
        task_graph.connect(value.id, scale.id)
    else:
        scale = create_compute(task_graph, input.shape, TaskBlockType.CVS, precision,
                               value)
    output = create_data(task_graph, input.shape, precision, is_output)
    task_graph.connect_tasks_in_sequence([input, scale, output])
    task_dict["compute"] = scale
    task_dict["output"] = output
    return output, task_dict

def create_concat(task_graph: TaskGraph, inputs: Sequence[TaskBlock],
                  shape: Shape, precision: Precision,
                  is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    shape.batch = inputs[0].shape.batch
    shape.token = inputs[0].shape.token
    concat_shape = copy(shape)
    concat_shape.branch = len(inputs)
    concat = create_compute(task_graph, concat_shape, 
                            TaskBlockType.MCONCAT, precision)
    output = create_data(task_graph, shape, precision, is_output)
    for input in inputs:
        task_graph.connect(input.id, concat.id)
    task_graph.connect(concat.id, output.id)
    if task_dict is not None:
        task_dict["move"] = concat
        task_dict["output"] = output
    return output, task_dict

def create_prefill_attention(task_graph: TaskGraph, embedding: TaskBlock,
                             d_key: int, d_value: int, 
                             precision: Precision,
                             is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    num_tokens = embedding.shape.token
    d_model = max(embedding.shape.nf, embedding.shape.nr)
    task_dict["query"] = {}
    query, _ = create_mlp(
        task_graph, embedding, 
        Shape(nr=d_model, nf=d_key), 
        precision, task_dict=task_dict["query"])
    task_dict["key"] = {}
    key, _ = create_mlp(task_graph, embedding, 
                        Shape(nr=d_model, nf=d_key), 
                        precision, task_dict=task_dict["key"])
    task_dict["value"] = {}
    value, _ = create_mlp(
        task_graph, embedding, 
        Shape(nr=d_model, nf=d_value), 
        precision, task_dict=task_dict["value"])
    task_dict["dot_product"] = {}
    dot_product_output, _ = create_mlp(
        task_graph, query, 
        Shape(nr=d_key, nf=num_tokens), 
        precision, task_dict=task_dict["dot_product"], weight=key)
    task_dict["add"] = {}
    mask = create_static(task_graph, Shape(token=num_tokens, nf=num_tokens),
                         precision)
    task_dict["add"]["mask"] = mask
    add_output, _ = create_add(task_graph, [dot_product_output, mask], 
                               dot_product_output.shape, precision, 
                               task_dict=task_dict["add"])
    task_dict["softmax"] = {}
    softmax_output, _ = create_pointwise(
        task_graph, add_output, precision, TaskBlockType.CSSoftMax,
        task_dict=task_dict["softmax"])
    task_dict["attention"] = {}
    output, _ = create_mlp(
        task_graph, softmax_output, 
        Shape(nr=num_tokens, nf=d_value), 
        precision, is_output, task_dict["attention"], value)
    return output, task_dict

def create_final_prefill_attention(task_graph: TaskGraph, embedding: TaskBlock,
                                   d_key: int, d_value: int, 
                                   precision: Precision, 
                                   is_output: bool = False, 
                                   task_dict: Dict = None):
    # 2MM + Attention
    task_dict = {} if task_dict is None else task_dict
    num_tokens = embedding.shape.token
    d_model = embedding.shape.nr
    last_embedding = create_data(
        task_graph, Shape(batch=embedding.shape.batch, nf=embedding.shape.nf), 
        precision)
    task_graph.connect(embedding.id, last_embedding.id)
    task_dict["query"] = {}
    query, _ = create_mlp(task_graph, last_embedding, 
                          Shape(nr=d_model, nf=d_key), 
                          precision, task_dict=task_dict["query"])
    task_dict["key"] = {}
    key, _ = create_mlp(task_graph, embedding, 
                        Shape(nr=d_model, nf=d_key), 
                        precision, task_dict=task_dict["key"])
    task_dict["value"] = {}
    value, _ = create_mlp(
        task_graph, embedding, 
        Shape(nr=d_model, nf=d_value), 
        precision, task_dict=task_dict["key"])
    task_dict["dot_product"] = {}
    dot_product_output, _ = create_mlp(task_graph, query,
                                       Shape(nr=d_key, nf=num_tokens), 
                                       precision,
                                       task_dict=task_dict["dot_product"], 
                                       weight=key)
    task_dict["softmax"] = {}
    softmax_output, _ = create_pointwise(
        task_graph, dot_product_output, precision, TaskBlockType.CSSoftMax,
        task_dict=task_dict["softmax"])
    task_dict["attention"] = {}
    output, _ = create_mlp(
        task_graph, softmax_output, Shape(nr=num_tokens, nf=d_value), precision, 
        is_output, task_dict["attention"], value)
    return output, task_dict

def create_attention(task_graph: TaskGraph, embedding: TaskBlock,
                     d_model: int, d_key: int, 
                     d_value: int, seq_len: int, precision: Precision, 
                     is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["query"] = {}
    query, _ = create_mlp(task_graph, embedding, Shape(nr=d_model, nf=d_key), 
                          precision, task_dict=task_dict["query"])
    task_dict["key"] = {}
    key, _ = create_mlp(task_graph, embedding, Shape(nr=d_model, nf=d_key), 
                        precision, task_dict=task_dict["key"])
    task_dict["value"] = {}
    value, _ = create_mlp(task_graph, embedding, Shape(nr=d_model, nf=d_value), 
                          precision, task_dict=task_dict["value"])
    key_cache = create_data(
        task_graph, Shape(nr=d_key, nf=seq_len - 1), precision)
    task_dict["concat_key"] = {}
    new_key_cache, _ = create_concat(task_graph, [key, key_cache], 
                                     Shape(nr=d_key, nf=seq_len), precision,
                                     task_dict=task_dict["concat_key"])
    task_dict["dot_product"] = {}
    dot_product_output, _ = create_mlp(task_graph, query,
                                       Shape(nr=d_key, nf=seq_len), precision,
                                       task_dict=task_dict["dot_product"], 
                                       weight=new_key_cache)
    # task_dict["scale"] = {}
    # scale_output, _ = create_scale(
    #     task_graph, dot_product_output, precision, 
    #     task_dict=task_dict["scale"], value=1 / sqrt(d_key))
    task_dict["softmax"] = {}
    # softmax_output, _ = create_softmax(
    #     task_graph, scale_output, precision, 
    #     task_dict=task_dict["softmax"])
    # including scale + softmax
    softmax_output, _ = create_pointwise(
        task_graph, dot_product_output, precision, TaskBlockType.CSSoftMax,
        task_dict=task_dict["softmax"])
    value_cache = create_data(task_graph, Shape(nr=seq_len - 1, nf=d_value), 
                              precision)
    task_dict["concat_value"] = {}
    new_value_cache, _ = create_concat(task_graph, [value, value_cache], 
                                       Shape(nr=seq_len, nf=d_value), precision,
                                       task_dict=task_dict["concat_value"])
    task_dict["attention"] = {}
    output, _ = create_mlp(
        task_graph, softmax_output, Shape(nr=seq_len, nf=d_value), precision, 
        is_output, task_dict["attention"], new_value_cache)
    
    task_dict["key_cache"] = {}
    task_dict["key_cache"]["old"] = key_cache
    task_dict["key_cache"]["new"] = new_key_cache
    task_dict["value_cache"] = {}
    task_dict["value_cache"]["old"] = value_cache
    task_dict["value_cache"]["new"] = new_value_cache

    return output, task_dict

def create_multi_head_attention(task_graph: TaskGraph, embedding: TaskBlock,
                                d_model: int, d_key: int, d_value: int, 
                                seq_len: int, head: int, precision: Precision, 
                                is_output: bool = False, 
                                task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    new_d_key = d_key // head
    new_d_value = d_value // head
    outputs = []
    for i in range(head):
        task_dict[i] = {}
        output, _ = create_attention(task_graph, embedding, d_model, 
                                     new_d_key, new_d_value, seq_len, 
                                     precision, task_dict=task_dict[i])
        outputs.append(output)
    task_dict["concat"] = {}
    concat_output, _ = create_concat(task_graph, outputs, 
                                     Shape(nf=d_value, nr=1), precision,
                                     task_dict=task_dict["concat"])
    task_dict["mlp"] = {}
    output, _ = create_mlp(task_graph, concat_output,
                           Shape(nr=d_value, nf=d_model), 
                           precision, is_output, task_dict=task_dict["mlp"])
    
    return output, task_dict

def create_prefill_multi_head_attention(task_graph: TaskGraph, 
                                        embedding: TaskBlock,
                                        d_key: int, d_value: int, 
                                        head: int, precision: Precision, 
                                        is_output: bool = False, 
                                        task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    num_tokens = embedding.shape.token
    d_model = embedding.shape.nf
    new_d_key = d_key // head
    new_d_value = d_value // head
    outputs = []
    for i in range(head):
        task_dict[i] = {}
        output, _ = create_prefill_attention(task_graph, embedding, 
                                             new_d_key, new_d_value,
                                             precision, task_dict=task_dict[i])
        outputs.append(output)
    task_dict["concat"] = {}
    concat_output, _ = create_concat(
        task_graph, outputs, 
        Shape(nf=d_value), 
        precision, task_dict=task_dict["concat"])
    task_dict["mlp"] = {}
    output, _ = create_mlp(task_graph, concat_output,
                           Shape(nr=d_value, nf=d_model), 
                           precision, is_output, task_dict=task_dict["mlp"])
    return output, task_dict

def create_final_prefill_multi_head_attention(task_graph: TaskGraph, 
                                              embedding: TaskBlock,
                                              d_key: int, d_value: int, 
                                              head: int, precision: Precision, 
                                              is_output: bool = False, 
                                              task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    d_model = embedding.shape.nf
    new_d_key = d_key // head
    new_d_value = d_value // head
    outputs = []
    for i in range(head):
        task_dict[i] = {}
        output, _ = create_final_prefill_attention(task_graph, embedding, 
                                                   new_d_key, new_d_value,
                                                   precision, 
                                                   task_dict=task_dict[i])
        outputs.append(output)
    task_dict["concat"] = {}
    concat_output, _ = create_concat(task_graph, outputs, 
                                     Shape(nf=d_value), 
                                     precision, task_dict=task_dict["concat"])
    task_dict["mlp"] = {}
    output, _ = create_mlp(task_graph, concat_output,
                           Shape(nr=d_value, nf=d_model), 
                           precision, is_output, task_dict=task_dict["mlp"])
    return output, task_dict

def create_layer_norm(task_graph: TaskGraph, input: TaskBlock,
                      precision: Precision, is_output: bool = False, 
                      task_dict: Dict = None, epsilon: float = 0):
    task_dict = {} if task_dict is None else task_dict
    task_dict["reduce_sum_mean"] = {}
    reduce_sum_mean_output, _ = create_reduce_sum(
        task_graph, input, precision, task_dict=task_dict["reduce_sum_mean"])
    task_dict["average"] = {}
    average_out, _ = create_scale(
        task_graph, reduce_sum_mean_output, precision, 
        task_dict=task_dict["average"], value=-1 / input.shape.volume)
    task_dict["add_mean"] = {}
    add_mean_out, _ = create_add(task_graph, [input, average_out], 
                                 input.shape, precision, 
                                 task_dict=task_dict["add_mean"])
    task_dict["product"] = {}
    product_output, _ = create_pointwise(
        task_graph, add_mean_out, precision, TaskBlockType.CVVH,
        task_dict=task_dict["product"])
    task_dict["reduce_sum_var"] = {}
    reduce_sum_var_output, _ = create_reduce_sum(
        task_graph, product_output, precision, 
        task_dict=task_dict["reduce_sum_var"])
    task_dict["add_var"] = {}
    add_var_out, _ = create_add(task_graph, [reduce_sum_var_output], 
                                reduce_sum_var_output.shape, precision, 
                                task_dict=task_dict["add_var"],
                                constant=epsilon)
    task_dict["sqrt"] = {}
    sqrt_out, _ = create_pointwise(
        task_graph, add_var_out, precision, TaskBlockType.CSQRT, 
        task_dict=task_dict["sqrt"])
    task_dict["div"] = {}
    output, _ = create_div(task_graph, [add_mean_out, sqrt_out],
                           add_mean_out.shape, precision, is_output,
                           task_dict["div"])
    return output, task_dict

def create_reduce_sum(task_graph: TaskGraph, input: TaskBlock,
                      precision: Precision, is_output: bool = False, 
                      task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    reduce_sum = create_compute(task_graph, input.shape, 
                                TaskBlockType.CReduceSum, precision)
    output = create_data(task_graph, Shape(nf=1), precision, is_output)
    task_graph.connect_tasks_in_sequence([input, reduce_sum, output])
    task_dict["compute"] = reduce_sum
    task_dict["output"] = output
    return output, task_dict

def create_add(task_graph: TaskGraph, inputs: Iterable[TaskBlock],
               shape: Shape, precision: Precision, is_output: bool = False,
               task_dict: Dict = None, constant: float = None):
    task_dict = {} if task_dict is None else task_dict
    compute_shape = copy(shape)
    compute_shape.branch = len(inputs)
    add = create_compute(task_graph, compute_shape, 
                         TaskBlockType.CADD, precision, constant)
    output = create_data(task_graph, shape, precision, is_output)
    for input in inputs:
        task_graph.connect(input.id, add.id)
    task_graph.connect(add.id, output.id)
    task_dict["compute"] = add
    task_dict["output"] = output
    return output, task_dict

def create_hadamard_product(task_graph: TaskGraph, inputs: Sequence[TaskBlock],
                            shape: Shape, precision: Precision, 
                            is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    product = create_compute(task_graph, shape, TaskBlockType.CVVH, precision)
    output = create_data(task_graph, shape, precision, is_output)
    for input in inputs:
        task_graph.connect(input.id, product.id)
    task_graph.connect(product.id, output.id)
    task_dict["compute"] = product
    task_dict["output"] = output
    return output, task_dict

def create_div(task_graph: TaskGraph, inputs: Sequence[TaskBlock],
               shape: Shape, precision: Precision, 
               is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    div = create_compute(task_graph, shape, TaskBlockType.CDIV, precision)
    output = create_data(task_graph, shape, precision, is_output)
    for input in inputs:
        task_graph.connect(input.id, div.id)
    task_graph.connect(div.id, output.id)
    task_dict["compute"] = div
    task_dict["output"] = output
    return output, task_dict

def create_attention_block(task_graph: TaskGraph, embedding: TaskBlock,
                           d_model: int, d_key: int, d_value: int, 
                           seq_len: int, head: int,
                           precision: Precision, is_output: bool = False, 
                           task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["multi_head_attention"] = {}
    attention_output, _ = create_multi_head_attention(
        task_graph, embedding, d_model, d_key, d_value, seq_len, head, 
        precision, task_dict=task_dict["multi_head_attention"])
    task_dict["add"] = {}
    add_output, _ = create_add(task_graph, [embedding, attention_output],
                               embedding.shape, precision,
                               task_dict=task_dict["add"])
    task_dict["layer_norm"] = {}
    # output, _ = create_layer_norm(task_graph, add_output, precision, is_output,
    #                               task_dict=task_dict["layer_norm"], 
    #                               epsilon=epsilon)
    output, _ = create_pointwise(task_graph, add_output, precision,
                                 TaskBlockType.CLayerNorm, is_output,
                                 task_dict["layer_norm"])
    return output, task_dict

def create_prefill_attention_block(task_graph: TaskGraph, embedding: TaskBlock,
                                   d_key: int, d_value: int, 
                                   head: int, precision: Precision, 
                                   is_output: bool = False, 
                                   task_dict: Dict = None,
                                   is_final: bool = False):
    '''
    This function can be applied to prefill and final prefill
    '''
    task_dict = {} if task_dict is None else task_dict
    task_dict["multi_head_attention"] = {}
    if is_final:
        attention_output, _ = create_final_prefill_multi_head_attention(
            task_graph, embedding, d_key, d_value, head, 
            precision, task_dict=task_dict["multi_head_attention"])
    else:
        attention_output, _ = create_prefill_multi_head_attention(
            task_graph, embedding, d_key, d_value, head, 
            precision, task_dict=task_dict["multi_head_attention"])
    task_dict["add"] = {}
    add_output, _ = create_add(task_graph, [embedding, attention_output],
                               embedding.shape, precision,
                               task_dict=task_dict["add"])
    task_dict["layer_norm"] = {}
    output, _ = create_pointwise(task_graph, add_output, precision,
                                 TaskBlockType.CLayerNorm, is_output,
                                 task_dict["layer_norm"])
    return output, task_dict
    
def create_ffn_block(task_graph: TaskGraph, input: TaskBlock, inner_dim: int,
                     precision: Precision, activation_type: TaskBlockType,
                     is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["ffn"] = {}
    ffn_output, _ = create_ffn(task_graph, input, inner_dim, precision,
                               activation_type, task_dict=task_dict["ffn"])
    task_dict["add"] = {}
    add_output, _ = create_add(task_graph, [input, ffn_output],
                               input.shape, precision,
                               task_dict=task_dict["add"])
    task_dict["layer_norm"] = {}
    output, _ = create_pointwise(task_graph, add_output, precision,
                                 TaskBlockType.CLayerNorm, is_output,
                                 task_dict["layer_norm"])
    return output, task_dict

def create_transformer_layer(task_graph: TaskGraph, embedding: TaskBlock,
                             d_key: int, d_value: int, 
                             seq_len: int, head: int, ffn_inner_dim: int, 
                             activation_type: TaskBlockType,
                             precision: Precision, is_output: bool = False, 
                             task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["attention"] = {}
    attention_output, _ = create_attention_block(
        task_graph, embedding, embedding.shape.nf, d_key, d_value, seq_len, 
        head, precision, task_dict=task_dict["attention"])
    task_dict["ffn"] = {}
    output, _ = create_ffn_block(task_graph, attention_output, ffn_inner_dim,
                                 precision, activation_type, is_output, 
                                 task_dict["ffn"])
    return output, task_dict

def create_prefill_transformer_layer(task_graph: TaskGraph, 
                                     embedding: TaskBlock,
                                     d_key: int, d_value: int, 
                                     head: int, ffn_inner_dim: int, 
                                     activation_type: TaskBlockType,
                                     precision: Precision, 
                                     is_output: bool = False, 
                                     task_dict: Dict = None,
                                     is_final: bool = False):
    task_dict = {} if task_dict is None else task_dict
    task_dict["attention"] = {}
    attention_output, _ = create_prefill_attention_block(
        task_graph, embedding, d_key, d_value, head, 
        precision, task_dict=task_dict["attention"], is_final=is_final)
    task_dict["ffn"] = {}
    output, _ = create_ffn_block(task_graph, attention_output, ffn_inner_dim,
                                 precision, activation_type, is_output, 
                                 task_dict["ffn"])
    return output, task_dict

def create_prefill_transformer(task_graph: TaskGraph, 
                               input_embedding: TaskBlock, 
                               position_encoding: TaskBlock, num_layers: int,
                               num_words: int, d_key: int, d_value: int, 
                               head: int, ffn_inner_dim: int, 
                               activation_type: TaskBlockType,
                               precision: Precision, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["add"] = {}
    embedding, _ = create_add(task_graph, [input_embedding, position_encoding],
                              input_embedding.shape, precision,
                              task_dict=task_dict["add"])
    embeddings = [embedding]
    for i in range(num_layers - 1):
        task_dict[i] = {}
        new_embedding, _ = create_prefill_transformer_layer(
            task_graph, embeddings[i], d_key, d_value, 
            head, ffn_inner_dim, activation_type, precision,
            task_dict=task_dict[i])
        embeddings.append(new_embedding)
    new_embedding, _ = create_prefill_transformer_layer(
        task_graph, embeddings[i], d_key, d_value, 
        head, ffn_inner_dim, activation_type, precision,
        task_dict=task_dict[i], is_final=True
    )
    task_dict["mlp"] = {}
    mlp_output, _ = create_mlp(task_graph, new_embedding, 
                               Shape(nr=embedding.shape.nf, nf=num_words),
                               precision, task_dict=task_dict["mlp"])
    task_dict["softmax"] = {}
    output, _ = create_pointwise(task_graph, mlp_output, precision, 
                                 TaskBlockType.CSoftMax,
                                 True, task_dict["softmax"])
    return output, task_dict

def create_transformer(task_graph: TaskGraph, input_embedding: TaskBlock, 
                       position_encoding: TaskBlock, num_layers: int,
                       num_words: int, d_key: int, d_value: int, 
                       seq_len: int, head: int, ffn_inner_dim: int, 
                       activation_type: TaskBlockType,
                       precision: Precision, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["add"] = {}
    embedding, _ = create_add(task_graph, [input_embedding, position_encoding],
                              input_embedding.shape, precision,
                              task_dict=task_dict["add"])
    embeddings = [embedding]
    for i in range(num_layers):
        task_dict[i] = {}
        new_embedding, _ = create_transformer_layer(
            task_graph, embeddings[i], d_key, d_value, 
            seq_len, head, ffn_inner_dim, activation_type, precision,
            task_dict=task_dict[i])
        embeddings.append(new_embedding)
    # task_dict[0] = {}
    # new_embedding, _ = create_transformer_layer(
    #     task_graph, embedding, d_key, d_value, 
    #     seq_len, head, epsilon, ffn_inner_dim, activation_type, precision,
    #     task_dict=task_dict[0])
    task_dict["mlp"] = {}
    mlp_output, _ = create_mlp(task_graph, new_embedding, 
                               Shape(nr=embedding.shape.nf, nf=num_words),
                               precision, task_dict=task_dict["mlp"])
    task_dict["softmax"] = {}
    output, _ = create_pointwise(task_graph, mlp_output, precision, 
                                 TaskBlockType.CSoftMax,
                                 True, task_dict["softmax"])
    return output, task_dict