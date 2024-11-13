from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import Coord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_cyclic_weight, create_tiled_elementwise, create_pointwise, create_data
import matplotlib.pyplot as plt
import toml
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from typing import Dict
import numpy as np
from copy import deepcopy


# Algorithm Parameters
BATCH = 8
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/distributed_many_core_board.toml")
config = BoardConfig(config["PCB"])
SIZE_X, SIZE_Y = config.chiplet.size
CHIP = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0


def simulate_tiled_mlp_weight_on_chip_2mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query, Key, Value
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=32, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False,
        weight_offchip=False,
        is_output=True
    )

    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_2mb_weight_on_chip.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(cyclic_local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(6, 7, 63)])
    return overall_latency


def simulate_tiled_mlp_weight_on_chip_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query, Key, Value
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=32, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False,
        weight_offchip=False,
        output_offchip=True,
        is_output=True
    )

    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb_weight_on_chip.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    env.put_in(dram, output.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(cyclic_local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(6, 7, 63)])
    return overall_latency


def simulate_tiled_mlp_weight_offchip_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query, Key, Value
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=64, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False,
        output_offchip=True,
        is_output=True
    )
    # weight_offchip.shape.nf = weight_offchip.shape.nf // 2
    weight_offchip: TaskBlock = mlp_task_dict["weight_offchip"]
    weight_offchip_shape = deepcopy(weight_offchip.shape)
    weight_offchip_shape.nf = weight_offchip_shape.nf // 2
    weight_offchip.shape = weight_offchip_shape

    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb_weight_offchip.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(cyclic_local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(7, 8, 63)])
    return overall_latency


def simulate_tiled_dot_product_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=D_MODEL),
        Precision.FLOAT_16)
    # Dot Product
    output, dot_product_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=32, nf=32),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        static_weight=False,
        is_input_on_chip=False
    )
    output, add_task_dict = create_tiled_elementwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CADD
    )
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output_offchip = create_data(task_graph, output.shape, Precision.FLOAT_16, 
                                 True)
    task_graph.connect_tasks_in_sequence([output, output_offchip])

    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb_dot_product.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Dot Product
    dram = create_mlcoord(DRAM)
    weight_offchip = dot_product_task_dict["weight_offchip"]
    env.put_in(dram, input_offchip.id)
    env.put_in(dram, weight_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    dot_product_input_on_chip = dot_product_task_dict["input_on_chip"]
    dot_product_weight_on_chip = dot_product_task_dict["weight_on_chip"]
    dot_product_cyclic_weight_on_chip = dot_product_task_dict["cyclic_weight_on_chip"]
    dot_product_output_on_chip = dot_product_task_dict["output_on_chip"]
    dot_product = dot_product_task_dict["compute"]
    cyclic_dot_product = dot_product_task_dict["cyclic_compute"]
    env.put_in(local_memory, dot_product_input_on_chip.id)
    env.put_in(local_memory, dot_product_weight_on_chip.id)
    env.put_in(local_memory, dot_product_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, dot_product_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_dot_product.id)
    env.sync(tensor_unit, sync_id2)

    # Add
    mask_offchip = add_task_dict["mask_offchip"]
    env.sync(dram, sync_id2)
    env.put_in(dram, mask_offchip.id)
    mask_on_chip = add_task_dict["mask_on_chip"]
    add = add_task_dict["compute"]
    add_output = add_task_dict["output"]
    softmax = softmax_task_dict["compute"]
    softmax_output = softmax_task_dict["output"]
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(local_memory, mask_on_chip.id)
    env.put_in(local_memory, add_output.id)
    env.put_in(local_memory, softmax_output.id)
    env.put_in(vector_unit, add.id)
    env.put_in(vector_unit, softmax.id)

    # output
    env.put_in(local_memory, output.id)
    env.put_in(dram, output_offchip.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(7, 8, 15)])
    return overall_latency


def simulate_tiled_dot_product_input_on_chip_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=D_MODEL),
        Precision.FLOAT_16)
    # Dot Product
    output, dot_product_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=32, nf=32),
        task_graph=task_graph,
        input=input,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        static_weight=False,
        is_input_on_chip=True,
        is_output=True
    )

    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb_dot_product_input_on_chip.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Dot Product
    dram = create_mlcoord(DRAM)
    weight_offchip = dot_product_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    dot_product_weight_on_chip = dot_product_task_dict["weight_on_chip"]
    dot_product_cyclic_weight_on_chip = dot_product_task_dict["cyclic_weight_on_chip"]
    dot_product_output_on_chip = dot_product_task_dict["output_on_chip"]
    dot_product = dot_product_task_dict["compute"]
    cyclic_dot_product = dot_product_task_dict["cyclic_compute"]
    env.put_in(local_memory, input.id)
    env.put_in(local_memory, dot_product_weight_on_chip.id)
    env.put_in(local_memory, dot_product_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, dot_product_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_dot_product.id)

    # output
    env.put_in(local_memory, output.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(6, 7, 15)])
    return overall_latency


def simulate_tiled_attention_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=SEQ_LEN),
        Precision.FLOAT_16)
    # Attention
    output, attention_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=64),
        task_graph=task_graph,
        input=input,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True,
        is_input_on_chip=True
    )
    output.shape.nf = output.shape.nf // 64
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    config.chiplet.core.local_memory["capacity"] = 2048
    config.chiplet.core.mac_array["fp16"]["parallelism"] = [64, 64]
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 512

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Attention
    dram = create_mlcoord(DRAM)
    weight_offchip = attention_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    attention_weight_on_chip = attention_task_dict["weight_on_chip"]
    attention_cyclic_weight_on_chip = attention_task_dict["cyclic_weight_on_chip"]
    attention_output_on_chip = attention_task_dict["output_on_chip"]
    attention = attention_task_dict["compute"]
    cyclic_attention = attention_task_dict["cyclic_compute"]
    env.put_in(local_memory, input.id)
    env.put_in(local_memory, attention_weight_on_chip.id)
    env.put_in(local_memory, attention_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, attention_cyclic_weight_on_chip.id)

    env.put_in(tensor_unit, attention.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_attention.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(6, 7, 15)])
    return overall_latency
    

def simulate(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=D_MODEL),
        Precision.FLOAT_16)
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=128),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False
    )
    output, dot_product_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        static_weight=False
    )
    output, add_task_dict = create_tiled_elementwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CADD
    )
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True
    )
    # STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
    #                     width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    config.chiplet.core.local_memory["capacity"] = 2560
    config.chiplet.core.mac_array["fp16"]["parallelism"] = [32, 32]
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
            config.chiplet.core.mac_array["fp16"]["latency"] = hardware_paramter_dict[type][0] * 2

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    # print(many_core_board.container[Coord(CHIP)].area)
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Graph Transformation
    # num_batch_split = 2
    # num_token_split = 2
    # split_embedding = env.split_task(embedding.id, 
    #                                  SplitVector(batch=num_batch_split, 
    #                                              token=num_token_split))
    # env.add_nodes_between(embedding, embedding.out_tasks, 
    #                       split_embedding)
    # split_task_dict = env.split_attention(
    #     task_dict=task_dict,
    #     split_embedding=split_embedding,
    #     query_split_vector=SplitVector(),
    #     key_split_vector=SplitVector(),
    #     value_split_vector=SplitVector(),
    #     dot_product_split_vector=SplitVector(),
    #     softmax_split_vector=SplitVector(),
    #     attention_split_vector=SplitVector(),
    #     attention_type=AttentionType.PREFILL,
    #     num_batch_split=num_batch_split,
    #     num_token_split=num_token_split
    # )
    # output.enable()
    # env.connect_tasks(split_task_dict["attention"]["output"], [output])
    # STDraw.draw_graph(task_graph, 
    #                     out_path='temp/tiled_prefill_attention.task.html',
    #                     width='1920px', height='1080px')

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    mlp_weight_offchip = mlp_task_dict["weight_offchip"]
    env.put_in(dram, mlp_weight_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)
    env.sync(tensor_unit, sync_id2)

    # Dot Product
    weight_offchip = dot_product_task_dict["weight_offchip"]
    env.sync(dram, sync_id2)
    env.put_in(dram, weight_offchip.id)

    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    dot_product_weight_on_chip = dot_product_task_dict["weight_on_chip"]
    dot_product_cyclic_weight_on_chip = dot_product_task_dict["cyclic_weight_on_chip"]
    dot_product_output_on_chip = dot_product_task_dict["output_on_chip"]
    dot_product = dot_product_task_dict["compute"]
    cyclic_dot_product = dot_product_task_dict["cyclic_compute"]
    env.put_in(local_memory, dot_product_weight_on_chip.id)
    env.put_in(local_memory, dot_product_output_on_chip.id)
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id3)
    env.put_in(cyclic_local_memory, dot_product_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product.id)
    env.sync(tensor_unit, sync_id3)
    env.put_in(tensor_unit, cyclic_dot_product.id)
    env.sync(tensor_unit, sync_id4)

    # Add
    mask_offchip = add_task_dict["mask_offchip"]
    env.sync(dram, sync_id4)
    env.put_in(dram, mask_offchip.id)
    mask_on_chip = add_task_dict["mask_on_chip"]
    add = add_task_dict["compute"]
    add_output = add_task_dict["output"]
    softmax = softmax_task_dict["compute"]
    softmax_output = softmax_task_dict["output"]
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(local_memory, mask_on_chip.id)
    env.put_in(local_memory, add_output.id)
    env.put_in(local_memory, softmax_output.id)
    env.put_in(vector_unit, add.id)
    env.put_in(vector_unit, softmax.id)

    # Attention
    weight_offchip = attention_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    attention_weight_on_chip = attention_task_dict["weight_on_chip"]
    attention_cyclic_weight_on_chip = attention_task_dict["cyclic_weight_on_chip"]
    attention_output_on_chip = attention_task_dict["output_on_chip"]
    attention = attention_task_dict["compute"]
    cyclic_attention = attention_task_dict["cyclic_compute"]
    env.put_in(local_memory, attention_weight_on_chip.id)
    env.put_in(local_memory, attention_output_on_chip.id)
    sync_id5 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id5)
    env.put_in(cyclic_local_memory, attention_cyclic_weight_on_chip.id)

    env.put_in(tensor_unit, attention.id)
    env.sync(tensor_unit, sync_id5)
    env.put_in(tensor_unit, cyclic_attention.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(7, 8, 127), 
                                             LoopInfo(13, 14, 15), 
                                             LoopInfo(25, 26, 15)])
    return overall_latency


def simulate_3mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=D_MODEL),
        Precision.FLOAT_16)
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=32),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False
    )
    # weight_offchip.shape.nf = weight_offchip.shape.nf // 2
    weight_offchip: TaskBlock = mlp_task_dict["weight_offchip"]
    weight_offchip_shape = deepcopy(weight_offchip.shape)
    weight_offchip_shape.nf = weight_offchip_shape.nf // 4
    weight_offchip.shape = weight_offchip_shape
    output, dot_product_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        static_weight=False
    )
    output, add_task_dict = create_tiled_elementwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CADD
    )
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True
    )
    # STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
    #                     width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    config.chiplet.core.local_memory["capacity"] = 3072
    config.chiplet.core.mac_array["fp16"]["parallelism"] = [16, 16]
    config.chiplet.core.mac_array["fp16"]["latency"] = 32
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
            config.chiplet.core.mac_array["fp16"]["latency"] = hardware_paramter_dict[type][0] * 2

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    # print(many_core_board.container[Coord(CHIP)].area)
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    mlp_weight_offchip = mlp_task_dict["weight_offchip"]
    env.put_in(dram, mlp_weight_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)
    env.sync(tensor_unit, sync_id2)

    # Dot Product
    weight_offchip = dot_product_task_dict["weight_offchip"]
    env.sync(dram, sync_id2)
    env.put_in(dram, weight_offchip.id)

    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    dot_product_weight_on_chip = dot_product_task_dict["weight_on_chip"]
    dot_product_cyclic_weight_on_chip = dot_product_task_dict["cyclic_weight_on_chip"]
    dot_product_output_on_chip = dot_product_task_dict["output_on_chip"]
    dot_product = dot_product_task_dict["compute"]
    cyclic_dot_product = dot_product_task_dict["cyclic_compute"]
    env.put_in(local_memory, dot_product_weight_on_chip.id)
    env.put_in(local_memory, dot_product_output_on_chip.id)
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id3)
    env.put_in(cyclic_local_memory, dot_product_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product.id)
    env.sync(tensor_unit, sync_id3)
    env.put_in(tensor_unit, cyclic_dot_product.id)
    env.sync(tensor_unit, sync_id4)

    # Add
    mask_offchip = add_task_dict["mask_offchip"]
    env.sync(dram, sync_id4)
    env.put_in(dram, mask_offchip.id)
    mask_on_chip = add_task_dict["mask_on_chip"]
    add = add_task_dict["compute"]
    add_output = add_task_dict["output"]
    softmax = softmax_task_dict["compute"]
    softmax_output = softmax_task_dict["output"]
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(local_memory, mask_on_chip.id)
    env.put_in(local_memory, add_output.id)
    env.put_in(local_memory, softmax_output.id)
    env.put_in(vector_unit, add.id)
    env.put_in(vector_unit, softmax.id)

    # Attention
    weight_offchip = attention_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    attention_weight_on_chip = attention_task_dict["weight_on_chip"]
    attention_cyclic_weight_on_chip = attention_task_dict["cyclic_weight_on_chip"]
    attention_output_on_chip = attention_task_dict["output_on_chip"]
    attention = attention_task_dict["compute"]
    cyclic_attention = attention_task_dict["cyclic_compute"]
    env.put_in(local_memory, attention_weight_on_chip.id)
    env.put_in(local_memory, attention_output_on_chip.id)
    sync_id5 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id5)
    env.put_in(cyclic_local_memory, attention_cyclic_weight_on_chip.id)

    env.put_in(tensor_unit, attention.id)
    env.sync(tensor_unit, sync_id5)
    env.put_in(tensor_unit, cyclic_attention.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(7, 8, 31), 
                                             LoopInfo(13, 14, 15), 
                                             LoopInfo(25, 26, 15)])
    return overall_latency


def simulate_2mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query, Key, Value
    output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=32, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_input_on_chip=False
    )
    # weight_offchip.shape.nf = weight_offchip.shape.nf // 2
    weight_offchip: TaskBlock = mlp_task_dict["weight_offchip"]
    weight_offchip_shape = deepcopy(weight_offchip.shape)
    weight_offchip_shape.nf = weight_offchip_shape.nf // 2
    weight_offchip.shape = weight_offchip_shape
    output.shape.token = output.shape.token * 2
    # Dot Product
    output, dot_product_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=32),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        static_weight=False
    )
    output, add_task_dict = create_tiled_elementwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CADD
    )
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=32),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True
    )
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_2mb.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"])
    config.chiplet.core.local_memory["capacity"] = 2048
    config.chiplet.core.mac_array["fp16"]["parallelism"] = [64, 64]
    config.chiplet.core.mac_array["fp16"]["latency"] = 128
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 512
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
            config.chiplet.core.mac_array["fp16"]["latency"] = hardware_paramter_dict[type][0] * 2

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    mlp_weight_offchip = mlp_task_dict["weight_offchip"]
    env.put_in(dram, mlp_weight_offchip.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
    mlp_input_on_chip = mlp_task_dict["input_on_chip"]
    mlp_weight_on_chip = mlp_task_dict["weight_on_chip"]
    mlp_cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
    mlp_output_on_chip = mlp_task_dict["output_on_chip"]
    mlp = mlp_task_dict["compute"]
    cyclic_mlp = mlp_task_dict["cyclic_compute"]
    env.put_in(local_memory, mlp_input_on_chip.id)
    env.put_in(local_memory, mlp_weight_on_chip.id)
    env.put_in(local_memory, mlp_output_on_chip.id)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id1)
    env.put_in(cyclic_local_memory, mlp_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp.id)
    env.sync(tensor_unit, sync_id1)
    env.put_in(tensor_unit, cyclic_mlp.id)
    env.sync(tensor_unit, sync_id2)

    # Dot Product
    weight_offchip = dot_product_task_dict["weight_offchip"]
    env.sync(dram, sync_id2)
    env.put_in(dram, weight_offchip.id)

    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    dot_product_weight_on_chip = dot_product_task_dict["weight_on_chip"]
    dot_product_cyclic_weight_on_chip = dot_product_task_dict["cyclic_weight_on_chip"]
    dot_product_output_on_chip = dot_product_task_dict["output_on_chip"]
    dot_product = dot_product_task_dict["compute"]
    cyclic_dot_product = dot_product_task_dict["cyclic_compute"]
    env.put_in(local_memory, dot_product_weight_on_chip.id)
    env.put_in(local_memory, dot_product_output_on_chip.id)
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id3)
    env.put_in(cyclic_local_memory, dot_product_cyclic_weight_on_chip.id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product.id)
    env.sync(tensor_unit, sync_id3)
    env.put_in(tensor_unit, cyclic_dot_product.id)
    env.sync(tensor_unit, sync_id4)

    # Add
    mask_offchip = add_task_dict["mask_offchip"]
    env.sync(dram, sync_id4)
    env.put_in(dram, mask_offchip.id)
    mask_on_chip = add_task_dict["mask_on_chip"]
    add = add_task_dict["compute"]
    add_output = add_task_dict["output"]
    softmax = softmax_task_dict["compute"]
    softmax_output = softmax_task_dict["output"]
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(local_memory, mask_on_chip.id)
    env.put_in(local_memory, add_output.id)
    env.put_in(local_memory, softmax_output.id)
    env.put_in(vector_unit, add.id)
    env.put_in(vector_unit, softmax.id)

    # Attention
    weight_offchip = attention_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    attention_weight_on_chip = attention_task_dict["weight_on_chip"]
    attention_cyclic_weight_on_chip = attention_task_dict["cyclic_weight_on_chip"]
    attention_output_on_chip = attention_task_dict["output_on_chip"]
    attention = attention_task_dict["compute"]
    cyclic_attention = attention_task_dict["cyclic_compute"]
    env.put_in(local_memory, attention_weight_on_chip.id)
    env.put_in(local_memory, attention_output_on_chip.id)
    sync_id5 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id5)
    env.put_in(cyclic_local_memory, attention_cyclic_weight_on_chip.id)

    env.put_in(tensor_unit, attention.id)
    env.sync(tensor_unit, sync_id5)
    env.put_in(tensor_unit, cyclic_attention.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[LoopInfo(7, 8, 63), 
                                             LoopInfo(9, 14, 1),
                                             LoopInfo(13, 14, 30),
                                             LoopInfo(21, 26, 1),
                                             LoopInfo(25, 26, 30)],
                                      extra_latencies=[simulate_tiled_mlp_weight_on_chip_2mb()]
                                      )
    return overall_latency


def simulate_1mb(hardware_paramter_dict: Dict[str, int] = []):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=SEQ_LEN),
        Precision.FLOAT_16)
    # Attention
    output, attention_task_dict = create_tiled_mlp_cyclic_weight(
        split_vector=SplitVector(batch=8, token=16, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True,
        is_input_on_chip=False
    )
    output.shape.nf = output.shape.nf // 64
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention_1mb.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    config.chiplet.core.local_memory["capacity"] = 1024
    config.chiplet.core.mac_array["fp16"]["parallelism"] = [128, 128]
    config.chiplet.core.mac_array["fp16"]["latency"] = 256
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
            config.chiplet.core.mac_array["fp16"]["latency"] = hardware_paramter_dict[type][0] * 2

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    # print(many_core_board.container[Coord(CHIP)].area)
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    # print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Attention
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    weight_offchip = attention_task_dict["weight_offchip"]
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output.id)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    attention_input_on_chip = attention_task_dict["input_on_chip"]
    attention_weight_on_chip = attention_task_dict["weight_on_chip"]
    attention_cyclic_weight_on_chip = attention_task_dict["cyclic_weight_on_chip"]
    attention_output_on_chip = attention_task_dict["output_on_chip"]
    attention = attention_task_dict["compute"]
    cyclic_attention = attention_task_dict["cyclic_compute"]
    env.put_in(local_memory, attention_input_on_chip.id)
    env.put_in(local_memory, attention_weight_on_chip.id)
    env.put_in(local_memory, attention_output_on_chip.id)
    sync_id5 = env.get_sync_id()
    env.sync(cyclic_local_memory, sync_id5)
    env.put_in(cyclic_local_memory, attention_cyclic_weight_on_chip.id)

    env.put_in(tensor_unit, attention.id)
    env.sync(tensor_unit, sync_id5)
    env.put_in(tensor_unit, cyclic_attention.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(
        loops=[LoopInfo(7, 8, 16)],
        extra_latencies=[simulate_tiled_attention_1mb(hardware_paramter_dict) * 3,
                         simulate_tiled_dot_product_1mb(hardware_paramter_dict) * 2,
                         simulate_tiled_dot_product_input_on_chip_1mb(hardware_paramter_dict) * 6,
                         simulate_tiled_mlp_weight_offchip_1mb(hardware_paramter_dict) * 3,
                         simulate_tiled_mlp_weight_on_chip_1mb(hardware_paramter_dict) * 9]
         )
    return overall_latency


# noc_bandwidth_paramters = []
# noc_bandwidth_latencies = []
# for noc_bandwidth in (4, 8, 16, 32, 64):
#     noc_bandwidth_paramters.append(noc_bandwidth)
#     noc_bandwidth_latencies.append(simulate(noc_bandwidth, "noc_bandwidth"))

# local_memory_latency_paramters = []
# local_memory_latency_latencies = []
# for local_memory_latency in range(1, 20):
#     local_memory_latency_paramters.append(local_memory_latency)
#     local_memory_latency_latencies.append(simulate(local_memory_latency, "local_memory_latency"))

# local_memory_bandwidth_paramters = []
# local_memory_bandwidth_latencies = []
# for local_memory_bandwidth in (4, 8, 16, 32, 64, 128):
#     local_memory_bandwidth_paramters.append(local_memory_bandwidth)
#     local_memory_bandwidth_latencies.append(simulate(local_memory_bandwidth, "local_memory_bandwidth"))

latency1 = simulate_1mb()
latency2 = simulate_2mb()
latency2_5 = simulate()
latency3 = simulate_3mb()

parameter_labels = ["(1024, 128, 512)", "(2048, 64, 512)", "(2560, 32, 128)", "(3072, 16, 128)"]
latencies = [latency1, latency2, latency2_5, latency3]

plt.plot(parameter_labels, latencies, marker='o')
plt.title("Distributed Many Core")
plt.xlabel("Hardware Parameter")
plt.ylabel("Latency")

plt.xticks(rotation=45)  # 旋转x轴标签，使其更易读
# plt.legend()
plt.tight_layout()       # 自动调整布局
plt.savefig('test/exp/distributed_many_core_compute_memory.png', dpi=300)


noc_bandwidth_paramters = []
local_memory_latency_paramters = []
local_memory_bandwidth_paramters = []
mac_array1 = {16: [146, 146], 32: [146, 146], 64: [145, 145], 128: [142, 142], 256: [138, 138], 512: [128, 128]}
mac_array2 = {8: [107, 107], 16: [105, 105], 32: [103, 103], 64: [98, 98], 128: [88, 88], 256: [64, 64]}
mac_array2_5 = {4: [73, 73], 8: [72, 72], 16: [69, 69], 32: [64, 64], 64: [51, 51], 128: [32, 32]}
mac_array3 = {4: [72, 72], 8: [71, 71], 16: [68, 68], 32: [60, 60], 64: [44, 44], 128: [16, 16]}
# paramters = []
latencies = []
local_memory_bandwidth1 = (16, 32, 64, 128, 256, 512)
local_memory_bandwidth2 = (8, 16, 32, 64, 128, 256)
local_memory_bandwidth3 = (4, 8, 16, 32, 64, 128)
for local_memory_bandwidth in local_memory_bandwidth1:
    for local_memory_latency in range(80, 1, -10):
        for noc_bandwidth in (4, 8, 16, 32, 64):
            noc_bandwidth_paramters.append(noc_bandwidth)
            local_memory_latency_paramters.append(local_memory_latency)
            local_memory_bandwidth_paramters.append(local_memory_bandwidth)
            # paramters.append((local_memory_bandwidth, local_memory_latency, noc_bandwidth))
            latencies.append(simulate_1mb({'noc_bandwidth': noc_bandwidth,
                                           'local_memory_latency': local_memory_latency,
                                           'local_memory_bandwidth': local_memory_bandwidth,
                                           'mac_array': mac_array1[local_memory_bandwidth]}))
               
np.savez('{:s}.npz'.format("test/exp/many_core_1_prefill"), 
         noc_bandwidth_paramters=np.array(noc_bandwidth_paramters), 
         local_memory_latency_paramters=np.array(local_memory_latency_paramters),
         local_memory_bandwidth_paramters=np.array(local_memory_bandwidth_paramters),
         latencies=np.array(latencies))

# parameter_labels = [f"({x}, {y}, {z})" for x, y, z in paramters]
# plt.plot(noc_bandwidth_paramters, noc_bandwidth_latencies, marker='o', label='NoC Bandwidth')
# plt.plot(local_memory_latency_paramters, local_memory_latency_latencies, marker='o', label='Local Memory Latency')
# plt.plot(local_memory_bandwidth_paramters, local_memory_bandwidth_latencies, marker='o', label='Local Memory Bandwidth')

# plt.plot(parameter_labels, latencies, marker='o')
# plt.title("Distributed Many Core")
# plt.xlabel("Hardware Parameter")
# plt.ylabel("Latency")

# plt.xticks(rotation=45)  # 旋转x轴标签，使其更易读
# # plt.legend()
# plt.tight_layout()       # 自动调整布局
# plt.savefig('temp/distributed_many_core.png', dpi=300)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置柱的参数
dx = dy = 0.4  # 柱的宽度
dz = latencies  # 柱的高度

# 将bandwidth取对数
local_memory_bandwidth_paramters = np.log2(local_memory_bandwidth_paramters)
noc_bandwidth_paramters = np.log2(noc_bandwidth_paramters)

# x1: local bandwidth x2: latency x3: noc bandwidth
# 定义偏移量，用于将x2编码成不同的层
offset = 0.5
x1_combined = [local_memory_bandwidth_paramters[i] * 10 + (20 - local_memory_latency_paramters[i]) * offset for i in range(len(local_memory_latency_paramters))]  # 将x1和x2组合

# 根据x1的值设置颜色，颜色随着数值变大而变化
colors = plt.cm.viridis((np.array(local_memory_bandwidth_paramters) - min(local_memory_bandwidth_paramters)) / (max(local_memory_bandwidth_paramters) - min(local_memory_bandwidth_paramters)))

# 绘制三维柱状图
ax.bar3d(x1_combined, noc_bandwidth_paramters, np.zeros_like(latencies), dx, dy, dz, color=colors)

# # 添加二维坐标标签(x1, x2)作为y轴上的标记
# for i in range(len(y)):
#     ax.text(x1[i], x2_combined[i], y[i] + 0.5, f"({x2[i]},{x3[i]})", color='black', ha='center')

# 设置坐标轴标签
ax.set_xlabel('Local Memory Bandwidth + Latency')
ax.set_ylabel('NoC Bandwidth')
ax.set_zlabel('Latency')

plt.title("Distributed Many Core")

# 添加颜色映射条
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(local_memory_bandwidth_paramters)
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_ticks([4, 8, 16, 32, 64, 128])

plt.savefig('temp/test.png', dpi=300)


