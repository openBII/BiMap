from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import Coord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_decoder_weight, create_tiled_elementwise, create_pointwise, create_tiled_data_arrange_16_to_1_broadcast
import matplotlib.pyplot as plt
import toml
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from typing import Dict
import numpy as np


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

# 2048-th Decoder

def simulate_decoder_1mb(hardware_paramter_dict: Dict[str, int], ram):
    assert ram == 1
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=1, nf=D_MODEL),
        Precision.FLOAT_16)
    # calculate Q,K,V. Loop-3.
    output, mlp_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=64),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=True,
        output_offchip=False,
        is_input_on_chip=False,
        cyclic=3,
        cyclic_from_offchip=False
    )
    # re-arragne output data together
    output, arrange_q_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Q * K
    output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=32),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=False,
        output_offchip=False,
        is_input_on_chip=True,
        cyclic=1,
        cyclic_from_offchip=True
    )
    # re-arragne output data together
    output, arrange_qk_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Softmax
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=32),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True,
        cyclic=1,
        cyclic_from_offchip=True
    )
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention1.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    config.chiplet.core.mac_array["fp16"]["parallelism"] = [128, 128]
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    config.chiplet.core.local_memory["capacity"] = 1 * 1024

    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    print(many_core_board.container[Coord(CHIP)].area)
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_qk_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()
    sync_id5 = env.get_sync_id()
    sync_id6 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["compute"].id)

    # sync 1
    env.sync(tensor_unit, sync_id1)
    env.sync(local_memory, sync_id1)

    # Q,K,V cyclic for 3 times
    env.put_in(local_memory, mlp_task_dict["cyclic_weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["cyclic_compute"].id)

    # sync 2
    env.sync(tensor_unit, sync_id2)
    env.sync(arrange_q_local_memory, sync_id2)
    env.sync(dram, sync_id2)

    # re-arrange Q
    env.put_in(local_memory, mlp_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_q_local_memory, arrange_q_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_q_arrange_memory, arrange_q_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_q_task_dict["output"].id)

    # Q * K
    env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # sync 3
    env.sync(tensor_unit, sync_id3)
    env.sync(dram, sync_id3)

    # Q * K cyclic
    env.put_in(dram, dot_product_task_dict["cyclic_weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["cyclic_weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product_task_dict["cyclic_compute"].id)

    # sync 4
    env.sync(tensor_unit, sync_id4)
    env.sync(arrange_qk_local_memory, sync_id4)

    # re-arrange Q*K result
    env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_qk_local_memory, arrange_qk_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_qk_arrange_memory, arrange_qk_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_qk_task_dict["output"].id)

    # softmax
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 5
    env.sync(vector_unit, sync_id5)
    env.sync(dram, sync_id5)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)

    # sync 6
    env.sync(tensor_unit, sync_id6)
    env.sync(dram, sync_id6)

    # result * V
    env.put_in(dram, attention_task_dict["cyclic_weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["cyclic_weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, attention_task_dict["cyclic_compute"].id)

    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    env.put_in(dram, attention_task_dict["output_offchip"].id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[
        LoopInfo(input_offchip.id,                                  arrange_q_task_dict["output"].id, 2),     # Q, K, V Loop
        LoopInfo(mlp_task_dict["cyclic_weight_on_chip"].id,         mlp_task_dict["cyclic_compute"].id, 12-3), # Q,K,V cyclic
        LoopInfo(dot_product_task_dict["cyclic_weight_offchip"].id, dot_product_task_dict["cyclic_compute"].id, 1), # Q * K cyclic
        LoopInfo(attention_task_dict["cyclic_weight_offchip"].id,   attention_task_dict["cyclic_compute"].id, 1)  # Q * K cyclic
        ])
    return overall_latency


def simulate_decoder_2mb(hardware_paramter_dict: Dict[str, int], ram):
    assert ram == 2
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=1, nf=D_MODEL),
        Precision.FLOAT_16)
    # calculate Q,K,V. Loop-3.
    output, mlp_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=32),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=True,
        output_offchip=False,
        is_input_on_chip=False,
        cyclic=1,
        cyclic_from_offchip=False
    )
    # re-arragne output data together
    output, arrange_q_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Q * K
    output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=False,
        output_offchip=False,
        is_input_on_chip=True,
    )
    # re-arragne output data together
    output, arrange_qk_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Softmax
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True
    )
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    config.chiplet.core.mac_array["fp16"]["parallelism"] = [64, 64]
    config.chiplet.core.vector_unit["fp16"]["parallelism"] = 512
    config.chiplet.core.local_memory["capacity"] = 2 * 1024
    
    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    print(many_core_board.container[Coord(CHIP)].area)
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_qk_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["compute"].id)

    # sync 1
    env.sync(tensor_unit, sync_id1)
    env.sync(local_memory, sync_id1)

    # Q,K,V cyclic
    env.put_in(local_memory, mlp_task_dict["cyclic_weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["cyclic_compute"].id)

    # sync 2
    env.sync(tensor_unit, sync_id2)
    env.sync(arrange_q_local_memory, sync_id2)
    env.sync(dram, sync_id2)

    # re-arrange Q
    env.put_in(local_memory, mlp_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_q_local_memory, arrange_q_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_q_arrange_memory, arrange_q_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_q_task_dict["output"].id)

    # Q * K
    env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # sync 3
    env.sync(tensor_unit, sync_id3)
    env.sync(arrange_qk_local_memory, sync_id3)

    # re-arrange Q*K result
    env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_qk_local_memory, arrange_qk_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_qk_arrange_memory, arrange_qk_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_qk_task_dict["output"].id)

    # softmax
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 4
    env.sync(vector_unit, sync_id4)
    env.sync(dram, sync_id4)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)
    env.put_in(dram, attention_task_dict["output_offchip"].id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[
        LoopInfo(input_offchip.id, arrange_q_task_dict["output"].id, 2)     # Q, K, V Loop
        ])
    return overall_latency


def simulate_decoder_2p5mb_3mb(hardware_paramter_dict: Dict[str, int], ram):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 8, token=1, nf=D_MODEL),
        Precision.FLOAT_16)
    # calculate Q,K,V. Loop-3.
    output, mlp_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=16),
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=True,
        output_offchip=False,
        is_input_on_chip=False
    )
    # re-arragne output data together
    output, arrange_q_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Q * K
    output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=False,
        output_offchip=False,
        is_input_on_chip=True,
    )
    # re-arragne output data together
    output, arrange_qk_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16
    )
    # Softmax
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=SplitVector(batch=8, token=1, nf=16),
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=True,
        output_offchip=True
    )
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    if ram == 2.5:
        config.chiplet.core.mac_array["fp16"]["parallelism"] = [32, 32]
        config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
        config.chiplet.core.local_memory["capacity"] = 2.5 * 1024
    elif ram == 3:
        config.chiplet.core.mac_array["fp16"]["parallelism"] = [16, 16]
        config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
        config.chiplet.core.local_memory["capacity"] = 3 * 1024
    else:
        raise NotImplementedError
    
    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.DISTRIBUTED_MANY_CORE)
    
    print(many_core_board.container[Coord(CHIP)].area)
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area())
    print(many_core_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area())

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_q_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    arrange_qk_arrange_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["compute"].id)

    # sync 1
    env.sync(tensor_unit, sync_id1)
    env.sync(arrange_q_local_memory, sync_id1)
    env.sync(dram, sync_id1)

    # re-arrange Q
    env.put_in(local_memory, mlp_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_q_local_memory, arrange_q_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_q_arrange_memory, arrange_q_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_q_task_dict["output"].id)

    # Q * K
    env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # sync 2
    env.sync(tensor_unit, sync_id2)
    env.sync(arrange_qk_local_memory, sync_id2)

    # re-arrange Q*K result
    env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
    for i in range(15):
        env.put_in(arrange_qk_local_memory, arrange_qk_task_dict["weight{}".format(i)].id)
    env.put_in(arrange_qk_arrange_memory, arrange_qk_task_dict["data_merged"].id)
    env.put_in(local_memory, arrange_qk_task_dict["output"].id)

    # softmax
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 3
    env.sync(vector_unit, sync_id3)
    env.sync(dram, sync_id3)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)
    env.put_in(dram, attention_task_dict["output_offchip"].id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    overall_latency = env.get_latency(loops=[
        LoopInfo(input_offchip.id, arrange_q_task_dict["output"].id, 2)     # Q, K, V Loop
        ])
    return overall_latency


def draw_one_ram_size(ram, file_name : str):

    if ram == 1:
        sim_func = simulate_decoder_1mb
    elif ram == 2:
        sim_func = simulate_decoder_2mb
    elif ram == 2.5 or ram == 3:
        sim_func = simulate_decoder_2p5mb_3mb
    else:
        raise NotImplementedError

    noc_bandwidth_paramters = []
    local_memory_latency_paramters = []
    local_memory_bandwidth_paramters = []
    nb = [128, 64, 32, 16, 8, 4]
    lml = [i for i in range(80, 1, -10)]
    lmb = [4, 8, 16, 32, 64]
    # paramters = []
    latencies = []
    for local_memory_bandwidth in lmb:
        for local_memory_latency in lml:
            for noc_bandwidth in nb:
                noc_bandwidth_paramters.append(noc_bandwidth)
                local_memory_latency_paramters.append(local_memory_latency)
                local_memory_bandwidth_paramters.append(local_memory_bandwidth)
                # paramters.append((local_memory_bandwidth, local_memory_latency, noc_bandwidth))
                latencies.append(sim_func({'noc_bandwidth': noc_bandwidth,
                                'local_memory_latency': local_memory_latency,
                                'local_memory_bandwidth': local_memory_bandwidth}, ram=ram))
    np.savez('{:s}.npz'.format(file_name), 
         noc_bandwidth_paramters=np.array(noc_bandwidth_paramters), 
         local_memory_latency_paramters=np.array(local_memory_latency_paramters),
         local_memory_bandwidth_paramters=np.array(local_memory_bandwidth_paramters),
         latencies=np.array(latencies))
    
    # data = np.load('{:s}.npz'.format(file_name))
    # noc_bandwidth_paramters = data["noc_bandwidth_paramters"]
    # local_memory_latency_paramters = data["local_memory_latency_paramters"]
    # local_memory_bandwidth_paramters = data["local_memory_bandwidth_paramters"]
    # latencies = data["latencies"]

    # import scienceplots
    # plt.style.use(['ieee'])
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置柱的参数
    dx = 1
    dy = 0.6
    dz = latencies  # 柱的高度
    font_size = 9

    # 将bandwidth取对数
    local_memory_bandwidth_paramters = np.log2(local_memory_bandwidth_paramters)
    noc_bandwidth_paramters = np.log2(max(noc_bandwidth_paramters)) - np.log2(noc_bandwidth_paramters) + 1

    # x1: local bandwidth x2: latency x3: noc bandwidth
    # 定义偏移量，用于将x2编码成不同的层
    # offset = 0.5
    # x1_combined = [local_memory_bandwidth_paramters[i] * 10 + (20 - local_memory_latency_paramters[i]) * offset for i in range(len(local_memory_latency_paramters))]  # 将x1和x2组合
    x1_combined = []
    offset = (10.0) / len(lml)
    for i, local_memory_bandwidth in enumerate(lmb):
        for j, local_memory_latency in enumerate(lml):
            for noc_bandwidth in nb:
                x1_combined.append(i * 13 + j * offset)

    # 排序，改变绘制柱子的顺序，避免重叠
    sorted_indices = np.argsort(x1_combined)
    x1_combined = np.array(x1_combined)[sorted_indices]
    noc_bandwidth_paramters = np.array(noc_bandwidth_paramters)[sorted_indices]
    dz = np.array(latencies)[sorted_indices]
    # 根据x1的值设置颜色，颜色随着数值变大而变化
    colors = plt.cm.viridis((np.array(local_memory_latency_paramters) - min(local_memory_latency_paramters)) / (max(local_memory_latency_paramters) - min(local_memory_latency_paramters)))

    # 绘制三维柱状图
    ax.bar3d(x1_combined, noc_bandwidth_paramters, np.zeros_like(latencies), dx, dy, dz, color=colors)
    ax.zaxis.set_tick_params(labelsize=font_size)
    # # 添加二维坐标标签(x1, x2)作为y轴上的标记
    # for i in range(len(y)):
    #     ax.text(x1[i], x2_combined[i], y[i] + 0.5, f"({x2[i]},{x3[i]})", color='black', ha='center')

    # 设置坐标轴标签
    ax.set_xlabel('Local Memory Bandwidth', fontsize=10)
    ax.set_ylabel('NoC Bandwidth', fontsize=10)
    ax.set_zlabel('Latency', fontsize=10)
    ax.set_xticks([i * 13 + 3 for i in range(len(lmb))])
    ax.set_xticklabels(lmb, fontsize=font_size)
    ax.set_yticks(np.log2(max(nb)) - np.log2(nb) + 1.5)
    ax.set_yticklabels(nb, fontsize=font_size)

    plt.title("Distributed Many Core")

    # 创建图例
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=color, label=f'{height}')
        for color, height in zip(colors[::len(nb)], lml)
    ]
    alegend = ax.legend(handles=legend_patches, bbox_to_anchor=(-0.25, 0.8), loc='upper left', fontsize=8,
                frameon=True, fancybox=False,
                title='Local Memory\nLatency', title_fontsize=9)
    alegend.get_title().set_ha('center')

    plt.savefig(file_name + '.png', dpi=1000)


def draw_diff_config():
    local_memory_bandwidth_paramters = [1] # [4, 8, 16, 32, 64, 128]
    local_memory_latency_paramters = [30] # [i for i in range(20, 1, -2)]
    noc_bandwidth_paramters = [4, 8, 16, 32, 64]
    # paramters = []
    latency = np.zeros((len(local_memory_bandwidth_paramters), 
                        len(local_memory_latency_paramters),
                        len(noc_bandwidth_paramters),
                        4))
    for imb, local_memory_bandwidth in enumerate(local_memory_bandwidth_paramters):
        for iml, local_memory_latency in enumerate(local_memory_latency_paramters):
            for inb, noc_bandwidth in enumerate(noc_bandwidth_paramters):
                latency[imb][iml][inb][0] = simulate_decoder_1mb(
                    {'noc_bandwidth': noc_bandwidth,
                     'local_memory_latency': local_memory_latency,
                     'local_memory_bandwidth': local_memory_bandwidth},
                     ram=1)
                latency[imb][iml][inb][1] = simulate_decoder_2mb(
                    {'noc_bandwidth': noc_bandwidth,
                     'local_memory_latency': local_memory_latency,
                     'local_memory_bandwidth': local_memory_bandwidth},
                     ram=2)
                latency[imb][iml][inb][2] = simulate_decoder_2p5mb_3mb(
                    {'noc_bandwidth': noc_bandwidth,
                     'local_memory_latency': local_memory_latency,
                     'local_memory_bandwidth': local_memory_bandwidth},
                     ram=2.5)
                latency[imb][iml][inb][3] = simulate_decoder_2p5mb_3mb(
                    {'noc_bandwidth': noc_bandwidth,
                     'local_memory_latency': local_memory_latency,
                     'local_memory_bandwidth': local_memory_bandwidth},
                     ram=3)

    # # 为了绘制3D图，我们需要创建网格数据
    # x = np.arange(4)
    # y = np.arange(5)
    # X, Y = np.meshgrid(x, y)  # 创建网格
    # Z = latency[0, 0, :, :]  # 气温数据

    # # 绘制3D图
    # fig = plt.figure(figsize=(14, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis')  # 使用颜色映射表

    # # 设置坐标轴
    # ax.set_title("Temperature Variation Over Time")
    # ax.set_xlabel("SRAM size")
    # ax.set_ylabel("NoC Bandwidth")
    # ax.set_zlabel("Latency")
    # ax.set_xticks(np.arange(4))
    # ax.set_xticklabels(["1", "2", "2.5", "3"])
    # ax.set_yticks(np.arange(5))  # 设置日期标签为整数
    # ax.set_yticklabels(noc_bandwidth_paramters, rotation=45, ha="right")

    # # 保存图表到文件
    # plt.tight_layout()
    # plt.savefig("temperature_changes.png", format="png", dpi=300)  # 保存为png格式，分辨率300dpi

    # 绘制二维折线图
    plt.figure(figsize=(14, 8))
    for day_idx in range(5):
        x = np.arange(4)              # 小时数，0-23
        y = latency[0, 0, day_idx, :]      # 当前日期的气温数据
        plt.plot(x, y, label=noc_bandwidth_paramters[day_idx])

    # 设置图例和坐标轴
    # plt.title("Hourly Temperature Variation Over a Month")
    plt.xlabel("Different mampping for different SRAM size")
    plt.ylabel("latency")
    plt.xticks(ticks=np.arange(0, 4, 1))  # 每个小时一个刻度
    plt.legend(loc="upper right", fontsize="small", ncol=2)  # 图例展示日期

    plt.grid(True)

    # 保存图表到文件
    plt.tight_layout()
    plt.savefig("temperature_changes.png", format="png", dpi=300)  # 保存为png格式，分辨率300dpi

if __name__ == '__main__':
    # draw_one_ram_size(ram=1, file_name='temp/many_core_decoder_1mb')
    # draw_one_ram_size(ram=2, file_name='temp/many_core_decoder_2mb')
    # draw_one_ram_size(ram=2.5, file_name='temp/many_core_decoder_2p5mb')
    # draw_one_ram_size(ram=3, file_name='temp/many_core_decoder_3mb')
    draw_diff_config()
