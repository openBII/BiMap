from math import prod
from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import Coord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_cyclic_weight, create_tiled_elementwise, create_pointwise, create_data, create_mlp, create_static, create_add
import matplotlib.pyplot as plt
import toml
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from typing import Dict
import numpy as np


CHIP = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0


def simulate_tiled_mlp_on_chip(config_file: str,
                               input_shape: Shape,
                               weight_shape: Shape,
                               is_input_l1: bool = False,
                               is_weight_l1: bool = False,
                               input_split_vector: SplitVector = None,
                               weight_split_vector: SplitVector = None):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()

    if is_input_l1:
         _, input_l1 = create_input(
            task_graph,
            Shape(batch=input_shape.batch // input_split_vector.batch, 
                  token=input_shape.token // input_split_vector.token, 
                  nf=input_shape.nf),
            Precision.FLOAT_16)
    else:
        _, input_l2 = create_input(
            task_graph,
            Shape(batch=input_shape.batch // input_split_vector.batch, 
                token=input_shape.token // input_split_vector.token, 
                nf=input_shape.nf),
            Precision.FLOAT_16)
        input_l1 = create_data(task_graph, input_l2.shape, Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([input_l2, input_l1])

    if is_weight_l1:
        weight_l1 = create_static(
            task_graph, 
            Shape(nf=weight_shape.nf // weight_split_vector.nf, 
                  nr=weight_shape.nr), 
            Precision.FLOAT_16)
    else:
        weight_l2 = create_static(
            task_graph, 
            Shape(nf=weight_shape.nf // weight_split_vector.nf, 
                  nr=weight_shape.nr),
            Precision.FLOAT_16)
        weight_l1 = create_data(task_graph, weight_l2.shape, Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([weight_l2, weight_l1])

    output, task_dict = create_mlp(
        task_graph=task_graph,
        input=input_l1,
        shape=weight_l1.shape,
        precision=Precision.FLOAT_16,
        weight=weight_l1
    )
    output_l2 = create_data(task_graph, output.shape, Precision.FLOAT_16, True)
    task_graph.connect_tasks_in_sequence([output, output_l2])

    STDraw.draw_graph(task_graph, out_path='temp/shared_tiled_mlp_on_chip.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/" + config_file + ".toml")
    config = BoardConfig(config["PCB"], config["process_node"])

    # Create Hardware
    shared_memory_board = BoardFactory.create_matrix(config, 
                                                     BoardType.SHARED_MEMORY)

    # Create DSE snvironment
    env = STEnv(task_graph, shared_memory_board)

    # Equivalent Hardware Parameter
    shared_memory_board.container[Coord(CHIP)].communication_networks[0].update_bandwidth(
        config.chiplet.network["bandwidth"] // prod(config.chiplet.size))
    
    SHARED_MEMORY = (config.chiplet.size[0] + 1, config.chiplet.size[1] // 2)

    # Mapping
    # L2
    shared_memory = create_mlcoord(CHIP, SHARED_MEMORY)
    if not is_input_l1:
        env.put_in(shared_memory, input_l2.id)
    if not is_weight_l1:
        env.put_in(shared_memory, weight_l2.id)
    env.put_in(shared_memory, output_l2.id)

    # L1
    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(local_memory, input_l1.id)
    env.put_in(local_memory, weight_l1.id)
    env.put_in(local_memory, output.id)
    mlp = task_dict["compute"]
    env.put_in(tensor_unit, mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    return env.get_compute_time()


def simulate_tiled_mlp(config_file: str,
                       input_shape: Shape,
                       weight_shape: Shape,
                       input_offchip_split_vector: SplitVector = None,
                       weight_offchip_split_vector: SplitVector = None,
                       input_l1_split_vector: SplitVector = None,
                       weight_l1_split_vector: SplitVector = None):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()

    _, input_offchip = create_input(
        task_graph,
        Shape(batch=input_shape.batch // input_offchip_split_vector.batch, 
              token=input_shape.token // input_offchip_split_vector.token, 
              nf=input_shape.nf),
        Precision.FLOAT_16)
    input_l2 = create_data(task_graph, input_offchip.shape, 
                           Precision.FLOAT_16)
    input_l1 = create_data(
        task_graph, 
        Shape(batch=input_shape.batch // input_l1_split_vector.batch, 
              token=input_shape.token // input_l1_split_vector.token, 
              nf=input_shape.nf),
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([input_offchip, input_l2, input_l1])

    weight_offchip = create_static(
        task_graph, 
        Shape(nf=weight_shape.nf // weight_offchip_split_vector.nf, 
              nr=weight_shape.nr), 
        Precision.FLOAT_16)
    weight_l2 = create_data(task_graph, weight_offchip.shape, 
                            Precision.FLOAT_16)
    weight_l1 = create_data(
        task_graph, 
        Shape(nf=weight_shape.nf // weight_l1_split_vector.nf, 
              nr=weight_shape.nr), 
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([weight_offchip, weight_l2, weight_l1])

    output, task_dict = create_mlp(
        task_graph=task_graph,
        input=input_l1,
        shape=Shape(nf=weight_shape.nf // weight_l1_split_vector.nf, 
                    nr=weight_shape.nr),
        precision=Precision.FLOAT_16,
        weight=weight_l1
    )
    output_l2 = create_data(task_graph, input_offchip.shape, Precision.FLOAT_16)
    output_offchip = create_data(task_graph, input_offchip.shape, 
                                Precision.FLOAT_16, is_output=True)
    task_graph.connect_tasks_in_sequence([output, output_l2, output_offchip])

    STDraw.draw_graph(task_graph, out_path='temp/shared_tiled_mlp.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/" + config_file + ".toml")
    config = BoardConfig(config["PCB"], config["process_node"])

    # Create Hardware
    shared_memory_board = BoardFactory.create_matrix(config, 
                                                     BoardType.SHARED_MEMORY)

    # Create DSE snvironment
    env = STEnv(task_graph, shared_memory_board)

    # Equivalent Hardware Parameter
    shared_memory_board.container[Coord(CHIP)].communication_networks[0].update_bandwidth(
        config.chiplet.network["bandwidth"] // prod(config.chiplet.size))
    
    SHARED_MEMORY = (config.chiplet.size[0] + 1, config.chiplet.size[1] // 2)

    # Mapping
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, output_offchip.id)

    # L2
    shared_memory = create_mlcoord(CHIP, SHARED_MEMORY)
    env.put_in(shared_memory, input_l2.id)
    env.put_in(shared_memory, weight_l2.id)
    env.put_in(shared_memory, output_l2.id)

    # L1
    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(local_memory, input_l1.id)
    env.put_in(local_memory, weight_l1.id)
    env.put_in(local_memory, output.id)
    mlp = task_dict["compute"]
    env.put_in(tensor_unit, mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    return env.get_latency(
        loops=[LoopInfo(1, 10, 4)],
        extra_latencies=[simulate_tiled_mlp_on_chip(
            config_file,
            input_shape,
            weight_shape,
            is_input_l1=True,
            is_weight_l1=False,
            input_split_vector=input_l1_split_vector,
            weight_split_vector=weight_l1_split_vector)] * (8 * 8 * 64 - 4))
    # return env.get_latency()


if __name__ == "__main__":
    latency1 = simulate_tiled_mlp(config_file="2080Ti",
                                  input_shape=Shape(token=2048, nf=2048),
                                  weight_shape=Shape(nf=2048, nr=2048),
                                  input_offchip_split_vector=SplitVector(token=2),
                                  weight_offchip_split_vector=SplitVector(nf=2),
                                  input_l1_split_vector=SplitVector(token=512),
                                  weight_l1_split_vector=SplitVector(nf=512))
    latency2 = simulate_tiled_mlp_on_chip(config_file="2080Ti",
                                         input_shape=Shape(token=2048, nf=2048),
                                         weight_shape=Shape(nf=2048, nr=2048),
                                         is_input_l1=True,
                                         is_weight_l1=False,
                                         input_split_vector=SplitVector(token=512),
                                         weight_split_vector=SplitVector(nf=512))
    print(latency1)
    print(latency2)