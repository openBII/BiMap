from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import Coord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_decoder_weight, create_tiled_elementwise, create_pointwise, create_tiled_data_arrange_16_to_1_broadcast, create_add
import matplotlib.pyplot as plt
import toml
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from typing import Dict
import numpy as np
from src.simulator.task_rabbit.task_model.transformer import create_static, create_data


# Algorithm Parameters
BATCH = 1           # batch = 1
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/distributed_many_core_board.toml")
config = BoardConfig(config["PCB"])
SIZE_X, SIZE_Y = config.chiplet.size
COMPUTE = 0
CHIP = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0

# 2048-th Decoder + 2MLP
# Use 1 CHIP
# 1. Weight initaled on Chip
# 2. Weight read from DRAM

def simulate_decoder_mlp_1mb(hardware_paramter_dict: Dict[str, int], 
                             is_weight_onchip: bool,
                             split_vector: SplitVector,
                             config_file: str = "top/distributed_many_core_package1.toml"):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH, token=1, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query MLP
    query, query_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=True,
        output_offchip=False,
        is_input_on_chip=False,
        is_weight_on_chip=is_weight_onchip
    )
    query.shape = Shape(batch=1, token=1, nf=query.shape.nf * split_vector.nf)
    # re-arragne output data together
    query_middle_arrange = create_data(
        task_graph, query.shape, Precision.FLOAT_16)
    query_after_arrange = create_data(
        task_graph, query.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([query, query_middle_arrange, 
                                          query_after_arrange])
    # Q * K
    output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=query_after_arrange,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=False,
        output_offchip=False,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip
    )
    output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
    # re-arragne output data together
    dot_product_middle_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    dot_product_after_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([output, 
                                          dot_product_middle_arrange, 
                                          dot_product_after_arrange])
    # Softmax
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=dot_product_after_arrange,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=False,
        output_offchip=False,
        static_weight=False,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip
    )
    # re-arragne output data together
    output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
    attention_middle_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    attention_after_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([output, attention_middle_arrange, attention_after_arrange])
    # Add
    add_output, add_task_dict = create_add(
        task_graph=task_graph, 
        inputs=[query_task_dict["input_on_chip"], attention_after_arrange],
        shape=attention_after_arrange.shape, 
        precision=Precision.FLOAT_16)
    # LayerNorm
    layer_norm_output, layer_norm_task_dict = create_pointwise(
        task_graph=task_graph,
        input=add_output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CLayerNorm)
    # MLP 4096 * 16384
    output, ffn1_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=layer_norm_output,
        shape=Shape(nf=16384, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        output_offchip=False,
        static_weight=True,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip
    )
    # ReLU
    output, relu_dict = create_pointwise(
        task_graph=task_graph,
        input=output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CRELU,
    )
    # re-arragne output data together
    output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
    ffn_middle_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    ffn_after_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([output, ffn_middle_arrange, 
                                          ffn_after_arrange])
    # MLP 16384 * 4096
    output, ffn2_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=ffn_after_arrange,
        shape=Shape(nf=4096, nr=16384),
        precision=Precision.FLOAT_16,
        is_output=not is_weight_onchip,
        output_offchip=not is_weight_onchip,
        static_weight=True,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip,
    )
    if is_weight_onchip:
        output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
        output_arrange = create_data(
            task_graph, output.shape, Precision.FLOAT_16)
        output_other_chip = create_data(task_graph, output.shape, Precision.FLOAT_16,
                                        is_output=True)
        task_graph.connect_tasks_in_sequence([output, output_arrange, output_other_chip])

    STDraw.draw_graph(task_graph, out_path='temp/tiled_transformer.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    if is_weight_onchip:
        config = toml.load("top/distributed_many_core_package1.toml")
    else:
        config = toml.load("top/distributed_many_core_package_temporal.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    # config.chiplet.core.local_memory["capacity"] = 3 * 1024
    # config.chiplet.core.mac_array["fp16"]["parallelism"] = [16, 16]
    # config.chiplet.core.mac_array["fp16"]["latency"] = 1
    # config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.compute_domain.package.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.compute_domain.package.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.compute_domain.package.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.compute_domain.package.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
        else:
            raise NotImplementedError
    
    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.PACKAGE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), SRAM_BUFFER)
    if not is_weight_onchip:
        arrange_memory = create_mlcoord(COMPUTE, (7, 0), (0, 0), (15, 7), SRAM_BUFFER)
    else:
        arrange_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (15, 7), SRAM_BUFFER)
    other_chip_memory = create_mlcoord(COMPUTE, (1, 0), (0, 0), (15, 7), SRAM_BUFFER)
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), TENSOR_UNIT)
    vector_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), VECTOR_UNIT)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, query_task_dict["input_on_chip"].id) 
    env.put_in(local_memory, query_task_dict["weight_on_chip"].id) 
    if not is_weight_onchip:
        env.put_in(dram, query_task_dict["weight_offchip"].id)
    env.put_in(tensor_unit, query_task_dict["compute"].id)

    if not is_weight_onchip:
        env.sync(tensor_unit, sync_id1)

    # re-arrange Q
    env.put_in(local_memory, query_task_dict["output_on_chip"].id)
    env.put_in(arrange_memory, query_middle_arrange.id)
    env.put_in(local_memory, query_after_arrange.id)

    # Q * K
    if not is_weight_onchip:
        env.sync(dram, sync_id1)
        env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # re-arrange Q*K result
    env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
    env.put_in(arrange_memory, dot_product_middle_arrange.id)
    env.put_in(local_memory, dot_product_after_arrange.id)

    # softmax
    env.put_in(vector_unit, softmax_task_dict["compute"].id)
    if not is_weight_onchip:
        env.sync(vector_unit, sync_id2)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    if not is_weight_onchip:
        env.sync(dram, sync_id2)
        env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)

    # arrange
    env.put_in(arrange_memory, attention_middle_arrange.id)
    env.put_in(local_memory, attention_after_arrange.id)

    # Add
    env.put_in(vector_unit, add_task_dict["compute"].id)
    env.put_in(local_memory, add_task_dict["output"].id)

    # LayerNorm
    env.put_in(vector_unit, layer_norm_task_dict["compute"].id)
    if not is_weight_onchip:
        env.sync(vector_unit, sync_id3)
    env.put_in(local_memory, layer_norm_task_dict["output"].id)

    # FFN MLP1
    if not is_weight_onchip:
        env.sync(dram, sync_id3)
        env.put_in(dram, ffn1_task_dict["weight_offchip"].id)
    env.put_in(local_memory, ffn1_task_dict["weight_on_chip"].id) 
    env.put_in(tensor_unit, ffn1_task_dict["compute"].id)
    env.put_in(local_memory, ffn1_task_dict["output_on_chip"].id)

    # ReLU
    env.put_in(vector_unit, relu_dict["compute"].id)
    if not is_weight_onchip:
        env.sync(vector_unit, sync_id4)
    env.put_in(local_memory, relu_dict["output"].id)

    # arrange
    env.put_in(arrange_memory, ffn_middle_arrange.id)
    env.put_in(local_memory, ffn_after_arrange.id)

    # FFN MLP2
    if not is_weight_onchip:
        env.sync(dram, sync_id4)
        env.put_in(dram, ffn2_task_dict["weight_offchip"].id)
    env.put_in(local_memory, ffn2_task_dict["weight_on_chip"].id) 
    env.put_in(tensor_unit, ffn2_task_dict["compute"].id)
    env.put_in(local_memory, ffn2_task_dict["output_on_chip"].id)

    if is_weight_onchip:
        env.put_in(arrange_memory, output_arrange.id)
        env.put_in(other_chip_memory, output_other_chip.id)
    else:
        env.put_in(dram, ffn2_task_dict["output_offchip"].id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    if not is_weight_onchip:
        sync_dict = {}
        sync_dict[11] = 3
        sync_dict[19] = 15
        sync_dict[29] = 25
        sync_dict[37] = 31
        recorder = env.collect_time(sync=sync_dict)
        compute_time = env.get_compute_time(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ], sync=sync_dict)
        overall_latency = env.get_latency_pipeline(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ], sync=sync_dict)
        return overall_latency - compute_time
    else:
        recorder = env.collect_time()
        overall_latency = env.get_latency_pipeline(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ])
        routing = recorder[task_graph[4], task_graph[6]]
        return overall_latency, routing.end - routing.start + 2
    

def simulate_decoder_mlp_1mb_attention(hardware_paramter_dict: Dict[str, int], 
                                       is_weight_onchip: bool,
                                       split_vector: SplitVector):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH, token=1, nf=D_MODEL),
        Precision.FLOAT_16)
    # Query MLP
    query, query_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=input_offchip,
        shape=Shape(nf=4096, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=True,
        output_offchip=False,
        is_input_on_chip=False,
        is_weight_on_chip=is_weight_onchip
    )
    query.shape = Shape(batch=1, token=1, nf=query.shape.nf * split_vector.nf)
    # re-arragne output data together
    query_middle_arrange = create_data(
        task_graph, query.shape, Precision.FLOAT_16)
    query_after_arrange = create_data(
        task_graph, query.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([query, query_middle_arrange, 
                                          query_after_arrange])
    # Q * K
    output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=query_after_arrange,
        shape=Shape(nf=2048, nr=4096),
        precision=Precision.FLOAT_16,
        is_output=False,
        static_weight=False,
        output_offchip=False,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip
    )
    output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
    # re-arragne output data together
    dot_product_middle_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    dot_product_after_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([output, 
                                          dot_product_middle_arrange, 
                                          dot_product_after_arrange])
    # Softmax
    output, softmax_task_dict = create_pointwise(
        task_graph=task_graph,
        input=dot_product_after_arrange,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CSoftMax,
    )
    output, attention_task_dict = create_tiled_mlp_decoder_weight(
        split_vector=split_vector,
        task_graph=task_graph,
        input=output,
        shape=Shape(nf=4096, nr=2048),
        precision=Precision.FLOAT_16,
        is_output=False,
        output_offchip=False,
        static_weight=False,
        is_input_on_chip=True,
        is_weight_on_chip=is_weight_onchip
    )
    # re-arragne output data together
    output.shape = Shape(batch=1, token=1, nf=output.shape.nf * split_vector.nf)
    attention_middle_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    attention_after_arrange = create_data(
        task_graph, output.shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([output, attention_middle_arrange, attention_after_arrange])
    # Add
    add_output, add_task_dict = create_add(
        task_graph=task_graph, 
        inputs=[query_task_dict["input_on_chip"], attention_after_arrange],
        shape=attention_after_arrange.shape, 
        precision=Precision.FLOAT_16)
    # LayerNorm
    layer_norm_output, layer_norm_task_dict = create_pointwise(
        task_graph=task_graph,
        input=add_output,
        precision=Precision.FLOAT_16,
        type=TaskBlockType.CLayerNorm,
        is_output=True)

    STDraw.draw_graph(task_graph, out_path='temp/tiled_decoding_attention.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    if is_weight_onchip:
        config = toml.load("top/distributed_many_core_package1.toml")
    else:
        config = toml.load("top/distributed_many_core_package_temporal.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    # config.chiplet.core.local_memory["capacity"] = 3 * 1024
    # config.chiplet.core.mac_array["fp16"]["parallelism"] = [16, 16]
    # config.chiplet.core.mac_array["fp16"]["latency"] = 1
    # config.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.compute_domain.package.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.compute_domain.package.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.compute_domain.package.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]
        elif type == "mac_array":
            config.compute_domain.package.chiplet.core.mac_array["fp16"]["parallelism"] = hardware_paramter_dict[type]
        else:
            raise NotImplementedError
    
    # Create Hardware
    many_core_board = BoardFactory.create_matrix(config, 
                                                 BoardType.PACKAGE)

    # Create DSE snvironment
    env = STEnv(task_graph, many_core_board)

    # Equivalent Hardware Parameter
    many_core_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)
    local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), SRAM_BUFFER)
    if not is_weight_onchip:
        arrange_memory = create_mlcoord(COMPUTE, (7, 0), (0, 0), (15, 7), SRAM_BUFFER)
    else:
        arrange_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (15, 7), SRAM_BUFFER)
    other_chip_memory = create_mlcoord(COMPUTE, (1, 0), (0, 0), (15, 7), SRAM_BUFFER)
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), TENSOR_UNIT)
    vector_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), VECTOR_UNIT)
    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, query_task_dict["input_on_chip"].id) 
    env.put_in(local_memory, query_task_dict["weight_on_chip"].id) 
    if not is_weight_onchip:
        env.put_in(dram, query_task_dict["weight_offchip"].id)
    env.put_in(tensor_unit, query_task_dict["compute"].id)

    if not is_weight_onchip:
        env.sync(tensor_unit, sync_id1)

    # re-arrange Q
    env.put_in(local_memory, query_task_dict["output_on_chip"].id)
    env.put_in(arrange_memory, query_middle_arrange.id)
    env.put_in(local_memory, query_after_arrange.id)

    # Q * K
    if not is_weight_onchip:
        env.sync(dram, sync_id1)
        env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # re-arrange Q*K result
    env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
    env.put_in(arrange_memory, dot_product_middle_arrange.id)
    env.put_in(local_memory, dot_product_after_arrange.id)

    # softmax
    env.put_in(vector_unit, softmax_task_dict["compute"].id)
    if not is_weight_onchip:
        env.sync(vector_unit, sync_id2)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    if not is_weight_onchip:
        env.sync(dram, sync_id2)
        env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)

    # arrange
    env.put_in(arrange_memory, attention_middle_arrange.id)
    env.put_in(local_memory, attention_after_arrange.id)

    # Add
    env.put_in(vector_unit, add_task_dict["compute"].id)
    env.put_in(local_memory, add_task_dict["output"].id)

    # LayerNorm
    env.put_in(vector_unit, layer_norm_task_dict["compute"].id)
    if not is_weight_onchip:
        env.sync(vector_unit, sync_id3)
    env.put_in(local_memory, layer_norm_task_dict["output"].id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    if not is_weight_onchip:
        sync_dict = {}
        sync_dict[11] = 3
        sync_dict[19] = 15
        sync_dict[29] = 25
        sync_dict[37] = 31
        recorder = env.collect_time(sync=sync_dict)
        compute_time = env.get_compute_time(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ], sync=sync_dict)
        overall_latency = env.get_latency_pipeline(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ], sync=sync_dict)
        return overall_latency - compute_time
    else:
        recorder = env.collect_time()
        overall_latency = env.get_latency_pipeline(loops=[
            LoopInfo(query_task_dict["compute"].id, dot_product_task_dict["compute"].id, 2, True)  # Q, K, V Loop
            ])
        return overall_latency



def draw_one_ram_size(ram, file_name : str):

    mac_array1 = {16: [146, 146], 32: [146, 146], 64: [145, 145], 128: [142, 142], 256: [138, 138], 512: [128, 128]}
    mac_array2 = {8: [107, 107], 16: [105, 105], 32: [103, 103], 64: [98, 98], 128: [88, 88], 256: [64, 64]}
    mac_array2_5 = {4: [73, 73], 8: [72, 72], 16: [69, 69], 32: [64, 64], 64: [51, 51], 128: [32, 32]}
    mac_array3 = {4: [72, 72], 8: [71, 71], 16: [68, 68], 32: [60, 60], 64: [44, 44], 128: [16, 16]}

    if ram == 1:
        sim_func = simulate_decoder_1mb
        mac_array = mac_array1
    elif ram == 2:
        sim_func = simulate_decoder_2mb
        mac_array = mac_array2
    elif ram == 2.5:
        sim_func = simulate_decoder_2p5mb_3mb
        mac_array = mac_array2_5
    elif ram == 3:
        sim_func = simulate_decoder_2p5mb_3mb
        mac_array = mac_array3
    else:
        raise NotImplementedError


    noc_bandwidth_paramters = []
    local_memory_latency_paramters = []
    local_memory_bandwidth_paramters = []
    nb = [128, 64, 32, 16, 8, 4]
    lml = [i for i in range(80, 1, -10)]
    lmb = [i for i in mac_array.keys()]
    # paramters = []
    latencies = []
    # for local_memory_bandwidth in lmb:
    #     for local_memory_latency in lml:
    #         for noc_bandwidth in nb:
    #             noc_bandwidth_paramters.append(noc_bandwidth)
    #             local_memory_latency_paramters.append(local_memory_latency)
    #             local_memory_bandwidth_paramters.append(local_memory_bandwidth)
    #             # paramters.append((local_memory_bandwidth, local_memory_latency, noc_bandwidth))
    #             latencies.append(sim_func({'noc_bandwidth': noc_bandwidth,
    #                             'local_memory_latency': local_memory_latency,
    #                             'local_memory_bandwidth': local_memory_bandwidth,
    #                             'mac_array': mac_array[local_memory_bandwidth]}, ram=ram))
    # np.savez('{:s}.npz'.format(file_name), 
    #      noc_bandwidth_paramters=np.array(noc_bandwidth_paramters), 
    #      local_memory_latency_paramters=np.array(local_memory_latency_paramters),
    #      local_memory_bandwidth_paramters=np.array(local_memory_bandwidth_paramters),
    #      latencies=np.array(latencies))
    
    data = np.load('{:s}.npz'.format(file_name))
    noc_bandwidth_paramters = data["noc_bandwidth_paramters"]
    local_memory_latency_paramters = data["local_memory_latency_paramters"]
    local_memory_bandwidth_paramters = data["local_memory_bandwidth_paramters"]
    latencies = data["latencies"]

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

    plt.savefig(file_name + '.png', dpi=300)


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
    plt.savefig("temp/temperature_changes.png", format="png", dpi=300)  # 保存为png格式，分辨率300dpi

if __name__ == '__main__':
    from src.simulator.resource_simulator.evaluation_model.chiplet.exploration import single_module_multiple_chiplets
    # draw_one_ram_size(ram=1,   file_name='temp/sch_1113/many_core_decoder_1mb')
    # draw_one_ram_size(ram=2,   file_name='temp/sch_1113/many_core_decoder_2mb')
    # draw_one_ram_size(ram=2.5, file_name='temp/sch_1113/many_core_decoder_2p5mb')
    # draw_one_ram_size(ram=3,   file_name='temp/sch_1113/many_core_decoder_3mb')
    # draw_diff_config()
    print(simulate_decoder_mlp_1mb(
        {}, is_weight_onchip=True, 
        split_vector=SplitVector(batch=1, token=1, nf=128)))
    print(simulate_decoder_mlp_1mb(
        {}, is_weight_onchip=False, 
        split_vector=SplitVector(batch=1, token=1, nf=512)))
    print(simulate_decoder_mlp_1mb_attention(
        {}, is_weight_onchip=True, 
        split_vector=SplitVector(batch=1, token=1, nf=128)))
    
    num_chiplets = (1, 2, 3, 4, 6, 8, 12)
    draw_dict = {1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6, 12: 7, 24: 8}
    mcm_latencies = []
    si_latencies = []
    mcm_routings = []
    si_routings = []
    for num in (1, 2, 3, 4, 6, 8, 12):
        latency, routing = simulate_decoder_mlp_1mb(
            {}, is_weight_onchip=True, 
            split_vector=SplitVector(batch=1, token=1, nf=128))
        compute = (latency - routing * 3) * 8
        routing = (24 / num - 1) * (300 + 4096 / 8) + (num - 1) * (24 / num) * (150 + 4096 / 16)
        latency = compute + routing
        mcm_latencies.append(latency)
        mcm_routings.append(routing)
        routing = (24 / num - 1) * (300 + 4096 / 8) + (num - 1) * (24 / num) * (100 + 4096 / 32)
        latency = compute + routing
        si_latencies.append(latency)
        si_routings.append(routing)

    _, _, cost_increasing_mcms, cost_increasing_sis = single_module_multiple_chiplets(numbers=num_chiplets,
                                                                                      module_area=400)
    fig, ax1 = plt.subplots()

    width = 0.35
    bar1 = ax1.bar(np.array([1, 2, 3, 4, 5, 6, 7]), mcm_latencies, width, color='#304E7E', label='Latency')
    bar2 = ax1.bar(np.array([1, 2, 3, 4, 5, 6, 7]) + width, mcm_routings, width, color='#4D1F03', label="Inter-Chip Communication")
    ax1.set_xlabel('Number of Chiplets', fontsize=14)
    ax1.set_ylabel('Latency', fontsize=14)
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax2 = ax1.twinx()
    line1, = ax1.plot(np.array([1, 2, 3, 4, 5, 6, 7]), 1600000000000 / np.array(mcm_latencies) / (cost_increasing_mcms[0:1] + cost_increasing_mcms[2:]), marker='o', linestyle='-', color='#81D0D6', label='Performance-Cost Ratio')
    line2, = ax2.plot([1, 2, 3, 4, 5, 6, 7], cost_increasing_mcms[0:1] + cost_increasing_mcms[2:], marker='o', linestyle='-', color='#FDBC63', label='Cost')
    ax2.set_ylabel('Cost', fontsize=14)

    # 添加标题和标签
    # plt.title('Multi-Chip-Modulo Integration', fontsize=18)
    # ax1.legend(loc="upper left")
    # ax2.legend()
    lns = [bar1, bar2, line1, line2]
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left', ncol=2, prop={'size': 9.5})
    ax1.set_ylim(0, 85000)
    ax2.set_ylim(0, 3000)
    plt.tight_layout()
    plt.savefig("test/exp/decode_chiplet_mcm" + '.png', dpi=300)


    plt.clf()
    fig, ax1 = plt.subplots()

    width = 0.35
    bar1 = ax1.bar(np.array([1, 2, 3, 4, 5, 6, 7]), si_latencies, width, color='#304E7E', label='Latency')
    bar2 = ax1.bar(np.array([1, 2, 3, 4, 5, 6, 7]) + width, si_routings, width, color='#4D1F03', label="Inter-Chip Communication")
    ax1.set_xlabel('Number of Chiplets', fontsize=14)
    ax1.set_ylabel('Latency', fontsize=14)
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax2 = ax1.twinx()
    line1, = ax1.plot(np.array([1, 2, 3, 4, 5, 6, 7]), 1600000000000 / np.array(si_latencies) / (cost_increasing_sis[0:1] + cost_increasing_sis[2:]), marker='o', linestyle='-', color='#81D0D6', label='Performance-Cost Ratio')
    line2, = ax2.plot([1, 2, 3, 4, 5, 6, 7], cost_increasing_sis[0:1] + cost_increasing_sis[2:], marker='o', linestyle='-', color='#FDBC63', label='Cost')
    ax2.set_ylabel('Cost', fontsize=14)

    # 添加标题和标签
    # plt.title('2.5D (CoWos) Integration', fontsize=18)
    # ax1.legend(loc="upper left")
    # ax2.legend()
    lns = [bar1, bar2, line1, line2]
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left', ncol=2, prop={'size': 9.5})
    ax1.set_ylim(0, 85000)
    ax2.set_ylim(0, 30000)
    plt.tight_layout()
    plt.savefig("test/exp/decode_chiplet_si" + '.png', dpi=300)

    
    mac_array1 = {256: [83, 83], 512: [82, 82], 1024: [80, 80], 2048: [74, 74], 4096: [64, 64], 8192: [32, 32]}
    latencies = []
    noc_bandwidth_parameters = []
    local_memory_latency_parameters = []
    local_memory_bandwidth_parameters = []
    chiplet_parameters = []
    local_memory_bandwidth1 = (8192, 4096, 2048, 1024, 512, 256)
    for local_memory_bandwidth in local_memory_bandwidth1:
        latency, routing = simulate_decoder_mlp_1mb(
            {'local_memory_bandwidth': local_memory_bandwidth,
             'mac_array': mac_array1[local_memory_bandwidth]}, 
            is_weight_onchip=True, 
            split_vector=SplitVector(batch=1, token=1, nf=128))
        compute = (latency - routing * 3) * 8
        for num in (1, 2, 3, 4, 6, 8, 12, 24):
            local_memory_bandwidth_parameters.append(local_memory_bandwidth)
            chiplet_parameters.append(num)
            latency = compute + (24 / num - 1) * (300 + 4096 / 8) + (num - 1) * (24 / num) * (150 + 4096 / 16)
            latencies.append(latency)
            print(latency)

    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    chiplet_positions = [draw_dict[p] for p in chiplet_parameters]

    # 绘制三维散点图
    scatter = ax.scatter(chiplet_positions, np.log2(local_memory_bandwidth_parameters), latencies, c=chiplet_positions, cmap='viridis', alpha=1)
    plt.xticks(ticks=range(len(num_chiplets)), labels=num_chiplets)
    plt.yticks(ticks=(13, 12, 11, 10, 9, 8), labels=(8192, 4096, 2048, 1024, 512, 256))

    # ax.plot(chiplet_positions, np.log2(local_memory_bandwidth_parameters), latencies, color='r', label='Line along x')

    # 添加坐标轴标签
    ax.set_xlabel('Number of Chiplets')
    ax.set_ylabel('Local Memory Bandwidth (B/cycle)')
    ax.set_zlabel('Latency (Cycles)')

    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label('Number of Chiplets')
    plt.gca().invert_yaxis()

    # 显示图形
    ax.view_init(elev=30, azim=-45)
    plt.savefig("test/exp/decode_chiplet_local_bandwidth" + '.png', dpi=300)

    noc_bandwidth_parameters = []
    local_memory_latency_parameters = []
    local_memory_bandwidth_parameters = []
    latencies = []
    for local_memory_bandwidth in local_memory_bandwidth1:
        for local_memory_latency in range(10, 1, -2):
            for noc_bandwidth in (4, 8, 16, 32, 64):
                noc_bandwidth_parameters.append(noc_bandwidth)
                local_memory_latency_parameters.append(local_memory_latency)
                local_memory_bandwidth_parameters.append(local_memory_bandwidth)
                # paramters.append((local_memory_bandwidth, local_memory_latency, noc_bandwidth))
                latencies.append(simulate_decoder_mlp_1mb_attention(
                    {'noc_bandwidth': noc_bandwidth,
                     'local_memory_latency': local_memory_latency,
                     'local_memory_bandwidth': local_memory_bandwidth,
                     'mac_array': mac_array1[local_memory_bandwidth]},
                     is_weight_onchip=True,
                     split_vector=SplitVector(batch=1, token=1, nf=128)))
                
    np.savez('{:s}.npz'.format("test/exp/decode_attention_1mb_64*64"), 
            noc_bandwidth_parameters=np.array(noc_bandwidth_parameters), 
            local_memory_latency_parameters=np.array(local_memory_latency_parameters),
            local_memory_bandwidth_parameters=np.array(local_memory_bandwidth_parameters),
            latencies=np.array(latencies))
    
    lmb = [128, 256, 512, 1024, 2048, 4096]
    lml = [i for i in range(10, 1, -2)]
    nb = [64, 32, 16, 8, 4]
    
    # data = np.load('{:s}.npz'.format("decode_attention_1mb_64*64"))
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
    local_memory_bandwidth_paramters = np.log2(local_memory_bandwidth_parameters)
    noc_bandwidth_paramters = np.log2(max(noc_bandwidth_parameters)) - np.log2(noc_bandwidth_parameters) + 1

    # x1: local bandwidth x2: latency x3: noc bandwidth
    # 定义偏移量，用于将x2编码成不同的层
    # offset = 0.5
    # x1_combined = [local_memory_bandwidth_paramters[i] * 10 + (20 - local_memory_latency_paramters[i]) * offset for i in range(len(local_memory_latency_paramters))]  # 将x1和x2组合
    x1_combined = []
    offset = (12.0) / len(lml)
    for i, local_memory_bandwidth in enumerate(lmb):
        for j, local_memory_latency in enumerate(lml):
            for noc_bandwidth in nb:
                x1_combined.append(i * 20 + j * offset)

    # 排序，改变绘制柱子的顺序，避免重叠
    sorted_indices = np.argsort(x1_combined)
    x1_combined = np.array(x1_combined)[sorted_indices]
    noc_bandwidth_paramters = np.array(noc_bandwidth_paramters)[sorted_indices]
    dz = np.array(latencies)[sorted_indices]
    # 根据x1的值设置颜色，颜色随着数值变大而变化
    colors = plt.cm.viridis((np.array(local_memory_latency_parameters) - min(local_memory_latency_parameters)) / (max(local_memory_latency_parameters) - min(local_memory_latency_parameters)))

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
    ax.set_xticks([i * 20 + 3 for i in range(len(lmb))])
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

    plt.savefig("test/exp/decode_attention_1mb_64*64" + '.png', dpi=300)