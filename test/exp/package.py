import toml
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_env import LoopInfo, STEnv
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.transformer import create_input, create_pointwise, create_tiled_data_arrange_16_to_1_broadcast, create_tiled_mlp_decoder_weight
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from typing import Dict


BATCH = 8
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/distributed_many_core_chiplet2.toml")
config = BoardConfig(config["PCB"])
CHIPLET_SIZE_X, CHIPLET_SIZE_Y = config.package.size
CORE_SIZE_X, CORE_SIZE_Y = config.package.chiplet.size
COMPUTE = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0


def simulate_chiplet_decoder_1mb(chip_num: int, hardware_paramter_dict: Dict[str, int], ram):
    assert ram == 1
    assert(chip_num in [1, 2, 4, 8, 16])
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
    STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/distributed_many_core_package{:d}.toml".format(chip_num))
    config = BoardConfig(config["PCB"], config["process_node"])
    chiplet_board = BoardFactory.create_matrix(config, 
                                               BoardType.PACKAGE)
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.package.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.package.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.package.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    config.compute_domain.package.chiplet.core.mac_array["fp16"]["parallelism"] = [128, 128]
    config.compute_domain.package.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
    config.compute_domain.package.chiplet.core.local_memory["capacity"] = 1 * 1024

    # Create DSE snvironment
    env = STEnv(task_graph, chiplet_board)

    # Equivalent Hardware Parameter
    chiplet_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // 128)     # 为了跟SoC的对比，这里还是除了128

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), SRAM_BUFFER)
    if chip_num == 1:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), (0, 0), SRAM_BUFFER)
    elif chip_num == 2:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), (0, 0), SRAM_BUFFER)
    elif chip_num == 4:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), (0, 0), SRAM_BUFFER)
    elif chip_num == 8:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), (0, 0), SRAM_BUFFER)
    elif chip_num == 16:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (0, 0), (15, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (0, 0), (15, 0), (0, 0), SRAM_BUFFER)

    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()
    sync_id5 = env.get_sync_id()
    sync_id6 = env.get_sync_id()

    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), TENSOR_UNIT)
    vector_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), (0, 0), VECTOR_UNIT)

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    env.put_in(tensor_unit, mlp_task_dict["compute"].id)

    # sync 1
    env.sync(tensor_unit, sync_id1)
    env.sync(local_memory, sync_id1)

    # Q,K,V cyclic for 3 times
    env.put_in(local_memory, mlp_task_dict["cyclic_weight_on_chip"].id)
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
    env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

    # sync 3
    env.sync(tensor_unit, sync_id3)
    env.sync(dram, sync_id3)

    # Q * K cyclic
    env.put_in(dram, dot_product_task_dict["cyclic_weight_offchip"].id)
    env.put_in(local_memory, dot_product_task_dict["cyclic_weight_on_chip"].id)
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
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 5
    env.sync(vector_unit, sync_id5)
    env.sync(dram, sync_id5)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(tensor_unit, attention_task_dict["compute"].id)

    # sync 6
    env.sync(tensor_unit, sync_id6)
    env.sync(dram, sync_id6)

    # result * V
    env.put_in(dram, attention_task_dict["cyclic_weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["cyclic_weight_on_chip"].id)
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


def simulate_chiplet_decoder_2mb(chip_num: int, hardware_paramter_dict: Dict[str, int], ram):
    assert ram == 2
    assert(chip_num in [2, 4, 8, 16])
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
    config = toml.load("top/distributed_many_core_chiplet{:d}.toml".format(chip_num))
    config = BoardConfig(config["PCB"], config["process_node"])
    chiplet_board = BoardFactory.create_matrix(config, 
                                            BoardType.CHIPLET)
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.package.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.package.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.package.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    config.package.chiplet.core.mac_array["fp16"]["parallelism"] = [64, 64]
    config.package.chiplet.core.vector_unit["fp16"]["parallelism"] = 512
    config.package.chiplet.core.local_memory["capacity"] = 2 * 1024

    # Create DSE snvironment
    env = STEnv(task_graph, chiplet_board)

    # Equivalent Hardware Parameter
    chiplet_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // 128)       # 为了跟SoC的延迟匹配，这里还是除了128

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    if chip_num == 2:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), SRAM_BUFFER)
    elif chip_num == 4:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), SRAM_BUFFER)
    elif chip_num == 8:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), SRAM_BUFFER)
    elif chip_num == 16:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), SRAM_BUFFER)

    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()
    sync_id4 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
    env.put_in(tensor_unit, mlp_task_dict["compute"].id)

    # sync 1
    env.sync(tensor_unit, sync_id1)
    env.sync(local_memory, sync_id1)

    # Q,K,V cyclic
    env.put_in(local_memory, mlp_task_dict["cyclic_weight_on_chip"].id)
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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
    vector_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), VECTOR_UNIT)
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 4
    env.sync(vector_unit, sync_id4)
    env.sync(dram, sync_id4)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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


# 2.5MB == 3MB
def simulate_chiplet_decoder_2p5mb(chip_num: int, hardware_paramter_dict: Dict[str, int], ram):  # 2, 4, 8, 16
    assert(chip_num in [2, 4, 8, 16])
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

    # Create Hardware
    config = toml.load("top/distributed_many_core_chiplet{:d}.toml".format(chip_num))
    config = BoardConfig(config["PCB"], config["process_node"])
    chiplet_board = BoardFactory.create_matrix(config, 
                                            BoardType.CHIPLET)
    for type in hardware_paramter_dict:
        if type == "noc_bandwidth":
            config.package.chiplet.network["bandwidth"] = hardware_paramter_dict[type]
        elif type == "local_memory_latency":
            config.package.chiplet.core.local_memory["latency"] = hardware_paramter_dict[type]
        elif type == "local_memory_bandwidth":
            config.package.chiplet.core.local_memory["bandwidth"] = hardware_paramter_dict[type]

    if ram == 2.5:
        config.package.chiplet.core.mac_array["fp16"]["parallelism"] = [32, 32]
        config.package.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
        config.package.chiplet.core.local_memory["capacity"] = 2.5 * 1024
    elif ram == 3:
        config.package.chiplet.core.mac_array["fp16"]["parallelism"] = [16, 16]
        config.package.chiplet.core.vector_unit["fp16"]["parallelism"] = 128
        config.package.chiplet.core.local_memory["capacity"] = 3 * 1024
    else:
        raise NotImplementedError

    # Create DSE snvironment
    env = STEnv(task_graph, chiplet_board)

    # Equivalent Hardware Parameter
    chiplet_board.communication_networks[0].update_bandwidth(
        config.network["bandwidth"] // 128)     # 为了跟SoC的延迟匹配，这里还是除了128

    # Mapping
    # Query, Key, Value
    # DRAM
    dram = create_mlcoord(DRAM)

    local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    arrange_q_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    arrange_qk_local_memory = create_mlcoord(COMPUTE, (0, 0), (0, 0), SRAM_BUFFER)
    if chip_num == 2:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (1, 0), (7, 0), SRAM_BUFFER)
    elif chip_num == 4:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (3, 0), (3, 0), SRAM_BUFFER)
    elif chip_num == 8:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (7, 0), (1, 0), SRAM_BUFFER)
    elif chip_num == 16:
        arrange_q_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), SRAM_BUFFER)
        arrange_qk_arrange_memory = create_mlcoord(COMPUTE, (15, 0), (0, 0), SRAM_BUFFER)

    sync_id1 = env.get_sync_id()
    sync_id2 = env.get_sync_id()
    sync_id3 = env.get_sync_id()

    # Q,K,V
    env.put_in(dram, input_offchip.id)
    env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
    env.put_in(dram, mlp_task_dict["weight_offchip"].id)
    env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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
    vector_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), VECTOR_UNIT)
    env.put_in(vector_unit, softmax_task_dict["compute"].id)

    # sync 3
    env.sync(vector_unit, sync_id3)
    env.sync(dram, sync_id3)

    # result * V
    env.put_in(local_memory, softmax_task_dict["output"].id)
    env.put_in(dram, attention_task_dict["weight_offchip"].id)
    env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
    env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
    tensor_unit = create_mlcoord(COMPUTE, (0, 0), (0, 0), TENSOR_UNIT)
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


if __name__ == '__main__':
    from test.exp.distributed_many_core_decoder import simulate_decoder_2p5mb_3mb
    # print("------3------")
    # print(simulate_chiplet_decoder_2p5mb(2, {},  ram=3))
    # print(simulate_chiplet_decoder_2p5mb(4, {},  ram=3))
    # print(simulate_chiplet_decoder_2p5mb(8, {},  ram=3))
    # print(simulate_chiplet_decoder_2p5mb(16, {}, ram=3))
    # print("------2.5----")
    # print(simulate_decoder_2p5mb_3mb({}, ram=2.5))
    # print(simulate_chiplet_decoder_2p5mb(2, {},  ram=2.5))
    # print(simulate_chiplet_decoder_2p5mb(4, {},  ram=2.5))
    # print(simulate_chiplet_decoder_2p5mb(8, {},  ram=2.5))
    # print(simulate_chiplet_decoder_2p5mb(16, {}, ram=2.5))
    # print("------2----")
    # print(simulate_chiplet_decoder_2mb(2, {},  ram=2))
    # print(simulate_chiplet_decoder_2mb(4, {},  ram=2))
    # print(simulate_chiplet_decoder_2mb(8, {},  ram=2))
    # print(simulate_chiplet_decoder_2mb(16, {}, ram=2))
    print("------1----")
    print(simulate_chiplet_decoder_1mb(1, {},  ram=1))
    print(simulate_chiplet_decoder_1mb(2, {},  ram=1))
    print(simulate_chiplet_decoder_1mb(4, {},  ram=1))
    print(simulate_chiplet_decoder_1mb(8, {},  ram=1))
    print(simulate_chiplet_decoder_1mb(16, {}, ram=1))