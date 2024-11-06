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


# Algorithm Parameters
BATCH = 8
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/shared_memory_board.toml")
config = BoardConfig(config["PCB"])
SIZE_X, SIZE_Y = config.chiplet.size
CHIP = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0
REGISTER_FILE = 4
SHARED_MEMORY = (SIZE_X + 1, SIZE_Y // 2)


def update_hardware(config: BoardConfig, hardware_parameter_dict={}):
    # Update Hardware Configuration
    for type in hardware_parameter_dict:
        if type == "shared_memory_bandwidth":
            config.chiplet.network["bandwidth"] = hardware_parameter_dict[type]
        elif type == "shared_memory_latency":
            config.chiplet.network["latency"] = hardware_parameter_dict[type]
        elif type == "local_memory_latency":
            config.chiplet.core.local_memory["latency"] = hardware_parameter_dict[type]
        elif type == "local_memory_bandwidth":
            config.chiplet.core.local_memory["bandwidth"] = hardware_parameter_dict[type]


def simulate_tiled_mlp(hardware_parameter_dict={},
                       is_weight_on_chip: bool = True,
                       is_input_on_chip: bool = False,
                       is_output_offchip: bool = True):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    if not is_input_on_chip:
        _, input_offchip = create_input(
            task_graph,
            Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL),
            Precision.FLOAT_16)
        input_l2 = create_data(task_graph, input_offchip.shape, 
                               Precision.FLOAT_16)
        input_l1 = create_data(
            task_graph, 
            Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL),
            Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([input_offchip, input_l2, input_l1])
    else:
        _, input_l2 = create_input(
            task_graph,
            Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL),
            Precision.FLOAT_16)
        input_l1 = create_data(
            task_graph, 
            Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL),
            Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([input_l2, input_l1])
    if not is_weight_on_chip:
        weight_offchip = create_static(task_graph, 
                                       Shape(nf=D_MODEL, nr=D_MODEL), 
                                       Precision.FLOAT_16)
        weight_l2 = create_data(task_graph, weight_offchip.shape, 
                                Precision.FLOAT_16)
        weight_l1 = create_data(
            task_graph, Shape(nf=D_MODEL // 128, nr=D_MODEL), Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([weight_offchip, weight_l2, weight_l1])
    else:
        weight_l1 = create_static(
            task_graph, Shape(nf=D_MODEL // 128, nr=D_MODEL), 
            Precision.FLOAT_16)
    if is_output_offchip:
        output, task_dict = create_mlp(
            task_graph=task_graph,
            input=input_l1,
            shape=Shape(nf=D_MODEL, nr=D_MODEL),
            precision=Precision.FLOAT_16,
            weight=weight_l1
        )
        output_l2 = create_data(task_graph, input_offchip.shape, Precision.FLOAT_16)
        output_offchip = create_data(task_graph, input_offchip.shape, 
                                    Precision.FLOAT_16, is_output=True)
        task_graph.connect_tasks_in_sequence([output, output_l2, output_offchip])
    else:
        output, task_dict = create_mlp(
            task_graph=task_graph,
            input=input_l1,
            shape=Shape(nf=D_MODEL // 128, nr=D_MODEL),
            precision=Precision.FLOAT_16,
            weight=weight_l1
        )
        output_l2 = create_data(task_graph, output.shape, Precision.FLOAT_16,
                                is_output=True)
        task_graph.connect_tasks_in_sequence([output, output_l2])
    STDraw.draw_graph(task_graph, out_path='temp/shared_tiled_mlp.task.html',
                        width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/shared_memory_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    update_hardware(config, hardware_parameter_dict)

    # Create Hardware
    shared_memory_board = BoardFactory.create_matrix(config, 
                                                     BoardType.SHARED_MEMORY)

    # Create DSE snvironment
    env = STEnv(task_graph, shared_memory_board)

    # Equivalent Hardware Parameter
    shared_memory_board.communication_networks[0].update_bandwidth(
        config.chiplet.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # DRAM
    dram = create_mlcoord(DRAM)
    if not is_input_on_chip:
        env.put_in(dram, input_offchip.id)
    if not is_weight_on_chip:
        env.put_in(dram, weight_offchip.id)
    if is_output_offchip:
        env.put_in(dram, output_offchip.id)

    # L2
    shared_memory = create_mlcoord(CHIP, SHARED_MEMORY)
    env.put_in(shared_memory, input_l2.id)
    if not is_weight_on_chip:
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
    return env.get_latency()


def simulate_tiled_dot_product(hardware_parameter_dict={},
                               transpose: bool = True,
                               is_weight_l1: bool = True):
    # Construct Task Graph
    if transpose:
        input_shape = Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL)
        weight_shape = Shape(batch=BATCH // 8, token=SEQ_LEN // 128, nf=D_MODEL)
    else:
        input_shape = Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=SEQ_LEN)
        weight_shape = Shape(batch=BATCH // 8, token=SEQ_LEN, nf=D_MODEL // 128)
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_l2 = create_input(task_graph, input_shape, Precision.FLOAT_16)
    input_l1 = create_data(task_graph, input_shape, Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([input_l2, input_l1])
    if not is_weight_l1:
        weight_l2 = create_static(task_graph, weight_shape, Precision.FLOAT_16)
        weight_l1 = create_data(task_graph, weight_shape, Precision.FLOAT_16)
        task_graph.connect_tasks_in_sequence([weight_l2, weight_l1])
    else:
        weight_l1 = create_static(task_graph, weight_shape, Precision.FLOAT_16)
    nf = weight_shape.token if transpose else weight_shape.nf
    nr = D_MODEL if transpose else SEQ_LEN
    output_l1, task_dict = create_mlp(
        task_graph=task_graph,
        input=input_l1,
        shape=Shape(nf=nf, nr=nr),
        precision=Precision.FLOAT_16,
        weight=weight_l1
    )
    output_l2 = create_data(task_graph, output_l1.shape, Precision.FLOAT_16,
                            is_output=True)
    task_graph.connect_tasks_in_sequence([output_l1, output_l2])

    STDraw.draw_graph(task_graph, out_path='temp/shared_tiled_dot_product.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/shared_memory_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    update_hardware(config, hardware_parameter_dict)

    # Create Hardware
    shared_memory_board = BoardFactory.create_matrix(config, 
                                                     BoardType.SHARED_MEMORY)

    # Create DSE snvironment
    env = STEnv(task_graph, shared_memory_board)

    # Equivalent Hardware Parameter
    shared_memory_board.communication_networks[0].update_bandwidth(
        config.chiplet.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # L2
    shared_memory = create_mlcoord(CHIP, SHARED_MEMORY)
    env.put_in(shared_memory, input_l2.id)
    if not is_weight_l1:
        env.put_in(shared_memory, weight_l2.id)
    env.put_in(shared_memory, output_l2.id)

    # L1
    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    env.put_in(local_memory, input_l1.id)
    env.put_in(local_memory, weight_l1.id)
    env.put_in(local_memory, output_l1.id)
    mlp = task_dict["compute"]
    env.put_in(tensor_unit, mlp.id)

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    return env.get_latency()


def simulate_tiled_attention(hardware_parameter_dict={}):
    # Construct Task Graph
    IDGenerator.set_base_task_id(0)
    task_graph = TaskGraph()
    _, input_offchip = create_input(
        task_graph,
        Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL),
        Precision.FLOAT_16)
    input_l2 = create_data(task_graph, input_offchip.shape, Precision.FLOAT_16)
    input_l1 = create_data(
        task_graph, Shape(batch=BATCH // 8, token=SEQ_LEN // 64, nf=D_MODEL),
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([input_offchip, input_l2, input_l1])
    weight_offchip = create_static(
        task_graph, Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL), 
        Precision.FLOAT_16)
    weight_l2 = create_data(task_graph, weight_offchip.shape, 
                            Precision.FLOAT_16)
    weight_l1 = create_data(
        task_graph, Shape(batch=BATCH // 8, token=SEQ_LEN // 128, nf=D_MODEL), 
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([weight_offchip, weight_l2, weight_l1])
    mlp_output_l1, task_dict = create_mlp(
        task_graph=task_graph,
        input=input_l1,
        shape=Shape(nf=SEQ_LEN // 128, nr=D_MODEL),
        precision=Precision.FLOAT_16,
        weight=weight_l1
    )
    add_input_l2 = create_data(
        task_graph, Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=SEQ_LEN), 
        Precision.FLOAT_16)
    add_input_l1 = create_data(
        task_graph, Shape(batch=BATCH // 8, token=SEQ_LEN // 32, nf=SEQ_LEN), 
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([mlp_output_l1, add_input_l2, add_input_l1])
    mask = create_static(task_graph, add_input_l1.shape, Precision.FLOAT_16)
    add_output, add_task_dict = create_add(task_graph, [add_input_l1, mask], 
                                       add_input_l1.shape, 
                                       Precision.FLOAT_16)
    softmax_output, softmax_task_dict = create_pointwise(
        task_graph, add_output, Precision.FLOAT_16, TaskBlockType.CSoftMax)
    # output_l2 = create_data(task_graph, output.shape, Precision.FLOAT_16)
    # task_graph.connect_tasks_in_sequence([output, output_l2])
    value_offchip = create_static(
        task_graph, Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL),
        Precision.FLOAT_16)
    value_l2 = create_data(task_graph, value_offchip.shape, Precision.FLOAT_16)
    value_l1 = create_data(
        task_graph, Shape(batch=BATCH // 8, token=SEQ_LEN, nf=D_MODEL // 128),
        Precision.FLOAT_16)
    task_graph.connect_tasks_in_sequence([value_offchip, value_l2, value_l1])
    attention_output_l1, attention_task_dict = create_mlp(
        task_graph=task_graph,
        input=softmax_output,
        shape=Shape(nf=D_MODEL // 128, nr=SEQ_LEN),
        precision=Precision.FLOAT_16,
        weight=value_l1
    )
    attention_output_l2 = create_data(
        task_graph, Shape(batch=BATCH // 2, token=SEQ_LEN, nf=D_MODEL), 
        Precision.FLOAT_16)
    attention_output_offchip = create_data(
        task_graph, attention_output_l2.shape, Precision.FLOAT_16,
        is_output=True)
    task_graph.connect_tasks_in_sequence(
        [attention_output_l1, attention_output_l2, attention_output_offchip])

    STDraw.draw_graph(task_graph, out_path='temp/shared_tiled_attention_dram.task.html',
                      width='1920px', height='1080px')

    # Update Hardware Configuration
    config = toml.load("top/shared_memory_board.toml")
    config = BoardConfig(config["PCB"], config["process_node"])
    update_hardware(config, hardware_parameter_dict)

    # Create Hardware
    shared_memory_board = BoardFactory.create_matrix(config, 
                                                     BoardType.SHARED_MEMORY)
    
    area = shared_memory_board.container[Coord(CHIP)].area
    shared_memory_area = shared_memory_board.container[Coord(CHIP)].container[Coord(SHARED_MEMORY)].evaluator.eval_area()
    register_file_area = shared_memory_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(REGISTER_FILE)].evaluator.eval_area()
    local_memory_area = shared_memory_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(SRAM_BUFFER)].evaluator.eval_area()
    tensor_unit_area = shared_memory_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(TENSOR_UNIT)].evaluator.eval_area()
    vector_unit_area = shared_memory_board.container[Coord(CHIP)].container[Coord((0, 0))].container[Coord(VECTOR_UNIT)].evaluator.eval_area()

    # Create DSE snvironment
    env = STEnv(task_graph, shared_memory_board)

    # Equivalent Hardware Parameter
    shared_memory_board.container[Coord(CHIP)].communication_networks[0].update_bandwidth(
        config.chiplet.network["bandwidth"] // (SIZE_X * SIZE_Y))

    # Mapping
    # DRAM
    dram = create_mlcoord(DRAM)
    env.put_in(dram, input_offchip.id)
    env.put_in(dram, weight_offchip.id)
    env.put_in(dram, value_offchip.id)
    env.put_in(dram, attention_output_offchip.id)

    # L2
    shared_memory = create_mlcoord(CHIP, SHARED_MEMORY)
    env.put_in(shared_memory, input_l2.id)
    env.put_in(shared_memory, weight_l2.id)
    env.put_in(shared_memory, value_l2.id)
    env.put_in(shared_memory, add_input_l2.id)
    env.put_in(shared_memory, attention_output_l2.id)

    # L1
    local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
    tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
    vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
    env.put_in(local_memory, input_l1.id)
    env.put_in(local_memory, weight_l1.id)
    env.put_in(local_memory, mlp_output_l1.id)
    env.put_in(local_memory, mask.id)
    env.put_in(local_memory, add_input_l1.id)
    env.put_in(local_memory, add_output.id)
    env.put_in(local_memory, softmax_output.id)
    env.put_in(local_memory, value_l1.id)
    env.put_in(local_memory, attention_output_l1.id)
    mlp1 = task_dict["compute"]
    add = add_task_dict["compute"]
    softmax = softmax_task_dict["compute"]
    mlp2 = attention_task_dict["compute"]
    env.put_in(tensor_unit, mlp1.id)
    env.put_in(vector_unit, add.id)
    env.put_in(vector_unit, softmax.id)
    env.put_in(tensor_unit, mlp2.id)
    # TODO: 差一个同步

    # Edge Mapping
    env.auto_edge_map()

    env.simulate()
    # env.show_overall_time()
    return env.get_latency(
        loops=[LoopInfo(start=1, end=22, num=1)],
        extra_latencies=[simulate_tiled_mlp(hardware_parameter_dict, is_input_on_chip=True, is_output_offchip=False) * (8 * 64 - 2),
                         simulate_tiled_mlp(hardware_parameter_dict, is_weight_on_chip=False),
                         simulate_tiled_mlp(hardware_parameter_dict),
                         simulate_tiled_dot_product(hardware_parameter_dict) * (64 * 8 - 8),
                         simulate_tiled_dot_product(hardware_parameter_dict, is_weight_l1=False) * (8 - 2),
                         simulate_tiled_dot_product(hardware_parameter_dict, transpose=False) * (32 * 8 - 8),
                         simulate_tiled_dot_product(hardware_parameter_dict, transpose=False, is_weight_l1=False) * (8 - 2)])


# latency1 = simulate_tiled_mlp(is_weight_on_chip=False)
# latency2 = simulate_tiled_mlp(is_input_on_chip=True, is_output_offchip=False)
# latency3 = simulate_tiled_mlp()
# latency4 = simulate_tiled_dot_product(is_weight_l1=False)
# latency5 = simulate_tiled_dot_product(is_weight_l1=False, transpose=False)
# latency6 = simulate_tiled_dot_product()
# latency7 = simulate_tiled_dot_product(transpose=False)

latency = simulate_tiled_attention()

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

shared_memory_bandwidth_parameters = []
shared_memory_latency_parameters = []
local_memory_bandwidth_parameters = []
local_memory_latency_parameters = []
latencies = []
for local_memory_bandwidth in (4, 8, 16, 32, 64, 128):
    # for local_memory_latency in range(80, 1, -10):
    for shared_memory_bandwidth in (256, 512, 1024, 2048, 4096, 8192):
        for shared_memory_latency in range(400, 50, -50):
            shared_memory_bandwidth_parameters.append(shared_memory_bandwidth)
            shared_memory_latency_parameters.append(shared_memory_latency)
            local_memory_bandwidth_parameters.append(local_memory_bandwidth)
            # local_memory_bandwidth_paramters.append(local_memory_bandwidth)
            hardware_parameter_dict = {'shared_memory_latency': shared_memory_latency,
                                       'local_memory_bandwidth': local_memory_bandwidth,
                                       'shared_memory_bandwidth': shared_memory_bandwidth}
            latency = simulate_tiled_attention(hardware_parameter_dict)
            latencies.append(latency)



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

np.savez('test/exp/shared_memory_data.npz', 
         shared_memory_bandwidth_parameters=np.array(shared_memory_bandwidth_parameters), 
         shared_memory_latency_parameters=np.array(shared_memory_latency_parameters),
         local_memory_bandwidth_parameters=np.array(local_memory_bandwidth_parameters),
         latencies=np.array(latencies))

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置柱的参数
dx = dy = 1  # 柱的宽度
dz = latencies  # 柱的高度

# 将bandwidth取对数
shared_memory_bandwidth_paramters = np.log2(shared_memory_bandwidth_parameters)
local_memory_bandwidth_parameters = np.log2(local_memory_bandwidth_parameters)

# x1: local bandwidth x2: latency x3: noc bandwidth
# 定义偏移量，用于将x2编码成不同的层
offset = 1 / 40
x1_combined = [shared_memory_bandwidth_parameters[i] * 10 + (400 - shared_memory_latency_parameters[i]) * offset for i in range(len(latencies))]  # 将x1和x2组合

# 根据x1的值设置颜色，颜色随着数值变大而变化
colors = plt.cm.viridis((np.array(shared_memory_bandwidth_paramters) - min(shared_memory_bandwidth_paramters)) / (max(shared_memory_bandwidth_paramters) - min(shared_memory_bandwidth_paramters)))

# 绘制三维柱状图
ax.bar3d(x1_combined, local_memory_bandwidth_parameters, np.zeros_like(latencies), dx, dy, dz, color=colors)

# # 添加二维坐标标签(x1, x2)作为y轴上的标记
# for i in range(len(y)):
#     ax.text(x1[i], x2_combined[i], y[i] + 0.5, f"({x2[i]},{x3[i]})", color='black', ha='center')

# 设置坐标轴标签
ax.set_xlabel('Shared Memory Bandwidth + Latency')
ax.set_ylabel('Local Memory Latency')
ax.set_zlabel('Latency')

plt.title("GPU-Like Shared Memory")

# 添加颜色映射条
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(shared_memory_bandwidth_paramters)
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_ticks([256, 512, 1024, 2048, 4096, 8192])

plt.savefig('temp/shared_memory.png', dpi=300)


