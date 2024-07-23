from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.transformer import create_input, create_mlp


# Construct a task graph
task_graph = TaskGraph()
input, mlp_input = create_input(task_graph, Shape(nf=2048), Precision.FLOAT_16)
output, task_dict = create_mlp(task_graph, mlp_input, Shape(nr=2048, nf=4096), 
                               Precision.FLOAT_16, True)

STDraw.draw_graph(task_graph, out_path='temp/mlp.task.html',
                  width='1920px', height='1080px')

# Construct a hardware
config = ServerConfig("top/gpu_server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
env = STEnv(task_graph, server)

# Define some constants for writing coordinates
SIZE_X, SIZE_Y = config.PCB.chiplet.size
CHIP = 0
DRAM = 2
PCIE_PHY = 3
TENSOR_UNIT = 1
ROUTER = 3
SHARED_MEMORY = (SIZE_X + 1, SIZE_Y // 2)
SRAM_BUFFER = 0

# Parse some tasks
mlp: TaskBlock = task_dict["compute"]
weight: TaskBlock = task_dict["weight"]

# Graph Transformation
split_inputs = env.split_task(mlp_input.id, SplitVector(nf=4))
env.add_nodes_between(mlp_input, mlp, split_inputs)
new_tasks = env.split_mlp(
    split_inputs=split_inputs,
    weight=weight,
    compute=mlp,
    output=output,
    split_vector=SplitVector(nr=4)
)
split_weight, split_mlp, split_mlp_output, split_add, split_add_output = new_tasks
output.enable()
env.connect_tasks(split_add_output, [output])

STDraw.draw_graph(task_graph, out_path='temp/tiled_mlp0.task.html',
                  width='1920px', height='1080px')

# 测试undo功能
# env.undo()
# STDraw.draw_graph(task_graph, out_path='temp/tiled_mlp1.task.html',
#                   width='1920px', height='1080px')

# Map the task graph onto the hardware
dram0 = create_mlcoord((0, 0), DRAM)
dram1 = create_mlcoord((0, 1), DRAM)
dram3 = create_mlcoord((1, 1), DRAM)
shared_memory1 = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
env.put_in(dram0, mlp_input.id)
env.put_tasks_in(shared_memory1, split_inputs)
env.put_in(dram1, weight.id)
env.put_tasks_in(shared_memory1, split_weight)
env.put_tasks_in(shared_memory1, split_mlp_output)
env.put_tasks_in(shared_memory1, split_add_output)
for i in range(2):
    for j in range(2):
        tensor_unit = create_mlcoord((0, 1), CHIP, (i, j), TENSOR_UNIT)
        env.put_in(tensor_unit, split_mlp[j + i * 2].id)
env.put_in(tensor_unit, split_add[0].id)
env.put_in(dram3, output.id)

# Edge mapping
env.auto_edge_map()

# pciephy0 = create_mlcoord((0, 0), PCIE_PHY)
# board1 = create_mlcoord((0, 1))
# pciephy1 = create_mlcoord((0, 1), PCIE_PHY)
# router = create_mlcoord((0, 1), CHIP, (0, 0), ROUTER)
# board3 = create_mlcoord((1, 1))
# pciephy3 = create_mlcoord((1, 1), PCIE_PHY)
# shared_memory = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
# buffer = create_mlcoord((0, 1), CHIP, (0, 0), SRAM_BUFFER)
# # level = min(ml_coord0.level, ml_coord1.level)
# # if ml_coord0[level - 1] != ml_coord1[level - 1]: 需要在中间加入后一个坐标的container坐标
# # 必须包含某一层次的跳出坐标或跳入坐标
# env.map_edge(mlp_input.output_edge, [dram0, pciephy0, board1, pciephy1, dram1, shared_memory, router, mac_array])
# env.undo()
# env.map_edge(mlp_input.output_edge, [dram0, pciephy0, board1, pciephy1, dram1, shared_memory, router, mac_array])
# env.map_edge(mlp.output_edge, [mac_array, router, shared_memory, dram1, pciephy1, board3, pciephy3, dram3])
# env.map_edge(weight.output_edge, [dram1, shared_memory, router, buffer, (mac_array, 0, 1)])  # domain, link
STDraw.draw_graph(task_graph, out_path='temp/tiled_mlp1.task.html',
                  width='1920px', height='1080px')

# Simulate
env.simulate()
env.show_overall_time()
print(env.get_latency())
