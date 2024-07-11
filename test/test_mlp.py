from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.transformer import create_input, create_data, create_compute, create_static


# Construct a task graph
task_graph = TaskGraph()
input = create_input(task_graph, Shape(nr=2048), Precision.FLOAT_16)
mlp_input = create_data(task_graph, Shape(nr=2048), Precision.FLOAT_16)
weight = create_static(task_graph, Shape(nr=2048, nf=4096), Precision.FLOAT_16)
mlp = create_compute(task_graph, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
output = create_data(task_graph, Shape(nf=4096), Precision.FLOAT_16, True)
task_graph.connect_tasks_in_sequence([input, mlp_input, mlp, output])
task_graph.connect(weight.id, mlp.id)
task_graph.topologize()

STDraw.draw_graph(task_graph, out_path='temp/mlp.task.html',
                width='1920px', height='1080px')

# Construct a hardware
config = ServerConfig("top/gpu_server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Define some constants for writing coordinates
CHIP = 0
DRAM = 2
PCIE_PHY = 3
TENSOR_UNIT = 1
ROUTER = 3
SHARED_MEMORY = (3, 1)
SRAM_BUFFER = 0

# Map the task graph onto the hardware
dram0 = create_mlcoord((0, 0), DRAM)
dram1 = create_mlcoord((0, 1), DRAM)
dram2 = create_mlcoord((1, 0), DRAM)
dram3 = create_mlcoord((1, 1), DRAM)
st_env.put_in(dram0, mlp_input.id)
mac_array = create_mlcoord((0, 1), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array, mlp.id)
st_env.put_in(dram3, output.id)
st_env.put_in(dram1, weight.id)
# Edge mapping
pciephy0 = create_mlcoord((0, 0), PCIE_PHY)
board1 = create_mlcoord((0, 1))
pciephy1 = create_mlcoord((0, 1), PCIE_PHY)
router = create_mlcoord((0, 1), CHIP, (0, 0), ROUTER)
board3 = create_mlcoord((1, 1))
pciephy3 = create_mlcoord((1, 1), PCIE_PHY)
shared_memory = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
buffer = create_mlcoord((0, 1), CHIP, (0, 0), SRAM_BUFFER)
# level = min(ml_coord0.level, ml_coord1.level)
# if ml_coord0[level - 1] != ml_coord1[level - 1]: 需要在中间加入后一个坐标的container坐标
# 必须包含某一层次的跳出坐标或跳入坐标
st_env.map_edge(mlp_input.output_edge, [dram0, pciephy0, board1, pciephy1, dram1, shared_memory, router, mac_array])
st_env.undo()
st_env.map_edge(mlp_input.output_edge, [dram0, pciephy0, board1, pciephy1, dram1, shared_memory, router, mac_array])
st_env.map_edge(mlp.output_edge, [mac_array, router, shared_memory, dram1, pciephy1, board3, pciephy3, dram3])
st_env.map_edge(weight.output_edge, [dram1, shared_memory, router, buffer, (mac_array, 0, 1)])  # domain, link
STDraw.draw_graph(task_graph, out_path='temp/mlp_mapped.task.html',
                  width='1920px', height='1080px')

# Simulate
st_env.simulate()
st_env.show_overall_time()
