from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.task_rabbit.task_model.transformer import create_input, create_data, create_mlp
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw


# Construct a task graph
task_graph = TaskGraph()
input = create_input(task_graph, Shape(nr=2048), Precision.FLOAT_16)
mlp0_input = create_data(task_graph, Shape(nr=2048), Precision.FLOAT_16)
weight0, mlp0, mlp0_output = create_mlp(task_graph, Shape(nr=2048, nf=4096), Precision.FLOAT_16, False)
weight1, mlp1, output = create_mlp(task_graph, Shape(nr=4096, nf=1024), Precision.FLOAT_16, True)
task_graph.connect_tasks_in_sequence([input, mlp0_input, mlp0])
task_graph.connect_tasks_in_sequence([mlp0_output, mlp1])

STDraw.draw_graph(task_graph, out_path='temp/2mlp.task.html',
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
# Task Mapping
dram0 = create_mlcoord((0, 0), DRAM)
st_env.put_in(dram0, mlp0_input.id)
st_env.put_in(dram0, weight0.id)
mac_array0 = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array0, mlp0.id)
dram1 = create_mlcoord((0, 1), DRAM)
st_env.put_in(dram1, mlp0_output.id)
st_env.put_in(dram1, weight1.id)
mac_array1 = create_mlcoord((0, 1), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array1, mlp1.id)
st_env.put_in(dram1, output.id)
# Edge Mapping
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
router0 = create_mlcoord((0, 0), CHIP, (0, 0), ROUTER)
phy0 = create_mlcoord((0, 0), PCIE_PHY)
board1 = create_mlcoord((0, 1))
phy1 = create_mlcoord((0, 1), PCIE_PHY)
shared_memory1 = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
router1 = create_mlcoord((0, 1), CHIP, (0, 0), ROUTER)
st_env.map_edge(mlp0_input.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(weight0.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(mlp0.output_edge, [mac_array0, router0, shared_memory0, dram0, phy0, board1, phy1, dram1])
st_env.map_edge(mlp0_output.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(weight1.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(mlp1.output_edge, [mac_array1, router1, shared_memory1, dram1])

# Simulate
st_env.simulate()
st_env.show_overall_time()
