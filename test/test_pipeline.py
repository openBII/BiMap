from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.task_rabbit.task_model.transformer import create_input, create_data, create_mlp


# Construct a task graph
task_graph = TaskGraph()
input0 = create_input(task_graph, Shape(nr=2048), Precision.FLOAT_16)
mlp_input0 = create_data(task_graph, Shape(nr=2048), Precision.FLOAT_16)
weight0_0, mlp0_0, mlp_output0 = create_mlp(task_graph, Shape(nr=2048, nf=4096), Precision.FLOAT_16, False)
weight0_1, mlp0_1, output0 = create_mlp(task_graph, Shape(nr=4096, nf=1024), Precision.FLOAT_16, True)
task_graph.connect_tasks_in_sequence([input0, mlp_input0, mlp0_0])
task_graph.connect_tasks_in_sequence([mlp_output0, mlp0_1])

input1 = create_input(task_graph, Shape(nr=2048), Precision.FLOAT_16)
mlp_input1 = create_data(task_graph, Shape(nr=2048), Precision.FLOAT_16)
weight1_0, mlp1_0, mlp_output1 = create_mlp(task_graph, Shape(nr=2048, nf=4096), Precision.FLOAT_16, False)
weight1_1, mlp1_1, output1 = create_mlp(task_graph, Shape(nr=4096, nf=1024), Precision.FLOAT_16, True)
task_graph.connect_tasks_in_sequence([input1, mlp_input1, mlp1_0])
task_graph.connect_tasks_in_sequence([mlp_output1, mlp1_1])

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
st_env.put_in(dram0, mlp_input0.id)
st_env.put_in(dram0, weight0_0.id)
sync_id0 = st_env.get_sync_id()
st_env.sync(dram0, sync_id0)
st_env.put_in(dram0, mlp_input1.id)
st_env.put_in(dram0, weight1_0.id)

mac_array0 = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array0, mlp0_0.id)
st_env.sync(mac_array0, sync_id0)
st_env.put_in(mac_array0, mlp1_0.id)
sync_id1 = st_env.get_sync_id()
st_env.sync(mac_array0, sync_id1)

dram1 = create_mlcoord((0, 1), DRAM)
st_env.put_in(dram1, mlp_output0.id)
st_env.put_in(dram1, weight0_1.id)
st_env.put_in(dram1, mlp_output1.id)
st_env.put_in(dram1, weight1_1.id)
mac_array1 = create_mlcoord((0, 1), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array1, mlp0_1.id)
st_env.sync(mac_array1, sync_id1)
st_env.put_in(dram1, output0.id)
st_env.put_in(mac_array1, mlp1_1.id)
st_env.put_in(dram1, output1.id)

# Edge Mapping
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
router0 = create_mlcoord((0, 0), CHIP, (0, 0), ROUTER)
phy0 = create_mlcoord((0, 0), PCIE_PHY)
board1 = create_mlcoord((0, 1))
phy1 = create_mlcoord((0, 1), PCIE_PHY)
shared_memory1 = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
router1 = create_mlcoord((0, 1), CHIP, (0, 0), ROUTER)
st_env.map_edge(mlp_input0.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(weight0_0.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(mlp0_0.output_edge, [mac_array0, router0, shared_memory0, dram0, phy0, board1, phy1, dram1])
st_env.map_edge(mlp_output0.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(weight0_1.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(mlp0_1.output_edge, [mac_array1, router1, shared_memory1, dram1])
st_env.map_edge(mlp_input1.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(weight1_0.output_edge, [dram0, shared_memory0, router0, mac_array0])
st_env.map_edge(mlp1_0.output_edge, [mac_array0, router0, shared_memory0, dram0, phy0, board1, phy1, dram1])
st_env.map_edge(mlp_output1.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(weight1_1.output_edge, [dram1, shared_memory1, router1, mac_array1])
st_env.map_edge(mlp1_1.output_edge, [mac_array1, router1, shared_memory1, dram1])

# Simulate
st_env.simulate()
st_env.show_overall_time()
