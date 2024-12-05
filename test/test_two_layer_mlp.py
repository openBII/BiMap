from src.simulator.task_rabbit.task_model.task_block import TaskBlock
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
_, input = create_input(task_graph, Shape(nr=2048), Precision.FLOAT_16)
output0, mlp0_dict = create_mlp(task_graph, input, 
                                Shape(nr=2048, nf=4096), 
                                Precision.FLOAT_16, False)
output, mlp1_dict = create_mlp(task_graph, output0, Shape(nr=4096, nf=1024), 
                               Precision.FLOAT_16, True)

STDraw.draw_graph(task_graph, out_path='temp/two_layer_mlp.task.html',
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
mlp0: TaskBlock = mlp0_dict["compute"]
mlp1: TaskBlock = mlp1_dict["compute"]
weight0: TaskBlock = mlp0_dict["weight"]
weight1: TaskBlock = mlp1_dict["weight"]
dram0 = create_mlcoord((0, 0), DRAM)
dram1 = create_mlcoord((0, 1), DRAM)
tensor_unit0 = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
tensor_unit1 = create_mlcoord((0, 1), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(dram0, input)
st_env.put_in(dram0, weight0)
st_env.put_in(tensor_unit0, mlp0)
st_env.put_in(dram1, output0)
st_env.put_in(dram1, weight1)
st_env.put_in(tensor_unit1, mlp1)
st_env.put_in(dram1, output)

# Edge Mapping
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
router0 = create_mlcoord((0, 0), CHIP, (0, 0), ROUTER)
phy0 = create_mlcoord((0, 0), PCIE_PHY)
board1 = create_mlcoord((0, 1))
phy1 = create_mlcoord((0, 1), PCIE_PHY)
shared_memory1 = create_mlcoord((0, 1), CHIP, SHARED_MEMORY)
router1 = create_mlcoord((0, 1), CHIP, (0, 0), ROUTER)
# [DRAM0, Chip] [Shared Memory0, Core0] [Router0, Tensor Unit0]
st_env.map_edge(input.output_edge, 
                [dram0, shared_memory0, router0, tensor_unit0])
st_env.map_edge(weight0.output_edge, 
                [dram0, shared_memory0, router0, tensor_unit0])
st_env.map_edge(mlp0.output_edge, 
                [tensor_unit0, router0, shared_memory0, dram0, 
                 phy0, board1, phy1, dram1])
st_env.map_edge(output0.output_edge, 
                [dram1, shared_memory1, router1, tensor_unit1])
st_env.map_edge(weight1.output_edge, 
                [dram1, shared_memory1, router1, tensor_unit1])
st_env.map_edge(mlp1.output_edge, 
                [tensor_unit1, router1, shared_memory1, dram1])
# st_env.auto_edge_map()

# Simulate
st_env.simulate()
st_env.show_overall_time()
