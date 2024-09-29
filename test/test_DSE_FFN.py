from __future__ import annotations
from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.compiler.mapper.split_vector_explorer import SplitVectorExplorer


# Construct a task graph
task_graph = TaskGraph()
input = InputTaskBlock(-1, Shape(nr=512), Precision.FLOAT_16)
up_input = STaskBlock(0, Shape(nr=512), Precision.FLOAT_16)
up_weight = StaticTaskBlock(1, Shape(nr=512, nf=2048), Precision.FLOAT_16)
up_mlp = CTaskBlock(2, Shape(nr=512, nf=2048), TaskBlockType.CVM, Precision.FLOAT_16)
relu_input = STaskBlock(3, Shape(nf=2048), Precision.FLOAT_16)
relu = CTaskBlock(4, Shape(nf=2048), TaskBlockType.CCMPB, Precision.FLOAT_16)
down_input = STaskBlock(5, Shape(nr=2048), Precision.FLOAT_16)
down_weight = StaticTaskBlock(6, Shape(nr=2048, nf=512), Precision.FLOAT_16)
down_mlp = CTaskBlock(7, Shape(nr=2048, nf=512), TaskBlockType.CVM, Precision.FLOAT_16)
output = OutputTaskBlock(8, Shape(nf=512), Precision.FLOAT_16)
task_graph.add_nodes([input, up_input, up_weight, up_mlp, relu_input, relu, down_input, down_weight, down_mlp, output])
task_graph.connect(input.id, up_input.id)
task_graph.connect(up_input.id, up_mlp.id)
task_graph.connect(up_weight.id, up_mlp.id)
task_graph.connect(up_mlp.id, relu_input.id)
task_graph.connect(relu_input.id, relu.id)
task_graph.connect(relu.id, down_input.id)
task_graph.connect(down_input.id, down_mlp.id)
task_graph.connect(down_weight.id, down_mlp.id)
task_graph.connect(down_mlp.id, output.id)

# Construct a hardware
config = ServerConfig("top/server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Mapping Exploration
# Define some coordinates
shared_memory_coord0 = MLCoord(Coord((0, 0)), Coord(0), Coord((3, 1)))
dram_coord0 = MLCoord(Coord((0, 0)), Coord(1))
chip_coord0 = MLCoord(Coord((0, 0)), Coord(0))
phy_coord0 = MLCoord(Coord((0, 0)), Coord(2))
board0 = MLCoord(Coord((0, 0)))
shared_memory_coord1 = MLCoord(Coord((0, 1)), Coord(0), Coord((3, 1)))
dram_coord1 = MLCoord(Coord((0, 1)), Coord(1))
chip_coord1 = MLCoord(Coord((0, 1)), Coord(0))
phy_coord1 = MLCoord(Coord((0, 1)), Coord(2))
board1 = MLCoord(Coord((0, 1)))
shared_memory_coord2 = MLCoord(Coord((1, 0)), Coord(0), Coord((3, 1)))
dram_coord2 = MLCoord(Coord((1, 0)), Coord(1))
chip_coord2 = MLCoord(Coord((1, 0)), Coord(0))
phy_coord2 = MLCoord(Coord((1, 0)), Coord(2))
board2 = MLCoord(Coord((1, 0)))
shared_memory_coord3 = MLCoord(Coord((1, 1)), Coord(0), Coord((3, 1)))
dram_coord3 = MLCoord(Coord((1, 1)), Coord(1))
chip_coord3 = MLCoord(Coord((1, 1)), Coord(0))
phy_coord3 = MLCoord(Coord((1, 1)), Coord(2))
board3 = MLCoord(Coord((1, 1)))

# Graph Transformation
up_input_on_chip = st_env.copy_task(up_input.id)
split_up_input = st_env.split_task(up_input_on_chip.id, SplitVector())
st_env.connect_tasks([up_input], split_up_input)
split_up_weight, split_up_mlp, split_up_mlp_output = st_env.split_mlp(up_input_on_chip, split_up_input, up_weight, up_mlp, relu_input, SplitVector(nf=4))
split_relu, split_down_input = st_env.split_pointwise(relu_input, split_up_mlp_output, relu, down_input, SplitVector(nf=4))
split_down_weight, split_down_mlp, split_down_mlp_output = st_env.split_mlp(down_input, split_down_input, down_weight, down_mlp, output, SplitVector(nf=4))
st_env.connect_tasks(split_down_mlp_output, [output])
st_env.enable_task(output.id)

STDraw.draw_graph(task_graph, out_path='test/FFN.task.html',
                  width='1920px', height='1080px')

# Map the task graph onto the hardware
st_env.put_in(dram_coord0, up_input.id)
st_env.put_in(dram_coord0, up_weight.id)
st_env.put_in(dram_coord2, down_weight.id)
st_env.put_in(dram_coord3, output.id)
st_env.put_tasks_in(shared_memory_coord0, split_up_input)
st_env.put_tasks_in(shared_memory_coord0, split_up_weight)
st_env.put_tasks_in(shared_memory_coord0, split_up_mlp_output)
st_env.put_tasks_in(shared_memory_coord1, split_down_input)
st_env.put_tasks_in(shared_memory_coord1, split_down_weight)
st_env.put_tasks_in(shared_memory_coord1, split_down_mlp_output)
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, split_up_mlp[j + i * 2].id)
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, split_relu[j + i * 2].id)
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((1, 0)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, split_down_mlp[j + i * 2].id)
# Edge Mapping
st_env.map_edges(up_input.enabled_output_edges, [dram_coord0, chip_coord0])
st_env.map_edges(up_weight.enabled_output_edges, [dram_coord0, chip_coord0])
st_env.map_edges(down_weight.enabled_output_edges, [dram_coord2, chip_coord2])
for i in range(2):
    for j in range(2):
        task: CTaskBlock = split_up_mlp[j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.enabled_input_edges, [shared_memory_coord0, router_coord, mac_array_coord])
        st_env.map_edges(task.enabled_output_edges, [mac_array_coord, router_coord, shared_memory_coord0, dram_coord0])
for i in range(2):
    for j in range(2):
        task = split_relu[j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.enabled_input_edges, [dram_coord0, phy_coord0, board1, phy_coord1, dram_coord1, shared_memory_coord1, router_coord, mac_array_coord])
        st_env.map_edges(task.enabled_output_edges, [mac_array_coord, router_coord, shared_memory_coord1, dram_coord1])
for i in range(2):
    for j in range(2):
        task = split_down_mlp[j + i * 2]
        mac_array_coord = MLCoord(Coord((1, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((1, 0)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.enabled_input_edges, [dram_coord1, phy_coord1, board2, phy_coord2, dram_coord2, shared_memory_coord2, router_coord, mac_array_coord])
        st_env.map_edges(task.enabled_output_edges, [mac_array_coord, router_coord, shared_memory_coord2, dram_coord2])
st_env.map_edges(output.enabled_input_edges, [dram_coord2, phy_coord2, board3, phy_coord3, dram_coord3])

st_env.simulate()
st_env.show_overall_time()