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
config = ServerConfig("top/gpu_server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Graph Transformation
up_input_on_chip = st_env.copy_task(up_input.id)
split_up_input = st_env.split_task(up_input_on_chip.id, SplitVector(nr=2))
st_env.connect_tasks([up_input], split_up_input)
split_up_weight, split_up_mlp, split_up_mlp_output, split_up_add, split_up_add_output = st_env.split_mlp(up_input_on_chip, split_up_input, up_weight, up_mlp, relu_input, SplitVector(nf=2, nr=2))
split_relu_input, split_relu, split_down_input = st_env.split_pointwise(relu_input, split_up_add_output, relu, down_input, SplitVector(nf=4))
split_down_weight, split_down_mlp, split_down_mlp_output, split_down_add, split_down_add_output = st_env.split_mlp(down_input, split_down_input, down_weight, down_mlp, output, SplitVector(nf=2, nr=2))
st_env.connect_tasks(split_down_add_output, [output])
st_env.enable_task(output.id)

# up_input_on_chip = st_env.copy_task(up_input.id)
# split_up_input = st_env.split_task(up_input_on_chip.id, SplitVector(nr=2))
# st_env.connect_tasks([up_input], split_up_input)
# task_dict = st_env.split_FFN(SplitVector(nf=2, nr=2), SplitVector(nf=4), SplitVector(nf=2, nr=2),
#                              up_input_on_chip, split_up_input, up_weight, up_mlp, relu_input,
#                              relu, down_input, down_weight, down_mlp, output)
# st_env.connect_tasks(task_dict["down"]["add_output"], [output])

st_env.undo()
st_env.undo()
st_env.undo()
st_env.delete_tasks(split_up_input)
STDraw.draw_graph(task_graph, out_path='test/FFN.task.html',
                  width='1920px', height='1080px')

# Map the task graph onto the hardware
dram_coord = MLCoord(Coord((0, 0)), Coord(1))
chip_coord = MLCoord(Coord((0, 0)), Coord(0))
st_env.put_in(dram_coord, up_input.id)
st_env.undo()
st_env.put_in(dram_coord, up_weight.id)
st_env.put_in(dram_coord, down_weight.id)
st_env.put_in(dram_coord, output.id)
shared_memory_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((3, 1)))
# st_env.put_tasks_in(shared_memory_coord, split_up_input)
# st_env.put_tasks_in(shared_memory_coord, split_up_weight)
# st_env.put_tasks_in(shared_memory_coord, split_up_mlp_output)
# st_env.put_tasks_in(shared_memory_coord, split_relu_input)
# st_env.put_tasks_in(shared_memory_coord, split_down_input)
# st_env.put_tasks_in(shared_memory_coord, split_down_weight)
# st_env.put_tasks_in(shared_memory_coord, split_down_mlp_output)
# st_env.put_tasks_in(shared_memory_coord, split_down_add_output)
# for i in range(2):
#     for j in range(2):
#         mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
#         st_env.put_in(mac_array_coord, split_up_mlp[j + i * 2].id)
# for i in range(2):
#     mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
#     st_env.put_in(mac_array_coord, split_up_add[i].id)
# for i in range(2):
#     for j in range(2):
#         mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
#         st_env.put_in(mac_array_coord, split_relu[j + i * 2].id)
#         st_env.put_in(mac_array_coord, split_down_mlp[j + i * 2].id)
# for i in range(2):
#     mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
#     st_env.put_in(mac_array_coord, split_down_add[i].id)
# # Edge Mapping
# st_env.map_edges(up_input.output_edges, [dram_coord, chip_coord])
# st_env.map_edges(up_weight.output_edges, [dram_coord, chip_coord])
# st_env.map_edges(down_weight.output_edges, [dram_coord, chip_coord])
# for i in range(2):
#     for j in range(2):
#         task = split_up_mlp[j + i * 2]
#         mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
#         router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
#         st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
#         st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
# for i in range(2):
#     add = split_up_add[i]
#     mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
#     router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(3))
#     st_env.map_edges(add.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
#     st_env.map_edges(add.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
# for i in range(2):
#     for j in range(2):
#         task = split_relu[j + i * 2]
#         mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
#         router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
#         st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
#         st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
# for i in range(2):
#     for j in range(2):
#         task = split_down_mlp[j + i * 2]
#         mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
#         router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
#         st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
#         st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
# for i in range(2):
#     add = split_down_add[i]
#     mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
#     router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(3))
#     st_env.map_edges(add.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
#     st_env.map_edges(add.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
# st_env.map_edges(output.input_edges, [chip_coord, dram_coord])
st_env.put_tasks_in(shared_memory_coord, task_dict["up"]["input"])
st_env.put_tasks_in(shared_memory_coord, task_dict["up"]["weight"])
st_env.put_tasks_in(shared_memory_coord, task_dict["up"]["mlp_output"])
st_env.put_tasks_in(shared_memory_coord, task_dict["activation"]["input"])
st_env.put_tasks_in(shared_memory_coord, task_dict["down"]["input"])
st_env.put_tasks_in(shared_memory_coord, task_dict["down"]["weight"])
st_env.put_tasks_in(shared_memory_coord, task_dict["down"]["mlp_output"])
st_env.put_tasks_in(shared_memory_coord, task_dict["down"]["add_output"])
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, task_dict["up"]["mlp"][j + i * 2].id)
for i in range(2):
    mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
    st_env.put_in(mac_array_coord, task_dict["up"]["add"][i].id)
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, task_dict["activation"]["compute"][j + i * 2].id)
        st_env.put_in(mac_array_coord, task_dict["down"]["mlp"][j + i * 2].id)
for i in range(2):
    mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
    st_env.put_in(mac_array_coord, task_dict["down"]["add"][i].id)
# Edge Mapping
st_env.map_edges(up_input.output_edges, [dram_coord, chip_coord])
st_env.map_edges(up_weight.output_edges, [dram_coord, chip_coord])
st_env.map_edges(down_weight.output_edges, [dram_coord, chip_coord])
for i in range(2):
    for j in range(2):
        task = task_dict["up"]["mlp"][j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
        st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
for i in range(2):
    add = task_dict["up"]["add"][i]
    mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
    router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(3))
    st_env.map_edges(add.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
    st_env.map_edges(add.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
for i in range(2):
    for j in range(2):
        task = task_dict["activation"]["compute"][j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
        st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
for i in range(2):
    for j in range(2):
        task = task_dict["down"]["mlp"][j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
        st_env.map_edges(task.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
        st_env.map_edges(task.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
for i in range(2):
    add = task_dict["down"]["add"][i]
    mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(1))
    router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, i)), Coord(3))
    st_env.map_edges(add.input_edges, [shared_memory_coord, router_coord, mac_array_coord])
    st_env.map_edges(add.output_edges, [mac_array_coord, router_coord, shared_memory_coord])
st_env.map_edges(output.input_edges, [chip_coord, dram_coord])

st_env.simulate()
st_env.show_overall_time()