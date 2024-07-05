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
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.resource_simulator.st_draw import STDraw
from copy import copy


# Construct a task graph
task_graph = TaskGraph()
input_task = InputTaskBlock(-1, Shape(nr=2048), Precision.FLOAT_16)
storage_task = STaskBlock(0, Shape(nr=2048), Precision.FLOAT_16)
compute_task = CTaskBlock(1, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
weight_task = StaticTaskBlock(2, Shape(nr=2048, nf=4096), Precision.FLOAT_16)
output_task = OutputTaskBlock(100, Shape(nr=4096), Precision.FLOAT_16)
task_graph.add_nodes([compute_task, storage_task, input_task, output_task, weight_task])
edge0 = task_graph.connect(storage_task.id, compute_task.id)
task_graph.connect(input_task.id, storage_task.id)
edge1 = task_graph.connect(compute_task.id, output_task.id)
edge2 = task_graph.connect(weight_task.id, compute_task.id)
task_graph.topologize()

# Construct a hardware
config = ServerConfig("top/server.toml")
server = ServerFactory.create_matrix(config)
# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Graph Transformation
storage_task_on_chip = st_env.copy_task(storage_task.id)
split_storage_tasks = st_env.split_and_copy_task(storage_task_on_chip.id, SplitVector(nr=2))
st_env.connect_tasks([storage_task], split_storage_tasks)
split_compute_tasks = st_env.split_task(compute_task.id, SplitVector(nf=2, nr=2))
st_env.connect_tasks(split_storage_tasks, split_compute_tasks)
mlp_storage_task = st_env.copy_task(output_task.id)
split_mlp_storage_tasks = st_env.split_and_copy_task(mlp_storage_task.id, SplitVector(nf=2))
st_env.connect_tasks(split_compute_tasks, split_mlp_storage_tasks)
add_task0 = CTaskBlock(IDGenerator.get_next_task_id(), Shape(nf=2048, branch=2), TaskBlockType.CADD, Precision.FLOAT_16)
task_graph.add_node(add_task0)
add_task1 = st_env.copy_task(add_task0.id)
st_env.connect_tasks(split_mlp_storage_tasks, [add_task0, add_task1])
split_add_storage_tasks = st_env.split_task(mlp_storage_task.id, SplitVector(nf=2))
st_env.connect_tasks([add_task0, add_task1], split_add_storage_tasks)
st_env.connect_tasks(split_add_storage_tasks, [output_task])
weight_task_on_chip = st_env.copy_task(weight_task.id)
split_weight_tasks = st_env.split_task(weight_task_on_chip.id, SplitVector(nf=2, nr=2))
st_env.connect_tasks([weight_task], split_weight_tasks)
st_env.connect_tasks(split_weight_tasks, split_compute_tasks)
st_env.delete_tasks([compute_task, storage_task_on_chip, mlp_storage_task, weight_task_on_chip])

# Map the task graph onto the hardware
dram_coord = MLCoord(Coord((0, 0)), Coord(1))
chip_coord = MLCoord(Coord((0, 0)), Coord(0))
st_env.put_in(dram_coord, storage_task.id)
st_env.put_in(dram_coord, weight_task.id)
st_env.put_in(dram_coord, output_task.id)
shared_memory_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((3, 1)))
for task in split_weight_tasks:
    st_env.put_in(shared_memory_coord, task.id)
for task in split_storage_tasks:
    st_env.put_in(shared_memory_coord, task.id)
for task in split_mlp_storage_tasks:
    st_env.put_in(shared_memory_coord, task.id)
for task in split_add_storage_tasks:
    st_env.put_in(shared_memory_coord, task.id)
for i in range(2):
    for j in range(2):
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        st_env.put_in(mac_array_coord, split_compute_tasks[j + i * 2].id)
mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 0)), Coord(1))
st_env.put_in(mac_array_coord, add_task0.id)
mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 1)), Coord(1))
st_env.put_in(mac_array_coord, add_task1.id)
# Edge Mapping
for edge in weight_task.output_edges:
    st_env.map_edge(edge, [dram_coord, chip_coord])
for edge in storage_task.output_edges:
    st_env.map_edge(edge, [dram_coord, chip_coord])
for i in range(2):
    for j in range(2):
        task = split_compute_tasks[j + i * 2]
        mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(1))
        router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((i, j)), Coord(3))
        input_edges = task.input_edges
        out_edge = task.output_edges[0]
        for edge in input_edges:
            st_env.map_edge(edge, [shared_memory_coord, router_coord, mac_array_coord])
        st_env.map_edge(out_edge, [mac_array_coord, router_coord, shared_memory_coord])
mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 0)), Coord(1))
router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 0)), Coord(3))
for edge in add_task0.input_edges:
    st_env.map_edge(edge, [shared_memory_coord, router_coord, mac_array_coord])
st_env.map_edge(add_task0.output_edges[0], [mac_array_coord, router_coord, shared_memory_coord])
mac_array_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 1)), Coord(1))
router_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 1)), Coord(3))
for edge in add_task1.input_edges:
    st_env.map_edge(edge, [shared_memory_coord, router_coord, mac_array_coord])
st_env.map_edge(add_task1.output_edges[0], [mac_array_coord, router_coord, shared_memory_coord])
for edge in output_task.input_edges:
    st_env.map_edge(edge, [chip_coord, dram_coord])

STDraw.draw_graph(task_graph, out_path='test/mlp.task.html',
                  width='1920px', height='1080px')
st_env.simulate()
st_env.show_overall_time()