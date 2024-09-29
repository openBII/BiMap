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
input_task = InputTaskBlock(-1, Shape(nr=2048), Precision.FLOAT_16)
storage_task = STaskBlock(0, Shape(nr=2048), Precision.FLOAT_16)
compute_task = CTaskBlock(1, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
weight_task = StaticTaskBlock(2, Shape(nr=2048, nf=4096), Precision.FLOAT_16)
output_task = OutputTaskBlock(100, Shape(nf=4096), Precision.FLOAT_16)
task_graph.add_nodes([compute_task, storage_task, input_task, output_task, weight_task])
edge0 = task_graph.connect(storage_task.id, compute_task.id)
task_graph.connect(input_task.id, storage_task.id)
edge1 = task_graph.connect(compute_task.id, output_task.id)
edge2 = task_graph.connect(weight_task.id, compute_task.id)
task_graph.topologize()

STDraw.draw_graph(task_graph, out_path='test/mlp.task.html',
                  width='1920px', height='1080px')

# Construct a hardware
config = ServerConfig("top/server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Map the task graph onto the hardware
shared_memory_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((3, 1)))

dram_coord0 = MLCoord(Coord((0, 0)), Coord(1))
st_env.put_in(dram_coord0, storage_task.id)
mac_array_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(1))
st_env.put_in(mac_array_coord, compute_task.id)
dram_coord1 = MLCoord(Coord((1, 1)), Coord(1))
st_env.put_in(dram_coord1, output_task.id)
weight_coord = MLCoord(Coord((0, 1)), Coord(1))
st_env.put_in(weight_coord, weight_task.id)
pciephy_coord0 = MLCoord(Coord((0, 0)), Coord(2))
board_coord0 = MLCoord(Coord((0, 1)))
pciephy_coord1 = MLCoord(Coord((0, 1)), Coord(2))
router_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(3))
board_coord1 = MLCoord(Coord((1, 1)))
pciephy_coord2 = MLCoord(Coord((1, 1)), Coord(2))
buffer_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(0))
llc_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((3, 1)))
# level = min(ml_coord0.level, ml_coord1.level)
# if ml_coord0[level - 1] != ml_coord1[level - 1]: 需要在中间加入后一个坐标的container坐标
# 必须包含某一层次的跳出坐标或跳入坐标
st_env.map_edge(edge0, [dram_coord0, pciephy_coord0, board_coord0, pciephy_coord1, llc_coord, router_coord, mac_array_coord])
STDraw.draw_graph(task_graph, out_path='test/mlp.task.html',
                  width='1920px', height='1080px')
st_env.undo()
STDraw.draw_graph(task_graph, out_path='test/mlp.task.html',
                  width='1920px', height='1080px')
st_env.map_edge(edge0, [dram_coord0, pciephy_coord0, board_coord0, pciephy_coord1, llc_coord, router_coord, mac_array_coord])
STDraw.draw_graph(task_graph, out_path='test/mlp.task.html',
                  width='1920px', height='1080px')
# Also OK: [dram_coord0, pciephy_coord0, board_coord0, router_coord, mac_array_coord]
st_env.map_edge(edge1, [mac_array_coord, router_coord, llc_coord, pciephy_coord1, board_coord1, pciephy_coord2, dram_coord1])
# Also OK: [mac_array_coord, router_coord, board_coord1, pciephy_coord2, dram_coord1]
st_env.map_edge(edge2, [weight_coord, llc_coord, router_coord, buffer_coord, (mac_array_coord, 0, 1)])

def update(split_vector: SplitVector, state: Shape | None):
    if state is None:
        nf = split_vector.nf * 2
        return SplitVector(nf=nf), Shape(nf=2)
    elif state.nf == 2:
        nr = split_vector.nr * 2
        return SplitVector(nf=split_vector.nf, nr=nr), Shape(nr=2, nf=1)
    else:  # state.nr == 2
        nf = split_vector.nf * 2
        return SplitVector(nf=nf, nr=split_vector.nr), Shape(nr=1, nf=2)

split_vector = SplitVectorExplorer.explore_legal_split_vector(
    update=update,
    tasks=[storage_task, weight_task, output_task],
    memory_coord=shared_memory_coord,
    overflow=st_env.does_mlp_memory_overflow
)

# Simulate
st_env.simulate(1)
st_env.show_overall_time()
