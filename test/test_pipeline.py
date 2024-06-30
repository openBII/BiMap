from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.static_task_block import StaticTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord
from src.simulator.resource_simulator.sync.sync_task import SyncTask

PIPELINE = False

# Construct a task graph
task_graph = TaskGraph()
input0 = InputTaskBlock(0, Shape(nr=2048), Precision.FLOAT_16)
storage0_0 = STaskBlock(1, Shape(nr=2048), Precision.FLOAT_16)
weight0_0 = StaticTaskBlock(2, Shape(nr=2048, nf=4096), Precision.FLOAT_16)
compute0_0 = CTaskBlock(3, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
storage0_1 = STaskBlock(4, Shape(nr=4096), Precision.FLOAT_16)
weight0_1 = StaticTaskBlock(5, Shape(nr=4096, nf=1024), Precision.FLOAT_16)
compute0_1 = CTaskBlock(6, Shape(nr=4096, nf=1024), TaskBlockType.CVM, Precision.FLOAT_16)
output0 = OutputTaskBlock(7, Shape(nf=1024), Precision.FLOAT_16)
task_graph.add_nodes([input0, storage0_0, weight0_0, compute0_0, storage0_1, weight0_1, compute0_1, output0])
task_graph.connect(input0.id, storage0_0.id)
edge0_0 = task_graph.connect(storage0_0.id, compute0_0.id)
edge0_1 = task_graph.connect(weight0_0.id, compute0_0.id)
edge0_2 = task_graph.connect(compute0_0.id, storage0_1.id)
edge0_3 = task_graph.connect(storage0_1.id, compute0_1.id)
edge0_4 = task_graph.connect(weight0_1.id, compute0_1.id)
edge0_5 = task_graph.connect(compute0_1.id, output0.id)

if PIPELINE:
    input1 = InputTaskBlock(8, Shape(nr=2048), Precision.FLOAT_16)
    storage1_0 = STaskBlock(9, Shape(nr=2048), Precision.FLOAT_16)
    weight1_0 = StaticTaskBlock(10, Shape(nr=2048, nf=4096), Precision.FLOAT_16)
    compute1_0 = CTaskBlock(11, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
    storage1_1 = STaskBlock(12, Shape(nr=4096), Precision.FLOAT_16)
    weight1_1 = StaticTaskBlock(13, Shape(nr=4096, nf=1024), Precision.FLOAT_16)
    compute1_1 = CTaskBlock(14, Shape(nr=4096, nf=1024), TaskBlockType.CVM, Precision.FLOAT_16)
    output1 = OutputTaskBlock(15, Shape(nf=1024), Precision.FLOAT_16)
    task_graph.add_nodes([input1, storage1_0, weight1_0, compute1_0, storage1_1, weight1_1, compute1_1, output1])
    task_graph.connect(input1.id, storage1_0.id)
    edge1_0 = task_graph.connect(storage1_0.id, compute1_0.id)
    edge1_1 = task_graph.connect(weight1_0.id, compute1_0.id)
    edge1_2 = task_graph.connect(compute1_0.id, storage1_1.id)
    edge1_3 = task_graph.connect(storage1_1.id, compute1_1.id)
    edge1_4 = task_graph.connect(weight1_1.id, compute1_1.id)
    edge1_5 = task_graph.connect(compute1_1.id, output1.id)

task_graph.topologize()

# Construct a hardware
config = ServerConfig("top/server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Map the task graph onto the hardware
# Task Mapping
storage0_coord = MLCoord(Coord((0, 0)), Coord(1))
st_env.put_in(storage0_coord, storage0_0.id)
st_env.put_in(storage0_coord, weight0_0.id)
compute0_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 0)), Coord(1))
st_env.put_in(compute0_coord, compute0_0.id)
storage1_coord = MLCoord(Coord((0, 1)), Coord(1))
st_env.put_in(storage1_coord, storage0_1.id)
st_env.put_in(storage1_coord, weight0_1.id)
compute1_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(1))
st_env.put_in(compute1_coord, compute0_1.id)
st_env.put_in(storage1_coord, output0.id)

if PIPELINE:
    st_env.put_in(storage0_coord, storage1_0.id)
    st_env.put_in(storage0_coord, weight1_0.id)
    st_env.put_in(compute0_coord, compute1_0.id)
    st_env.put_in(storage1_coord, storage1_1.id)
    st_env.put_in(storage1_coord, weight1_1.id)
    st_env.put_in(compute1_coord, compute1_1.id)
    st_env.put_in(storage1_coord, output1.id)

# Edge Mapping
llc0_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((3, 1)))
router0_coord = MLCoord(Coord((0, 0)), Coord(0), Coord((0, 0)), Coord(3))
dramphy0_coord = MLCoord(Coord((0, 0)), Coord(2))
board_coord = MLCoord(Coord((0, 1)))
dramphy1_coord = MLCoord(Coord((0, 1)), Coord(2))
llc1_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((3, 1)))
router1_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(3))

st_env.map_edge(edge0_0, [storage0_coord, llc0_coord, router0_coord, compute0_coord])
st_env.map_edge(edge0_1, [storage0_coord, llc0_coord, router0_coord, compute0_coord])

st_env.map_edge(edge0_2, [compute0_coord, router0_coord, llc0_coord, storage0_coord, dramphy0_coord, board_coord, dramphy1_coord, storage1_coord])

st_env.map_edge(edge0_3, [storage1_coord, llc1_coord, router1_coord, compute1_coord])
st_env.map_edge(edge0_4, [storage1_coord, llc1_coord, router1_coord, compute1_coord])
st_env.map_edge(edge0_5, [compute1_coord, router1_coord, llc1_coord, storage1_coord])

if PIPELINE:
    st_env.map_edge(edge1_0, [storage0_coord, llc0_coord, router0_coord, compute0_coord])
    st_env.map_edge(edge1_1, [storage0_coord, llc0_coord, router0_coord, compute0_coord])
    st_env.map_edge(edge1_2, [compute0_coord, router0_coord, llc0_coord, storage0_coord, dramphy0_coord, board_coord, dramphy1_coord, storage1_coord])
    st_env.map_edge(edge1_3, [storage1_coord, llc1_coord, router1_coord, compute1_coord])
    st_env.map_edge(edge1_4, [storage1_coord, llc1_coord, router1_coord, compute1_coord])
    st_env.map_edge(edge1_5, [compute1_coord, router1_coord, llc1_coord, storage1_coord])

# Simulate
st_env.simulate(1)
st_env.show_overall_time()
