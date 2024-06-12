from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord


# Construct a task graph
task_graph = TaskGraph()
input_task = InputTaskBlock(-1, Shape(nr=2048), Precision.FLOAT_16)
storage_task = STaskBlock(0, Shape(nr=2048), Precision.FLOAT_16)
compute_task = CTaskBlock(1, Shape(nr=2048, nf=4096), TaskBlockType.CVM, Precision.FLOAT_16)
output_task = OutputTaskBlock(100, Shape(nr=4096), Precision.FLOAT_16)
task_graph.add_nodes([compute_task, storage_task, input_task, output_task])
edge0 = task_graph.connect(storage_task.id, compute_task.id)
task_graph.connect(input_task.id, storage_task.id)
edge1 = task_graph.connect(compute_task.id, output_task.id)
task_graph.topologize()

# Construct a hardware
config = ServerConfig("top/server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)

# Map the task graph onto the hardware
dram_coord0 = MLCoord(Coord((0, 0)), Coord(1))
st_env.put_in(dram_coord0, storage_task.id)
mac_array_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(1))
st_env.put_in(mac_array_coord, compute_task.id)
dram_coord1 = MLCoord(Coord((1, 1)), Coord(1))
st_env.put_in(dram_coord1, output_task.id)
pciephy_coord0 = MLCoord(Coord((0, 0)), Coord(2))
board_coord0 = MLCoord(Coord((0, 1)))
pciephy_coord1 = MLCoord(Coord((0, 1)), Coord(2))
router_coord = MLCoord(Coord((0, 1)), Coord(0), Coord((0, 0)), Coord(3))
board_coord1 = MLCoord(Coord((1, 1)))
pciephy_coord2 = MLCoord(Coord((1, 1)), Coord(2))
# level = min(ml_coord0.level, ml_coord1.level)
# if ml_coord0[level - 1] != ml_coord1[level - 1]: 需要在中间加入后一个坐标的container坐标
# 必须包含某一层次的跳出坐标或跳入坐标
st_env.map_edge(edge0, [dram_coord0, pciephy_coord0, board_coord0, pciephy_coord1, router_coord, mac_array_coord])
# Also OK: [dram_coord0, pciephy_coord0, board_coord0, router_coord, mac_array_coord]
st_env.map_edge(edge1, [mac_array_coord, router_coord, pciephy_coord1, board_coord1, pciephy_coord2, dram_coord1])
# Also OK: [mac_array_coord, router_coord, board_coord1, pciephy_coord2, dram_coord1]

# Simulate
st_env.simulate(1)
