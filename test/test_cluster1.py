from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.input_task_block import InputTaskBlock
from src.simulator.task_rabbit.task_model.output_task_block import OutputTaskBlock
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.factory.cluster_factory import ClusterFactory
from top.config import GlobalConfig
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord
from src.simulator.task_rabbit.task_model.input_type import InputType


# Construct a task graph
# TODO: 处理静态数据
# TODO: 加入边映射
task_graph = TaskGraph()
shape = Shape(2048, 4096)
input_task = InputTaskBlock(-1, shape, Precision.FLOAT_16)
storage_task = STaskBlock(0, shape, Precision.FLOAT_16)
compute_task = CTaskBlock(1, shape, TaskBlockType.CVM, Precision.FLOAT_16)
output_task = OutputTaskBlock(100, shape, Precision.FLOAT_16)
task_graph.add_node(compute_task)
task_graph.add_node(storage_task)
task_graph.add_node(input_task)
task_graph.add_node(output_task)

edge0 = task_graph.connect(storage_task.id, compute_task.id)
task_graph.connect(input_task.id, storage_task.id)
edge1 = task_graph.connect(compute_task.id, output_task.id)

task_graph.topologize()

# Construct a hardware
cluster = ClusterFactory.create_st_matrix()

# Construct a simulation environment
st_env = STEnv(task_graph, cluster)

# Map the task graph onto the hardware
server_coord = Coord(0)
card_coord = Coord((0, 0, 0))
chip_ml_coord0 = MLCoord(server_coord, card_coord, Coord(0))
st_env.put_in(chip_ml_coord0, compute_task.id)
ddr_ml_coord1 = MLCoord(server_coord, card_coord, Coord(2))
st_env.put_in(ddr_ml_coord1, storage_task.id)
ddr_ml_coord2 = MLCoord(server_coord, card_coord, Coord(3))
st_env.put_in(ddr_ml_coord2, output_task.id)
st_env.map_edge(edge0, [ddr_ml_coord1, chip_ml_coord0])
st_env.map_edge(edge1, [chip_ml_coord0, ddr_ml_coord2])

# Simulate
st_env.simulate(2, input_type=InputType.BATCH)
# st_env.simulate(2, input_type=InputType.PIPELINE)
print(task_graph)

