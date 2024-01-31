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
for level3_count in range(GlobalConfig.GroupLevel3['GROUP2_NUM']):
    server_coord = Coord(level3_count)
    card_x, card_y, card_z = 0, 0, 0
    for _ in range(GlobalConfig.GroupLevel2['GROUP1_NUM']):
        card_coord = Coord((card_z, card_y, card_x))
        card_x += 1
        if card_x == GlobalConfig.GroupLevel2['GROUP1_X']:
            card_x = 0
            card_y += 1
            if card_y == GlobalConfig.GroupLevel2['GROUP1_Y']:
                card_y = 0
                card_z += 1
                
        point_count = 0
        for _ in range(GlobalConfig.GroupLevel1['CHIP_NUM']):
            coord = Coord(point_count)
            point_count += 1
            chip_ml_coord = MLCoord(server_coord, card_coord, coord)
            # TODO: 加入address
            st_env.put_in(chip_ml_coord, compute_task.id)
        for _ in range(GlobalConfig.GroupLevel1['DDR_NUM']):
            coord = Coord(point_count)
            point_count += 1
            ddr_ml_coord = MLCoord(server_coord, card_coord, coord)
            st_env.put_in(ddr_ml_coord, storage_task.id)
            st_env.put_in(ddr_ml_coord, output_task.id)

# Simulate
st_env.simulate(2, input_type=InputType.BATCH)
# st_env.simulate(2, input_type=InputType.PIPELINE)
print(task_graph)

