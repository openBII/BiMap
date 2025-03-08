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
from src.simulator.task_rabbit.task_model.transformer import *
from src.simulator.resource_simulator.config.matrix_config import ServerConfig, HybridPackageConfig, CompAirPackageConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.resource_simulator.st_model.space_matrix.package_factory import HybridPackageFactory, CompAirPackageFactory
# Parameter
trm_qkv_len = 256
cur_precision = Precision.INT_8

# Construct a task graph
task_graph = TaskGraph()

# Map the task graph onto the hardware
#                <chiplet-id> - <core-id> - <device-id>
# create_mlcoord(    (0, 0),      (0, 0),        3)
device_dict = {
    "sram": 0,      # SRAM (Shadow)
    "mac_array": 1, # SRAM-PIM
    "vec": 2,       # DRAM-PIM
    "dram": 3,      # DRAM (Shadow)
    "router": 5     # Router
}

# Input
_, trm_input = create_input(task_graph, Shape(nr=trm_qkv_len), Precision.INT_8, True)

# Q
q_output, task_q_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# Construct a hardware
config = CompAirPackageConfig("top/comp_air_package.toml")
package = CompAirPackageFactory.create_matrix(config)
chip_x = config.chiplet.size[0]
chip_y = config.chiplet.size[1]

# Construct a simulation environment
st_env = STEnv(task_graph, package)
split_num = 4
core_x = config.chiplet.size[0]
core_y = config.chiplet.size[1]

# Graph Transformation
task_q_splt_dict = st_env.split_mlp(split_inputs=[trm_input], weight=task_q_dict["weight"], \
                                    compute=task_q_dict["compute"], output=task_q_dict["output"], \
                                    split_vector=SplitVector(nf=split_num))

split_weights = st_env.split_task(task_q_dict["weight"].id, SplitVector(nf=split_num), True)
st_env.connect_tasks(split_weights, task_q_splt_dict["weight"])
task_q_dict["weight"].disable()

# Visualize
STDraw.draw_graph(task_graph, out_path='temp/mapped_q.task.html',
                  width='1920px', height='1080px')

# Print
print("[trm_input]", "I", trm_input.id)
print("[task_q_dict]", "C", task_q_dict["compute"].id, "O", task_q_dict["output"].id, "W", task_q_dict["weight"].id)
print("[Chiplet Number]", config.size, "[Cores/Chiplet]", config.chiplet.size)
print("[trm_input]", trm_input.output_edges[0].in_task, trm_input.output_edges[0].out_task, trm_input.output_edges[0].is_enable()) # Q
print("[trm_input]", len(trm_input.output_edges))
print("[task_q_splt_dict]")
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
    print(sub_q_key, len(sub_q_val_list), sub_q_val_list[0].shape, sub_q_val_list[0].id)

# Map Node
# TODO: ...

# Map Edge
# TODO: ...
exit()

print("------------     Simulate      ---------------")
st_env.simulate()
print("------------ Show Overall Time ---------------")
st_env.show_overall_time()
