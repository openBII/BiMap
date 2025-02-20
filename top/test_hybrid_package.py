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
from src.simulator.resource_simulator.config.matrix_config import ServerConfig, HybridPackageConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord, create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.resource_simulator.st_model.space_matrix.package_factory import HybridPackageFactory

# Parameter
trm_context_len = 8 * 1024
trm_head = 32
trm_qkv_len = 4096
trm_ffn_len = 11008
trm_output_size = 1024
cur_precision = Precision.UINT_4

# Construct a task graph
task_graph = TaskGraph()

# Map the task graph onto the hardware
#                <chiplet-id> - <core-id> - <device-id>
# create_mlcoord(    (0, 0),      (0, 0),        3)
device_dict = {
    "sram": 0,
    "core": 1,
    "dram": 3,
    "flash": 4
}

# Input
_, trm_input = create_input(task_graph, Shape(nr=trm_qkv_len), Precision.INT_8, True)

# Q
q_output, task_q_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# K
# k_output, task_k_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
#                                  cur_precision, True)

# V
# v_output, task_v_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
#                                  cur_precision, True)

# Q K
# qk_output, task_qk_dict = create_mlp(task_graph, q_output, Shape(nf=trm_context_len, nr=trm_qkv_len), \
#                                  Precision.INT_8, True)

# (QK^T) V
# qkv_output, task_qkv_dict = create_mlp(task_graph, qk_output, Shape(nf=trm_qkv_len, nr=trm_context_len), \
#                                  Precision.INT_8, True)

# O
# o_output, task_o_dict = create_mlp(task_graph, qkv_output, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
#                                  cur_precision, True)

# FFN (Up & Gate)
# ffn1_output, task_ffn1_dict = create_mlp(task_graph, o_output, 
#                               Shape(nr=o_output.shape.nf, nf=2 * trm_ffn_len),
#                               cur_precision, None)

# FFN (Down)
# ffn2_output, task_ffn2_dict = create_mlp(task_graph, qkv_output, 
#                               Shape(nr=qkv_output.shape.nf, nf=trm_ffn_len),
#                               cur_precision, None)

# Visualize
STDraw.draw_graph(task_graph, out_path='temp/mapped_mlp.task.html',
                  width='1920px', height='1080px')

# type(task_q_dict["weight"]) = StaticTaskBlock
print("Model Shape Info (y、x、f<o-channel>、r<i-channel>、ky、kx)")
print("[trm_input]", "I", trm_input.id)
print("[task_q_dict]", "C", task_q_dict["compute"].id, "O", task_q_dict["output"].id, "W", task_q_dict["weight"].id)
# print("[task_k_dict]", "C", task_k_dict["compute"].id, "O", task_k_dict["output"].id, "W", task_k_dict["weight"].id)
# print("[task_v_dict]", "C", task_v_dict["compute"].id, "O", task_v_dict["output"].id, "W", task_v_dict["weight"].id)
# print("[task_o_dict]", "C", task_o_dict["compute"].id, "O", task_o_dict["output"].id, "W", task_o_dict["weight"].id)
# print("[task_qk_dict]", "C", task_qk_dict["compute"].id, "O", task_qk_dict["output"].id, "W", task_qk_dict["weight"].id)
# print("[task_qkv_dict]", "C", task_qkv_dict["compute"].id, "O", task_qkv_dict["output"].id, "W", task_qkv_dict["weight"].id)
# print("[task_ffn1_dict]", "C", task_ffn1_dict["compute"].id, "O", task_ffn1_dict["output"].id, "W", task_ffn1_dict["weight"].id)
# print("[task_ffn2_dict]", "C", task_ffn2_dict["compute"].id, "O", task_ffn2_dict["output"].id, "W", task_ffn2_dict["weight"].id)

# Construct a hardware
config = HybridPackageConfig("top/hybrid_package.toml")
package = HybridPackageFactory.create_matrix(config)
pe_x = config.chiplet.core.mac_array["int8"]["parallelism"][0]
pe_y = config.chiplet.core.mac_array["int8"]["parallelism"][1]
pe_t = config.chiplet.core.mac_array["int8"]["latency"]
print("[Chiplet Number]", config.size, "[Cores/Chiplet]", config.chiplet.size, "[PEs/Core]", pe_x, pe_y, pe_t)

# Construct a simulation environment
st_env = STEnv(task_graph, package)

# Graph Transformation
# task_q_splt_dict = st_env.split_mlp(split_inputs=[trm_input], weight=task_q_dict["weight"], \
#                              compute=task_q_dict["compute"], output=task_q_dict["output"], \
#                              split_vector=SplitVector(nf=(task_q_dict["compute"].shape.nf // pe_x), \
#                                                       nr=(task_q_dict["compute"].shape.nr // pe_y)))
# print("[task_q_splt_dict]")
# for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
#     print(sub_q_key, len(sub_q_val_list), sub_q_val_list[0].shape, sub_q_val_list[0].id)

dram_coord = create_mlcoord((0, 0), (0, 0), device_dict["dram"])
st_env.put_in(dram_coord, task_q_dict["weight"].id)

flash_coord = create_mlcoord((0, 0), (0, 0), device_dict["flash"])
st_env.put_in(flash_coord, trm_input.id)

sram_coord = create_mlcoord((0, 0), (0, 0), device_dict["sram"])
st_env.put_in(sram_coord, task_q_dict["output"].id)

mac_array_coord = create_mlcoord((0, 0), (0, 0), device_dict["core"])
st_env.put_in(mac_array_coord, task_q_dict["compute"].id)

# router_0_0_coord = create_mlcoord((0, 0), (0, 0), 5)

# Edges
# env.auto_edge_map()
print("[trm_input]", trm_input.output_edges[0].in_task, trm_input.output_edges[0].out_task) # V
# print("[trm_input]", trm_input.output_edges[1].in_task, trm_input.output_edges[1].out_task) # K
# print("[trm_input]", trm_input.output_edges[2].in_task, trm_input.output_edges[2].out_task) # Q

st_env.map_edge(trm_input.output_edges[0], [flash_coord, mac_array_coord])
st_env.map_edge(task_q_dict["weight"].output_edge, [dram_coord, mac_array_coord])
st_env.map_edge(task_q_dict["compute"].output_edge, [mac_array_coord, sram_coord])

st_env.simulate()
st_env.show_overall_time()

exit()

# Example Core
dram_coord = create_mlcoord((0, 0), (0, 0), device_dict["dram"])
st_env.put_in(dram_coord, task_q_dict["weight"].id)

flash_coord = create_mlcoord((0, 0), (1, 0), device_dict["flash"])
st_env.put_in(flash_coord, trm_input.id)

mac_array_coord = create_mlcoord((1, 0), (0, 0), device_dict["core"])
st_env.put_in(mac_array_coord, task_q_dict["compute"].id)

sram_coord = create_mlcoord((1, 1), (0, 0), device_dict["sram"])
st_env.put_in(sram_coord, task_q_dict["output"].id)

router_coord = create_mlcoord((0, 0), (1, 0), 5)
core_coord = create_mlcoord((0, 0), (15, 0))
chiplet_coord = create_mlcoord((1, 0))

st_env.map_edge(trm_input.output_edge, [flash_coord,  router_coord, core_coord, \
                                        chiplet_coord, create_mlcoord((1, 0), (0, 0)), \
                                        create_mlcoord((1, 0), (0, 0), 5), mac_array_coord])