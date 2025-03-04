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
    "sram": 0,
    "mac_array": 1,
    "vec": 2,
    "dram": 3,
    "flash": 4,
    "router": 5
}

# Input
_, trm_input = create_input(task_graph, Shape(nr=trm_qkv_len), Precision.INT_8, True)

# Q
q_output, task_q_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# type(task_q_dict["weight"]) = StaticTaskBlock
print("Model Shape Info (y、x、f<o-channel>、r<i-channel>、ky、kx)")
print("[trm_input]", "I", trm_input.id)
print("[task_q_dict]", "C", task_q_dict["compute"].id, "O", task_q_dict["output"].id, "W", task_q_dict["weight"].id)

# Construct a hardware
config = HybridPackageConfig("top/hybrid_package.toml")
package = CompAirPackageFactory.create_matrix(config)
chip_x = config.chiplet.size[0]
chip_y = config.chiplet.size[1]
print("[Chiplet Number]", config.size, "[Cores/Chiplet]", config.chiplet.size)

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

print("[trm_input]", trm_input.output_edges[0].in_task, trm_input.output_edges[0].out_task, trm_input.output_edges[0].is_enable()) # Q

# Visualize
STDraw.draw_graph(task_graph, out_path='temp/mapped_q.task.html',
                  width='1920px', height='1080px')

print("[trm_input]", len(trm_input.output_edges))
print("[task_q_splt_dict]")
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
    print(sub_q_key, len(sub_q_val_list), sub_q_val_list[0].shape, sub_q_val_list[0].id)

# Map Node
mac_array_coord_q = []
flash_coord_q = []
sram_coord_q = []

dram_w_coord_q = []
sram_w_coord_q = []
dram_i_coord_q = []

for splict_id in range(len(split_weights)):
    chip_pos = (0, 0)
    core_pos = (splict_id // core_x, splict_id % core_x)
    dram_w_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["dram"]))
    st_env.put_in(dram_w_coord_q[-1], split_weights[splict_id].id)
    print("dram_w", split_weights[splict_id].id)

for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            mac_array_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["mac_array"]))
            st_env.put_in(mac_array_coord_q[-1], sub_q_val_list[splict_id].id)
            print("mac_array", sub_q_val_list[splict_id].id)
            
    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            sram_w_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["sram"]))
            st_env.put_in(sram_w_coord_q[-1], sub_q_val_list[splict_id].id)
            print("sram_w", sub_q_val_list[splict_id].id)

    elif sub_q_key == "input":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            dram_i_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["sram"])) 
            st_env.put_in(dram_i_coord_q[-1], sub_q_val_list[splict_id].id)
            print("dram_i", sub_q_val_list[splict_id].id)

    elif sub_q_key == "mlp_output":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            flash_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["flash"]))
            st_env.put_in(flash_coord_q[-1], sub_q_val_list[splict_id].id)
            print("flash", sub_q_val_list[splict_id].id)

    else:

        pass

# Map Edge
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [mac_array_coord_q[splict_id], flash_coord_q[splict_id]])

    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):

            # DRAM -> SRAM
            st_env.map_edge(split_weights[splict_id].output_edge, [dram_w_coord_q[splict_id], sram_w_coord_q[splict_id]])

            # SRAM -> MAC Array
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [sram_w_coord_q[splict_id], mac_array_coord_q[splict_id]])

    elif sub_q_key == "input":
        
        splict_id = 0
        map_edge_num = 0
        select_edge = []

        for i in range(split_num):
            select_edge.append(sub_q_val_list[0].output_edges[i])

        for out_edge in select_edge:
            # Input Path
            src_router = create_mlcoord((0,0), (0,0), device_dict["router"])
            dst_pos = (splict_id // core_x, splict_id % core_x)
            dst_core = create_mlcoord((0,0), dst_pos)
            dst_router = create_mlcoord((0,0), dst_pos, device_dict["router"])


            # st_env.map_edge(out_edge, send_pth)
            # print("[input]", splict_id, "[Before map_edge]", out_edge.is_enable(), out_edge._edge_id)
            
            if splict_id == 0:
                st_env.map_edge(out_edge, [dram_i_coord_q[0], mac_array_coord_q[splict_id]])
            else:
                st_env.map_edge(out_edge, [dram_i_coord_q[0], src_router, dst_core, dst_router, mac_array_coord_q[splict_id]])
            
            # print("[input]", splict_id, "[After map_edge]", out_edge.is_enable(), out_edge._edge_id)
            
            splict_id += 1

st_env.simulate()
print("------------ show_overall_time ---------------")
st_env.show_overall_time()
exit()

# --------------------------------------------------

dram_input_q = create_mlcoord((0, 0), (0, 0), device_dict["flash"])
st_env.put_in(dram_input_q, trm_input.id)

dram_coord_q = create_mlcoord((0, 0), (0, 1), device_dict["dram"])
st_env.put_in(dram_coord_q, task_q_dict["weight"].id)

sram_coord_q = create_mlcoord((0, 0), (0, 1), device_dict["sram"])
st_env.put_in(sram_coord_q, task_q_dict["output"].id)

mac_array_coord_q = create_mlcoord((0, 0), (0, 1), device_dict["mac_array"])
st_env.put_in(mac_array_coord_q, task_q_dict["compute"].id)

# Map Edges

print("[trm_input]", trm_input.output_edges[0].in_task, trm_input.output_edges[0].out_task, trm_input.output_edges[0].is_enable()) # Q

# I->Q: Remote
src_router = create_mlcoord((0, 0), (0,0), device_dict["router"])
dst_core = create_mlcoord((0, 0), (0, 1))
dst_router = create_mlcoord((0, 0), (0, 1), device_dict["router"])

st_env.map_edge(trm_input.output_edges[0], [dram_input_q, src_router, dst_core, dst_router, mac_array_coord_q])
st_env.map_edge(task_q_dict["weight"].output_edge, [dram_coord_q, mac_array_coord_q])
st_env.map_edge(task_q_dict["compute"].output_edge, [mac_array_coord_q, sram_coord_q])

st_env.simulate()
st_env.show_overall_time()
exit()

# --------------------------------------------------

dram_input_q = create_mlcoord((0, 0), (0, 0), device_dict["flash"])
st_env.put_in(dram_input_q, trm_input.id)

dram_coord_q = create_mlcoord((0, 0), (0, 0), device_dict["dram"])
st_env.put_in(dram_coord_q, task_q_dict["weight"].id)

sram_coord_q = create_mlcoord((0, 0), (0, 0), device_dict["sram"])
st_env.put_in(sram_coord_q, task_q_dict["output"].id)

mac_array_coord_q = create_mlcoord((0, 0), (0, 0), device_dict["mac_array"])
st_env.put_in(mac_array_coord_q, task_q_dict["compute"].id)

# Map Edges

print("[trm_input]", trm_input.output_edges[0].in_task, trm_input.output_edges[0].out_task, trm_input.output_edges[0].is_enable()) # Q

# I->Q: Local
st_env.map_edge(trm_input.output_edges[0], [dram_input_q, mac_array_coord_q])
st_env.map_edge(task_q_dict["weight"].output_edge, [dram_coord_q, mac_array_coord_q])
st_env.map_edge(task_q_dict["compute"].output_edge, [mac_array_coord_q, sram_coord_q])

st_env.simulate()
st_env.show_overall_time()

# --------------------------------------------------

# Example Core
# dram_coord = create_mlcoord((0, 0), (0, 0), device_dict["dram"])
# st_env.put_in(dram_coord, task_q_dict["weight"].id)

# flash_coord = create_mlcoord((0, 0), (1, 0), device_dict["flash"])
# st_env.put_in(flash_coord, trm_input.id)

# mac_array_coord = create_mlcoord((1, 0), (0, 0), device_dict["mac_array"])
# st_env.put_in(mac_array_coord, task_q_dict["compute"].id)

# sram_coord = create_mlcoord((1, 1), (0, 0), device_dict["sram"])
# st_env.put_in(sram_coord, task_q_dict["output"].id)

# router_coord = create_mlcoord((0, 0), (1, 0), 5)
# core_coord = create_mlcoord((0, 0), (15, 0))
# chiplet_coord = create_mlcoord((1, 0))

# st_env.map_edge(trm_input.output_edge, [flash_coord,  router_coord, core_coord, \
#                                         chiplet_coord, create_mlcoord((1, 0), (0, 0)), \
#                                         create_mlcoord((1, 0), (0, 0), 5), mac_array_coord])