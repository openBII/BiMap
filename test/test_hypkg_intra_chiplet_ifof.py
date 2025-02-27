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
    "mac_array": 1,
    "vec": 2,
    "dram": 3,
    "flash": 4,
    "router": 5
}

# Input
_, trm_input = create_input(task_graph, Shape(nr=trm_qkv_len), Precision.INT_8, True)

q_output, task_q_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# Construct a hardware
config = HybridPackageConfig("top/hybrid_package.toml")
package = HybridPackageFactory.create_matrix(config)
print("[Chiplet Number]", config.size, "[Cores/Chiplet]", config.chiplet.size)

# Construct a simulation environment
st_env = STEnv(task_graph, package)

# [nf(o), nr(i)]
split_num = [8, 7]

chip_x = config.size[0]
chip_y = config.size[1]
core_x = config.chiplet.size[0]
core_y = config.chiplet.size[1]

# Graph Transformation
task_q_splt_dict = st_env.split_mlp(split_inputs=[trm_input], weight=task_q_dict["weight"], \
                                    compute=task_q_dict["compute"], output=task_q_dict["output"], \
                                    split_vector=SplitVector(nf=split_num[0], nr=split_num[1]))

split_weights = st_env.split_task(task_q_dict["weight"].id, SplitVector(nf=split_num[0], nr=split_num[1]), True)
st_env.connect_tasks(split_weights, task_q_splt_dict["weight"])
task_q_dict["weight"].disable()

# Visualize
STDraw.draw_graph(task_graph, out_path='temp/mapped_q_if.task.html',
                  width='1920px', height='1080px')

print("[task_q_splt_dict]")
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
    print(sub_q_key, len(sub_q_val_list), sub_q_val_list[0].shape, sub_q_val_list[0].id, "~", sub_q_val_list[-1].id)

# Map Node
mac_array_coord_q = []
vec_add_coord_q = []
flash_coord_q = []
sram_mlp_o_coord_q = []

dram_w_coord_q = []
sram_w_coord_q = []
dram_i_coord_q = []

comp_mem_map_dict = {}

print("------------------------ Node Mapping ------------------------")

for splict_id in range(len(split_weights)):
    # Chiplet / Core ID ------------------------------
    core_id = splict_id % (core_x * core_y)
    chiplet_id = splict_id // (core_x * core_y)
    # ------------------------------------------------
    chip_pos = (chiplet_id // chip_x, chiplet_id % chip_x)
    core_pos = (core_id // core_x, core_id % core_x)
    # ------------------------------------------------
    dram_w_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["dram"]))
    st_env.put_in(dram_w_coord_q[-1], split_weights[splict_id].id)
    # ------------------------------------------------
    print("dram - weight", split_weights[splict_id].id, chip_pos, core_pos)

# First Map Compute Node / Weight(S) Node / Mlp Out Node / Add Node
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            mac_array_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["mac_array"]))
            st_env.put_in(mac_array_coord_q[-1], sub_q_val_list[splict_id].id)
            print("mac_array - compute", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            sram_w_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["sram"]))
            st_env.put_in(sram_w_coord_q[-1], sub_q_val_list[splict_id].id)
            print("sram - weight", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    elif sub_q_key == "mlp_output":

        for splict_id in range(len(sub_q_val_list)):
            chip_pos = (0, 0)
            core_pos = (splict_id // core_x, splict_id % core_x)
            sram_mlp_o_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["sram"]))
            st_env.put_in(sram_mlp_o_coord_q[-1], sub_q_val_list[splict_id].id)
            print("sram - mlp_output", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    elif sub_q_key == "add":

        num_comp_cores = split_num[0] * split_num[1]

        for splict_id in range(len(sub_q_val_list)):
            map_id = num_comp_cores + splict_id
            chip_pos = (0, 0)
            core_pos = (map_id // core_x, map_id % core_x)
            vec_add_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["vec"])) 
            st_env.put_in(vec_add_coord_q[-1], sub_q_val_list[splict_id].id)
            
            # Debug Code:
            i_edges_list = [] 
            for i_edge in sub_q_val_list[splict_id]._input_edges:
                i_edges_list.append(i_edge._in_task._id)

            print("vec - add", sub_q_val_list[splict_id].id, chip_pos, core_pos, i_edges_list)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    else:

        pass

# Then Map input / 
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
            
    if sub_q_key == "output":

        for splict_id in range(len(sub_q_val_list)):

            # Get the pos from the vec_add_coord_q
            chip_pos = vec_add_coord_q[splict_id][0]
            core_pos = vec_add_coord_q[splict_id][1]

            flash_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["flash"]))
            st_env.put_in(flash_coord_q[-1], sub_q_val_list[splict_id].id)
            print("add_output (= final_output) - weight", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    elif sub_q_key == "input":

        for splict_id in range(len(sub_q_val_list)):
            
            # Get the pos from the output of input task
            task_id = sub_q_val_list[splict_id]._output_edges[-1]._out_task._id
            chip_pos = comp_mem_map_dict[task_id][0]
            core_pos = comp_mem_map_dict[task_id][1]

            dram_i_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["dram"]))
            st_env.put_in(dram_i_coord_q[-1], sub_q_val_list[splict_id].id)
            print("dram_i - input", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            comp_mem_map_dict.update({int(sub_q_val_list[splict_id].id) : [chip_pos, core_pos]})

    else:

        pass

print("------------------------ Edge Mapping ------------------------")

# Map Edge
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):

            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [mac_array_coord_q[splict_id], sram_mlp_o_coord_q[splict_id]])
            dst_task_id = sub_q_val_list[splict_id].output_edge._out_task._id
            src_task_id = sub_q_val_list[splict_id].output_edge._in_task._id

            print("Connect compute (Write-back)", src_task_id, "->", dst_task_id,
                  [mac_array_coord_q[splict_id], sram_mlp_o_coord_q[splict_id]])

    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):

            # DRAM -> SRAM
            st_env.map_edge(split_weights[splict_id].output_edge, [dram_w_coord_q[splict_id], sram_w_coord_q[splict_id]])
            # SRAM -> MAC Array
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [sram_w_coord_q[splict_id], mac_array_coord_q[splict_id]])

            print("Connect weight (D-S-M)", dram_w_coord_q[splict_id], "->", sram_w_coord_q[splict_id], "->", mac_array_coord_q[splict_id])


    elif sub_q_key == "input":
        
        select_edge = []
        for splict_id in range(len(sub_q_val_list)):
            for i in range(split_num[0]):
                select_edge.append(sub_q_val_list[splict_id].output_edges[i])

        for out_edge in select_edge:

            dst_task_id = out_edge._out_task._id
            src_task_id = out_edge._in_task._id

            src_chip = comp_mem_map_dict[src_task_id][0]
            src_core = comp_mem_map_dict[src_task_id][1]
            dst_chip = comp_mem_map_dict[dst_task_id][0]
            dst_core = comp_mem_map_dict[dst_task_id][1]

            if (src_chip == dst_chip and src_core == dst_core):
                
                path = [create_mlcoord(src_chip, src_core, device_dict["dram"]), create_mlcoord(dst_chip, dst_core, device_dict["mac_array"])]
                
                st_env.map_edge(out_edge, path)
                print("Mac [Local]", src_task_id, "->", dst_task_id, path)

            else:

                src_router = create_mlcoord(src_chip, src_core, device_dict["router"])
                dst_pos = create_mlcoord(dst_chip, dst_core)
                dst_router = create_mlcoord(dst_chip, dst_core, device_dict["router"])
                path = [create_mlcoord(src_chip, src_core, device_dict["dram"]), src_router, dst_pos, dst_router, create_mlcoord(dst_chip, dst_core, device_dict["mac_array"])]

                st_env.map_edge(out_edge, path)
                print("Mac [Remote]", src_task_id, "->", dst_task_id, path)
    
    elif sub_q_key == "add":

        for splict_id in range(len(sub_q_val_list)):
            dst_task_id = sub_q_val_list[splict_id].output_edge._out_task._id
            src_task_id = sub_q_val_list[splict_id].output_edge._in_task._id
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [vec_add_coord_q[splict_id], flash_coord_q[splict_id]])
            print("Add out (Write-back)", src_task_id, "->", dst_task_id, \
                  [vec_add_coord_q[splict_id], flash_coord_q[splict_id]])

    elif sub_q_key == "mlp_output":

        select_edge = []
        for splict_id in range(len(sub_q_val_list)):
            select_edge.append(sub_q_val_list[splict_id].output_edges[0])

        for out_edge in select_edge:

            dst_task_id = out_edge._out_task._id
            src_task_id = out_edge._in_task._id

            src_chip = comp_mem_map_dict[src_task_id][0]
            src_core = comp_mem_map_dict[src_task_id][1]
            dst_chip = comp_mem_map_dict[dst_task_id][0]
            dst_core = comp_mem_map_dict[dst_task_id][1]

            if (src_chip == dst_chip and src_core == dst_core):
                path = [create_mlcoord(src_chip, src_core, device_dict["sram"]), create_mlcoord(dst_chip, dst_core, device_dict["vec"])]
                st_env.map_edge(out_edge, path)
                print("Mlp-out -> Add [Local]", src_task_id, "->", dst_task_id, path)
            else:
                src_router = create_mlcoord(src_chip, src_core, device_dict["router"])
                dst_pos = create_mlcoord(dst_chip, dst_core)
                dst_router = create_mlcoord(dst_chip, dst_core, device_dict["router"])
                path = [create_mlcoord(src_chip, src_core, device_dict["sram"]), src_router, dst_pos, dst_router, create_mlcoord(dst_chip, dst_core, device_dict["vec"])]
                st_env.map_edge(out_edge, path)
                print("Mlp-out -> Add [Remote]", src_task_id, "->", dst_task_id, path)

    else:

        pass

print("------------------------ Run Mapping ------------------------")

st_env.simulate()
st_env.show_overall_time()