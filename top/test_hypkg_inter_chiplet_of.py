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

# type(task_q_dict["weight"]) = StaticTaskBlock

# print("Model Shape Info (y、x、f<o-channel>、r<i-channel>、ky、kx)")
# print("[trm_input]", "I", trm_input.id)
# print("[task_q_dict]",  "C", task_q_dict["compute"].id, 
#                         "O", task_q_dict["output"].id, 
#                         "W", task_q_dict["weight"].id)

# Construct a hardware
config = HybridPackageConfig("top/hybrid_package.toml")
package = HybridPackageFactory.create_matrix(config)
print("[Chiplet Number]", config.size, "[Cores/Chiplet]", config.chiplet.size)

# Construct a simulation environment
st_env = STEnv(task_graph, package)
split_num = 24
chip_x = config.size[0]
chip_y = config.size[1]
core_x = config.chiplet.size[0]
core_y = config.chiplet.size[1]

# Graph Transformation
task_q_splt_dict = st_env.split_mlp(split_inputs=[trm_input], weight=task_q_dict["weight"], \
                                    compute=task_q_dict["compute"], output=task_q_dict["output"], \
                                    split_vector=SplitVector(nf=split_num))

split_weights = st_env.split_task(task_q_dict["weight"].id, SplitVector(nf=split_num), True)
st_env.connect_tasks(split_weights, task_q_splt_dict["weight"])
task_q_dict["weight"].disable()

# print("[trm_inpt]", trm_input.output_edges[0].in_task, \
#                     trm_input.output_edges[0].out_task, \
#                     trm_input.output_edges[0].is_enable())

# Visualize
STDraw.draw_graph(task_graph, out_path='temp/mapped_q.task.html',
                  width='1920px', height='1080px')

print("[trm_input]", len(trm_input.output_edges))
print("[task_q_splt_dict]")
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():
    print(sub_q_key, len(sub_q_val_list), sub_q_val_list[0].shape, sub_q_val_list[0].id, "~", sub_q_val_list[-1].id)

# Map Node
mac_array_coord_q = []
flash_coord_q = []
sram_coord_q = []

dram_w_coord_q = []
sram_w_coord_q = []
dram_i_coord_q = []

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
    print("dram_weight", split_weights[splict_id].id, chip_pos, core_pos)

for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):
            # Chiplet / Core ID ------------------------------
            core_id = splict_id % (core_x * core_y)
            chiplet_id = splict_id // (core_x * core_y)
            # ------------------------------------------------
            chip_pos = (chiplet_id // chip_x, chiplet_id % chip_x)
            core_pos = (core_id // core_x, core_id % core_x)
            # ------------------------------------------------
            mac_array_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["mac_array"]))
            st_env.put_in(mac_array_coord_q[-1], sub_q_val_list[splict_id].id)
            # ------------------------------------------------
            print("mac_array", sub_q_val_list[splict_id].id, chip_pos, core_pos)
            
    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):
            # Chiplet / Core ID ------------------------------
            core_id = splict_id % (core_x * core_y)
            chiplet_id = splict_id // (core_x * core_y)
            # ------------------------------------------------
            chip_pos = (chiplet_id // chip_x, chiplet_id % chip_x)
            core_pos = (core_id // core_x, core_id % core_x)
            # ------------------------------------------------
            sram_w_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["sram"]))
            st_env.put_in(sram_w_coord_q[-1], sub_q_val_list[splict_id].id)
            # ------------------------------------------------
            print("dram_w", sub_q_val_list[splict_id].id, chip_pos, core_pos)

    elif sub_q_key == "input":

        for splict_id in range(len(sub_q_val_list)):
            # Chiplet / Core ID ------------------------------
            core_id = splict_id % (core_x * core_y)
            chiplet_id = splict_id // (core_x * core_y)
            # ------------------------------------------------
            chip_pos = (chiplet_id // chip_x, chiplet_id % chip_x)
            core_pos = (core_id // core_x, core_id % core_x)
            # ------------------------------------------------
            dram_i_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["dram"])) 
            st_env.put_in(dram_i_coord_q[-1], sub_q_val_list[splict_id].id)
            print("dram_i", sub_q_val_list[splict_id].id, chip_pos, core_pos)

    elif sub_q_key == "mlp_output":

        for splict_id in range(len(sub_q_val_list)):
            # Chiplet / Core ID ------------------------------
            core_id = splict_id % (core_x * core_y)
            chiplet_id = splict_id // (core_x * core_y)
            # ------------------------------------------------
            chip_pos = (chiplet_id // chip_x, chiplet_id % chip_x)
            core_pos = (core_id // core_x, core_id % core_x)
            # ------------------------------------------------
            flash_coord_q.append(create_mlcoord(chip_pos, core_pos, device_dict["flash"]))
            st_env.put_in(flash_coord_q[-1], sub_q_val_list[splict_id].id)
            print("flash", sub_q_val_list[splict_id].id, chip_pos, core_pos)

    else:

        pass

# Map Edge
for sub_q_key, sub_q_val_list in task_q_splt_dict.items():

    if sub_q_key == "compute":

        for splict_id in range(len(sub_q_val_list)):
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [mac_array_coord_q[splict_id], flash_coord_q[splict_id]])
            print("[compute]", [mac_array_coord_q[splict_id], flash_coord_q[splict_id]])

    elif sub_q_key == "weight":

        for splict_id in range(len(sub_q_val_list)):

            # DRAM -> SRAM
            st_env.map_edge(split_weights[splict_id].output_edge, [dram_w_coord_q[splict_id], sram_w_coord_q[splict_id]])

            # SRAM -> MAC Array
            st_env.map_edge(sub_q_val_list[splict_id].output_edge, [sram_w_coord_q[splict_id], mac_array_coord_q[splict_id]])

            print("[weight-dram]", [dram_w_coord_q[splict_id], sram_w_coord_q[splict_id]])
            print("[weight-sram]", [sram_w_coord_q[splict_id], mac_array_coord_q[splict_id]])

    elif sub_q_key == "input":
        
        splict_id = 0
        map_edge_num = 0
        select_edge = []

        for i in range(split_num):
            select_edge.append(sub_q_val_list[0].output_edges[i])
            print("[sel]", select_edge[-1].is_enable(), select_edge[-1]._edge_id)

        for out_edge in select_edge:
            # Input Path
            
            send_pth = [dram_i_coord_q[0]]
            
            # Chiplet / Core ID ------------------------------
            dst_core_id = splict_id % (core_x * core_y)
            dst_chiplet_id = splict_id // (core_x * core_y)
            # ------------------------------------------------
            src_chip_pos = (0, 0)
            src_core_pos = (0, 0)
            dst_chip_pos = (dst_chiplet_id // chip_x, dst_chiplet_id % chip_x)
            dst_core_pos = (dst_core_id // core_x, dst_core_id % core_x)
            # ------------------------------------------------

            # [0. Cross-Core] src.device -> src.router
            src_router = create_mlcoord((0,0), (0,0), device_dict["router"])
            if (splict_id != 0): 
                send_pth.append(src_router)

            # ----------------------------------------------------------------
            
            # Cross Row Edge
            cross_row_edge_chiplet = (src_chip_pos[1] != dst_chip_pos[1])
            if (cross_row_edge_chiplet):

                # [1. Cross-Chiplet] src.router -> row_edge_core_tx
                row_edge_core_tx = create_mlcoord((0, 0), (0, (core_y - 1)))
                send_pth.append(row_edge_core_tx)
                
                # [2. Cross-Chiplet] row_edge_core_tx -> row_chiplet
                row_chiplet = create_mlcoord((0, 1))
                send_pth.append(row_chiplet)
                
                # [3. Cross-Chiplet] row_chiplet -> row_edge_core_rx
                row_edge_core_rx = create_mlcoord((0, 1), (0, 0))
                send_pth.append(row_edge_core_rx)

            # ----------------------------------------------------------------
            
            # Mid-Node
            same_row = (src_core_pos[0] == dst_core_pos[0]) and (src_chip_pos[0] == dst_chip_pos[0])
            same_col = (src_core_pos[1] == dst_core_pos[1]) and (src_chip_pos[1] == dst_chip_pos[1])
            if (same_row == False and same_col == False):
                
                if (cross_row_edge_chiplet):
                    mid_core = create_mlcoord((0, 1), (0, dst_core_pos[1]))
                else:
                    mid_core = create_mlcoord((0, 0), (0, dst_core_pos[1]))
                
                send_pth.append(mid_core)
            
            # ----------------------------------------------------------------
                
            # Cross Col Edge
            cross_col_edge_chiplet = (src_chip_pos[0] != dst_chip_pos[0])
            if (cross_col_edge_chiplet):

                # [1. Cross-Chiplet] src.router -> col_edge_core_tx
                if (cross_row_edge_chiplet):
                    col_edge_core_tx = create_mlcoord((src_chip_pos[0], 1), ((core_x - 1), dst_core_pos[1]))
                else:
                    col_edge_core_tx = create_mlcoord((src_chip_pos[0], 0), ((core_x - 1), dst_core_pos[1]))
                
                send_pth.append(col_edge_core_tx)
                
                # [2. Cross-Chiplet] col_edge_core_tx -> col_chiplet
                if (cross_row_edge_chiplet):
                    col_chiplet = create_mlcoord((dst_chip_pos[0], 1))
                else:
                    col_chiplet = create_mlcoord((dst_chip_pos[0], 0))
                send_pth.append(col_chiplet)
                
                # [3. Cross-Chiplet] col_chiplet -> col_edge_core_rx
                if (cross_row_edge_chiplet):
                    col_edge_core_rx = create_mlcoord((dst_chip_pos[0], 1), (0, dst_core_pos[1]))
                else:
                    col_edge_core_rx = create_mlcoord((dst_chip_pos[0], 0), (0, dst_core_pos[1]))
                send_pth.append(col_edge_core_rx)

            # ----------------------------------------------------------------

            dst_core = create_mlcoord(dst_chip_pos, dst_core_pos)
            if (splict_id != 0 and dst_core != send_pth[-1]): send_pth.append(dst_core)

            dst_router = create_mlcoord(dst_chip_pos, dst_core_pos, device_dict["router"])
            if (splict_id != 0): send_pth.append(dst_router)

            send_pth.append(mac_array_coord_q[splict_id])

            print("[input]", splict_id, "[Before map_edge]", out_edge.is_enable(), out_edge._edge_id)
            st_env.map_edge(out_edge, send_pth)
            print("[input]", splict_id, "[After map_edge]", out_edge.is_enable(), out_edge._edge_id)
            
            splict_id += 1

st_env.simulate()
st_env.show_overall_time()