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
trm_context_len = 6 * 1024
trm_head = 32
trm_qkv_len = 4096
trm_ffn_len = 11008
trm_output_size = 1024
cur_precision = Precision.UINT_4


def create_ffn(task_graph: TaskGraph, input: TaskBlock, inner_dim: int,
               precision: Precision, is_output: bool = False, task_dict: Dict = None):
    task_dict = {} if task_dict is None else task_dict
    task_dict["up"] = {}
    up_output, _ = create_mlp(task_graph, input, 
                              Shape(nr=input.shape.nf, nf=2 * inner_dim),
                              precision,
                              task_dict=task_dict["up"])
    task_dict["down"] = {}
    output, _ = create_mlp(task_graph, up_output, 
                           Shape(nf=input.shape.nf, nr=inner_dim),
                           precision, is_output,
                           task_dict=task_dict["down"])
    
    # Omit Hard-Product
    return output, task_dict

# Construct a task graph
task_graph = TaskGraph()

_, trm_input = create_input(task_graph, Shape(nr=trm_qkv_len), Precision.INT_8, True)

# Q
q_output, task_q_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# K
k_output, task_k_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# V
v_output, task_v_dict = create_mlp(task_graph, trm_input, Shape(nf=trm_qkv_len, nr=trm_qkv_len), \
                                 cur_precision, True)

# Q K
qk_output, task_qk_dict = create_mlp(task_graph, q_output, Shape(nf=trm_context_len, nr=trm_qkv_len), \
                                 Precision.INT_8, True)

# (QK^T) V
qkv_output, task_qkv_dict = create_mlp(task_graph, qk_output, Shape(nf=trm_qkv_len, nr=trm_context_len), \
                                 Precision.INT_8, True)

# FFN
# ffn_output, task_dict = create_ffn(task_graph, qkv_output, trm_ffn_len, cur_precision)


# Construct a hardware
config = HybridPackageConfig("top/hybrid_package.toml")
package = HybridPackageFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, package)

# Graph Transformation
# task_dict = st_env.split_mlp(split_inputs=[trm_input], weight=task_q_dict["weight"],
#                              compute=task_q_dict["compute"], output=task_q_dict["output"],
#                              split_vector=SplitVector(nf=2))

# Map the task graph onto the hardware
dram_coord = create_mlcoord((0, 0), (0, 0), 3)
st_env.put_in(dram_coord, task_q_dict["weight"].id)
flash_coord = create_mlcoord((0, 0), (1, 0), 4)
st_env.put_in(flash_coord, trm_input.id)
mac_array_coord = create_mlcoord((1, 0), (0, 0), 1)
st_env.put_in(mac_array_coord, task_q_dict["compute"].id)
sram_coord = create_mlcoord((1, 1), (0, 0), 0)
st_env.put_in(sram_coord, task_q_dict["output"].id)

# Edge Mapping
router_coord = create_mlcoord((0, 0), (1, 0), 5)
core_coord = create_mlcoord((0, 0), (15, 0))
chiplet_coord = create_mlcoord((1, 0))

st_env.map_edge(trm_input.output_edge, [flash_coord,  router_coord, core_coord, chiplet_coord, create_mlcoord((1, 0), (0, 0)), create_mlcoord((1, 0), (0, 0), 5), mac_array_coord])
st_env.map_edge()

st_env.simulate()
st_env.show_overall_time()