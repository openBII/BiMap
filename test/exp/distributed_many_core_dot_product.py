from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_cyclic_weight, create_pointwise, create_tiled_elementwise
import matplotlib.pyplot as plt
import toml
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType


# Algorithm Parameters
BATCH = 8
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/distributed_many_core_board.toml")
config = BoardConfig(config["PCB"])
SIZE_X, SIZE_Y = config.chiplet.size
CHIP = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0

# Construct Task Graph
task_graph = TaskGraph()
_, input_offchip = create_input(
    task_graph,
    Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=D_MODEL),
    Precision.FLOAT_16)
output, mlp_task_dict = create_tiled_mlp_cyclic_weight(
    split_vector=SplitVector(batch=8, token=16, nf=16),
    task_graph=task_graph,
    input=input_offchip,
    shape=Shape(nf=2048, nr=4096),
    precision=Precision.FLOAT_16,
    static_weight=False,
    output_offchip=False,
    is_output=False,
    is_input_on_chip=False
)
output, add_task_dict = create_tiled_elementwise(
    task_graph=task_graph,
    input=output,
    precision=Precision.FLOAT_16,
    type=TaskBlockType.CADD
)
output, softmax_task_dict = create_pointwise(
    task_graph=task_graph,
    input=output,
    precision=Precision.FLOAT_16,
    type=TaskBlockType.CSoftMax,
    is_output=True
)
STDraw.draw_graph(task_graph, out_path='temp/tiled_dot_product.task.html',
                    width='1920px', height='1080px')

# Create Hardware
many_core_board = BoardFactory.create_matrix(config, 
                                             BoardType.DISTRIBUTED_MANY_CORE)

# Create DSE E snvironment
env = STEnv(task_graph, many_core_board)

# Equivalent Hardware Parameter
many_core_board.communication_networks[0].update_bandwidth(
    config.network["bandwidth"] // (SIZE_X * SIZE_Y))

# Graph Transformation
# num_batch_split = 2
# num_token_split = 2
# split_embedding = env.split_task(embedding.id, 
#                                  SplitVector(batch=num_batch_split, 
#                                              token=num_token_split))
# env.add_nodes_between(embedding, embedding.out_tasks, 
#                       split_embedding)
# split_task_dict = env.split_attention(
#     task_dict=task_dict,
#     split_embedding=split_embedding,
#     query_split_vector=SplitVector(),
#     key_split_vector=SplitVector(),
#     value_split_vector=SplitVector(),
#     dot_product_split_vector=SplitVector(),
#     softmax_split_vector=SplitVector(),
#     attention_split_vector=SplitVector(),
#     attention_type=AttentionType.PREFILL,
#     num_batch_split=num_batch_split,
#     num_token_split=num_token_split
# )
# output.enable()
# env.connect_tasks(split_task_dict["attention"]["output"], [output])
# STDraw.draw_graph(task_graph, 
#                     out_path='temp/tiled_prefill_attention.task.html',
#                     width='1920px', height='1080px')

# Mapping
# Map inputs, outputs, and static data (weights, mask) into DRAM
sync_id1 = env.get_sync_id()
sync_id2 = env.get_sync_id()
dram = create_mlcoord(DRAM)
env.put_in(dram, input_offchip.id)
weight_offchip = mlp_task_dict["weight_offchip"]
env.put_in(dram, weight_offchip.id)
mask_offchip = add_task_dict["mask_offchip"]
env.sync(dram, sync_id2)
env.put_in(dram, mask_offchip.id)
env.put_in(dram, output.id)
# Map on-chip computations and data
local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
cyclic_local_memory = create_mlcoord(CHIP, (15, 0), SRAM_BUFFER)
input_on_chip = mlp_task_dict["input_on_chip"]
weight_on_chip = mlp_task_dict["weight_on_chip"]
cyclic_weight_on_chip = mlp_task_dict["cyclic_weight_on_chip"]
output_on_chip = mlp_task_dict["output_on_chip"]
compute = mlp_task_dict["compute"]
cyclic_compute = mlp_task_dict["cyclic_compute"]
env.put_in(local_memory, input_on_chip.id)
env.put_in(local_memory, weight_on_chip.id)
env.sync(cyclic_local_memory, sync_id1)
env.put_in(cyclic_local_memory, cyclic_weight_on_chip.id)
env.put_in(local_memory, output_on_chip.id)
tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, compute.id)
env.sync(tensor_unit, sync_id1)
env.put_in(tensor_unit, cyclic_compute.id)
env.sync(tensor_unit, sync_id2)
mask_on_chip = add_task_dict["mask_on_chip"]
add = add_task_dict["compute"]
add_output = add_task_dict["output"]
softmax = softmax_task_dict["compute"]
softmax_output = softmax_task_dict["output"]
vector_unit = create_mlcoord(CHIP, (0, 0), VECTOR_UNIT)
env.put_in(local_memory, mask_on_chip.id)
env.put_in(local_memory, add_output.id)
env.put_in(local_memory, softmax_output.id)
env.put_in(vector_unit, add.id)
env.put_in(vector_unit, softmax.id)

# Edge Mapping
env.auto_edge_map()

env.simulate()
env.show_overall_time()
print(env.get_latency(loops=LoopInfo(7, 8, 15)))
# latencies.append(env.get_latency())

# plt.plot([1, 2, 4, 8, 16], latencies)
# plt.savefig('temp/tiling.png')