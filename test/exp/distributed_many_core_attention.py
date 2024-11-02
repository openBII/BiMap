from src.simulator.resource_simulator.st_env import STEnv, LoopInfo
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import BoardConfig
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_prefill_attention, create_input, AttentionType, create_tiled_mlp_cyclic_weight
import matplotlib.pyplot as plt
import toml


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
ROUTER = 3
SRAM_BUFFER = 0

# Construct Task Graph
task_graph = TaskGraph()
input = create_input(
    task_graph,
    Shape(batch=BATCH // 8, token=SEQ_LEN // 16, nf=2048),
    Precision.FLOAT_16,
    generate_data=False)
output, task_dict = create_tiled_mlp_cyclic_weight(
    split_vector=SplitVector(batch=8, token=16, nf=16),
    task_graph=task_graph,
    input=input,
    shape=Shape(nf=4096, nr=2048),
    precision=Precision.FLOAT_16,
    is_input_on_chip=False,
    is_output=True,
    output_offchip=True
)
STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
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
# Map inputs, outputs, and weights into DRAM
sync_id = env.get_sync_id()
dram = create_mlcoord(DRAM)
env.put_in(dram, output.id)
weight_offchip = task_dict["weight_offchip"]
env.put_in(dram, weight_offchip.id)
# Map on-chip computations and data
local_memory = create_mlcoord(CHIP, (0, 0), SRAM_BUFFER)
cyclic_local_memory = create_mlcoord(CHIP, (1, 0), SRAM_BUFFER)
input_on_chip = task_dict["input_on_chip"]
weight_on_chip = task_dict["weight_on_chip"]
cyclic_weight_on_chip = task_dict["cyclic_weight_on_chip"]
output_on_chip = task_dict["output_on_chip"]
compute = task_dict["compute"]
cyclic_compute = task_dict["cyclic_compute"]
env.put_in(local_memory, input_on_chip.id)
env.put_in(local_memory, weight_on_chip.id)
env.sync(cyclic_local_memory, sync_id)
env.put_in(cyclic_local_memory, cyclic_weight_on_chip.id)
env.put_in(local_memory, output_on_chip.id)
tensor_unit = create_mlcoord(CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, compute.id)
env.sync(tensor_unit, sync_id)
env.put_in(tensor_unit, cyclic_compute.id)

# Edge Mapping
env.auto_edge_map()

env.simulate()
env.show_overall_time()
print(env.get_latency(loops=LoopInfo(6, 7, 15)))
# latencies.append(env.get_latency())

# plt.plot([1, 2, 4, 8, 16], latencies)
# plt.savefig('temp/tiling.png')