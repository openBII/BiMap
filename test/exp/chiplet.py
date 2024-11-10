import toml
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_env import LoopInfo, STEnv
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.transformer import create_input, create_pointwise, create_tiled_data_arrange_16_to_1_broadcast, create_tiled_mlp_decoder_weight
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.resource_simulator.st_model.space_matrix.board_factory import BoardFactory, BoardType
from src.simulator.resource_simulator.config.matrix_config import BoardConfig


BATCH = 8
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
# Initial Hardware Configuration
config = toml.load("top/distributed_many_core_chiplet2.toml")
config = BoardConfig(config["PCB"])
CHIPLET_SIZE_X, CHIPLET_SIZE_Y = config.package.size
CORE_SIZE_X, CORE_SIZE_Y = config.package.chiplet.size
PACKAGE = 0
DRAM = 2
TENSOR_UNIT = 1
VECTOR_UNIT = 2
ROUTER = 3
SRAM_BUFFER = 0

# Construct Task Graph
IDGenerator.set_base_task_id(0)
task_graph = TaskGraph()
_, input_offchip = create_input(
    task_graph,
    Shape(batch=BATCH // 8, token=1, nf=D_MODEL),
    Precision.FLOAT_16)
# calculate Q,K,V. Loop-3.
output, mlp_task_dict = create_tiled_mlp_decoder_weight(
    split_vector=SplitVector(batch=8, token=1, nf=16),
    task_graph=task_graph,
    input=input_offchip,
    shape=Shape(nf=4096, nr=4096),
    precision=Precision.FLOAT_16,
    is_output=False,
    static_weight=True,
    output_offchip=False,
    is_input_on_chip=False
)
# re-arragne output data together
output, arrange_q_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
    task_graph=task_graph,
    input=output,
    precision=Precision.FLOAT_16
)
# Q * K
output, dot_product_task_dict = create_tiled_mlp_decoder_weight(
    split_vector=SplitVector(batch=8, token=1, nf=16),
    task_graph=task_graph,
    input=output,
    shape=Shape(nf=2048, nr=4096),
    precision=Precision.FLOAT_16,
    is_output=False,
    static_weight=False,
    output_offchip=False,
    is_input_on_chip=True,
)
# re-arragne output data together
output, arrange_qk_task_dict = create_tiled_data_arrange_16_to_1_broadcast(
    task_graph=task_graph,
    input=output,
    precision=Precision.FLOAT_16
)
# Softmax
output, softmax_task_dict = create_pointwise(
    task_graph=task_graph,
    input=output,
    precision=Precision.FLOAT_16,
    type=TaskBlockType.CSoftMax,
)
output, attention_task_dict = create_tiled_mlp_decoder_weight(
    split_vector=SplitVector(batch=8, token=1, nf=16),
    task_graph=task_graph,
    input=output,
    shape=Shape(nf=4096, nr=2048),
    precision=Precision.FLOAT_16,
    is_output=True,
    output_offchip=True
)
STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
                    width='1920px', height='1080px')

# Create Hardware
config = toml.load("top/distributed_many_core_chiplet2.toml")
config = BoardConfig(config["PCB"], config["process_node"])
chiplet_board = BoardFactory.create_matrix(config, 
                                           BoardType.CHIPLET)

# Create DSE snvironment
env = STEnv(task_graph, chiplet_board)

# # Equivalent Hardware Parameter
# chiplet_board.communication_networks[0].update_bandwidth(
#     config.network["bandwidth"] // (SIZE_X * SIZE_Y))

# Mapping
# Query, Key, Value
# DRAM
dram = create_mlcoord(DRAM)

local_memory = create_mlcoord(PACKAGE, (0, 0), (0, 0), SRAM_BUFFER)
arrange_q_arrange_memory = create_mlcoord(PACKAGE, (1, 0) (7, 0), SRAM_BUFFER)
arrange_q_local_memory = create_mlcoord(PACKAGE, (0, 0), (0, 0), SRAM_BUFFER)
arrange_q_arrange_memory = create_mlcoord(PACKAGE, (1, 0), (7, 0), SRAM_BUFFER)
arrange_qk_local_memory = create_mlcoord(PACKAGE, (0, 0), (0, 0), SRAM_BUFFER)
arrange_qk_arrange_memory = create_mlcoord(PACKAGE, (1, 0), (7, 0), SRAM_BUFFER)
sync_id1 = env.get_sync_id()
sync_id2 = env.get_sync_id()
sync_id3 = env.get_sync_id()

# Q,K,V
env.put_in(dram, input_offchip.id)
env.put_in(local_memory, mlp_task_dict["input_on_chip"].id)
env.put_in(dram, mlp_task_dict["weight_offchip"].id)
env.put_in(local_memory, mlp_task_dict["weight_on_chip"].id) 
tensor_unit = create_mlcoord(PACKAGE, (0, 0), (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, mlp_task_dict["compute"].id)

# sync 1
env.sync(tensor_unit, sync_id1)
env.sync(arrange_q_local_memory, sync_id1)
env.sync(dram, sync_id1)

# re-arrange Q
env.put_in(local_memory, mlp_task_dict["output_on_chip"].id)
for i in range(15):
    env.put_in(arrange_q_local_memory, arrange_q_task_dict["weight{}".format(i)].id)
env.put_in(arrange_q_arrange_memory, arrange_q_task_dict["data_merged"].id)
env.put_in(local_memory, arrange_q_task_dict["output"].id)

# Q * K
env.put_in(dram, dot_product_task_dict["weight_offchip"].id)
env.put_in(local_memory, dot_product_task_dict["weight_on_chip"].id)
tensor_unit = create_mlcoord(PACKAGE, (0, 0), (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, dot_product_task_dict["compute"].id)

# sync 2
env.sync(tensor_unit, sync_id2)
env.sync(arrange_qk_local_memory, sync_id2)

# re-arrange Q*K result
env.put_in(local_memory, dot_product_task_dict["output_on_chip"].id)
for i in range(15):
    env.put_in(arrange_qk_local_memory, arrange_qk_task_dict["weight{}".format(i)].id)
env.put_in(arrange_qk_arrange_memory, arrange_qk_task_dict["data_merged"].id)
env.put_in(local_memory, arrange_qk_task_dict["output"].id)

# softmax
vector_unit = create_mlcoord(PACKAGE, (0, 0), (0, 0), VECTOR_UNIT)
env.put_in(vector_unit, softmax_task_dict["compute"].id)

# sync 3
env.sync(vector_unit, sync_id3)
env.sync(dram, sync_id3)

# result * V
env.put_in(local_memory, softmax_task_dict["output"].id)
env.put_in(dram, attention_task_dict["weight_offchip"].id)
env.put_in(local_memory, attention_task_dict["weight_on_chip"].id)
env.put_in(local_memory, attention_task_dict["output_on_chip"].id)
tensor_unit = create_mlcoord(PACKAGE, (0, 0), (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, attention_task_dict["compute"].id)
env.put_in(dram, attention_task_dict["output_offchip"].id)

# Edge Mapping
env.auto_edge_map()

env.simulate()
# env.show_overall_time()
overall_latency = env.get_latency(loops=[
    LoopInfo(input_offchip.id, arrange_q_task_dict["output"].id, 2)     # Q, K, V Loop
    ])

print(overall_latency)