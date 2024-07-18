from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_attention, create_input
import matplotlib.pyplot as plt

SEQ_LEN = 2048
D_KEY = 4096
D_VALUE = 4096

# latencies = []

# Construct a task graph
task_graph = TaskGraph()
IDGenerator.set_base_task_id(task_graph)
input, embedding = create_input(task_graph, Shape(nf=4096), Precision.FLOAT_16)
input_key_cache = create_input(task_graph, Shape(nf=SEQ_LEN - 1, nr=D_KEY), 
                            Precision.FLOAT_16, False)
input_value_cache = create_input(task_graph, Shape(nf=D_VALUE, nr=SEQ_LEN - 1),
                                Precision.FLOAT_16, False)
output, task_dict = create_attention(
    task_graph=task_graph,
    embedding=embedding,
    d_model=4096,
    d_key=D_KEY,
    d_value=D_VALUE,
    seq_len=SEQ_LEN,
    precision=Precision.FLOAT_16,
    is_output=True
)
task_graph.connect(input_key_cache.id, task_dict["key_cache"]["old"].id)
task_graph.connect(input_value_cache.id, task_dict["value_cache"]["old"].id)

# Create a hardware
config = ServerConfig("top/gpu_server.toml")
SIZE_X, SIZE_Y = config.PCB.chiplet.size
server = ServerFactory.create_matrix(config)

# Create a DSE environment
env = STEnv(task_graph, server)

# # Update hardware parameter
# board0 = create_mlcoord((0, 0))
# board_network = env.get_communication_network(board0)
# board_network.update_bandwidth(bandwidth)

# Define some constants for writing coordinates
CHIP = 0
DRAM = 2
PCIE_PHY = 3
TENSOR_UNIT = 1
ROUTER = 3
SHARED_MEMORY = (SIZE_X + 1, SIZE_Y // 2)
SRAM_BUFFER = 0

# Graph transformation
embedding_on_chip = env.copy_task(embedding.id)
task_graph.add_node_between(embedding, embedding.out_tasks, 
                            embedding_on_chip)
split_embedding_on_chip = env.split_task(embedding_on_chip.id, 
                                         SplitVector())
env.connect_tasks([embedding], split_embedding_on_chip)

SIZE_X = 2
SIZE_Y = 1

split_task_dict = env.split_attention(task_dict,
                                      embedding_on_chip,
                                      split_embedding_on_chip,
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y),
                                      SplitVector(nf=SIZE_X * SIZE_Y))

output.enable()
split_attention_output = split_task_dict["attention"]["output"]
env.connect_tasks(split_attention_output, [output])

STDraw.draw_graph(task_graph, out_path='temp/attention.task.html',
                width='1920px', height='1080px')

# Mapping
dram0 = create_mlcoord((0, 0), DRAM)
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
env.put_in(dram0, embedding.id)
env.put_tasks_in(shared_memory0, split_embedding_on_chip)
env.put_in(dram0, output.id)
query_weight = task_dict["query"]["weight"]
env.put_in(dram0, query_weight.id)
key_weight = task_dict["key"]["weight"]
env.put_in(dram0, key_weight.id)
value_weight = task_dict["value"]["weight"]
env.put_in(dram0, value_weight.id)
key_cache = task_dict["key_cache"]["old"]
env.put_in(dram0, key_cache.id)
value_cache = task_dict["value_cache"]["old"]
env.put_in(dram0, value_cache.id)

# Map MLP of query
split_query_weight = split_task_dict["query"]["weight"]
split_query_output = split_task_dict["query"]["output"]
split_query_mlp = split_task_dict["query"]["compute"]
env.put_tasks_in(shared_memory0, split_query_weight)
env.put_tasks_in(shared_memory0, split_query_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_query_mlp[k + j * SIZE_Y].id)

# Map MLP of key
split_key_mlp = split_task_dict["key"]["compute"]
split_key_weight = split_task_dict["key"]["weight"]
split_key_output = split_task_dict["key"]["output"]
env.put_tasks_in(shared_memory0, split_key_weight)
env.put_tasks_in(shared_memory0, split_key_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_key_mlp[k + j * SIZE_Y].id)

# Map MLP of value
split_value_mlp = split_task_dict["value"]["compute"]
split_value_weight = split_task_dict["value"]["weight"]
split_value_output = split_task_dict["value"]["output"]
env.put_tasks_in(shared_memory0, split_value_weight)
env.put_tasks_in(shared_memory0, split_value_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_value_mlp[k + j * SIZE_Y].id)

# Map the creation of new key cache
concat_key = task_dict["concat_key"]["move"]
env.put_in(dram0, concat_key.id)

# Map the dot product between query and key cache
split_dot_product = split_task_dict["dot_product"]["compute"]
split_key_cache = split_task_dict["dot_product"]["weight"]
split_dot_product_output = split_task_dict["dot_product"]["output"]
env.put_tasks_in(shared_memory0, split_key_cache)
env.put_tasks_in(shared_memory0, split_dot_product_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_dot_product[k + j * SIZE_Y].id)

# Map scale
split_scale = split_task_dict["scale"]["compute"]
split_scale_output = split_task_dict["scale"]["output"]
env.put_tasks_in(shared_memory0, split_scale_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_scale[k + j * SIZE_Y].id)

# Map SoftMax
split_softmax_exp = split_task_dict["softmax"]["exp"]["compute"]
split_softmax_exp_output = split_task_dict["softmax"]["exp"]["output"]
env.put_tasks_in(shared_memory0, split_softmax_exp_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_softmax_exp[k + j * SIZE_Y].id)

softmax_reduce = task_dict["softmax"]["reduction"]["compute"]
softmax_reduce_output = task_dict["softmax"]["reduction"]["output"]
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, softmax_reduce.id)
env.put_in(shared_memory0, softmax_reduce_output.id)

split_softmax_div = split_task_dict["softmax"]["div"]["compute"]
split_softmax_div_output = split_task_dict["softmax"]["div"]["output"]
env.put_tasks_in(shared_memory0, split_softmax_div_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_softmax_div[k + j * SIZE_Y].id)

# Map the creation of new key cache
concat_value = task_dict["concat_value"]["move"]
env.put_in(dram0, concat_value.id)

# Map attention
split_attention = split_task_dict["attention"]["compute"]
split_value_cache = split_task_dict["attention"]["weight"]
env.put_tasks_in(shared_memory0, split_value_cache)
env.put_tasks_in(shared_memory0, split_attention_output)
for j in range(SIZE_X):
    for k in range(SIZE_Y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_attention[k + j * SIZE_Y].id)

# Edge Mapping
env.auto_edge_map()

env.simulate()
env.show_overall_time()
print(env.get_latency())
# latencies.append(env.get_latency())

# plt.plot(list(range(100, 1600, 100)), latencies)
# plt.savefig('temp/dram.png')