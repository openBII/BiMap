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

SEQ_LEN = 2048
D_KEY = 4096
D_VALUE = 4096
config = ServerConfig("top/gpu_server.toml")
SIZE_X, SIZE_Y = config.PCB.chiplet.size
CHIP = 0
DRAM = 2
PCIE_PHY = 3
TENSOR_UNIT = 1
ROUTER = 3
SHARED_MEMORY = (SIZE_X + 1, SIZE_Y // 2)
SRAM_BUFFER = 0

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
server = ServerFactory.create_matrix(config)

# Create a DSE environment
env = STEnv(task_graph, server)

# Update hardware parameter
# board0 = create_mlcoord((0, 0))
# board_network = env.get_communication_network(board0)
# board_network.update_bandwidth(bandwidth)

# Mapping
dram0 = create_mlcoord((0, 0), DRAM)
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
env.put_in(dram0, embedding.id)
split_embedding= env.split_task(embedding.id, SplitVector())
task_graph.add_node_between(embedding, embedding.out_tasks, 
                            split_embedding[0])
env.put_tasks_in(shared_memory0, split_embedding)
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

size_x = 2
size_y = 1

split_task_dict = env.split_attention(
    task_dict=task_dict,
    split_embedding=split_embedding,
    query_split_vector=SplitVector(nf=size_x * size_y),
    key_split_vector=SplitVector(nf=size_x * size_y),
    value_split_vector=SplitVector(nf=size_x * size_y),
    dot_product_split_vector=SplitVector(nf=size_x * size_y),
    scale_split_vector=SplitVector(nf=size_x * size_y),
    softmax_exp_split_vector=SplitVector(nf=size_x * size_y),
    softmax_div_split_vector=SplitVector(nf=size_x * size_y),
    attention_split_vector=SplitVector(nf=size_x * size_y)
)
output.enable()
env.connect_tasks(split_task_dict["attention"]["output"], [output])
STDraw.draw_graph(task_graph, out_path='temp/tiled_attention.task.html',
                width='1920px', height='1080px')

# Map MLP of query
query_mlp = task_dict["query"]["compute"]
query = task_dict["query"]["output"]
split_query_weight = split_task_dict["query"]["weight"]
split_query_output = split_task_dict["query"]["output"]
split_query_mlp = split_task_dict["query"]["compute"]
env.put_tasks_in(shared_memory0, split_query_weight)
env.put_tasks_in(shared_memory0, split_query_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_query_mlp[k + j * size_y].id)

# Map MLP of key
key_mlp = task_dict["key"]["compute"]
key = task_dict["key"]["output"]
split_key_weight = split_task_dict["key"]["weight"]
split_key_output = split_task_dict["key"]["output"]
split_key_mlp = split_task_dict["key"]["compute"]
env.put_tasks_in(shared_memory0, split_key_weight)
env.put_tasks_in(shared_memory0, split_key_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_key_mlp[k + j * size_y].id)

# Map MLP of value
value_mlp = task_dict["value"]["compute"]
value = task_dict["value"]["output"]
split_value_weight = split_task_dict["value"]["weight"]
split_value_output = split_task_dict["value"]["output"]
split_value_mlp = split_task_dict["value"]["compute"]
env.put_tasks_in(shared_memory0, split_value_weight)
env.put_tasks_in(shared_memory0, split_value_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_value_mlp[k + j * size_y].id)

# Map the creation of new key cache
concat_key = task_dict["concat_key"]["move"]
env.put_in(dram0, concat_key.id)

# Map the dot product between query and key cache
dot_product = task_dict["dot_product"]["compute"]
dot_product_output = task_dict["dot_product"]["output"]
split_key_cache = split_task_dict["dot_product"]["weight"]
split_dot_product = split_task_dict["dot_product"]["compute"]
split_dot_product_output = split_task_dict["dot_product"]["output"]
split_dot_product_concat = split_task_dict["dot_product"]["concat"]
env.put_tasks_in(shared_memory0, split_key_cache)
env.put_tasks_in(shared_memory0, split_dot_product_output)
env.put_tasks_in(shared_memory0, split_dot_product_concat)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_dot_product[k + j * size_y].id)

# Map scale
scale = task_dict["scale"]["compute"]
scale_output = task_dict["scale"]["output"]
split_scale = split_task_dict["scale"]["compute"]
split_scale_output = split_task_dict["scale"]["output"]
env.put_tasks_in(shared_memory0, split_scale_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_scale[k + j * size_y].id)

# Map SoftMax
softmax_exp = task_dict["softmax"]["exp"]["compute"]
softmax_exp_output = task_dict["softmax"]["exp"]["output"]
split_softmax_exp = split_task_dict["softmax"]["exp"]["compute"]
split_softmax_exp_output = split_task_dict["softmax"]["exp"]["output"]
env.put_tasks_in(shared_memory0, split_softmax_exp_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_softmax_exp[k + j * size_y].id)

softmax_reduce = task_dict["softmax"]["reduction"]["compute"]
softmax_reduce_output = task_dict["softmax"]["reduction"]["output"]
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, softmax_reduce.id)
env.put_in(shared_memory0, softmax_reduce_output.id)

softmax_div = task_dict["softmax"]["div"]["compute"]
softmax_div_output = task_dict["softmax"]["div"]["output"]
split_softmax_div = split_task_dict["softmax"]["div"]["compute"]
split_softmax_div_output = split_task_dict["softmax"]["div"]["output"]
env.put_tasks_in(shared_memory0, split_softmax_div_output)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_softmax_div[k + j * size_y].id)

# Map the creation of new key cache
concat_value = task_dict["concat_value"]["move"]
env.put_in(dram0, concat_value.id)

# Map attention
attention = task_dict["attention"]["compute"]
attention_output = task_dict["attention"]["output"]
split_value_cache = split_task_dict["attention"]["weight"]
split_attention = split_task_dict["attention"]["compute"]
split_attention_output = split_task_dict["attention"]["output"]
split_attention_concat = split_task_dict["attention"]["concat"]
env.put_tasks_in(shared_memory0, split_value_cache)
env.put_tasks_in(shared_memory0, split_attention_output)
env.put_tasks_in(shared_memory0, split_attention_concat)
for j in range(size_x):
    for k in range(size_y):
        tensor_unit = create_mlcoord((0, 0), CHIP, (j, k), TENSOR_UNIT)
        env.put_in(tensor_unit, split_attention[k + j * size_y].id)

# Edge Mapping
env.auto_edge_map()

# Simulation
env.simulate()
env.show_overall_time()
print(env.get_latency())
