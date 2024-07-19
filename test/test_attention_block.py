from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_attention_block, create_input


# Algorithm parameters
SEQ_LEN = 2048
D_MODEL = 4096
D_KEY = 4096
D_VALUE = 4096
HEAD = 2
# Parse hardware configuration
config = ServerConfig("top/gpu_server.toml")
SIZE_X, SIZE_Y = config.PCB.chiplet.size
# Some constants for writing coordinates
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
input, embedding = create_input(task_graph, Shape(nf=D_MODEL), 
                                Precision.FLOAT_16)
output, task_dict = create_attention_block(
    task_graph=task_graph,
    embedding=embedding,
    d_model=D_MODEL,
    d_key=D_KEY,
    d_value=D_VALUE,
    seq_len=SEQ_LEN,
    head=HEAD,
    epsilon=1e-5,
    precision=Precision.FLOAT_16,
    is_output=True
)

for i in range(HEAD):
    input_key_cache = create_input(task_graph, Shape(nf=SEQ_LEN - 1, nr=D_KEY), 
                                   Precision.FLOAT_16, False)
    input_value_cache = create_input(task_graph, 
                                     Shape(nf=D_VALUE, nr=SEQ_LEN - 1),
                                     Precision.FLOAT_16, False)
    task_graph.connect(input_key_cache.id, 
                       task_dict["multi_head_attention"][i]["key_cache"]["old"].id)
    task_graph.connect(input_value_cache.id, 
                       task_dict["multi_head_attention"][i]["value_cache"]["old"].id)

STDraw.draw_graph(task_graph, out_path='temp/attention_block.task.html',
                  width='1920px', height='1080px')

# Create a hardware
server = ServerFactory.create_matrix(config)

# Create a DSE environment
env = STEnv(task_graph, server)

# # Update hardware parameter
# board0 = create_mlcoord((0, 0))
# board_network = env.get_communication_network(board0)
# board_network.update_bandwidth(bandwidth)

size_x = 2
size_y = 1

# Graph transformation
embedding_on_chip = env.copy_task(embedding.id)
task_graph.add_node_between(embedding, embedding.out_tasks, embedding_on_chip)
split_embedding_on_chip = env.split_task(embedding_on_chip.id, SplitVector())
env.connect_tasks([embedding], split_embedding_on_chip)

split_task_dict = env.split_attention_block(
    task_dict,
    embedding_on_chip,
    split_embedding_on_chip,
    HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    [SplitVector(nf=size_x * size_y)] * HEAD,
    SplitVector(nf=size_x * size_y * HEAD),
    SplitVector(),
    SplitVector(nf=size_x * size_y * HEAD),
    SplitVector(nf=size_x * size_y * HEAD),
    SplitVector(nf=size_x * size_y * HEAD)
    )
output.enable()
env.connect_tasks(split_task_dict["layer_norm"]["div"]["output"], [output])

STDraw.draw_graph(task_graph, 
                  out_path='temp/tiled_attention_block.task.html',
                  width='1920px', height='1080px')

# Mapping
dram0 = create_mlcoord((0, 0), DRAM)
shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
env.put_in(dram0, embedding.id)
env.put_tasks_in(shared_memory0, split_embedding_on_chip)
env.put_in(dram0, output.id)
env.put_in(dram0, task_dict["multi_head_attention"]["mlp"]["weight"].id)
for i in range(HEAD):
    query_weight = task_dict["multi_head_attention"][i]["query"]["weight"]
    env.put_in(dram0, query_weight.id)
    key_weight = task_dict["multi_head_attention"][i]["key"]["weight"]
    env.put_in(dram0, key_weight.id)
    value_weight = task_dict["multi_head_attention"][i]["value"]["weight"]
    env.put_in(dram0, value_weight.id)
    key_cache = task_dict["multi_head_attention"][i]["key_cache"]["old"]
    env.put_in(dram0, key_cache.id)
    value_cache = task_dict["multi_head_attention"][i]["value_cache"]["old"]
    env.put_in(dram0, value_cache.id)

    block_idx = i // (SIZE_Y // size_y)
    block_idy = i % (SIZE_Y // size_y)

    core_idx = block_idx * size_x
    core_idy = block_idy * size_y

    # Map MLP of query
    split_query_weight = split_task_dict["multi_head_attention"][i]["query"]["weight"]
    split_query_output = split_task_dict["multi_head_attention"][i]["query"]["output"]
    split_query_mlp = split_task_dict["multi_head_attention"][i]["query"]["compute"]
    env.put_tasks_in(shared_memory0, split_query_weight)
    env.put_tasks_in(shared_memory0, split_query_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_query_mlp[k + j * size_y].id)

    # Map MLP of key
    split_key_mlp = split_task_dict["multi_head_attention"][i]["key"]["compute"]
    split_key_weight = split_task_dict["multi_head_attention"][i]["key"]["weight"]
    split_key_output = split_task_dict["multi_head_attention"][i]["key"]["output"]
    env.put_tasks_in(shared_memory0, split_key_weight)
    env.put_tasks_in(shared_memory0, split_key_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_key_mlp[k + j * size_y].id)

    # Map MLP of value
    split_value_mlp = split_task_dict["multi_head_attention"][i]["value"]["compute"]
    split_value_weight = split_task_dict["multi_head_attention"][i]["value"]["weight"]
    split_value_output = split_task_dict["multi_head_attention"][i]["value"]["output"]
    env.put_tasks_in(shared_memory0, split_value_weight)
    env.put_tasks_in(shared_memory0, split_value_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_value_mlp[k + j * size_y].id)

    # Map the creation of new key cache
    concat_key = task_dict["multi_head_attention"][i]["concat_key"]["move"]
    env.put_in(dram0, concat_key.id)

    # Map the dot product between query and key cache
    split_dot_product = split_task_dict["multi_head_attention"][i]["dot_product"]["compute"]
    split_key_cache = split_task_dict["multi_head_attention"][i]["dot_product"]["weight"]
    split_dot_product_output = split_task_dict["multi_head_attention"][i]["dot_product"]["output"]
    env.put_tasks_in(shared_memory0, split_key_cache)
    env.put_tasks_in(shared_memory0, split_dot_product_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_dot_product[k + j * size_y].id)

    # Map scale
    split_scale = split_task_dict["multi_head_attention"][i]["scale"]["compute"]
    split_scale_output = split_task_dict["multi_head_attention"][i]["scale"]["output"]
    env.put_tasks_in(shared_memory0, split_scale_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_scale[k + j * size_y].id)

    # Map SoftMax
    split_softmax_exp = split_task_dict["multi_head_attention"][i]["softmax"]["exp"]["compute"]
    split_softmax_exp_output = split_task_dict["multi_head_attention"][i]["softmax"]["exp"]["output"]
    env.put_tasks_in(shared_memory0, split_softmax_exp_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_softmax_exp[k + j * size_y].id)

    softmax_reduce = task_dict["multi_head_attention"][i]["softmax"]["reduction"]["compute"]
    softmax_reduce_output = task_dict["multi_head_attention"][i]["softmax"]["reduction"]["output"]
    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx, core_idy), TENSOR_UNIT)
    env.put_in(tensor_unit, softmax_reduce.id)
    env.put_in(shared_memory0, softmax_reduce_output.id)

    split_softmax_div = split_task_dict["multi_head_attention"][i]["softmax"]["div"]["compute"]
    split_softmax_div_output = split_task_dict["multi_head_attention"][i]["softmax"]["div"]["output"]
    env.put_tasks_in(shared_memory0, split_softmax_div_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_softmax_div[k + j * size_y].id)

    # Map the creation of new key cache
    concat_value = task_dict["multi_head_attention"][i]["concat_value"]["move"]
    env.put_in(dram0, concat_value.id)

    # Map attention
    split_attention = split_task_dict["multi_head_attention"][i]["attention"]["compute"]
    split_value_cache = split_task_dict["multi_head_attention"][i]["attention"]["weight"]
    split_attention_output = split_task_dict["multi_head_attention"][i]["attention"]["output"]
    env.put_tasks_in(shared_memory0, split_value_cache)
    env.put_tasks_in(shared_memory0, split_attention_output)
    for j in range(size_x):
        for k in range(size_y):
            tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
            env.put_in(tensor_unit, split_attention[k + j * size_y].id)

concat = task_dict["multi_head_attention"]["concat"]["move"]
split_concat_output = split_task_dict["multi_head_attention"]["concat"]["output"]
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, concat.id)
env.put_tasks_in(shared_memory0, split_concat_output)

split_mlp = split_task_dict["multi_head_attention"]["mlp"]["compute"]
split_mlp_weight = split_task_dict["multi_head_attention"]["mlp"]["weight"]
split_mlp_output = split_task_dict["multi_head_attention"]["mlp"]["output"]
env.put_tasks_in(shared_memory0, split_mlp_weight)
env.put_tasks_in(shared_memory0, split_mlp_output)
for i in range(size_x * size_y * HEAD):
    idx = i // SIZE_Y
    idy = i % SIZE_Y
    tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
    env.put_in(tensor_unit, split_mlp[i].id)

add = split_task_dict["add"]["compute"]
split_add_output = split_task_dict["add"]["output"]
env.put_tasks_in(shared_memory0, split_add_output)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, add[0].id)

reduce_sum = task_dict["layer_norm"]["reduce_sum_mean"]["compute"]
reduce_sum_output = task_dict["layer_norm"]["reduce_sum_mean"]["output"]
env.put_in(shared_memory0, reduce_sum_output.id)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, reduce_sum.id)

scale = task_dict["layer_norm"]["average"]["compute"]
scale_output = task_dict["layer_norm"]["average"]["output"]
env.put_in(shared_memory0, scale_output.id)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, scale.id)

split_layer_norm_add = split_task_dict["layer_norm"]["add"]["compute"]
split_layer_norm_add_output = split_task_dict["layer_norm"]["add"]["output"]
env.put_tasks_in(shared_memory0, split_layer_norm_add_output)
for i in range(size_x * size_y * HEAD):
    idx = i // SIZE_Y
    idy = i % SIZE_Y
    tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
    env.put_in(tensor_unit, split_layer_norm_add[i].id)

split_product = split_task_dict["layer_norm"]["product"]["compute"]
split_product_output = split_task_dict["layer_norm"]["product"]["output"]
env.put_tasks_in(shared_memory0, split_product_output)
for i in range(size_x * size_y * HEAD):
    idx = i // SIZE_Y
    idy = i % SIZE_Y
    tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
    env.put_in(tensor_unit, split_product[i].id)

reduce_sum_var = task_dict["layer_norm"]["reduce_sum_var"]["compute"]
reduce_sum_var_output = task_dict["layer_norm"]["reduce_sum_var"]["output"]
env.put_in(shared_memory0, reduce_sum_var_output.id)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, reduce_sum_var.id)

add_var = task_dict["layer_norm"]["add_var"]["compute"]
add_var_output = task_dict["layer_norm"]["add_var"]["output"]
env.put_in(shared_memory0, add_var_output.id)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, add_var.id)

sqrt = task_dict["layer_norm"]["sqrt"]["compute"]
sqrt_output = task_dict["layer_norm"]["sqrt"]["output"]
env.put_in(shared_memory0, sqrt_output.id)
tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
env.put_in(tensor_unit, sqrt.id)

split_div = split_task_dict["layer_norm"]["div"]["compute"]
split_div_output = split_task_dict["layer_norm"]["div"]["output"]
env.put_tasks_in(shared_memory0, split_div_output)
for i in range(size_x * size_y * HEAD):
    idx = i // SIZE_Y
    idy = i % SIZE_Y
    tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
    env.put_in(tensor_unit, split_div[i].id)

# Edge Mapping
env.auto_edge_map()

env.simulate()
env.show_overall_time()
print(env.get_latency())