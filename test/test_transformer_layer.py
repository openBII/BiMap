from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_transformer_layer, create_input
import cProfile
import pstats
from matplotlib import pyplot as plt


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
VECTOR_UNIT = 2
ROUTER = 3
SHARED_MEMORY = (SIZE_X + 1, SIZE_Y // 2)
SRAM_BUFFER = 0

latencies = {}

for size_x, size_y in ((2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)):
    latencies[(size_x, size_y)] = {}
    for bandwidth in range(1024, 4096, 256):
        # Construct a task graph
        task_graph = TaskGraph()
        IDGenerator.set_base_task_id(task_graph)
        input, embedding = create_input(task_graph, Shape(nf=D_MODEL), 
                                        Precision.FLOAT_16)
        output, task_dict = create_transformer_layer(
            task_graph=task_graph,
            embedding=embedding,
            d_key=D_KEY,
            d_value=D_VALUE,
            seq_len=SEQ_LEN,
            head=HEAD,
            ffn_inner_dim=D_VALUE * 4,
            activation_type=TaskBlockType.CRELU,
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
                            task_dict["attention"]["multi_head_attention"][i]["key_cache"]["old"].id)
            task_graph.connect(input_value_cache.id, 
                            task_dict["attention"]["multi_head_attention"][i]["value_cache"]["old"].id)

        # Create a hardware
        server = ServerFactory.create_matrix(config)

        # Create a DSE environment
        env = STEnv(task_graph, server)

        # chip = create_mlcoord((0, 0), CHIP)
        # print(env.eval_area(chip))

        # Update hardware parameter
        board0 = create_mlcoord((0, 0))
        board_network = env.get_communication_network(board0)
        board_network.update_bandwidth(bandwidth)

        # Graph transformation
        split_embedding = env.split_task(embedding.id, SplitVector())
        task_graph.add_node_between(embedding, embedding.out_tasks, split_embedding[0])
        env.connect_tasks([embedding], split_embedding)

        split_task_dict = env.split_transformer_layer(
            task_dict=task_dict,
            split_embedding=split_embedding,
            head=HEAD,
            query_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            key_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            value_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            dot_product_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            softmax_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            attention_split_vectors=[SplitVector(nf=size_x * size_y)] * HEAD,
            mlp_split_vector=SplitVector(nf=size_x * size_y * HEAD),
            add_split_vector=SplitVector(),
            layer_norm_split_vector=SplitVector(nf=size_x * size_y * HEAD),
            split_vector_up=SplitVector(nf=size_x * size_y * HEAD),
            split_vector_act=SplitVector(nf=size_x * size_y * HEAD),
            split_vector_down=SplitVector(nf=size_x * size_y * HEAD),
            ffn_add_split_vector=SplitVector(),
            ffn_layer_norm_split_vector=SplitVector(nf=size_x * size_y * HEAD)
            )
        output.enable()
        env.connect_tasks(split_task_dict["ffn"]["layer_norm"]["output"], [output])

        # STDraw.draw_graph(task_graph, 
        #                 out_path='temp/tiled_transformer.task.html',
        #                 width='1920px', height='1080px')

        # Mapping
        dram0 = create_mlcoord((0, 0), DRAM)
        shared_memory0 = create_mlcoord((0, 0), CHIP, SHARED_MEMORY)
        env.put_in(dram0, embedding.id)
        env.put_tasks_in(shared_memory0, split_embedding)
        env.put_in(dram0, output.id)
        env.put_in(dram0, task_dict["attention"]["multi_head_attention"]["mlp"]["weight"].id)
        env.put_in(dram0, task_dict["ffn"]["ffn"]["up"]["weight"].id)
        env.put_in(dram0, task_dict["ffn"]["ffn"]["down"]["weight"].id)
        for i in range(HEAD):
            query_weight = task_dict["attention"]["multi_head_attention"][i]["query"]["weight"]
            env.put_in(dram0, query_weight.id)
            key_weight = task_dict["attention"]["multi_head_attention"][i]["key"]["weight"]
            env.put_in(dram0, key_weight.id)
            value_weight = task_dict["attention"]["multi_head_attention"][i]["value"]["weight"]
            env.put_in(dram0, value_weight.id)
            key_cache = task_dict["attention"]["multi_head_attention"][i]["key_cache"]["old"]
            env.put_in(dram0, key_cache.id)
            value_cache = task_dict["attention"]["multi_head_attention"][i]["value_cache"]["old"]
            env.put_in(dram0, value_cache.id)

            block_idx = i // (SIZE_Y // size_y)
            block_idy = i % (SIZE_Y // size_y)

            core_idx = block_idx * size_x
            core_idy = block_idy * size_y

            # Map MLP of query
            split_query_weight = split_task_dict["attention"]["multi_head_attention"][i]["query"]["weight"]
            split_query_output = split_task_dict["attention"]["multi_head_attention"][i]["query"]["output"]
            split_query_mlp = split_task_dict["attention"]["multi_head_attention"][i]["query"]["compute"]
            env.put_tasks_in(shared_memory0, split_query_weight)
            env.put_tasks_in(shared_memory0, split_query_output)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
                    env.put_in(tensor_unit, split_query_mlp[k + j * size_y].id)

            # Map MLP of key
            split_key_mlp = split_task_dict["attention"]["multi_head_attention"][i]["key"]["compute"]
            split_key_weight = split_task_dict["attention"]["multi_head_attention"][i]["key"]["weight"]
            split_key_output = split_task_dict["attention"]["multi_head_attention"][i]["key"]["output"]
            env.put_tasks_in(shared_memory0, split_key_weight)
            env.put_tasks_in(shared_memory0, split_key_output)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
                    env.put_in(tensor_unit, split_key_mlp[k + j * size_y].id)

            # Map MLP of value
            split_value_mlp = split_task_dict["attention"]["multi_head_attention"][i]["value"]["compute"]
            split_value_weight = split_task_dict["attention"]["multi_head_attention"][i]["value"]["weight"]
            split_value_output = split_task_dict["attention"]["multi_head_attention"][i]["value"]["output"]
            env.put_tasks_in(shared_memory0, split_value_weight)
            env.put_tasks_in(shared_memory0, split_value_output)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
                    env.put_in(tensor_unit, split_value_mlp[k + j * size_y].id)

            # Map the creation of new key cache
            concat_key = task_dict["attention"]["multi_head_attention"][i]["concat_key"]["move"]
            env.put_in(dram0, concat_key.id)

            # Map the dot product between query and key cache
            split_dot_product = split_task_dict["attention"]["multi_head_attention"][i]["dot_product"]["compute"]
            split_key_cache = split_task_dict["attention"]["multi_head_attention"][i]["dot_product"]["weight"]
            split_dot_product_output = split_task_dict["attention"]["multi_head_attention"][i]["dot_product"]["output"]
            split_dot_product_concat = split_task_dict["attention"]["multi_head_attention"][i]["dot_product"]["concat"]
            env.put_tasks_in(shared_memory0, split_key_cache)
            env.put_tasks_in(shared_memory0, split_dot_product_output)
            env.put_tasks_in(shared_memory0, split_dot_product_concat)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
                    env.put_in(tensor_unit, split_dot_product[k + j * size_y].id)

            # Map SoftMax
            split_softmax = split_task_dict["attention"]["multi_head_attention"][i]["softmax"]["compute"]
            split_softmax_output = split_task_dict["attention"]["multi_head_attention"][i]["softmax"]["output"]
            env.put_tasks_in(shared_memory0, split_softmax_output)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), VECTOR_UNIT)
                    env.put_in(tensor_unit, split_softmax[k + j * size_y].id)

            # Map the creation of new key cache
            concat_value = task_dict["attention"]["multi_head_attention"][i]["concat_value"]["move"]
            env.put_in(dram0, concat_value.id)

            # Map attention
            split_attention = split_task_dict["attention"]["multi_head_attention"][i]["attention"]["compute"]
            split_value_cache = split_task_dict["attention"]["multi_head_attention"][i]["attention"]["weight"]
            split_attention_output = split_task_dict["attention"]["multi_head_attention"][i]["attention"]["output"]
            split_attention_concat = split_task_dict["attention"]["multi_head_attention"][i]["attention"]["concat"]
            env.put_tasks_in(shared_memory0, split_value_cache)
            env.put_tasks_in(shared_memory0, split_attention_output)
            env.put_tasks_in(shared_memory0, split_attention_concat)
            for j in range(size_x):
                for k in range(size_y):
                    tensor_unit = create_mlcoord((0, 0), CHIP, (core_idx + j, core_idy + k), TENSOR_UNIT)
                    env.put_in(tensor_unit, split_attention[k + j * size_y].id)

        concat = task_dict["attention"]["multi_head_attention"]["concat"]["move"]
        split_concat_output = split_task_dict["attention"]["multi_head_attention"]["concat"]["output"]
        env.put_in(shared_memory0, concat.id)
        env.put_tasks_in(shared_memory0, split_concat_output)

        split_mlp = split_task_dict["attention"]["multi_head_attention"]["mlp"]["compute"]
        split_mlp_weight = split_task_dict["attention"]["multi_head_attention"]["mlp"]["weight"]
        split_mlp_output = split_task_dict["attention"]["multi_head_attention"]["mlp"]["output"]
        env.put_tasks_in(shared_memory0, split_mlp_weight)
        env.put_tasks_in(shared_memory0, split_mlp_output)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
            env.put_in(tensor_unit, split_mlp[i].id)

        add = split_task_dict["attention"]["add"]["compute"]
        split_add_output = split_task_dict["attention"]["add"]["output"]
        env.put_tasks_in(shared_memory0, split_add_output)
        tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), VECTOR_UNIT)
        env.put_in(tensor_unit, add[0].id)

        split_layer_norm = split_task_dict["attention"]["layer_norm"]["compute"]
        split_layer_norm_output = split_task_dict["attention"]["layer_norm"]["output"]
        env.put_tasks_in(shared_memory0, split_layer_norm_output)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), VECTOR_UNIT)
            env.put_in(tensor_unit, split_layer_norm[i].id)

        # FFN up sampling
        split_up = split_task_dict["ffn"]["ffn"]["up"]["compute"]
        split_up_weight = split_task_dict["ffn"]["ffn"]["up"]["weight"]
        split_up_output = split_task_dict["ffn"]["ffn"]["up"]["output"]
        split_up_concat = split_task_dict["ffn"]["ffn"]["up"]["concat"]
        env.put_tasks_in(shared_memory0, split_up_weight)
        env.put_tasks_in(shared_memory0, split_up_output)
        env.put_tasks_in(shared_memory0, split_up_concat)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
            env.put_in(tensor_unit, split_up[i].id)

        # FFN activation
        split_activation = split_task_dict["ffn"]["ffn"]["activation"]["compute"]
        split_activation_output = split_task_dict["ffn"]["ffn"]["activation"]["output"]
        env.put_tasks_in(shared_memory0, split_activation_output)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), VECTOR_UNIT)
            env.put_in(tensor_unit, split_activation[i].id)

        # FFN down sampling
        split_down = split_task_dict["ffn"]["ffn"]["down"]["compute"]
        split_down_weight = split_task_dict["ffn"]["ffn"]["down"]["weight"]
        split_down_output = split_task_dict["ffn"]["ffn"]["down"]["output"]
        split_down_concat = split_task_dict["ffn"]["ffn"]["down"]["concat"]
        env.put_tasks_in(shared_memory0, split_down_weight)
        env.put_tasks_in(shared_memory0, split_down_output)
        env.put_tasks_in(shared_memory0, split_down_concat)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), TENSOR_UNIT)
            env.put_in(tensor_unit, split_down[i].id)

        ffn_add = split_task_dict["ffn"]["add"]["compute"]
        split_ffn_add_output = split_task_dict["ffn"]["add"]["output"]
        env.put_tasks_in(shared_memory0, split_ffn_add_output)
        tensor_unit = create_mlcoord((0, 0), CHIP, (0, 0), VECTOR_UNIT)
        env.put_in(tensor_unit, ffn_add[0].id)

        split_ffn_layer_norm = split_task_dict["ffn"]["layer_norm"]["compute"]
        split_ffn_layer_norm_output = split_task_dict["ffn"]["layer_norm"]["output"]
        env.put_tasks_in(shared_memory0, split_ffn_layer_norm_output)
        for i in range(size_x * size_y * HEAD):
            idx = i // SIZE_Y
            idy = i % SIZE_Y
            tensor_unit = create_mlcoord((0, 0), CHIP, (idx, idy), VECTOR_UNIT)
            env.put_in(tensor_unit, split_ffn_layer_norm[i].id)

        # Edge Mapping
        env.auto_edge_map()

        env.simulate()
        # cProfile.run("env.simulate()", 'temp/restats')
        # env.show_overall_time()
        latencies[(size_x, size_y)][bandwidth] = env.get_latency()

# p = pstats.Stats('temp/restats')
# p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

colors = ["red", "green", "black", "orange", "purple", "cyan"]
for i, (size_x, size_y) in enumerate(latencies):
    bandwidth_list = list(latencies[(size_x, size_y)].keys())
    latency_list = list(latencies[(size_x, size_y)].values())
    plt.scatter(bandwidth_list, latency_list, color=colors[i])
    plt.savefig('temp/dse.png')