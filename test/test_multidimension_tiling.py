from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory \
    import ServerFactory
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.transformer import create_input, \
    create_mlp, create_pointwise, create_attention, create_prefill_attention, \
    AttentionType, create_prefill_multi_head_attention, \
    create_prefill_attention_block, create_prefill_transformer_layer


def test_split_mlp():
    split_vector = SplitVector(batch=2, token=2, nf=2, nr=2)

    # Construct a task graph
    task_graph = TaskGraph()
    _, mlp_input = create_input(task_graph, Shape(batch=64, token=128, nf=2048), 
                                Precision.FLOAT_16)
    output, task_dict = create_mlp(task_graph, mlp_input, 
                                   Shape(nr=2048, nf=4096), 
                                   Precision.FLOAT_16, True)
    STDraw.draw_graph(task_graph, out_path='temp/mlp.task.html',
                    width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)
    # Construct a simulation environment
    env = STEnv(task_graph, server)

    split_inputs = env.split_task(
        mlp_input.id, 
        SplitVector(batch=split_vector.batch, token=split_vector.token))
    env.add_nodes_between(mlp_input, task_dict["compute"], split_inputs)
    task_dict = env.split_mlp(split_inputs, task_dict["weight"], 
                              task_dict["compute"], output, split_vector)
    env.connect_tasks(task_dict["output"], [output])
    env.enable_task(output.id)
    STDraw.draw_graph(task_graph, out_path='temp/tiled_mlp.task.html',
                    width='1920px', height='1080px')
    
def test_split_mlps():
    split_vector0 = SplitVector(batch=2, token=2, nf=2)
    split_vector1 = SplitVector(batch=2, token=2, nr=2)

    # Construct a task graph
    task_graph = TaskGraph()
    _, mlp_input = create_input(task_graph, Shape(batch=64, token=128, nf=2048), 
                                Precision.FLOAT_16)
    task_dict = {}
    task_dict[0] = {}
    mlp_output, _ = create_mlp(task_graph, mlp_input, 
                               Shape(nr=2048, nf=4096), 
                               Precision.FLOAT_16, 
                               task_dict=task_dict[0])
    task_dict[1] = {}
    output, _ = create_mlp(task_graph, mlp_output, 
                           Shape(nr=4096, nf=1024), 
                           Precision.FLOAT_16, True, task_dict[1])
    STDraw.draw_graph(task_graph, out_path='temp/mlp2.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)
    # Construct a simulation environment
    env = STEnv(task_graph, server)

    split_inputs = env.split_task(
        mlp_input.id, 
        SplitVector(batch=split_vector0.batch, token=split_vector0.token))
    env.add_nodes_between(mlp_input, task_dict[0]["compute"], split_inputs)
    split_task_dict = {}
    split_task_dict[0] = {}
    env.split_mlp(split_inputs, task_dict[0]["weight"], 
                  task_dict[0]["compute"], mlp_output, split_vector0, 
                  task_dict=split_task_dict[0])
    split_task_dict[1] = {}
    env.split_mlp(split_task_dict[0]["output"], task_dict[1]["weight"],
                  task_dict[1]["compute"], output, split_vector1,
                  task_dict=split_task_dict[1])
    env.connect_tasks(split_task_dict[1]["output"], [output])
    env.enable_task(output.id)
    env.undo()
    env.undo()

    split_vector0 = SplitVector(batch=2, token=2, nf=2)
    split_vector1 = SplitVector(batch=2, token=2, nf=2)
    split_task_dict = {}
    split_task_dict[0] = {}
    env.split_mlp(split_inputs, task_dict[0]["weight"], 
                  task_dict[0]["compute"], mlp_output, split_vector0, 
                  task_dict=split_task_dict[0])
    split_task_dict[1] = {}
    env.split_mlp(split_task_dict[0]["output"], task_dict[1]["weight"],
                  task_dict[1]["compute"], output, split_vector1,
                  task_dict=split_task_dict[1])
    env.connect_tasks(split_task_dict[1]["output"], [output])
    env.enable_task(output.id)
    env.undo()
    env.undo()


    split_vector0 = SplitVector(batch=2, token=2, nr=2)
    split_vector1 = SplitVector(batch=2, token=2, nr=2)
    split_task_dict = {}
    split_task_dict[0] = {}
    env.split_mlp(split_inputs, task_dict[0]["weight"], 
                  task_dict[0]["compute"], mlp_output, split_vector0, 
                  task_dict=split_task_dict[0])
    split_task_dict[1] = {}
    env.split_mlp(split_task_dict[0]["output"], task_dict[1]["weight"],
                  task_dict[1]["compute"], output, split_vector1,
                  task_dict=split_task_dict[1])
    env.connect_tasks(split_task_dict[1]["output"], [output])
    env.enable_task(output.id)
    STDraw.draw_graph(task_graph, out_path='temp/tiled_mlp2.task.html',
                      width='1920px', height='1080px')
    
def test_split_pointwise():
    split_vector = SplitVector(batch=2, token=2, nf=2)

    # Construct a task graph
    task_graph = TaskGraph()
    _, input = create_input(task_graph, Shape(batch=64, token=128, nf=2048), 
                            Precision.FLOAT_16)
    output, task_dict = create_pointwise(task_graph, input, Precision.FLOAT_16,
                                         TaskBlockType.CRELU, True)
    STDraw.draw_graph(task_graph, out_path='temp/pointwise.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)
    # Construct a simulation environment
    env = STEnv(task_graph, server)

    split_inputs = env.split_task(
        input.id, 
        SplitVector(batch=split_vector.batch, token=split_vector.token))
    env.add_nodes_between(input, task_dict["compute"], split_inputs)
    task_dict = env.split_pointwise(split_inputs, task_dict["compute"],
                                    output, split_vector)
    env.connect_tasks(task_dict["output"], [output])
    env.enable_task(output.id)
    STDraw.draw_graph(task_graph, out_path='temp/tiled_pointwise.task.html',
                      width='1920px', height='1080px')
    
def test_split_prefill_attention():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(batch=64, token=2048, nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_prefill_attention(
        task_graph=task_graph,
        embedding=embedding,
        d_key=4096,
        d_value=4096,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, out_path='temp/prefill_attention.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, 
                                     SplitVector(batch=1, token=2))
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_attention(
        task_dict=task_dict,
        split_embedding=split_embedding,
        query_split_vector=SplitVector(),
        key_split_vector=SplitVector(),
        value_split_vector=SplitVector(),
        dot_product_split_vector=SplitVector(nf=2, nr=1),
        softmax_split_vector=SplitVector(),
        attention_split_vector=SplitVector(nf=2, nr=1),
        attention_type=AttentionType.PREFILL,
        num_batch_split=1,
        num_token_split=2
    )
    output.enable()
    env.connect_tasks(split_task_dict["attention"]["output"], [output])
    STDraw.draw_graph(task_graph, 
                      out_path='temp/tiled_prefill_attention.task.html',
                      width='1920px', height='1080px')
    
def test_split_prefill_multi_head_attention():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(batch=64, token=512, nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_prefill_multi_head_attention(
        task_graph=task_graph,
        embedding=embedding,
        d_key=4096,
        d_value=4096,
        head=2,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, 
                      out_path='temp/prefill_multi_head_attention.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, 
                                     SplitVector(batch=1, token=2))
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_multi_head_attention(
        task_dict=task_dict,
        split_embedding=split_embedding,
        head=2,
        query_split_vectors=[SplitVector()] * 2,
        key_split_vectors=[SplitVector()] * 2,
        value_split_vectors=[SplitVector()] * 2,
        dot_product_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        softmax_split_vectors=[SplitVector()] * 2,
        attention_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        mlp_split_vector=SplitVector(),
        attention_type=AttentionType.PREFILL,
        embedding=embedding,
        num_batch_split=1,
        num_token_split=2
    )
    embedding.enable()
    output.enable()
    env.connect_tasks(split_task_dict["mlp"]["output"], [output])
    STDraw.draw_graph(
        task_graph, 
        out_path='temp/tiled_prefill_multi_head_attention.task.html',
        width='1920px', height='1080px')
    
def test_split_prefill_attention_block():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(batch=64, token=512, nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_prefill_attention_block(
        task_graph=task_graph,
        embedding=embedding,
        d_key=4096,
        d_value=4096,
        head=2,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, 
                      out_path='temp/prefill_attention_block.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, 
                                     SplitVector(batch=1, token=2))
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_attention_block(
        task_dict=task_dict,
        split_embedding=split_embedding,
        head=2,
        query_split_vectors=[SplitVector()] * 2,
        key_split_vectors=[SplitVector()] * 2,
        value_split_vectors=[SplitVector()] * 2,
        dot_product_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        softmax_split_vectors=[SplitVector()] * 2,
        attention_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        mlp_split_vector=SplitVector(),
        add_split_vector=SplitVector(),
        layer_norm_split_vector=SplitVector(),
        attention_type=AttentionType.PREFILL,
        embedding=embedding,
        num_batch_split=1,
        num_token_split=2
    )
    embedding.enable()
    output.enable()
    env.connect_tasks(split_task_dict["layer_norm"]["output"], [output])
    STDraw.draw_graph(
        task_graph, 
        out_path='temp/tiled_prefill_attention_block.task.html',
        width='1920px', height='1080px')
    
def test_split_prefill_transformer():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(batch=64, token=512, nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_prefill_transformer_layer(
        task_graph=task_graph,
        embedding=embedding,
        d_key=4096,
        d_value=4096,
        head=2,
        ffn_inner_dim=8192,
        activation_type=TaskBlockType.CRELU,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, 
                      out_path='temp/prefill_transformer.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, 
                                     SplitVector(batch=1, token=2))
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_transformer_layer(
        task_dict=task_dict,
        split_embedding=split_embedding,
        head=2,
        query_split_vectors=[SplitVector()] * 2,
        key_split_vectors=[SplitVector()] * 2,
        value_split_vectors=[SplitVector()] * 2,
        dot_product_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        softmax_split_vectors=[SplitVector()] * 2,
        attention_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        mlp_split_vector=SplitVector(),
        layer_norm_split_vector=SplitVector(),
        attention_type=AttentionType.PREFILL,
        split_vector_up=SplitVector(),
        split_vector_act=SplitVector(),
        split_vector_down=SplitVector(),
        ffn_layer_norm_split_vector=SplitVector(),
        embedding=embedding,
        num_batch_split=1,
        num_token_split=2
    )
    embedding.enable()
    output.enable()
    env.connect_tasks(split_task_dict["ffn"]["layer_norm"]["output"], [output])
    STDraw.draw_graph(
        task_graph, 
        out_path='temp/tiled_prefill_transformer.task.html',
        width='1920px', height='1080px')

def test_split_decode_attention():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_attention(
        task_graph=task_graph,
        embedding=embedding,
        d_model=4096,
        d_key=4096,
        d_value=4096,
        seq_len=2048,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, out_path='temp/attention.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, SplitVector())
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_attention(
        task_dict=task_dict,
        split_embedding=split_embedding,
        query_split_vector=SplitVector(nf=2, nr=2),
        key_split_vector=SplitVector(nf=2, nr=2),
        value_split_vector=SplitVector(nf=2, nr=2),
        dot_product_split_vector=SplitVector(nf=2, nr=2),
        softmax_split_vector=SplitVector(),
        attention_split_vector=SplitVector(nf=2, nr=2),
        attention_type=AttentionType.DECODE
    )
    output.enable()
    env.connect_tasks(split_task_dict["attention"]["output"], [output])
    STDraw.draw_graph(task_graph, 
                      out_path='temp/tiled_attention.task.html',
                      width='1920px', height='1080px')
    
def test_prefill_transformer_mapping():
    # Construct a task graph
    task_graph = TaskGraph()
    _, embedding = create_input(task_graph, 
                                Shape(batch=64, token=512, nf=4096), 
                                Precision.FLOAT_16)
    output, task_dict = create_prefill_transformer_layer(
        task_graph=task_graph,
        embedding=embedding,
        d_key=4096,
        d_value=4096,
        head=2,
        ffn_inner_dim=8192,
        activation_type=TaskBlockType.CRELU,
        precision=Precision.FLOAT_16,
        is_output=True
    )

    STDraw.draw_graph(task_graph, 
                      out_path='temp/prefill_transformer.task.html',
                      width='1920px', height='1080px')

    # Construct a hardware
    config = ServerConfig("top/gpu_server.toml")
    server = ServerFactory.create_matrix(config)

    # Create a DSE environment
    env = STEnv(task_graph, server)

    # Mapping
    split_embedding = env.split_task(embedding.id, 
                                     SplitVector(batch=1, token=2))
    env.add_nodes_between(embedding, embedding.out_tasks, 
                          split_embedding)

    split_task_dict = env.split_transformer_layer(
        task_dict=task_dict,
        split_embedding=split_embedding,
        head=2,
        query_split_vectors=[SplitVector()] * 2,
        key_split_vectors=[SplitVector()] * 2,
        value_split_vectors=[SplitVector()] * 2,
        dot_product_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        softmax_split_vectors=[SplitVector()] * 2,
        attention_split_vectors=[SplitVector(nf=2, nr=1)] * 2,
        mlp_split_vector=SplitVector(),
        layer_norm_split_vector=SplitVector(),
        attention_type=AttentionType.PREFILL,
        split_vector_up=SplitVector(),
        split_vector_act=SplitVector(),
        split_vector_down=SplitVector(),
        ffn_layer_norm_split_vector=SplitVector(),
        embedding=embedding,
        num_batch_split=1,
        num_token_split=2
    )
    embedding.enable()
    output.enable()
    env.connect_tasks(split_task_dict["ffn"]["layer_norm"]["output"], [output])
    STDraw.draw_graph(
        task_graph, 
        out_path='temp/tiled_prefill_transformer.task.html',
        width='1920px', height='1080px')

if __name__ == "__main__":
    test_split_prefill_transformer()
