from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape, SplitVector
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.transformer import create_input, create_mlp


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


if __name__ == "__main__":
    test_split_mlps()
