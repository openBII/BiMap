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
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import Coord, MLCoord
from src.simulator.resource_simulator.st_draw import STDraw
from src.simulator.task_rabbit.task_model.id_generator import IDGenerator
from src.simulator.task_rabbit.task_model.transformer import create_transformer, create_input, create_data


# Construct a task graph
task_graph = TaskGraph()
IDGenerator.set_base_task_id(task_graph)
input0, input_embedding = create_input(task_graph, Shape(nf=4096), 
                                       Precision.FLOAT_16)
input1, position_encoding = create_input(task_graph, Shape(nf=4096), 
                                         Precision.FLOAT_16)
output, task_dict = create_transformer(
    task_graph=task_graph,
    input_embedding=input_embedding,
    position_encoding=position_encoding,
    num_layers=8,
    num_words=32000,
    d_key=4096,
    d_value=4096,
    seq_len=2048,
    head=8,
    ffn_inner_dim=4 * 4096,
    activation_type=TaskBlockType.CRELU,
    precision=Precision.FLOAT_16
)

STDraw.draw_graph(task_graph, out_path='temp/transformer.task.html',
                  width='1920px', height='1080px')
