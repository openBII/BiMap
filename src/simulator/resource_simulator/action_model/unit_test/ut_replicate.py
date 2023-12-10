from cmath import phase
from top.config import GlobalConfig
from task_rabbit.task_model.shape import Shape
from resource_simulator.st_model.st_coord import Coord, MLCoord
from task_rabbit.initial_pass import execute_initial_pass
from resource_simulator.st_env import STMatrix, STEnv

from resource_simulator.st_draw import STDraw
from task_rabbit.task_model.task_block_type import TaskBlockType


case_name = 'ut_cvm'
task_path = GlobalConfig.Path["test_lib"] + 'task_lib/1P/cvm.task'
init = execute_initial_pass(
    case_path=task_path, case_name=case_name, input_type='task')
task_graph = init.task_graph
st_matrix = STMatrix()
st_env = STEnv(task_graph, st_matrix)
st_env.check_graph()

st_env.replicate_task(task_id=3)
st_env.replicate_group(task_id_list=[3, 3])

STDraw.draw_graph(task_graph)