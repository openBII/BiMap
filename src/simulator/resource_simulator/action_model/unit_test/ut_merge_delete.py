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

original_task_id = 3
original_task = st_env.get_task(original_task_id)
new_task_ids = st_env.split_task(task_id=3, split_vector=Shape(1, 1, 2, 1, 1, 1))

core_id = 0
phase_id = 0
space_coord_list = []
for task_id in new_task_ids:
    task = st_env.get_task(task_id)
    if task.task_type == original_task.task_type:
        space_coord = Coord((0, 0, 0, core_id))
        st_env.put_group_in(ml_coord=MLCoord(space_coord, (0, phase_id, PIIndex.AXON.value)), task_id=task_id)
        space_coord_list.append(space_coord)
        core_id += 1
        phase_id += 1

for task_id in st_env.get_all_task_id():
    task = st_env.get_task(task_id)
    if task.task_type == TaskBlockType.SO:
        space_coord = Coord((0, 0, 0, core_id))
        st_env.put_in(ml_coord=MLCoord(space_coord, (0, phase_id, PIIndex.MEMORY.value)), task_id=task_id)
        space_coord_list.append(space_coord)

# Core0在Phase0完成一个CVM
# Core1在Phase1完成一个CVM
# 最终结果在Core2 Phase2得到
STDraw.draw_matrix_sankey(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_sankey.html')

new_task_id_list = st_env.replicate_group(task_id_list=[5, 5])
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 3), (0, 0, PIIndex.AXON.value)), task_id=new_task_id_list[0])
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 3), (0, 1, PIIndex.AXON.value)), task_id=new_task_id_list[0])
st_env.delete_column(Coord((0, 0, 0, 3)))

st_env.merge_column(space_coord_list)

STDraw.draw_matrix_sankey(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_merge_sankey.html')
