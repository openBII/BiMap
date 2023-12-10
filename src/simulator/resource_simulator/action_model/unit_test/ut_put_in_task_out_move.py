from top.config import GlobalConfig
from src.compiler.mapper.passes.final_pass import FinalPass
from task_rabbit.task_model.shape import Shape
from resource_simulator.st_model.st_coord import MLCoord
from task_rabbit.initial_pass import execute_initial_pass
from resource_simulator.st_env import create_st_env

from resource_simulator.st_draw import STDraw
from task_rabbit.task_model.task_block_type import TaskBlockType

case_name = 'ut_cvm'
task_path = GlobalConfig.Path["test_lib"] + 'task_lib/1P/cvm.task.txt'
st_env = create_st_env(task_path, case_name)

st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)), task_id=0)
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.MEMORY.value)), task_id=1, end_ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)))
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.MEMORY.value)), task_id=2, end_ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)))
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.AXON.value)), task_id=3)
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)), task_id=4)

# STDraw.draw_matrix_table(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_table.html')

FinalPass(st_env=st_env, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm.map')

st_env = create_st_env(task_path, case_name)

st_env.put_group_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.AXON.value)), task_id=3)
st_env.take_out(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)), task_id=1)
st_env.take_out(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)), task_id=2)
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.MEMORY.value)), task_id=1, end_ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)))
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.MEMORY.value)), task_id=2, end_ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)))
st_env.put_in(ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)), task_id=4)

st_env.take_out(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.MEMORY.value)), task_id=1, end_ml_coord=MLCoord((0, 0, 0, 0), (0, 1, PIIndex.MEMORY.value)))

# STDraw.draw_matrix_table(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_sankey.html')

st_env = create_st_env(task_path, case_name)

original_task_id = 3
original_task = st_env.get_task(original_task_id)
new_task_ids = st_env.split_task(task_id=3, split_vector=Shape(1, 1, 2, 1, 1, 1))

core_id = 0
for task_id in new_task_ids:
    task = st_env.get_task(task_id)
    if task.task_type == original_task.task_type:
        st_env.put_group_in(ml_coord=MLCoord((0, 0, 0, core_id), (0, 0, PIIndex.AXON.value)), task_id=task_id)
        core_id += 1

for task_id in st_env.get_all_task_id():
    task = st_env.get_task(task_id)
    if task.task_type == TaskBlockType.SO:
        st_env.put_in(ml_coord=MLCoord((0, 0, 0, core_id), (0, 1, PIIndex.MEMORY.value)), task_id=task_id)

STDraw.draw_matrix_table(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_put_in_table.html')
STDraw.draw_matrix_sankey(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_put_in_sankey.html')
STDraw.draw_graph_table(st_env.st_matrix, st_env.task_graph,out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_put_in_graph_table.html')

# st_env.move_group(src_coord=MLCoord((0, 0, 0, 1), (0, 0, PIIndex.AXON.value)),
#                   dml_coord=MLCoord((0, 0, 0, 3), (0, 0, PIIndex.AXON.value)))

# st_env.take_out(ml_coord=MLCoord((0, 0, 0, 3), (0, 0, PIIndex.AXON.value)))

# st_env.move(src_coord=MLCoord((0, 0, 0, 3), (0, 0, PIIndex.MEMORY.value)),
#             dml_coord=MLCoord((0, 0, 0, 1), (0, 1, PIIndex.MEMORY.value)))

# st_env.take_group_out(ml_coord=MLCoord((0, 0, 0, 0), (0, 0, PIIndex.AXON.value)))

# STDraw.draw_matrix_sankey(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_take_out_sankey.html')
# STDraw.draw_matrix_table(st_env.st_matrix, out_path=GlobalConfig.Path["temp"] + 'ut_cvm/ut_cvm_take_out_table.html')
