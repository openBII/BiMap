from top.global_config import GlobalConfig
from src.simulator.task_rabbit.task_graph_parser import TaskGraphParser
from src.utils.st_draw import STDraw
from src.simulator.resource_simulator.st_env import STMatrix, STEnv
from src.simulator.task_rabbit.task_model.shape import Shape
from typing import List


def draw(case_name: str, split_task_id_list: List[int], split_vector_list: List[List[int]], out_path: str):
    task_path = GlobalConfig.Path['test_lib'] + '/task_lib/1P/' + case_name + '.task'
    init = TaskGraphParser(
        case_path=task_path, case_name=case_name, ir_type='task')
    task_graph = init.task_graph
    st_matrix = STMatrix()
    st_env = STEnv(task_graph, st_matrix)
    st_env.check_graph()

    assert len(split_task_id_list) == len(split_vector_list)
    split_times = len(split_task_id_list)
    for i in range(split_times):
        split_vector = split_vector_list[i]
        split_task_id = split_task_id_list[i]
        split_vector = Shape(ny=split_vector[0], nx=split_vector[1], nf=split_vector[2],
                             nr=split_vector[3], nky=split_vector[4], nkx=split_vector[5])
        st_env.split_task(task_id=split_task_id, split_vector=split_vector)
        st_env.check_graph()

    STDraw.draw_graph(task_graph, out_path=out_path,
                      width='1920px', height='1080px')


if __name__ == '__main__':
    # CVM
    draw(case_name='ut_cvm', split_task_id_list=[3], split_vector_list=[
         [1, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cvm.html')
    draw(case_name='ut_cvm', split_task_id_list=[3], split_vector_list=[
         [1, 1, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cvm_f.html')
    draw(case_name='ut_cvm', split_task_id_list=[3], split_vector_list=[
         [1, 1, 1, 2, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cvm_r.html')
    draw(case_name='ut_cvm', split_task_id_list=[3], split_vector_list=[
         [1, 1, 2, 2, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cvm_fr.html')
    # CC CC2D
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [1, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc.html')
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [2, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc_y.html')
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [1, 2, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc_x.html')
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [1, 1, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc_f.html')
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [1, 1, 1, 2, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc_r.html')
    draw(case_name='ut_cc', split_task_id_list=[3], split_vector_list=[
         [2, 1, 2, 2, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cc_yfr.html')
    # CCMPB CCMPS CAVG CADD CAX CVVH CVS
    draw(case_name='ut_cavg', split_task_id_list=[2], split_vector_list=[
         [1, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cavg.html')
    draw(case_name='ut_cavg', split_task_id_list=[2], split_vector_list=[
         [2, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cavg_y.html')
    draw(case_name='ut_cavg', split_task_id_list=[2], split_vector_list=[
         [1, 2, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cavg_x.html')
    draw(case_name='ut_cavg', split_task_id_list=[2], split_vector_list=[
         [1, 1, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cavg_f.html')
    draw(case_name='ut_cavg', split_task_id_list=[2], split_vector_list=[
         [2, 2, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/cavg_yxf.html')
    # CLUT类
    draw(case_name='ut_clut', split_task_id_list=[2], split_vector_list=[
         [1, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/clut.html')
    draw(case_name='ut_clut', split_task_id_list=[2], split_vector_list=[
         [2, 1, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/clut_y.html')
    draw(case_name='ut_clut', split_task_id_list=[2], split_vector_list=[
         [1, 2, 1, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/clut_x.html')
    draw(case_name='ut_clut', split_task_id_list=[2], split_vector_list=[
         [1, 1, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/clut_f.html')
    draw(case_name='ut_clut', split_task_id_list=[2], split_vector_list=[
         [2, 2, 2, 1, 1, 1]], out_path='docs/sphinx/source/仿真栈/资源级仿真器/_static/clut_yxf.html')
