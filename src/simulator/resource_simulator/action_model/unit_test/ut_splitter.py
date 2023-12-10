from task_rabbit.initial_pass import execute_initial_pass
from src.compiler.mapper.passes.final_pass import FinalPass
from resource_simulator.st_draw import STDraw
import numpy as np
from top.config import GlobalConfig
from flow.execute import exe_task_rabbit_with_task, exe_task_rabbit_with_map
from resource_simulator.st_env import STMatrix, STEnv, create_st_env
from task_rabbit.task_model.shape import Shape
from typing import List


def compare(task_path, map_path, task_name, map_name, task_id):
    exe_task_rabbit_with_task(
        case_path=task_path, case_name=task_name)
    exe_task_rabbit_with_map(
        case_path=map_path, case_name=map_name)
    ref_result = np.fromfile(
        GlobalConfig.Path["temp"] + task_name + '/task_out/task_block' + str(task_id) + '.dat', dtype=np.int32)
    split_result = np.fromfile(
        GlobalConfig.Path["temp"] + map_name + '/map_out/task_block' + str(task_id) + '.dat', dtype=np.int32)
    assert (ref_result == split_result).all(), 'Comparison failed!'
    print('Comparison successful!')


def unit_test(case_name: str, split_task_id_list: List[int], split_vector_list: List[List[int]], compare_task_id: int):
    task_path = GlobalConfig.Path["test_lib"] + 'task_lib/1P/' + case_name + '.task.txt'
    init = execute_initial_pass(
        case_path=task_path, case_name=case_name, input_type='task')
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

    STDraw.draw_graph(task_graph, out_path=GlobalConfig.Path["temp"] + case_name + '/' + case_name + '.task.html',
                      width='1920px', height='1080px')

    map_path = GlobalConfig.Path["test_lib"] + 'mapping_lib/' + case_name + '.map'
    FinalPass(st_env=st_env, out_path=map_path)

    compare(task_path, map_path, task_name=case_name,
            map_name=case_name + '_split', task_id=compare_task_id)


if __name__ == '__main__':
    # CVM类
    unit_test(case_name='cvm', split_task_id_list=[3], split_vector_list=[[1, 1, 2, 2, 1, 1]], compare_task_id=4)
    # CC类
    unit_test(case_name='cc', split_task_id_list=[3], split_vector_list=[[2, 2, 2, 2, 1, 1]], compare_task_id=4)
    # CCMPB类
    unit_test(case_name='ccmpb', split_task_id_list=[1], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=2)
    unit_test(case_name='cavg', split_task_id_list=[2], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=3)
    unit_test(case_name='cadd', split_task_id_list=[3], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=4)
    unit_test(case_name='cvvh', split_task_id_list=[3], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=4)
    unit_test(case_name='cvs', split_task_id_list=[2], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=3)
    unit_test(case_name='cax', split_task_id_list=[3], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=4)
    # CLUT类
    unit_test(case_name='clut', split_task_id_list=[2], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=3)
    # ResBlock
    unit_test(case_name='resblock', split_task_id_list=[1], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=19)
    unit_test(case_name='resblock', split_task_id_list=[5], split_vector_list=[[2, 2, 2, 2, 1, 1]], compare_task_id=19)
    unit_test(case_name='resblock', split_task_id_list=[16], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=19)
    unit_test(case_name='resblock', split_task_id_list=[18], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=19)
    unit_test(
        case_name='resblock',
        split_task_id_list=[1, 5, 16, 18],
        split_vector_list=[[2, 2, 2, 1, 1, 1], [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1], [2, 2, 2, 1, 1, 1]],
        compare_task_id=19
    )
    # ccmpb_cc_ccmpb
    unit_test(case_name='ccmpb_cc_ccmpb', split_task_id_list=[1], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=8)
    unit_test(case_name='ccmpb_cc_ccmpb', split_task_id_list=[5], split_vector_list=[[2, 2, 2, 2, 1, 1]], compare_task_id=8)
    unit_test(case_name='ccmpb_cc_ccmpb', split_task_id_list=[7], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=8)
    unit_test(
        case_name='ccmpb_cc_ccmpb',
        split_task_id_list=[1, 5, 7],
        split_vector_list=[[2, 2, 2, 1, 1, 1], [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1]],
        compare_task_id=8
    )
    # ccmpb_cc_ccmpb_cavg_ccmpb_cvm
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              1], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=16)
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              5], split_vector_list=[[2, 2, 2, 2, 1, 1]], compare_task_id=16)
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              7], split_vector_list=[[2, 2, 2, 1, 1, 1]], compare_task_id=16)
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              9], split_vector_list=[[1, 1, 2, 1, 1, 1]], compare_task_id=16)
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              11], split_vector_list=[[1, 1, 2, 1, 1, 1]], compare_task_id=16)
    unit_test(case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm', split_task_id_list=[
              15], split_vector_list=[[1, 1, 2, 2, 1, 1]], compare_task_id=16)
    unit_test(
        case_name='ccmpb_cc_ccmpb_cavg_ccmpb_cvm',
        split_task_id_list=[1, 5, 7, 9, 11, 15],
        split_vector_list=[[2, 2, 2, 1, 1, 1], [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1], [
            1, 1, 2, 1, 1, 1], [1, 1, 2, 1, 1, 1], [1, 1, 2, 2, 1, 1]],
        compare_task_id=16
    )
    # split_group
    case_name = 'ccmpb_cc_ccmpb'
    task_path = GlobalConfig.Path["test_lib"] + 'task_lib/MP/ccmpb_cc_ccmpb.task.txt'
    st_env = create_st_env(task_path=task_path, case_name=case_name)
    st_env.split_group(
        task_id_list=[1, 7],
        split_vector=Shape(1, 1, 2, 1, 1, 1)
    )
    st_env.check_graph()
    map_path = GlobalConfig.Path["test_lib"] + 'mapping_lib/' + case_name + '.map'
    FinalPass(st_env=st_env, out_path=map_path)
    compare(task_path, map_path, task_name=case_name,
            map_name=case_name + '_split', task_id=8)

    case_name = 'ccmpb_cc_ccmpb'
    task_path = GlobalConfig.Path["test_lib"] + 'task_lib/MP/ccmpb_cc_ccmpb.task.txt'
    st_env = create_st_env(task_path=task_path, case_name=case_name)
    st_env.split_group(
        task_id_list=[1, 5, 7],
        split_vector=[
            Shape(1, 1, 2, 1, 1, 1),
            Shape(1, 1, 1, 2, 1, 1),
            Shape(1, 1, 2, 1, 1, 1)
        ]
    )
    st_env.check_graph()
    map_path = GlobalConfig.Path["test_lib"] + 'mapping_lib/' + case_name + '.map'
    FinalPass(st_env=st_env, out_path=map_path)
    compare(task_path, map_path, task_name=case_name,
            map_name=case_name + '_split', task_id=8)
