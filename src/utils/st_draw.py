# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import copy
import os

from pyecharts import options as opts
from pyecharts.charts import Graph, Page, Sankey, TreeMap
from pyecharts.components import Table
from pyecharts.options import (ComponentTitleOpts, InitOpts, ItemStyleOpts,
                               LabelOpts, LineStyleOpts)

from src.simulator.resource_simulator.st_model.st_matrix import (Coord,
                                                                 SpaceColumn,
                                                                 STMatrix)
from src.simulator.resource_simulator.st_model.st_point import STPoint
from src.simulator.task_rabbit.task_model.ctask_block import CTaskBlock
from src.simulator.task_rabbit.task_model.stask_block import STaskBlock
from src.simulator.task_rabbit.task_model.task_block import TaskBlock
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from top.global_config import GlobalConfig


class STDraw:
    """
    采用pyecharts库，将TaskGraph和STMatrix可视化为html文件
    """

    @staticmethod
    def draw_graph(graph: TaskGraph, out_path=GlobalConfig.Path['temp'] + 'task_graph_draw.html',
                   width='8000px', height='4000px'):
        """
        可视化Task Graph
        out_path为输出html文件的路径
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        graph = STDraw.obtain_graph(graph=graph, width=width, height=height)
        graph.render(out_path)

    @staticmethod
    def obtain_graph(graph: TaskGraph, width='1500px', height='800px') -> Graph:
        """
        获取针对TaskGraph的pyecharts图对象
        在本绘图中，存储任务块为绿色字体；计算任务块为蓝色字体
        """
        def get_node_color(node: TaskBlock):
            if isinstance(node, CTaskBlock):
                return 'blue'
            elif isinstance(node, STaskBlock):
                return 'green'
            else:
                return 'red'

        nodes_data = []
        for node_id in graph.get_all_node_ids():
            node = graph.get_node(node_id)
            node_id = node_id if type(node_id) is str else str(node_id)
            label_cc = LabelOpts(is_show=True, position='inside', color=get_node_color(node),
                                 font_size=7, font_weight='bold',
                                 formatter=node.task_type.name + '\n' + node_id + '\n' + str(node.shape)[1:-1])
            nodes_data.append(opts.GraphNode(name=node.task_type.name + '\n' + node_id, symbol_size=[80, 40],
                                             symbol='roundRect', label_opts=label_cc))

        links_data = []
        # link_color = ['rgb(61, 145, 64)', 'rgb(106, 90, 205)', 'rgb(56, 94, 15)', 'rgb(0, 255, 0)',
        #               'rgb(135, 206, 235)', 'rgb(107, 142, 35)', 'rgb(160, 32, 240)', 'rgb(218, 112, 214)',
        #               'rgb(3, 168, 158)', 'rgb(65, 105, 225)', 'rgb(255, 192, 203)', 'rgb(250, 128, 114)',
        #               'rgb(255, 0, 255)',  'rgb(156, 102, 31)', 'rgb(112, 128, 105)']
        for _, node in graph:
            # random.shuffle(link_color)
            for icc_idx, icc in enumerate(node.input_clusters):
                link_type = LineStyleOpts(is_show=True, width=1, opacity=1, curve=0, type_='solid',
                                          color='black')  # color=link_color[icc_idx % len(link_color)]
                for ic in icc.all_enable_edges:
                    if type(ic.in_task.id) is str:
                        in_task_id = ic.in_task.task_type.name + '\n' + ic.in_task.id
                        out_task_id = ic.out_task.task_type.name + '\n' + ic.out_task.id
                    else:
                        in_task_id = ic.in_task.task_type.name + \
                            '\n' + str(ic.in_task.id)
                        out_task_id = ic.out_task.task_type.name + \
                            '\n' + str(ic.out_task.id)
                    links_data.append(opts.GraphLink(source=in_task_id, target=out_task_id, value=None,
                                                     linestyle_opts=link_type))

        init_opts = InitOpts(width=width, height=height,
                             renderer='canvas', bg_color='white')
        c = (
            Graph(init_opts=init_opts)
            .add(
                "",
                nodes_data,
                links_data,
                is_draggable=True,
                repulsion=300,  # 节点之间的斥力因子
                edge_symbol=[None, 'arrow'],
                edge_symbol_size=8,
                itemstyle_opts=ItemStyleOpts(
                    color='rgb(230, 230, 230)', border_color='black', opacity=0.7)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Task_Graph")
            )
            # .render(out_path)
        )
        return c

    @ staticmethod
    def draw_matrix_table(matrix: STMatrix, out_path=GlobalConfig.Path['temp'] + 'st_matrix_draw_sankey.html'):
        """
        可视化ST Matrix为表格
        out_path为输出html文件的路径
        """
        table = STDraw.obtain_matrix_table(matrix=matrix)
        table.render(out_path)

    @staticmethod
    def draw_matrix_map_tree(matrix: STMatrix, leaf_depth=1, width=900, height=500, out_path=GlobalConfig.Path['temp'] + 'st_matrix_draw_map_tree.html'):
        def get_all_item(st, sub_data, pre_name='Space: ('):
            if isinstance(st, STMatrix):
                children_idx = 0
                for s_top in st.container.keys():
                    sub_data.append(
                        {'value': 100, 'name': pre_name + str(s_top) + ', ', 'children': []})
                    get_all_item(
                        st.container[s_top], sub_data[children_idx]['children'], pre_name + str(s_top) + ', ')
                    children_idx += 1
            elif isinstance(st, SpaceColumn):
                children_idx = 0
                if 'Time' not in pre_name:
                    pre_name = pre_name[:-2] + ')\nTime: ('
                for t_top in st.container.keys():
                    sub_data.append(
                        {'value': 100, 'name': pre_name + str(t_top) + ', ', 'children': []})
                    get_all_item(
                        st.container[t_top], sub_data[children_idx]['children'], pre_name + str(t_top) + ', ')
                    children_idx += 1
            elif isinstance(st, STPoint):
                for pi_id in range(len(st.index_to_item)):
                    pi = st.get_time(Coord((pi_id,)))
                    if pi:
                        if type(pi) is not dict:
                            pi_name = pi.task_type if type(
                                pi.task_type) == str else str(pi.task_type)
                            sub_data.append({'value': 100, 'name': pre_name + str(pi_id) + ')\n' + st.index_to_item[
                                pi_id] + ': ' + pi_name})
                        else:
                            sub_data.append(
                                {'value': 100, 'name': pre_name + str(pi_id) + ')\n' + st.index_to_item[pi_id]})

        def set_value(sub_data):  # 将data中的value值按照该层children的个数设置
            for i in range(len(sub_data)):
                if sub_data[i].get('children') is None:
                    sub_data[i]['value'] = 1
                else:
                    sub_data[i]['value'] = len(sub_data[i]['children'])
                    set_value(sub_data[i]['children'])

        data = []
        get_all_item(matrix, data)
        set_value(data)

        init_opts = opts.InitOpts(
            width=str(width) + 'px', height=str(height) + 'px')  # 初始化配置项
        treemap = (
            TreeMap(init_opts=init_opts)
            .add(
                series_name="",
                data=data,
                leaf_depth=leaf_depth,
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="ST_Matrix"))
        )
        treemap.render(path=out_path)

    @staticmethod
    def obtain_matrix_sankey(matrix: STMatrix, width=2000, height=500) -> Sankey:

        def get_value(st):
            if isinstance(st, STMatrix):
                value = 0
                for top in st.container.keys():
                    value += get_value(st.container[top])
                return value
            elif isinstance(st, SpaceColumn):
                value = 0
                for top in st.container.keys():
                    value += get_value(st.container[top])
                return value
            elif isinstance(st, STPoint):
                value = 0
                for pi_id in range(len(st.index_to_item)):
                    pi = st.get_time(Coord((pi_id,)))
                    if pi:
                        value += 1
                    else:
                        value += 0
                return value

        def get_all_item(st, node, link, pre_name='Space: ('):
            if isinstance(st, STMatrix):
                for top in st.container.keys():
                    this_name = pre_name
                    if pre_name[-1] == '(':
                        this_name += str(top)
                        node.append({'name': this_name})
                        pre_value = get_value(st.container[top])
                        get_all_item(st.container[top], node, link, this_name)
                    else:
                        this_name += ', ' + str(top)
                        node.append({'name': this_name})
                        post_value = get_value(st.container[top])
                        link.append(
                            {'source': pre_name, 'target': this_name, 'value': post_value})
                        get_all_item(st.container[top], node, link, this_name)
            elif isinstance(st, SpaceColumn):
                for top in st.container.keys():
                    this_name = pre_name
                    if 'Time' not in pre_name:
                        this_name = pre_name + ')\nTime: ('
                    if this_name == pre_name:
                        this_name += ', ' + str(top)
                    else:
                        this_name += str(top)
                    node.append({'name': this_name})
                    post_value = get_value(st.container[top])
                    link.append(
                        {'source': pre_name, 'target': this_name, 'value': post_value})
                    get_all_item(st.container[top], node, link, this_name)
            elif isinstance(st, STPoint):
                for pi_id in range(len(st.index_to_item)):
                    pi = st.get_time(Coord((pi_id,)))
                    if pi:
                        if type(pi) is not dict:
                            pi_name = pi.task_type if type(
                                pi.task_type) == str else str(pi.task_type)
                            this_name = pre_name + ', ' + str(pi_id) + ')\n' + st.index_to_item[
                                pi_id] + ': ' + pi_name + '\nID: ' + str(pi.id)
                            node.append({'name': this_name})
                            link.append(
                                {'source': pre_name, 'target': this_name, 'value': 1})
                        else:
                            for task_id in pi:
                                this_name = pre_name + ', ' + \
                                    str(pi_id) + ')\n' + st.index_to_item[pi_id] + ': ' + str(
                                        pi[task_id].task_type) + '\nID: ' + str(task_id)
                                node.append({'name': this_name})
                                link.append(
                                    {'source': pre_name, 'target': this_name, 'value': 1})
        nodes = []
        links = []
        get_all_item(matrix, nodes, links)
        total_value = 0
        for s_top in matrix.container.keys():
            total_value += get_value(matrix.container[s_top])
        height = total_value * 150
        init_opts = opts.InitOpts(
            width=str(width) + 'px', height=str(height) + 'px')
        label_opts = opts.LabelOpts(is_show=True, position='inside')
        c = (
            Sankey(init_opts=init_opts)
            .add(
                "sankey",
                nodes=nodes,
                links=links,
                pos_top="10%",
                node_width=150,
                node_gap=10,
                focus_node_adjacency='allEdges',
                levels=[
                    opts.SankeyLevelsOpts(
                        depth=0,
                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=1,
                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=2,
                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=3,
                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=4,
                        itemstyle_opts=opts.ItemStyleOpts(color="#EE82EE"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=5,
                        itemstyle_opts=opts.ItemStyleOpts(color="#FFDAB9"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=6,
                        itemstyle_opts=opts.ItemStyleOpts(color="#63B8FF"),
                        linestyle_opts=opts.LineStyleOpts(
                            color="source", opacity=0.6),
                    ),
                ],
                linestyle_opt=opts.LineStyleOpts(curve=0.5),
                label_opts=label_opts,
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="ST Matrix"),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", trigger_on="mousemove"),
            )
            # .render(out_path)
        )
        return c

    @ staticmethod
    def draw_matrix_sankey(matrix: STMatrix, width=2000, height=500, out_path=GlobalConfig.Path['temp'] + 'st_matrix_draw_sankey.html'):
        c = STDraw.obtain_matrix_sankey(
            matrix=matrix, width=width, height=height)
        c. render(out_path)

    @staticmethod
    def obtain_matrix_table(matrix: STMatrix) -> Table:
        """
        获取针对ST Matrix的pyecharts图对象
        """
        all_columns = []

        def get_all_item(st, column, phase=0):

            if isinstance(st, STMatrix):
                for top in st.container.keys():
                    column.append(top)
                    get_all_item(st.container[top], column)
                    column.pop()
            if isinstance(st, SpaceColumn):
                has_st_point = False

                for top in st.container.keys():
                    if isinstance(st.container[top], STPoint):
                        has_st_point = True
                        break

                if has_st_point:
                    for i in range(max(st.container.keys()) + 1):
                        if i in st.container:
                            get_all_item(st.container[i], column)
                        else:
                            column.extend([''] * 4)
                    has_content = False
                    for c in column[4:]:
                        if c:
                            has_content = True
                    if has_content:
                        all_columns.append(copy.copy(column))
                    for i in range(4 * max(st.container.keys()) + 4):
                        column.pop()
                else:
                    for top in st.container.keys():
                        get_all_item(st.container[top], column)
            if isinstance(st, STPoint):
                if st.axon is not None:
                    column.append(str(st.axon.id))
                else:
                    column.append('')

                if st.soma1 is not None:
                    column.append(str(st.soma1.id))
                else:
                    column.append('')

                if st.soma2 is not None:
                    column.append(str(st.soma2.id))
                else:
                    column.append('')

                if st.memory is not None:
                    memory_str = ''
                    for memory_key in st.memory.keys():
                        memory_str += str(memory_key) + ' '
                    column.append(memory_str)
                else:
                    column.append('')

        column = []
        get_all_item(matrix, column)
        # print(all_columns)
        table_column_value = len(all_columns)
        table_row_cnt = 0
        # print(all_columns[0])
        # for i in all_columns[i]:
        #     if len(all_columns[i]) > table_row_cnt:
        #         table_row_cnt = len(all_columns[i])
        # table_row_value = table_row_cnt
        for i in range(table_column_value):
            if len(all_columns[i]) > table_row_cnt:
                table_row_cnt = len(all_columns[i])
        table_row_value = table_row_cnt

        rows = []
        for i in range(table_row_value):
            rows.append([])
            for j in range(table_column_value + 1):
                rows[i].append(str(''))

        for i in range(table_row_value):
            if int(i % 4) == 0:
                if int(i//4) == 0:
                    rows[i][0] = 'chip'
                else:
                    rows[i][0] = 'phase' + ' ' + str(int(i//4) - 1) + ' - Axon'
            elif int(i % 4) == 1:
                if int(i//4) == 0:
                    rows[i][0] = 'step_group'
                else:
                    rows[i][0] = 'phase' + ' ' + \
                        str(int(i//4) - 1) + ' - Soma1'
            elif int(i % 4) == 2:
                if int(i//4) == 0:
                    rows[i][0] = 'phase_group'
                else:
                    rows[i][0] = 'phase' + ' ' + \
                        str(int(i//4) - 1) + ' - Soma2'
            elif int(i % 4) == 3:
                if int(i//4) == 0:
                    rows[i][0] = 'core'
                else:
                    rows[i][0] = 'phase' + ' ' + \
                        str(int(i//4) - 1) + ' - memory'

        # for i in range(table_row_value):
        #     for j in range(table_column_value + 1):
        #         if j <= (len(all_columns[j])):
        #             rows[i][j+1] = all_columns[j][i]

        for j in range(table_column_value + 1):
            if j < table_column_value:
                for i in range(len(all_columns[j])):
                    rows[i][j+1] = all_columns[j][i]

        table = Table()
        headers = []
        table.add(headers, rows)
        table.set_global_opts(
            title_opts=ComponentTitleOpts(title="ST Matrix"))

        return table
        # table.render(out_path)

    @staticmethod
    def draw_graph_table(matrix: STMatrix, graph: TaskGraph, out_path=GlobalConfig.Path['temp'] + 'st_matrix_draw_merge.html', width='1500px', height='800px'):
        """
        将Task Graph和STMatrix可视化到一个html文件中
        out_path为输出html文件的路径
        """
        page = Page(layout=Page.SimplePageLayout)
        page.add(
            STDraw.obtain_graph(graph=graph, width=width, height=height),
            STDraw.obtain_matrix_table(matrix=matrix),
        )
        page.render(out_path)
