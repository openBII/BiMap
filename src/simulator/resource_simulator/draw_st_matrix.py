#!/usr/bin/env python
# coding: utf-8

"""
    绘制 ST Matrix 的桑基图 使用visitor方式
"""
from task_rabbit.task_model.task_graph import TaskGraph
from task_rabbit.task_model.task_block import TaskBlock

from resource_simulator.st_model.st_matrix import STMatrix, SpaceColumn, Coord

from pyecharts import options as opts
from pyecharts.charts import Sankey
from resource_simulator.evaluation_model.task_matrix_visitor import \
    TaskMatrixVisitor


class DrawSTMatrix(TaskMatrixVisitor):
    def __init__(self, out_path='./st_matrix_draw_sankey.html'):
        super(DrawSTMatrix, self).__init__()
        self.all_levels_value = {'all': 0}
        self.data = DrawDataManage()
        self.out_path = out_path

    # def visit(self, task):
    #     self.current_type = task.task_type.name + '(' + str(task.id) + ')'

    @staticmethod
    def get_task_name(item, base_name=''):
        if isinstance(item, TaskBlock):
            return base_name + item.task_type.name + '(' + str(item.id) + ')'
        else:
            assert type(item) is dict
            for idx, task in enumerate(item.values()):
                if idx % 3 == 2:
                    base_name += DrawSTMatrix.get_task_name(task) + '\n'
                else:
                    base_name += DrawSTMatrix.get_task_name(task) + ' '
            return base_name

    def visit_matrix(self, st_matrix):
        for key, item in st_matrix:
            self._space_coord = Coord(self._space_coord + (key,))
            dn = DrawName(self._space_coord, self._time_coord)
            this_name = dn.get_self_name
            fa_name = dn.get_fa_name
            self.data.add_node(this_name)
            self.all_levels_value[this_name] = 0
            item.accept(self)
            if len(self._space_coord) > 1:
                self.data.add_link(fa_name, this_name,
                                   self.all_levels_value[this_name])
            self._space_coord = self._space_coord.outer_coord

    def visit_column(self, space_column):
        for key, item in space_column:
            self._time_coord = Coord(self._time_coord + (key,))
            dn = DrawName(self._space_coord, self._time_coord)
            this_name = dn.get_self_name
            fa_name = dn.get_fa_name
            self.data.add_node(this_name)
            self.all_levels_value[this_name] = 0
            item.accept(self)
            self.data.add_link(fa_name, this_name,
                               self.all_levels_value[this_name])
            self._time_coord = self._time_coord.outer_coord

    def visit_point(self, st_point):
        for idx, name in enumerate(st_point.index_to_item):
            self._time_coord = Coord(self._time_coord + (idx,))
            dn = DrawName(self._space_coord, self._time_coord)
            this_name = dn.get_self_name
            self.all_levels_value[this_name] = 0
            all_fa_name = dn.get_all_fa_name
            fa_name = all_fa_name[1]
            item = st_point.get_time(Coord((idx,)), None)
            if item:
                for f in all_fa_name:
                    self.all_levels_value[f] += 1
                current_name = DrawSTMatrix.get_task_name(item, name + '\n')
                self.data.add_node(this_name + current_name)
                self.data.add_link(fa_name, this_name + current_name,
                                   self.all_levels_value[this_name])
                # if isinstance(item, TaskBlock):
                #     current_name = DrawSTMatrix.get_task_name(item)
                #     self.data.add_node(this_name + '\n' + current_name)
                #     self.data.add_link(fa_name,
                #                        this_name + '\n' + current_name,
                #                        self.all_levels_value[this_name])
                # else:
                #     dict_name = ''
                #     for item_value in item.values():
                #         dict_name = dict_name + '\n' + \
                #                     DrawSTMatrix.get_task_name(item_value)
                #     self.data.add_node(this_name + '\n' + name + ': '
                #                        + dict_name)
                #     self.data.add_link(fa_name, this_name + '\n' + name + ': '
                #                        + dict_name,
                #                        self.all_levels_value[this_name])
            self._time_coord = self._time_coord.outer_coord

    def draw(self):
        height = self.all_levels_value['all'] * 100
        init_opts = opts.InitOpts(width='2500px', height=str(height) + 'px')
        label_opts = opts.LabelOpts(is_show=True, position='inside')
        c = (
            Sankey(init_opts=init_opts)
                .add(
                "sankey",
                nodes=self.data.nodes,
                links=self.data.links,
                pos_top="10%",
                node_width=200,
                node_gap=15,
                focus_node_adjacency='allEdges',
                levels=[
                    opts.SankeyLevelsOpts(
                        depth=0,
                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=1,
                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=2,
                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=3,
                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=4,
                        itemstyle_opts=opts.ItemStyleOpts(color="#EE82EE"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=5,
                        itemstyle_opts=opts.ItemStyleOpts(color="#FFDAB9"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                    opts.SankeyLevelsOpts(
                        depth=6,
                        itemstyle_opts=opts.ItemStyleOpts(color="#63B8FF"),
                        linestyle_opts=opts.LineStyleOpts(color="source",
                                                          opacity=0.6),
                    ),
                ],
                linestyle_opt=opts.LineStyleOpts(curve=0.5),
                label_opts=label_opts,
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="ST Matrix"),
                tooltip_opts=opts.TooltipOpts(trigger="item",
                                              trigger_on="click"),
            )
                .render(self.out_path)
        )


class DrawName(object):
    def __init__(self, s_coord: Coord, t_coord: Coord):
        self.s_coord = s_coord
        self.t_coord = t_coord
        self.all_fa_list = []

    @property
    def get_self_name(self):
        if len(self.s_coord) > 0:
            if len(self.t_coord) > 0:
                return 'Space: ' + str(self.s_coord) + '\n' + 'Time: ' + str(
                    self.t_coord)
            else:
                return 'Space: ' + str(self.s_coord)
        else:
            return None

    @property
    def get_fa_name(self):
        if self.coord_go_up():
            return self.get_self_name

    @property
    def get_all_fa_name(self):
        res = self.get_self_name
        if res is not None:
            self.all_fa_list.append(res)
            if self.coord_go_up():
                return self.get_all_fa_name
            else:
                raise ValueError('Should not be here!')
        else:
            self.all_fa_list.append('all')
            return self.all_fa_list

    def coord_go_up(self):
        if len(self.s_coord) > 0:
            if len(self.t_coord) > 0:
                self.t_coord = self.t_coord.outer_coord
                return True
            else:
                self.s_coord = self.s_coord.outer_coord
                return True
        else:
            return False


class DrawDataManage(object):
    def __init__(self):
        self.nodes = []
        self.links = []

    def add_node(self, node: str):
        self.nodes.append({'name': node})

    def add_link(self, src: str, dst: str, value: int):
        self.links.append({'source': src, 'target': dst, 'value': value})
