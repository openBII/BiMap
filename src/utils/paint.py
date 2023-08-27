# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from top.global_config import GlobalConfig
import src.compiler.ir.paint_pb2 as tianjic_paint
from google.protobuf.text_format import ParseLines
import math
fig = plt.figure()
ax1 = fig.add_subplot(111, aspect='equal')


tasks = {}
id2tasks = {}
cores = set()
taskpos = {}
graph = tianjic_paint.TaskGraph()
total_phase = 0
with open("temp/paint.txt", "r") as f:
    ParseLines(f.readlines(), graph)


core_map = {}
core_num = 0
for task in graph.tasks:
    if task.core_id in core_map:
        core_id = core_map[task.core_id]
    else:
        core_map[task.core_id] = core_num
        core_id = core_num
        core_num += 1
    cores.add(core_id)
    if core_id not in tasks:
        tasks[core_id] = []
    tasks[core_id].append(task)
    id2tasks[task.id] = task
    total_phase = max(total_phase, task.end_phase)


ax1.axis([0, 400*len(tasks), 0, 400*(total_phase+1)])
ax1.axes.get_xaxis().set_visible(False)  # 隐藏x坐标轴
ax1.axes.get_yaxis().set_visible(False)  # 隐藏y坐标轴
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

def paint_core(core_id):
    length = 400

    phase2task = [[] for _ in range(total_phase+1)]
    for task in tasks[core_id]:
        for t in range(task.start_phase, task.end_phase+1):
            phase2task[t].append(task)

    for t in range(total_phase+1):
        n = math.ceil(math.sqrt(len(phase2task[t])))
        k = 2
        l = length / ((k+1)*n+1)

        ax1.add_patch(
            patches.Rectangle(
                (core_id*length, t*length),   # (x,y)
                width=length,          # width
                height=length,          # height
                fill=False,
                linewidth=2
            )
        )

        i = 0
        
        for task in phase2task[t]:
            if task.core_id % 2 == 0:
                x = int(i / n)
                y = int(i % n)
            else:
                x = n-1-int(i / n)
                y = int(i % n)
                
            kind = "S"
            if task.kind == tianjic_paint.E:
                kind = "E"
                if task.type == 3:
                    kind = "M"
            kind += str(task.id)
            if (task.length > 0):
                kind += "\n" + str(task.length)
            offset_x = (k+1)*l*x+l
            offset_y = (k+1)*l*y+l
            pos_x = core_id*length + offset_x
            pos_y = t*length + offset_y
            plt.text(pos_x, pos_y, kind, fontsize=3)

            ax1.add_patch(
                patches.Rectangle(
                    (pos_x, pos_y),   # (x,y)
                    l*k,          # width
                    l*k,          # height,
                    fill=False,
                    linewidth=0.1,
                )
            )

            if task.id not in taskpos:
             taskpos[task.id] = {}
            taskpos[task.id][t] = [(pos_x+l*k/2, pos_y), (pos_x, pos_y+l*k/2),
                                (pos_x+l*k, pos_y+l*k/2), (pos_x+l*k/2, pos_y+l*k)]
            i += 1


def paint_arrow():
    for task in id2tasks.values():
        for router in task.router:
            out_id = router.id
            out = id2tasks[out_id]
            src_t, dst_t = router.phase, router.phase
            x, y = taskpos[out_id][dst_t][0]
            x_, y_ = taskpos[task.id][src_t][0]
            dx, dy = x - x_, y - y_
            if dy > 0:
                src_idx = 3
                dst_idx = 0
            if dy < 0:
                src_idx = 0
                dst_idx = 3
            if dx > 0:
                src_idx = 2
                dst_idx = 1
            if dx < 0:
                src_idx = 1
                dst_idx = 2

            src_x, src_y = taskpos[task.id][src_t][src_idx]
            dst_x, dst_y = taskpos[out_id][dst_t][dst_idx]


            ax1.add_patch(
                patches.FancyArrow(
                    src_x, src_y,
                    dst_x - src_x, dst_y - src_y,
                    lw=0.1, width=2, length_includes_head=True,
                    head_length=10,
                )
            )
        for out_id in task.output:
            out = id2tasks[out_id]
            if task.kind == tianjic_paint.S:
                if task.start_phase == task.end_phase:
                    src_t, dst_t = task.start_phase, task.start_phase
                else:
                    src_t, dst_t = out.start_phase, out.start_phase
                if task.end_phase < out.start_phase:
                    src_t = task.end_phase
            else:
                src_t, dst_t = task.start_phase, task.start_phase
                if task.start_phase < out.start_phase:
                    dst_t = out.start_phase
            
            x, y = taskpos[out_id][dst_t][0]
            x_, y_ = taskpos[task.id][src_t][0]
            dx, dy = x - x_, y - y_
            if dy > 0:
                src_idx = 3
                dst_idx = 0
            if dy < 0:
                src_idx = 0
                dst_idx = 3
            if dx > 0:
                src_idx = 2
                dst_idx = 1
            if dx < 0:
                src_idx = 1
                dst_idx = 2

            src_x, src_y = taskpos[task.id][src_t][src_idx]
            dst_x, dst_y = taskpos[out_id][dst_t][dst_idx]


            ax1.add_patch(
                patches.FancyArrow(
                    src_x, src_y,
                    dst_x - src_x, dst_y - src_y,
                    lw=0.1, width=2, length_includes_head=True,
                    head_length=10,
                )
            )


for core in cores:
    paint_core(core)

paint_arrow()
fig.savefig(GlobalConfig.Path['temp'] + 'paint.png', bbox_inches='tight', dpi=1000)
