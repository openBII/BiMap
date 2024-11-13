import matplotlib.pyplot as plt
import toml
import numpy as np
# from test.exp.distributed_many_core_memory_compute import simulate as many_core_prefill_simulate_2p5mb
# from test.exp.distributed_many_core_memory_compute import simulate_1mb as many_core_prefill_simulate_1mb
# from test.exp.distributed_many_core_memory_compute import simulate_2mb as many_core_prefill_simulate_2mb
# from test.exp.distributed_many_core_memory_compute import simulate_3mb as many_core_prefill_simulate_3mb


def draw_one_ram_size(ram, file_name: str):

    # if ram == 1:
    #     sim_func = many_core_prefill_simulate_1mb
    # elif ram == 2:
    #     sim_func = many_core_prefill_simulate_2mb
    # elif ram == 2.5 or ram == 3:
    #     sim_func = many_core_prefill_simulate_3mb
    # else:
    #     raise NotImplementedError

    # noc_bandwidth_paramters = []
    # local_memory_latency_paramters = []
    # local_memory_bandwidth_paramters = []
    # latencies = []
    if ram == 1:
        lmb = [16, 32, 64, 128, 256, 512]
    elif ram == 2:
        lmb = [8, 16, 32, 64, 128, 256]
    else:
        lmb = [4, 8, 16, 32, 64, 128]
    lml = [i for i in range(80, 1, -10)]
    nb = [64, 32, 16, 8, 4]
    # for local_memory_bandwidth in lmb:
    #     for local_memory_latency in lml:
    #         for noc_bandwidth in nb:
    #             noc_bandwidth_paramters.append(noc_bandwidth)
    #             local_memory_latency_paramters.append(local_memory_latency)
    #             local_memory_bandwidth_paramters.append(local_memory_bandwidth)
    #             latencies.append(sim_func({'noc_bandwidth': noc_bandwidth,
    #                             'local_memory_latency': local_memory_latency,
    #                             'local_memory_bandwidth': local_memory_bandwidth}))
    # np.savez('{:s}.npz'.format(file_name), 
    #      noc_bandwidth_paramters=np.array(noc_bandwidth_paramters), 
    #      local_memory_latency_paramters=np.array(local_memory_latency_paramters),
    #      local_memory_bandwidth_paramters=np.array(local_memory_bandwidth_paramters),
    #      latencies=np.array(latencies))
    
    data = np.load('{:s}.npz'.format(file_name))
    noc_bandwidth_paramters = data["noc_bandwidth_paramters"]
    local_memory_latency_paramters = data["local_memory_latency_paramters"]
    local_memory_bandwidth_paramters = data["local_memory_bandwidth_paramters"]
    latencies = data["latencies"]

    # import scienceplots
    # plt.style.use(['ieee'])
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置柱的参数
    dx = 1
    dy = 0.6
    dz = latencies  # 柱的高度
    font_size = 9

    # 将bandwidth取对数
    local_memory_bandwidth_paramters = np.log2(local_memory_bandwidth_paramters)
    noc_bandwidth_paramters = np.log2(max(noc_bandwidth_paramters)) - np.log2(noc_bandwidth_paramters) + 1

    # x1: local bandwidth x2: latency x3: noc bandwidth
    # 定义偏移量，用于将x2编码成不同的层
    # offset = 0.5
    # x1_combined = [local_memory_bandwidth_paramters[i] * 10 + (20 - local_memory_latency_paramters[i]) * offset for i in range(len(local_memory_latency_paramters))]  # 将x1和x2组合
    x1_combined = []
    offset = (12.0) / len(lml)
    for i, local_memory_bandwidth in enumerate(lmb):
        for j, local_memory_latency in enumerate(lml):
            for noc_bandwidth in nb:
                x1_combined.append(i * 20 + j * offset)

    # 排序，改变绘制柱子的顺序，避免重叠
    sorted_indices = np.argsort(x1_combined)
    x1_combined = np.array(x1_combined)[sorted_indices]
    noc_bandwidth_paramters = np.array(noc_bandwidth_paramters)[sorted_indices]
    dz = np.array(latencies)[sorted_indices]
    # 根据x1的值设置颜色，颜色随着数值变大而变化
    colors = plt.cm.viridis((np.array(local_memory_latency_paramters) - min(local_memory_latency_paramters)) / (max(local_memory_latency_paramters) - min(local_memory_latency_paramters)))

    # 绘制三维柱状图
    ax.bar3d(x1_combined, noc_bandwidth_paramters, np.zeros_like(latencies), dx, dy, dz, color=colors)
    ax.zaxis.set_tick_params(labelsize=font_size)
    # # 添加二维坐标标签(x1, x2)作为y轴上的标记
    # for i in range(len(y)):
    #     ax.text(x1[i], x2_combined[i], y[i] + 0.5, f"({x2[i]},{x3[i]})", color='black', ha='center')

    # 设置坐标轴标签
    ax.set_xlabel('Local Memory Bandwidth', fontsize=10)
    ax.set_ylabel('NoC Bandwidth', fontsize=10)
    ax.set_zlabel('Latency', fontsize=10)
    ax.set_xticks([i * 20 + 3 for i in range(len(lmb))])
    ax.set_xticklabels(lmb, fontsize=font_size)
    ax.set_yticks(np.log2(max(nb)) - np.log2(nb) + 1.5)
    ax.set_yticklabels(nb, fontsize=font_size)

    plt.title("Distributed Many Core")

    # 创建图例
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=color, label=f'{height}')
        for color, height in zip(colors[::len(nb)], lml)
    ]
    alegend = ax.legend(handles=legend_patches, bbox_to_anchor=(-0.25, 0.8), loc='upper left', fontsize=8,
                frameon=True, fancybox=False,
                title='Local Memory\nLatency', title_fontsize=9)
    alegend.get_title().set_ha('center')

    plt.savefig(file_name + '.png', dpi=1000)


def draw_calculate_mem_ratio(latencies: np.array, param_ticks: list, file_name: str):

    # dim1 -> 4 mapping scheme for different SRAM size
    # dim2 -> 3 different parameters (local_mem_latency, local_mem_bandwidth, noc_bandwidth)
    #                              (shared_mem_latency, shared_mem_bandwidth, local_mem_bandwidth)
    # dim3 -> 2 different kind of latency (calculate, access memory)
    # latencies = np.zeros((4, 3, 2))
    # latencies = np.random.rand(*latencies.shape)
    # latencies = np.array([
    #     [[1, 1], [2, 2], [3, 3]],
    #     [[4, 4], [5, 5], [6, 6]],
    #     [[7, 7], [8, 8], [9, 9]],
    #     [[10, 10], [11, 11], [12, 12]]
    # ])

    # 生成 x 轴的位置
    x = np.arange(latencies.shape[0])  # 4种存算比
    x_ticks = ['1MB/', '2MB/', '2.5MB', '3MB']
    # param_ticks = ['p1', 'p2', 'p3', ]  # 不同的mem latency 和 bandwidth
    # 颜色映射
    colors = {
        'calcute'    : "skyblue",
        'mem access' : "lightgreen"
    }

    assert(len(x_ticks) == latencies.shape[0] and 
           len(param_ticks) == latencies.shape[1] and
           len(colors) == latencies.shape[2])

    width = (1.0 - 0.3) / latencies.shape[1]  # 设置柱状图宽度

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(latencies.shape[1]):
        # 每次绘制一种parameter下，四种存算比的结果
        for j in range(latencies.shape[2]):
            ax.bar(x + i * width, latencies[:, i, j], width, color=list(colors.values())[j], edgecolor='black',
                   bottom=latencies[:, i, j-1] if j > 0 else None,
                   label=list(colors.keys())[j] if i == 0 else '')
        # 计算每种param下，总延迟，绘制折线
        total_latency = np.sum(latencies[:, i, :], axis=1)
        ax.plot(x + i * width, total_latency, marker='o', color='black', linestyle='-', alpha=0.7)

    # 设置 x 轴标签
    ax.set_xticks(x + width * ( latencies.shape[1] - 1.0) / 2.0 )
    ax.set_xticklabels(x_ticks)
    ax.tick_params(axis='x', labelsize=12, pad=65)

    # 设置param横坐标标签
    for i in range(latencies.shape[0]):
        for j in range(latencies.shape[1]):
            ax.text(x[i] + j * width - width * 0.5, -0.5, param_ticks[j], ha='center', va='top', 
                    rotation=45, fontsize=8)

    # 设置标签和标题
    ax.set_xlabel("Exams")
    ax.set_ylabel("Scores")
    ax.set_title("Scores by Subject for Each Student Across 4 Exams")

    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=12, frameon=True, fancybox=False,
              title='', title_fontsize=9)

    # 显示图形
    plt.tight_layout()
    plt.savefig(file_name + '.png', dpi=1000)


def draw_cal_mem_ratio_test():
    # 把所有240个参数的折线都画出来，看一下趋势
    latency = []
    for sss in ['1mb', '2mb', '2p5mb', '3mb']:
        data = np.load('temp/many_core_decoder_{:s}.npz'.format(sss))
        noc_bandwidth_paramters = data["noc_bandwidth_paramters"]
        local_memory_latency_paramters = data["local_memory_latency_paramters"]
        local_memory_bandwidth_paramters = data["local_memory_bandwidth_paramters"]
        latency.append(data["latencies"])

    # 数据
    x = [1, 2, 3, 4]

    latency = np.array(latency)

    for i in range(len(latency[0])):
        plt.plot(x, latency[:, i])

    # 添加标题和标签
    plt.title("Line Plot with Two Groups")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    # 显示图例
    plt.legend()

    # 显示图表
    plt.savefig('temp/many_core_decoder_cal_mem_ration_test.png', dpi=1000)



if __name__ == '__main__':
    # latencies = np.zeros((4, 8, 2))
    # param_ticks = []
    # for i, sss in enumerate(['1mb', '2mb', '2p5mb', '3mb']):
    #     data = np.load('temp/many_core_decoder_{:s}.npz'.format(sss))
    #     noc_bandwidth_paramters = data["noc_bandwidth_paramters"]
    #     local_memory_latency_paramters = data["local_memory_latency_paramters"]
    #     local_memory_bandwidth_paramters = data["local_memory_bandwidth_paramters"]
    #     latency = np.array(data["latencies"])
    
    #     idx = 0
    #     for local_mem_band in [4, 64]:
    #         for noc_band in [4, 128]:
    #             for local_mem_latency in [80, 10]:
    #                 i1 = np.where(local_memory_bandwidth_paramters == local_mem_band)
    #                 i2 = np.where(noc_bandwidth_paramters == noc_band)
    #                 i3 = np.where(local_memory_latency_paramters == local_mem_latency)
    #                 i4 = np.intersect1d(i1[0], i2[0])
    #                 i5 = np.intersect1d(i4, i3[0])
    #                 latencies[i, idx, 0] = latency[i5]
    #                 latencies[i, idx, 1] = 0
    #                 idx += 1
    #                 if i == 0:
    #                     param_ticks.append('({:d}, {:d}, {:d})'.format(local_mem_band, noc_band, local_mem_latency))
    # draw_calculate_mem_ratio(latencies=latencies, file_name='temp/many_core_decoder_cal_mem_ratio', param_ticks=param_ticks)

    draw_one_ram_size(ram=1, file_name='test/exp/many_core_1_prefill')