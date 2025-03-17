import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 定义颜色
color = {
    "DRAM_PIM": "grey",
    "SRAM_PIM": "#003f7c",
    "SRAM_PIM_PIPE": "#9400d3",
}

# 读取CSV文件
data = pd.read_csv('figs/CompAir/SRAMvsDRAM.csv')

# 提取数据
batch_sizes = data['BatchSize']
dram_pim = data['DRAM-PIM']
sram_pim = data['SRAM-PIM']
sram_pim_pipe = data['SRAM-PIM Pipe']

# 设置柱状图的宽度
bar_width = 0.2

# 设置x轴的位置
plt.figure(figsize=(8, 2.5))
x = np.arange(len(batch_sizes))
# 设置透明度
alpha = 0.5
plt.grid(alpha=alpha)

# 创建柱状图
plt.bar(x - bar_width, dram_pim, width=bar_width, color=color['DRAM_PIM'], label='DRAM-PIM')
plt.bar(x, sram_pim, width=bar_width, color=color['SRAM_PIM'], label='SRAM-PIM')
plt.bar(x + bar_width, sram_pim_pipe, width=bar_width, color=color['SRAM_PIM_PIPE'], label='SRAM-PIM Pipe')

# 设置x轴的刻度和标签
plt.xticks(x, batch_sizes)
plt.xlabel('Batch Size', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.yscale('log')
plt.title('Comparison of DRAM-PIM, SRAM-PIM for Different Batch Sizes', fontsize=16)

# 添加图例
plt.legend()
plt.tight_layout()
# 显示图形
plt.savefig('figs/CompAir/SRAMvsDRAM.pdf')