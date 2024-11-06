import matplotlib.pyplot as plt
import numpy as np

data = np.load('test/exp/shared_memory_data.npz')
shared_memory_bandwidth_parameters = data["shared_memory_bandwidth_parameters"]
shared_memory_latency_parameters = data["shared_memory_latency_parameters"]
local_memory_bandwidth_parameters = data["local_memory_bandwidth_parameters"]
latencies = data["latencies"]

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置柱的参数
dx = dy = 1  # 柱的宽度
dz = latencies  # 柱的高度

# 将bandwidth取对数
shared_memory_bandwidth_parameters = np.log2(shared_memory_bandwidth_parameters)
local_memory_bandwidth_parameters = np.log2(local_memory_bandwidth_parameters)

# x1: local bandwidth x2: latency x3: noc bandwidth
# 定义偏移量，用于将x2编码成不同的层
offset = 1 / 40
x1_combined = [shared_memory_bandwidth_parameters[i] * 10 + (400 - shared_memory_latency_parameters[i]) * offset for i in range(len(latencies))]  # 将x1和x2组合

# 根据x1的值设置颜色，颜色随着数值变大而变化
colors = plt.cm.viridis((np.array(shared_memory_bandwidth_parameters) - min(shared_memory_bandwidth_parameters)) / (max(shared_memory_bandwidth_parameters) - min(shared_memory_bandwidth_parameters)))

# 绘制三维柱状图
ax.bar3d(x1_combined, local_memory_bandwidth_parameters, np.zeros_like(latencies), dx, dy, dz, color=colors)

# # 添加二维坐标标签(x1, x2)作为y轴上的标记
# for i in range(len(y)):
#     ax.text(x1[i], x2_combined[i], y[i] + 0.5, f"({x2[i]},{x3[i]})", color='black', ha='center')

# 设置坐标轴标签
ax.set_xlabel('Shared Memory Bandwidth + Latency')
ax.set_ylabel('Local Memory Bandwidth')
ax.set_zlabel('Latency')

plt.title("GPU-Like Shared Memory")

# 添加颜色映射条
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(shared_memory_bandwidth_parameters)
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_ticks([256, 512, 1024, 2048, 4096, 8192])

plt.savefig('temp/shared_memory.png', dpi=300)


