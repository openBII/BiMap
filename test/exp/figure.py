import matplotlib.pyplot as plt
import numpy as np


# 示例数据
x1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 第一组x1坐标
x2 = [2, 4, 6, 8, 2, 4, 6, 8]  # 第二组x2坐标
x3 = [1, 1, 2, 2, 3, 3, 4, 4]  # 表示不同的组别
y = [5, 3, 7, 6, 2, 8, 4, 5]   # y值

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置柱的参数
dx = dy = 0.5   # 柱的宽度
dz = y         # 柱的高度

# 根据x3的值设置颜色，颜色随着数值变大而变化
colors = plt.cm.viridis((np.array(x3) - min(x3)) / (max(x3) - min(x3)))

# 绘制三维柱状图
ax.bar3d(x1, x2, np.zeros_like(x3), dx, dy, dz, color=colors)

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.title("3D Bar Chart with Color Variation by X3")

# 添加颜色映射条
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(x3)
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])

plt.savefig('temp/test.png', dpi=300)
