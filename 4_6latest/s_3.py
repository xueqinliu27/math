import matplotlib.pyplot as plt
import numpy as np

# 读取性能数据
data = np.loadtxt('t3.txt', skiprows=1)
dimensions = data[:, 0]  # 第一列是矩阵维度
block_sizes = data[:, 1]  # 第二列是分块大小
times = data[:, 2]  # 第三列是运行时间

# 创建图表
plt.figure(figsize=(12, 8))

# 定义不同分块大小的颜色和标记
block_size_values = np.unique(block_sizes)  # 获取所有分块大小的唯一值
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# 遍历每个分块大小，绘制折线图
for i, block_size in enumerate(block_size_values):
    # 筛选出当前分块大小的数据
    mask = (block_sizes == block_size)
    plt.plot(dimensions[mask], times[mask], label=f'Block Size {int(block_size)}',
             color=colors[i], marker=markers[i], linewidth=2, markersize=8)

# 设置图表标题和标签
plt.title('Matrix Multiplication Time vs Dimension for Different Block Sizes')
plt.xlabel('Matrix Dimension (n x n)')
plt.ylabel('Execution Time (seconds)')
plt.xticks(np.arange(100, 1100, 100))  # 设置x轴刻度间隔为100
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('t3.png', dpi=300, bbox_inches='tight')
plt.show()