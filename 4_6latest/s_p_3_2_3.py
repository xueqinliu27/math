import matplotlib.pyplot as plt
import numpy as np

# 读取 benchmark_results.txt 数据
benchmark_data = np.loadtxt('benchmark_results.txt', skiprows=1)
benchmark_sizes = benchmark_data[:, 0]
np_times = benchmark_data[:, 1]
semi_vector_times = benchmark_data[:, 2]

# 读取 performance_fixed_block.txt 数据
fixed_block_data = np.loadtxt('performance_fixed_block.txt', skiprows=1)
fixed_block_sizes = fixed_block_data[:, 0]
fixed_block_times = fixed_block_data[:, 1]

# 创建图表
plt.figure(figsize=(12, 8))

# 定义颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# 绘制 NumPy 全向量化的时间数据
plt.plot(benchmark_sizes, np_times, label='NumPy (Full Vectorization)', color=colors[0], marker=markers[0], linestyle='-', linewidth=2, markersize=8)

# 绘制 半向量化的时间数据
plt.plot(benchmark_sizes, semi_vector_times, label='Semi-Vectorization', color=colors[1], marker=markers[1], linestyle='--', linewidth=2, markersize=8)

# 绘制 固定分块大小为64的时间数据
plt.plot(fixed_block_sizes, fixed_block_times, label='Block Size 64', color=colors[2], marker=markers[2], linestyle='-.', linewidth=2, markersize=8)

# 设置图表标题和标签
plt.title('Matrix Multiplication Performance Comparison', fontsize=14)
plt.xlabel('Matrix Size (N x N)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)

# 设置 x 轴刻度
max_size = max(max(benchmark_sizes), max(fixed_block_sizes))
plt.xticks(np.arange(100, max_size + 1, 100), rotation=45)

# 添加图例
plt.legend(fontsize=10)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图表
plt.tight_layout()
plt.savefig('performance_plot_combined.png', dpi=300)

# 显示图表
plt.show()