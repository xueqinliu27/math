import matplotlib.pyplot as plt
import numpy as np
import csv

# 使用 matplotlib 自带的样式
plt.style.use('ggplot')

def plot_combined_performance():
    # 读取 performance.txt 数据（不同分块大小）
    data = np.loadtxt('t3.txt', skiprows=1)
    dimensions = data[:, 0]  # 第一列是矩阵维度
    block_sizes = data[:, 1]  # 第二列是分块大小
    times = data[:, 2]  # 第三列是运行时间

    # 读取 timings.csv 数据（标准矩阵乘法）
    sizes1, times1 = [], []
    with open('t1.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # 跳过标题行
        for row in plots:
            sizes1.append(int(row[0]))
            times1.append(float(row[1]))

    # 读取 timings_colmajor.csv 数据（列优先优化）
    sizes2, times2 = [], []
    with open('t2.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # 跳过标题行
        for row in plots:
            sizes2.append(int(row[0]))
            times2.append(float(row[1]))

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # 绘制不同分块大小的数据
    block_size_values = np.unique(block_sizes)
    for i, block_size in enumerate(block_size_values):
        mask = (block_sizes == block_size)
        plt.plot(dimensions[mask], times[mask], label=f'Block Size {int(block_size)}',
                 color=colors[i], marker=markers[i], linewidth=2, markersize=8)

    # 绘制标准矩阵乘法的数据
    plt.plot(sizes1, times1, label='Standard Multiplication', color=colors[len(block_size_values)], 
             marker=markers[len(block_size_values)], linewidth=2, markersize=8)

    # 绘制列优先优化的数据
    plt.plot(sizes2, times2, label='ColMajor Optimized', color=colors[len(block_size_values) + 1], 
             marker=markers[len(block_size_values) + 1], linewidth=2, markersize=8)

    # 设置图表标题和标签
    plt.title('Matrix Multiplication Performance Comparison', fontsize=14)
    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)

    # 设置x轴刻度
    max_size = max(max(dimensions), max(sizes1), max(sizes2))
    plt.xticks(np.arange(100, max_size + 1, 100), rotation=45)

    # 设置y轴刻度（更精细）
    min_time = min(np.min(times), np.min(times1), np.min(times2))
    max_time = max(np.max(times), np.max(times1), np.max(times2))
    plt.yticks(np.arange(min_time, max_time + 0.01, 0.1))  # 以0.1为间隔

    # 添加图例
    plt.legend(fontsize=10)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plt.tight_layout()
    plt.savefig('performance_plot_combined_3.png', dpi=300)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    plot_combined_performance()