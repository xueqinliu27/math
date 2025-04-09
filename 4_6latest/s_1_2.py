import matplotlib.pyplot as plt
import csv
import numpy as np

# 使用 matplotlib 自带的样式
plt.style.use('ggplot')

def plot_combined_performance():
    # 读取 t1.csv 数据
    sizes1, times1 = [], []
    with open('t1.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # 跳过标题行
        for row in plots:
            sizes1.append(int(row[0]))
            times1.append(float(row[1]))

    # 读取 t2.csv 数据
    sizes2, times2 = [], []
    with open('t2.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)  # 跳过标题行
        for row in plots:
            sizes2.append(int(row[0]))
            times2.append(float(row[1]))

    # 创建图表
    plt.figure(figsize=(12, 7))

    # 绘制第一组数据
    plt.plot(sizes1, times1, 'b-o', label='Standard Multiplication', linewidth=2, markersize=8)

    # 绘制第二组数据
    plt.plot(sizes2, times2, 'r-s', label='ColMajor Optimized', linewidth=2, markersize=8)

    # 设置图表标题和标签
    plt.title('Matrix Multiplication Performance Comparison', fontsize=14)
    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)

    # 设置x轴刻度
    plt.xticks(np.arange(100, max(max(sizes1), max(sizes2)) + 1, 100), rotation=45)

    # 添加图例
    plt.legend(fontsize=10)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plt.tight_layout()
    plt.savefig('performance_plot_combined.png', dpi=300)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    plot_combined_performance()