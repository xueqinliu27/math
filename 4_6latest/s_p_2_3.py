import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_python(matrix_sizes):
    # 用于保存运行时间的数据
    sizes = []
    np_times = []
    semi_vector_times = []

    for size in matrix_sizes:
        # 生成全1矩阵（与C实验一致）
        A = np.ones((size, size), dtype=np.float64)
        B = np.ones((size, size), dtype=np.float64)
        
        # NumPy全向量化
        start = time.time()
        C_np = A @ B
        np_time = time.time() - start
        
        # 半向量化
        C = np.zeros((size, size), dtype=np.float64)
        start = time.time()
        for i in range(size):
            row = A[i, :]
            for k in range(size):
                C[i, :] += row[k] * B[k, :]
        semi_vector_time = time.time() - start
        
        # 保存数据
        sizes.append(size)
        np_times.append(np_time)
        semi_vector_times.append(semi_vector_time)
        
        print(f"Size {size}x{size}: "
              f"NumPy {np_time:.4f}s, "
              f"Semi-Vector {semi_vector_time:.2f}s")

    # 保存数据到文件
    np.savetxt("benchmark_results.txt", np.column_stack((sizes, np_times, semi_vector_times)), 
               header="Size NumPy_Time SemiVector_Time", comments='', fmt='%d %.6f %.6f')

# 测试不同规模矩阵
benchmark_python([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

# 绘制折线图
def plot_results():
    data = np.loadtxt('benchmark_results.txt', skiprows=1)
    sizes = data[:, 0]
    np_times = data[:, 1]
    semi_vector_times = data[:, 2]

    plt.figure(figsize=(12, 8))
    plt.plot(sizes, np_times, label='NumPy (Full Vectorization)', color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.plot(sizes, semi_vector_times, label='Semi-Vectorization', color='red', marker='s', linestyle='--', linewidth=2, markersize=8)

    plt.title('Matrix Multiplication Time vs Matrix Size')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('benchmark_plot.png', dpi=300)
    plt.show()

# 绘制结果
plot_results()