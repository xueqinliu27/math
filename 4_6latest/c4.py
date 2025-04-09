import numpy as np
import time
import matplotlib.pyplot as plt

# 定义矩阵乘法的三种方式
def pure_loop_multiplication(A, B):
    N = A.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

def semi_vectorized_multiplication(A, B):
    N = A.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        row = A[i, :]
        for k in range(N):
            C[i, :] += row[k] * B[k, :]
    return C

def numpy_multiplication(A, B):
    return A @ B

# 测试不同规模矩阵
def benchmark_matrix_multiplication(matrix_sizes):
    sizes = []
    pure_loop_times = []
    semi_vector_times = []
    numpy_times = []

    for size in matrix_sizes:
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        # 纯循环
        start = time.time()
        C = pure_loop_multiplication(A, B)
        end = time.time()
        pure_loop_times.append(end - start)

        # 半向量化
        start = time.time()
        C_optimized = semi_vectorized_multiplication(A, B)
        end = time.time()
        semi_vector_times.append(end - start)

        # NumPy 内置运算
        start = time.time()
        C_numpy = numpy_multiplication(A, B)
        end = time.time()
        numpy_times.append(end - start)

        sizes.append(size)

        print(f"Size {size}x{size}: "
              f"Pure Loop {pure_loop_times[-1]:.4f}s, "
              f"Semi-Vector {semi_vector_times[-1]:.4f}s, "
              f"NumPy {numpy_times[-1]:.4f}s")

    # 保存数据到文件
    np.savetxt("benchmark_results.txt", np.column_stack((sizes, pure_loop_times, semi_vector_times, numpy_times)), 
               header="Size PureLoop_Time SemiVector_Time NumPy_Time", comments='', fmt='%d %.6f %.6f %.6f')

# 测试不同规模矩阵
benchmark_matrix_multiplication([100, 200, 300, 400, 500, 600])

# 绘制折线图
def plot_results():
    data = np.loadtxt('benchmark_results.txt', skiprows=1)
    sizes = data[:, 0]
    pure_loop_times = data[:, 1]
    semi_vector_times = data[:, 2]
    numpy_times = data[:, 3]

    plt.figure(figsize=(12, 8))
    plt.plot(sizes, pure_loop_times, 'b-o', label='Pure Loop', linewidth=2, markersize=8)
    plt.plot(sizes, semi_vector_times, 'r-s', label='Semi-Vectorized', linewidth=2, markersize=8)
    plt.plot(sizes, numpy_times, 'g-^', label='NumPy', linewidth=2, markersize=8)

    plt.title('Matrix Multiplication Performance Comparison')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark_plot_combined.png', dpi=300)
    plt.show()

# 绘制结果
plot_results()