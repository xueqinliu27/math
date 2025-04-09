#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))  // 定义最小值宏

/* 时间测量与格式化输出函数 */
void print_elapsed_time(double seconds) {
    const double nano = 1e9, micro = 1e6, milli = 1e3;
    const char* units[] = {"s", "ms", "μs", "ns"};  // 单位标签
    double scales[] = {1.0, milli, micro, nano};    // 单位换算系数
    
    int unit_idx = 0;
    // 自动选择合适的时间单位
    while (seconds * scales[unit_idx] < 1.0 && unit_idx < 3) {
        unit_idx++;
    }
    printf("%.3f %s", seconds * scales[unit_idx], units[unit_idx]);
}

/* 列优先分块矩阵乘法核心算法 */
void blocked_colB_multiply(const double *A, const double *B_col, double *C,
                          int m, int n, int p, int block_size) {
    // 初始化结果矩阵为全零
    for (int i = 0; i < m * p; ++i) C[i] = 0.0;

    // 三重分块循环结构
    for (int i_blk = 0; i_blk < m; i_blk += block_size) {  // 行分块
        int i_end = MIN(i_blk + block_size, m);            // 当前行块结束位置
        for (int k_blk = 0; k_blk < n; k_blk += block_size) {  // 中间维度分块
            int k_end = MIN(k_blk + block_size, n);
            for (int j_blk = 0; j_blk < p; j_blk += block_size) {  // 列分块
                int j_end = MIN(j_blk + block_size, p);
                
                // 块内计算（优化后的访问顺序）
                for (int i = i_blk; i < i_end; ++i) {         // 遍历行块
                    for (int k = k_blk; k < k_end; ++k) {     // 遍历中间维度块
                        const double a = A[i * n + k];       // 缓存A矩阵元素
                        for (int j = j_blk; j < j_end; ++j) {  // 遍历列块
                            C[i * p + j] += a * B_col[j * n + k];  // 列优先访问B
                        }
                    }
                }
            }
        }
    }
}

/* 列优先矩阵初始化函数 */
void init_colmajor_matrix(double *mat, int rows, int cols, double value) {
    for (int j = 0; j < cols; ++j) {         // 外层循环列
        for (int i = 0; i < rows; ++i) {     // 内层循环行
            mat[j * rows + i] = value;       // 列优先存储位置计算
        }
    }
}

/* 结果验证函数 */
int verify_result(const double *C1, const double *C2, int size) {
    const double eps = 1e-6;  // 允许误差范围
    for (int i = 0; i < size; ++i) {
        if (fabs(C1[i] - C2[i]) > eps) {     // 浮点数精确比较
            printf("Mismatch at %d: %.2f vs %.2f\n", i, C1[i], C2[i]);
            return 0;  // 验证失败
        }
    }
    return 1;  // 验证通过
}

/* 性能基准测试框架 */
void run_benchmark(int dim, int max_block) {
    int m = dim, n = dim, p = dim;
    
    // 内存分配（A行优先，B_col列优先）
    double *A = malloc(m * n * sizeof(double));         // 矩阵A
    double *B_col = malloc(n * p * sizeof(double));     // 矩阵B（列优先）
    double *C_base = malloc(m * p * sizeof(double));    // 基准结果
    double *C_blocked = malloc(m * p * sizeof(double)); // 分块结果

    // 矩阵初始化（使用伪随机数）
    for (int i = 0; i < m * n; ++i) A[i] = drand48();  // 初始化A矩阵
    init_colmajor_matrix(B_col, n, p, drand48());      // 初始化B矩阵（列优先）

    // 测试结果表头
    printf("\nBenchmarking %dx%d matrices:\n", dim, dim);
    printf("Block Size |  Base Time  | Blocked Time | Speedup |  GFLOPS\n");
    printf("-----------------------------------------------------------\n");

    // 分块大小测试循环
    for (int bs = 16; bs <= max_block; bs *= 2) {
        // 基准版本（不分块，使用最大块尺寸）
        clock_t start = clock();
        blocked_colB_multiply(A, B_col, C_base, m, n, p, m);
        double t1 = (double)(clock() - start) / CLOCKS_PER_SEC;

        // 分块版本
        start = clock();
        blocked_colB_multiply(A, B_col, C_blocked, m, n, p, bs);
        double t2 = (double)(clock() - start) / CLOCKS_PER_SEC;

        // 结果验证
        if (!verify_result(C_base, C_blocked, m * p)) {
            printf("Validation failed for block size %d!\n", bs);
            break;
        }

        // 性能指标计算
        double flops = 2.0 * dim * dim * dim / t2 / 1e9;  // GFLOPS计算
        printf("%9d | ", bs);
        print_elapsed_time(t1); printf(" | ");
        print_elapsed_time(t2); printf(" | %5.2fx | %6.2f\n", t1/t2, flops);
    }

    // 释放内存
    free(A);
    free(B_col);
    free(C_base);
    free(C_blocked);
}

int main() {
    // ================== 正确性验证 ==================
    double A[] = {1,2,3,4};       // 2x2行优先矩阵
    double B_col[] = {5,7,6,8};   // 2x2列优先矩阵（实际矩阵：[5,6;7,8]）
    double C[4];
    
    blocked_colB_multiply(A, B_col, C, 2, 2, 2, 2);
    printf("Small matrix test:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", C[0], C[1], C[2], C[3]);

    // ================== 性能测试 ==================
    run_benchmark(100, 128);    // 100x100矩阵测试
    run_benchmark(500, 128);    // 500x500矩阵测试
    run_benchmark(1000, 256);   // 1000x1000矩阵测试

    return 0;
}