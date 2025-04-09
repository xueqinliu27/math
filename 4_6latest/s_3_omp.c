#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>  // OpenMP并行计算库

#define MIN(a, b) ((a) < (b) ? (a) : (b))  // 最小值宏定义

/* 时间格式化输出函数 */
void print_elapsed_time(double seconds) {
    const double nano = 1e9, micro = 1e6, milli = 1e3;
    // 自动选择合适的时间单位
    if (seconds >= 1.0) {
        printf("%.3f s", seconds);
    } else if (seconds >= 1.0/milli) {
        printf("%.3f ms", seconds * milli);
    } else if (seconds >= 1.0/micro) {
        printf("%.3f μs", seconds * micro);
    } else {
        printf("%.3f ns", seconds * nano);
    }
}

/* 并行分块矩阵乘法（列优先优化） */
void matrix_multiply_colB(const double *A, const double *B_col, double *C,
                         int m, int n, int p, int block_size) {
    // 并行初始化结果矩阵
    #pragma omp parallel for  // OpenMP并行化
    for (int i = 0; i < m*p; i++) C[i] = 0.0;

    // 三重分块并行计算
    #pragma omp parallel for collapse(2) schedule(dynamic)  // 双层循环展开并行
    for (int i_blk = 0; i_blk < m; i_blk += block_size) {   // 行分块
        for (int j_blk = 0; j_blk < p; j_blk += block_size) { // 列分块
            for (int k_blk = 0; k_blk < n; k_blk += block_size) { // 中间维度分块
                // 计算当前块边界
                int i_end = MIN(i_blk + block_size, m);
                int j_end = MIN(j_blk + block_size, p);
                int k_end = MIN(k_blk + block_size, n);
                
                // 块内计算核心
                for (int i = i_blk; i < i_end; i++) {        // 行遍历
                    for (int j = j_blk; j < j_end; j++) {    // 列遍历
                        double sum = 0.0;
                        for (int k = k_blk; k < k_end; k++) { // 累加计算
                            sum += A[i*n + k] * B_col[j*n + k]; // 列优先访问
                        }
                        // 原子操作保证写安全
                        #pragma omp atomic  // 防止写冲突
                        C[i*p + j] += sum;
                    }
                }
            }
        }
    }
}

/* 并行列优先矩阵初始化 */
void init_colmajor_matrix(double *mat, int rows, int cols, double value) {
    #pragma omp parallel for  // 列维度并行
    for (int j = 0; j < cols; j++) {       // 外层循环列
        for (int i = 0; i < rows; i++) {   // 内层循环行
            mat[j*rows + i] = value;       // 列优先存储
        }
    }
}

/* 结果验证函数 */
int verify_result(const double *C, int m, int p, double expected) {
    const double eps = 1e-6;  // 允许误差
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            if (fabs(C[i*p + j] - expected) > eps) {  // 浮点比较
                printf("验证失败 at (%d,%d): %.2f != %.2f\n", i, j, C[i*p + j], expected);
                return 0;
            }
        }
    }
    return 1;
}

/* 性能基准测试框架 */
void run_benchmark(int dim, int max_block, int threads) {
    int m = dim, n = dim, p = dim;
    
    // 内存分配
    double *A = (double*)malloc(m * n * sizeof(double));   // 矩阵A
    double *B_col = (double*)malloc(n * p * sizeof(double)); // 矩阵B（列优先）
    double *C = (double*)malloc(m * p * sizeof(double));    // 结果矩阵

    // 并行初始化
    #pragma omp parallel for  // 行优先并行填充
    for (int i = 0; i < m*n; i++) A[i] = 1.0;
    init_colmajor_matrix(B_col, n, p, 1.0);  // 列优先初始化

    omp_set_num_threads(threads);  // 设置线程数
    printf("\n矩阵大小 %dx%d (%d线程):\n", dim, dim, threads);
    printf("块大小 | 计算时间 | GFLOPS\n");
    printf("--------------------------\n");

    // 分块大小测试循环
    for (int bs = 16; bs <= max_block; bs *= 2) {
        double start = omp_get_wtime();  // 高精度计时
        matrix_multiply_colB(A, B_col, C, m, n, p, bs);
        double elapsed = omp_get_wtime() - start;

        // 结果验证（全矩阵应为dim值）
        if (!verify_result(C, m, p, (double)dim)) {
            printf("验证失败!\n");
            break;
        }

        // 性能计算
        double gflops = 2.0 * dim * dim * dim / elapsed / 1e9;
        printf("%6d | ", bs);
        print_elapsed_time(elapsed);
        printf(" | %6.2f\n", gflops);
    }

    // 释放资源
    free(A);
    free(B_col);
    free(C);
}

int main() {
    // ============== 正确性验证 ==============
    double A[] = {1,2,3,4};     // 2x2行优先
    double B_col[] = {5,7,6,8}; // 2x2列优先（实际矩阵：[5,6;7,8]）
    double C[4];
    
    matrix_multiply_colB(A, B_col, C, 2, 2, 2, 2);
    printf("验证测试:\n");
    printf("%.1f %.1f\n%.1f %.1f\n", C[0], C[1], C[2], C[3]);

    // ============== 性能测试 ==============
    printf("\n===== 性能基准测试 =====\n");
    run_benchmark(1000, 128, 1);  // 单线程
    run_benchmark(1000, 128, 4);  // 4线程
    run_benchmark(1000, 128, 8);  // 8线程

    return 0;
}