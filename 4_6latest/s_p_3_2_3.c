#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* 时间测量与格式化输出函数 */
void print_elapsed_time(double seconds) {
    const double nano = 1e9, micro = 1e6, milli = 1e3;
    const char* units[] = {"s", "ms", "μs", "ns"};
    double scales[] = {1.0, milli, micro, nano};
    
    int unit_idx = 0;
    while (seconds * scales[unit_idx] < 1.0 && unit_idx < 3) {
        unit_idx++;
    }
    printf("%.3f %s", seconds * scales[unit_idx], units[unit_idx]);
}

/* 列优先分块矩阵乘法核心算法 */
void blocked_colB_multiply(const double *A, const double *B_col, double *C,
                          int m, int n, int p, int block_size) {
    for (int i = 0; i < m * p; ++i) C[i] = 0.0;

    for (int i_blk = 0; i_blk < m; i_blk += block_size) {
        int i_end = MIN(i_blk + block_size, m);
        for (int k_blk = 0; k_blk < n; k_blk += block_size) {
            int k_end = MIN(k_blk + block_size, n);
            for (int j_blk = 0; j_blk < p; j_blk += block_size) {
                int j_end = MIN(j_blk + block_size, p);
                
                for (int i = i_blk; i < i_end; ++i) {
                    for (int k = k_blk; k < k_end; ++k) {
                        const double a = A[i * n + k];
                        for (int j = j_blk; j < j_end; ++j) {
                            C[i * p + j] += a * B_col[j * n + k];
                        }
                    }
                }
            }
        }
    }
}

/* 列优先矩阵初始化函数 */
void init_colmajor_matrix(double *mat, int rows, int cols, double value) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mat[j * rows + i] = value;
        }
    }
}

/* 性能测试：不同矩阵尺寸 */
void run_dimension_benchmark() {
    FILE *fp = fopen("performance_fixed_block.txt", "w");
    if (!fp) {
        perror("无法打开文件");
        return;
    }
    fprintf(fp, "Dimension Time(s)\n");

    const int block_size = 64;  // 固定分块大小
    srand48(time(NULL));         // 初始化随机数生成器

    for (int dim = 100; dim <= 1000; dim += 100) {
        int m = dim, n = dim, p = dim;

        double *A = malloc(m * n * sizeof(double));
        double *B_col = malloc(n * p * sizeof(double));
        double *C = malloc(m * p * sizeof(double));

        // 矩阵初始化
        for (int i = 0; i < m * n; ++i) A[i] = drand48();
        init_colmajor_matrix(B_col, n, p, drand48());

        // 时间测量
        clock_t start = clock();
        blocked_colB_multiply(A, B_col, C, m, n, p, block_size);
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

        fprintf(fp, "%d %.6f\n", dim, elapsed);

        free(A);
        free(B_col);
        free(C);
    }

    fclose(fp);
    printf("性能数据已保存到performance_fixed_block.txt\n");
}

int main() {
    // ================== 正确性验证 ==================
    double A[] = {1, 2, 3, 4};
    double B_col[] = {5, 7, 6, 8};
    double C[4];
    
    blocked_colB_multiply(A, B_col, C, 2, 2, 2, 2);
    printf("Small matrix test:\n");
    printf("%.2f %.2f\n%.2f %.2f\n\n", C[0], C[1], C[2], C[3]);

    // ================== 性能测试 ==================
    run_dimension_benchmark();

    return 0;
}