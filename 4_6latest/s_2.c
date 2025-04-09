#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_elapsed_time(double seconds) {
    const double nano = 1e9, micro = 1e6, milli = 1e3;
    
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

void matrix_multiply_colB(const double *A, const double *B_col, double *C, 
                         int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B_col[j * n + k];
            }
            C[i * p + j] = sum;
        }
    }
}

void init_matrix_colmajor(double *matrix, int rows, int cols, double value) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            matrix[j * rows + i] = value;
        }
    }
}

int main() {
    // ================== 正确性验证 ==================
    int m = 2, n = 2, p = 2;
    double A[] = {1, 2, 3, 4};
    double B_col[] = {5, 7, 6, 8};
    double C[4];

    clock_t start = clock();
    matrix_multiply_colB(A, B_col, C, m, n, p);
    clock_t end = clock();
    
    double small_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Small matrix test (2x2):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%8.2f", C[i * p + j]);
        }
        printf("\n");
    }
    printf("Small matrix time: ");
    print_elapsed_time(small_time);
    printf("\n");

    // ================== 多尺寸性能测试 ==================
    FILE *fp = fopen("t2.csv", "w");
    if (!fp) {
        perror("Failed to open t2.csv");
        return 1;
    }
    fprintf(fp, "size,time\n");

    int sizes[] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        printf("\nTesting size: %4d x %4d", N, N);
        
        // 分配内存
        double *A = malloc(N * N * sizeof(double));
        double *B_col = malloc(N * N * sizeof(double));
        double *C = malloc(N * N * sizeof(double));

        // 初始化矩阵
        for (int i = 0; i < N*N; i++) A[i] = 1.0;
        init_matrix_colmajor(B_col, N, N, 1.0);

        // 执行计算并计时
        start = clock();
        matrix_multiply_colB(A, B_col, C, N, N, N);
        end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        // 记录数据
        fprintf(fp, "%d,%f\n", N, elapsed);
        printf(" - Time: ");
        print_elapsed_time(elapsed);

        // 释放内存
        free(A);
        free(B_col);
        free(C);
    }

    fclose(fp);
    printf("\n\nTiming data saved to t2.csv\n");

    return 0;
}