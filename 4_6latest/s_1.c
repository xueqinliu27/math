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

void matrix_multiply(const double *A, const double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void init_matrix(double *matrix, int rows, int cols, double value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = value;
    }
}

int main() {
    // 正确性验证保持不变
    int m = 2, n = 2, p = 2;
    double A[] = {1, 2, 3, 4};
    double B[] = {5, 6, 7, 8};
    double C[4];
    
    clock_t start = clock();
    matrix_multiply(A, B, C, m, n, p);
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

    // 性能测试不同尺寸矩阵
    FILE *fp = fopen("t1.csv", "w");
    if (!fp) {
        perror("Failed to open t1.csv");
        return 1;
    }
    fprintf(fp, "size,time\n");

    // 测试从100到1000，步长100的方阵
    int sizes[] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int current_size = sizes[i];
        double *A = malloc(current_size * current_size * sizeof(double));
        double *B = malloc(current_size * current_size * sizeof(double));
        double *C = malloc(current_size * current_size * sizeof(double));

        init_matrix(A, current_size, current_size, 1.0);
        init_matrix(B, current_size, current_size, 1.0);

        start = clock();
        matrix_multiply(A, B, C, current_size, current_size, current_size);
        end = clock();
        
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        fprintf(fp, "%d,%f\n", current_size, elapsed);

        free(A);
        free(B);
        free(C);

        printf("Size: %4d - Time: ", current_size);
        print_elapsed_time(elapsed);
        printf("\n");
    }

    fclose(fp);
    printf("\nTiming data saved to t1.csv\n");

    return 0;
}