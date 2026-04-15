"""Create benchmark C programs and compile them to LLVM IR."""
import os
import subprocess
from pathlib import Path

BENCHMARK_DIR = Path("/home/nw366/ResearchArena/outputs/claude_v5_compiler_optimization/idea_01/experiments/benchmarks")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Dictionary of benchmark name -> C source code
BENCHMARKS = {}

# === PolyBench-style kernels ===

BENCHMARKS["matmul"] = """
#include <stdlib.h>
#define N 128
double A[N][N], B[N][N], C[N][N];
void kernel_matmul() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}
int main() { kernel_matmul(); return 0; }
"""

BENCHMARKS["gemm"] = """
#include <stdlib.h>
#define N 100
double alpha, beta;
double A[N][N], B[N][N], C[N][N];
void kernel_gemm() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] *= beta;
            for (int k = 0; k < N; k++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
}
int main() { kernel_gemm(); return 0; }
"""

BENCHMARKS["jacobi_2d"] = """
#define N 64
#define TSTEPS 10
double A[N][N], B[N][N];
void kernel_jacobi_2d() {
    for (int t = 0; t < TSTEPS; t++) {
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
    }
}
int main() { kernel_jacobi_2d(); return 0; }
"""

BENCHMARKS["seidel_2d"] = """
#define N 64
#define TSTEPS 10
double A[N][N];
void kernel_seidel_2d() {
    for (int t = 0; t < TSTEPS; t++)
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                         + A[i][j-1] + A[i][j] + A[i][j+1]
                         + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0;
}
int main() { kernel_seidel_2d(); return 0; }
"""

BENCHMARKS["fdtd_2d"] = """
#define NX 64
#define NY 64
#define TMAX 10
double ex[NX][NY], ey[NX][NY], hz[NX][NY], fict[TMAX];
void kernel_fdtd_2d() {
    for (int t = 0; t < TMAX; t++) {
        for (int j = 0; j < NY; j++) ey[0][j] = fict[t];
        for (int i = 1; i < NX; i++)
            for (int j = 0; j < NY; j++)
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]);
        for (int i = 0; i < NX; i++)
            for (int j = 1; j < NY; j++)
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
        for (int i = 0; i < NX-1; i++)
            for (int j = 0; j < NY-1; j++)
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
    }
}
int main() { kernel_fdtd_2d(); return 0; }
"""

BENCHMARKS["lu"] = """
#define N 64
double A[N][N];
void kernel_lu() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][j] /= A[j][j];
        }
        for (int j = i; j < N; j++)
            for (int k = 0; k < i; k++)
                A[i][j] -= A[i][k] * A[k][j];
    }
}
int main() { kernel_lu(); return 0; }
"""

BENCHMARKS["cholesky"] = """
#include <math.h>
#define N 64
double A[N][N];
void kernel_cholesky() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++)
                A[i][j] -= A[i][k] * A[j][k];
            A[i][j] /= A[j][j];
        }
        for (int k = 0; k < i; k++)
            A[i][i] -= A[i][k] * A[i][k];
        A[i][i] = sqrt(A[i][i]);
    }
}
int main() { kernel_cholesky(); return 0; }
"""

BENCHMARKS["mvt"] = """
#define N 128
double A[N][N], x1[N], x2[N], y1[N], y2[N];
void kernel_mvt() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            x1[i] += A[i][j] * y1[j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            x2[i] += A[j][i] * y2[j];
}
int main() { kernel_mvt(); return 0; }
"""

BENCHMARKS["atax"] = """
#define M 100
#define N 120
double A[M][N], x[N], y[N], tmp[M];
void kernel_atax() {
    for (int i = 0; i < N; i++) y[i] = 0;
    for (int i = 0; i < M; i++) {
        tmp[i] = 0.0;
        for (int j = 0; j < N; j++) tmp[i] += A[i][j] * x[j];
        for (int j = 0; j < N; j++) y[j] += A[i][j] * tmp[i];
    }
}
int main() { kernel_atax(); return 0; }
"""

BENCHMARKS["bicg"] = """
#define M 100
#define N 110
double A[N][M], s[M], q[N], p[M], r[N];
void kernel_bicg() {
    for (int i = 0; i < M; i++) s[i] = 0;
    for (int i = 0; i < N; i++) {
        q[i] = 0.0;
        for (int j = 0; j < M; j++) {
            s[j] += r[i] * A[i][j];
            q[i] += A[i][j] * p[j];
        }
    }
}
int main() { kernel_bicg(); return 0; }
"""

BENCHMARKS["gesummv"] = """
#define N 100
double alpha, beta;
double A[N][N], B[N][N], tmp[N], x[N], y[N];
void kernel_gesummv() {
    for (int i = 0; i < N; i++) {
        tmp[i] = 0.0; y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            tmp[i] += A[i][j] * x[j];
            y[i] += B[i][j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}
int main() { kernel_gesummv(); return 0; }
"""

BENCHMARKS["symm"] = """
#define M 60
#define N 80
double alpha, beta;
double A[M][M], B[M][N], C[M][N];
void kernel_symm() {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double temp2 = 0;
            for (int k = 0; k < i; k++) {
                C[k][j] += alpha * B[i][j] * A[i][k];
                temp2 += B[k][j] * A[i][k];
            }
            C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
}
int main() { kernel_symm(); return 0; }
"""

BENCHMARKS["syrk"] = """
#define M 60
#define N 80
double alpha, beta;
double A[N][M], C[N][N];
void kernel_syrk() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) C[i][j] *= beta;
        for (int k = 0; k < M; k++)
            for (int j = 0; j <= i; j++)
                C[i][j] += alpha * A[i][k] * A[j][k];
    }
}
int main() { kernel_syrk(); return 0; }
"""

BENCHMARKS["trmm"] = """
#define M 60
#define N 80
double alpha;
double A[M][M], B[M][N];
void kernel_trmm() {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            for (int k = i+1; k < M; k++)
                B[i][j] += A[k][i] * B[k][j];
            B[i][j] = alpha * B[i][j];
        }
}
int main() { kernel_trmm(); return 0; }
"""

BENCHMARKS["covariance"] = """
#define M 80
#define N 100
double float_n;
double data[N][M], cov[M][M], mean[M];
void kernel_covariance() {
    for (int j = 0; j < M; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < N; i++) mean[j] += data[i][j];
        mean[j] /= float_n;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            data[i][j] -= mean[j];
    for (int i = 0; i < M; i++)
        for (int j = i; j < M; j++) {
            cov[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                cov[i][j] += data[k][i] * data[k][j];
            cov[i][j] /= (float_n - 1.0);
            cov[j][i] = cov[i][j];
        }
}
int main() { kernel_covariance(); return 0; }
"""

BENCHMARKS["correlation"] = """
#include <math.h>
#define M 80
#define N 100
double float_n;
double data[N][M], corr[M][M], mean[M], stddev[M];
void kernel_correlation() {
    double eps = 0.1;
    for (int j = 0; j < M; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < N; i++) mean[j] += data[i][j];
        mean[j] /= float_n;
    }
    for (int j = 0; j < M; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < N; i++)
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        stddev[j] /= float_n;
        stddev[j] = sqrt(stddev[j]);
        stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            data[i][j] -= mean[j];
            data[i][j] /= sqrt(float_n) * stddev[j];
        }
    for (int i = 0; i < M-1; i++) {
        corr[i][i] = 1.0;
        for (int j = i+1; j < M; j++) {
            corr[i][j] = 0.0;
            for (int k = 0; k < N; k++) corr[i][j] += data[k][i] * data[k][j];
            corr[j][i] = corr[i][j];
        }
    }
    corr[M-1][M-1] = 1.0;
}
int main() { kernel_correlation(); return 0; }
"""

# === Recursive algorithms ===

BENCHMARKS["fibonacci"] = """
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(30); }
"""

BENCHMARKS["quicksort"] = """
void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++)
        if (arr[j] < pivot) { i++; swap(&arr[i], &arr[j]); }
    swap(&arr[i+1], &arr[high]);
    return i + 1;
}
void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi-1);
        quicksort(arr, pi+1, high);
    }
}
int arr[1000];
int main() { quicksort(arr, 0, 999); return 0; }
"""

BENCHMARKS["mergesort"] = """
int temp[1000];
void merge(int arr[], int l, int m, int r) {
    int i = l, j = m+1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; i++) arr[i] = temp[i];
}
void mergesort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r-l)/2;
        mergesort(arr, l, m);
        mergesort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}
int arr[1000];
int main() { mergesort(arr, 0, 999); return 0; }
"""

BENCHMARKS["binary_search"] = """
int bsearch_arr(int arr[], int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}
int arr[1024];
int main() {
    for (int i = 0; i < 1024; i++) arr[i] = i*2;
    return bsearch_arr(arr, 1024, 500);
}
"""

# === String processing ===

BENCHMARKS["string_match"] = """
int strstr_naive(const char *text, const char *pattern) {
    for (int i = 0; text[i]; i++) {
        int j;
        for (j = 0; pattern[j]; j++)
            if (text[i+j] != pattern[j]) break;
        if (!pattern[j]) return i;
    }
    return -1;
}
char text[] = "the quick brown fox jumps over the lazy dog";
char pat[] = "fox";
int main() { return strstr_naive(text, pat); }
"""

BENCHMARKS["string_reverse"] = """
void reverse(char *s, int n) {
    for (int i = 0; i < n/2; i++) {
        char t = s[i]; s[i] = s[n-1-i]; s[n-1-i] = t;
    }
}
char str[] = "abcdefghijklmnopqrstuvwxyz";
int main() { reverse(str, 26); return str[0]; }
"""

BENCHMARKS["histogram"] = """
#define SIZE 10000
int data[SIZE];
int hist[256];
void compute_histogram() {
    for (int i = 0; i < 256; i++) hist[i] = 0;
    for (int i = 0; i < SIZE; i++) hist[data[i] & 0xFF]++;
}
int main() { compute_histogram(); return hist[0]; }
"""

# === Control flow heavy ===

BENCHMARKS["state_machine"] = """
int process(const char *input, int len) {
    int state = 0, count = 0;
    for (int i = 0; i < len; i++) {
        char c = input[i];
        switch (state) {
            case 0: if (c == 'a') state = 1; else if (c == 'b') state = 2; break;
            case 1: if (c == 'b') { state = 3; count++; } else state = 0; break;
            case 2: if (c == 'a') state = 1; else state = 0; break;
            case 3: if (c == 'c') { state = 0; count++; } else if (c == 'a') state = 1; else state = 0; break;
        }
    }
    return count;
}
char input[1000];
int main() { return process(input, 1000); }
"""

BENCHMARKS["calculator"] = """
int eval(const char *expr, int *pos) {
    int result = 0, sign = 1;
    while (expr[*pos]) {
        char c = expr[*pos];
        if (c >= '0' && c <= '9') {
            int num = 0;
            while (expr[*pos] >= '0' && expr[*pos] <= '9')
                num = num * 10 + (expr[(*pos)++] - '0');
            result += sign * num;
            continue;
        }
        if (c == '+') { sign = 1; (*pos)++; }
        else if (c == '-') { sign = -1; (*pos)++; }
        else (*pos)++;
    }
    return result;
}
char expr[] = "123+456-78+9";
int main() { int p = 0; return eval(expr, &p); }
"""

BENCHMARKS["nested_conditions"] = """
int classify(int a, int b, int c) {
    if (a > 0) {
        if (b > 0) {
            if (c > 0) return 7;
            else if (c == 0) return 6;
            else return 5;
        } else if (b == 0) {
            if (c > 0) return 4;
            else return 3;
        } else {
            return c > 0 ? 2 : 1;
        }
    } else if (a == 0) {
        return b + c;
    } else {
        if (b > a) return -1;
        if (c > b) return -2;
        return -3;
    }
}
int main() {
    int sum = 0;
    for (int i = -10; i <= 10; i++)
        for (int j = -10; j <= 10; j++)
            for (int k = -10; k <= 10; k++)
                sum += classify(i, j, k);
    return sum;
}
"""

# === Arithmetic heavy ===

BENCHMARKS["polynomial"] = """
double horner(double x, double *coeffs, int n) {
    double result = coeffs[n-1];
    for (int i = n-2; i >= 0; i--)
        result = result * x + coeffs[i];
    return result;
}
double coeffs[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
int main() {
    double sum = 0;
    for (int i = 0; i < 1000; i++)
        sum += horner(i * 0.001, coeffs, 20);
    return (int)sum;
}
"""

BENCHMARKS["dot_product"] = """
#define N 4096
double a[N], b[N];
double dot_product() {
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}
int main() { return (int)dot_product(); }
"""

BENCHMARKS["saxpy"] = """
#define N 4096
double x[N], y[N];
void saxpy(double a) {
    for (int i = 0; i < N; i++) y[i] += a * x[i];
}
int main() { saxpy(2.5); return 0; }
"""

BENCHMARKS["prefix_sum"] = """
#define N 1024
int arr[N], prefix[N];
void prefix_sum() {
    prefix[0] = arr[0];
    for (int i = 1; i < N; i++) prefix[i] = prefix[i-1] + arr[i];
}
int main() { prefix_sum(); return prefix[N-1]; }
"""

BENCHMARKS["matrix_transpose"] = """
#define N 128
double A[N][N], B[N][N];
void transpose() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j][i] = A[i][j];
}
int main() { transpose(); return 0; }
"""

# === Struct/pointer heavy ===

BENCHMARKS["linked_list"] = """
#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node nodes[100];
int sum_list(struct Node *head) {
    int sum = 0;
    while (head) { sum += head->val; head = head->next; }
    return sum;
}
int main() {
    for (int i = 0; i < 99; i++) { nodes[i].val = i; nodes[i].next = &nodes[i+1]; }
    nodes[99].val = 99; nodes[99].next = 0;
    return sum_list(&nodes[0]);
}
"""

BENCHMARKS["tree_traversal"] = """
struct TreeNode { int val; int left, right; };
struct TreeNode tree[31];
int inorder_sum(int idx) {
    if (idx < 0 || idx >= 31) return 0;
    return inorder_sum(tree[idx].left) + tree[idx].val + inorder_sum(tree[idx].right);
}
int main() {
    for (int i = 0; i < 31; i++) {
        tree[i].val = i;
        tree[i].left = 2*i+1 < 31 ? 2*i+1 : -1;
        tree[i].right = 2*i+2 < 31 ? 2*i+2 : -1;
    }
    return inorder_sum(0);
}
"""

BENCHMARKS["struct_array"] = """
struct Point { double x, y, z; };
#define N 500
struct Point points[N];
double total_distance() {
    double sum = 0.0;
    for (int i = 1; i < N; i++) {
        double dx = points[i].x - points[i-1].x;
        double dy = points[i].y - points[i-1].y;
        double dz = points[i].z - points[i-1].z;
        sum += dx*dx + dy*dy + dz*dz;
    }
    return sum;
}
int main() { return (int)total_distance(); }
"""

# === Loop patterns ===

BENCHMARKS["loop_reduction"] = """
#define N 10000
double data[N];
double reduce() {
    double sum = 0, min_val = data[0], max_val = data[0];
    for (int i = 0; i < N; i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    return sum + min_val + max_val;
}
int main() { return (int)reduce(); }
"""

BENCHMARKS["loop_interchange_candidate"] = """
#define N 64
double A[N][N], B[N][N];
void bad_access() {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            A[i][j] = B[i][j] * 2.0 + 1.0;
}
int main() { bad_access(); return 0; }
"""

BENCHMARKS["loop_invariant"] = """
#define N 1000
double A[N], B[N];
void invariant_motion(double scale, double offset) {
    double factor = scale * scale + offset;
    for (int i = 0; i < N; i++)
        B[i] = A[i] * factor + offset * scale;
}
int main() { invariant_motion(2.0, 3.0); return 0; }
"""

BENCHMARKS["loop_unroll_candidate"] = """
#define N 256
int A[N], B[N], C[N];
void add_arrays() {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}
int main() { add_arrays(); return 0; }
"""

BENCHMARKS["nested_loops"] = """
#define N 32
int A[N][N][N];
void fill_3d() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                A[i][j][k] = i * N * N + j * N + k;
}
int main() { fill_3d(); return A[0][0][0]; }
"""

# === Dead code and constant propagation targets ===

BENCHMARKS["dead_code"] = """
int used_func(int x) { return x * 2 + 1; }
int unused_func(int x) { return x * x * x; }
int partially_dead(int x) {
    int a = x + 1;
    int b = x * 2;  // used
    int c = a + b;  // dead
    int d = b + 3;
    return d;
}
int main() { return used_func(5) + partially_dead(10); }
"""

BENCHMARKS["const_prop"] = """
int compute() {
    int a = 10;
    int b = 20;
    int c = a + b;
    int d = c * 2;
    int e = d - a;
    int f = e / 5;
    int arr[10];
    for (int i = 0; i < 10; i++) arr[i] = f + i;
    int sum = 0;
    for (int i = 0; i < 10; i++) sum += arr[i];
    return sum;
}
int main() { return compute(); }
"""

BENCHMARKS["strength_reduce"] = """
#define N 1000
int results[N];
void strength_reduce_test() {
    for (int i = 0; i < N; i++) {
        results[i] = i * 15;  // can be strength-reduced to addition
    }
}
int main() { strength_reduce_test(); return results[0]; }
"""

# === Memory access patterns ===

BENCHMARKS["memcpy_like"] = """
#define N 4096
char src[N], dst[N];
void my_memcpy() {
    for (int i = 0; i < N; i++) dst[i] = src[i];
}
int main() { my_memcpy(); return dst[0]; }
"""

BENCHMARKS["memset_like"] = """
#define N 4096
int arr[N];
void my_memset(int val) {
    for (int i = 0; i < N; i++) arr[i] = val;
}
int main() { my_memset(42); return arr[0]; }
"""

BENCHMARKS["gather_scatter"] = """
#define N 1024
double data[N], result[N];
int indices[N];
void gather() {
    for (int i = 0; i < N; i++) result[i] = data[indices[i]];
}
void scatter() {
    for (int i = 0; i < N; i++) data[indices[i]] = result[i];
}
int main() { gather(); scatter(); return 0; }
"""

# === Mixed/Complex ===

BENCHMARKS["image_blur"] = """
#define W 64
#define H 64
int image[H][W], blurred[H][W];
void blur() {
    for (int i = 1; i < H-1; i++)
        for (int j = 1; j < W-1; j++)
            blurred[i][j] = (image[i-1][j-1] + image[i-1][j] + image[i-1][j+1] +
                             image[i][j-1] + image[i][j] + image[i][j+1] +
                             image[i+1][j-1] + image[i+1][j] + image[i+1][j+1]) / 9;
}
int main() { blur(); return blurred[1][1]; }
"""

BENCHMARKS["convolution_1d"] = """
#define N 1024
#define K 5
double input[N], kernel[K], output[N];
void conv1d() {
    for (int i = K/2; i < N-K/2; i++) {
        output[i] = 0;
        for (int j = 0; j < K; j++)
            output[i] += input[i-K/2+j] * kernel[j];
    }
}
int main() { conv1d(); return 0; }
"""

BENCHMARKS["bubble_sort"] = """
#define N 200
int arr[N];
void bubble_sort() {
    for (int i = 0; i < N-1; i++)
        for (int j = 0; j < N-1-i; j++)
            if (arr[j] > arr[j+1]) {
                int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t;
            }
}
int main() { bubble_sort(); return arr[0]; }
"""

BENCHMARKS["insertion_sort"] = """
#define N 200
int arr[N];
void insertion_sort() {
    for (int i = 1; i < N; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}
int main() { insertion_sort(); return arr[0]; }
"""

BENCHMARKS["matrix_vector"] = """
#define M 100
#define N 100
double A[M][N], x[N], y[M];
void matvec() {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x[j];
    }
}
int main() { matvec(); return 0; }
"""

BENCHMARKS["dgemv"] = """
#define N 120
double A[N][N], x[N], y[N], alpha, beta_val;
void dgemv() {
    for (int i = 0; i < N; i++) {
        double temp = 0.0;
        for (int j = 0; j < N; j++) temp += A[i][j] * x[j];
        y[i] = alpha * temp + beta_val * y[i];
    }
}
int main() { alpha = 1.0; beta_val = 0.0; dgemv(); return 0; }
"""

BENCHMARKS["heat_equation"] = """
#define N 100
#define STEPS 50
double u[N], u_new[N];
void heat_1d(double dt, double dx) {
    double r = dt / (dx * dx);
    for (int t = 0; t < STEPS; t++) {
        for (int i = 1; i < N-1; i++)
            u_new[i] = u[i] + r * (u[i-1] - 2*u[i] + u[i+1]);
        for (int i = 1; i < N-1; i++) u[i] = u_new[i];
    }
}
int main() { heat_1d(0.01, 0.1); return 0; }
"""

BENCHMARKS["nbody_simple"] = """
#define N 50
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], mass[N];
void nbody_step(double dt) {
    for (int i = 0; i < N; i++) {
        double fx = 0, fy = 0, fz = 0;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double dx = px[j]-px[i], dy = py[j]-py[i], dz = pz[j]-pz[i];
            double dist2 = dx*dx + dy*dy + dz*dz + 0.01;
            double inv = mass[j] / (dist2 * dist2);  // simplified
            fx += dx*inv; fy += dy*inv; fz += dz*inv;
        }
        vx[i] += fx*dt; vy[i] += fy*dt; vz[i] += fz*dt;
    }
    for (int i = 0; i < N; i++) {
        px[i] += vx[i]*dt; py[i] += vy[i]*dt; pz[i] += vz[i]*dt;
    }
}
int main() { nbody_step(0.01); return 0; }
"""

BENCHMARKS["crc32"] = """
unsigned int crc_table[256];
void init_crc_table() {
    for (unsigned int i = 0; i < 256; i++) {
        unsigned int crc = i;
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        crc_table[i] = crc;
    }
}
unsigned int crc32(const unsigned char *data, int len) {
    unsigned int crc = 0xFFFFFFFF;
    for (int i = 0; i < len; i++)
        crc = (crc >> 8) ^ crc_table[(crc ^ data[i]) & 0xFF];
    return crc ^ 0xFFFFFFFF;
}
unsigned char data[1024];
int main() { init_crc_table(); return (int)crc32(data, 1024); }
"""

BENCHMARKS["bitcount"] = """
int popcount(unsigned int x) {
    int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}
int main() {
    int total = 0;
    for (unsigned int i = 0; i < 10000; i++) total += popcount(i);
    return total;
}
"""

BENCHMARKS["gcd_lcm"] = """
int gcd(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }
int lcm(int a, int b) { return a / gcd(a, b) * b; }
int main() {
    int sum = 0;
    for (int i = 1; i <= 100; i++)
        for (int j = 1; j <= 100; j++)
            sum += gcd(i, j);
    return sum;
}
"""

BENCHMARKS["sieve"] = """
#define N 10000
char is_prime[N];
int primes[N];
int sieve() {
    int count = 0;
    for (int i = 0; i < N; i++) is_prime[i] = 1;
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; i < N; i++)
        if (is_prime[i]) {
            primes[count++] = i;
            for (int j = i*i; j < N; j += i) is_prime[j] = 0;
        }
    return count;
}
int main() { return sieve(); }
"""

def create_and_compile():
    """Create all C benchmarks and compile to LLVM IR."""
    success = 0
    failed = []
    for name, code in BENCHMARKS.items():
        c_path = BENCHMARK_DIR / f"{name}.c"
        ll_path = BENCHMARK_DIR / f"{name}.ll"

        # Write C file
        with open(c_path, 'w') as f:
            f.write(code)

        # Compile to LLVM IR at -O0
        cmd = [
            "clang", "-O0", "-Xclang", "-disable-O0-optnone",
            "-emit-llvm", "-S", "-o", str(ll_path), str(c_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            failed.append((name, result.stderr))
            continue

        # Verify IR
        verify = subprocess.run(
            ["opt", "--passes=verify", str(ll_path), "-S", "-o", "/dev/null"],
            capture_output=True, text=True, timeout=10
        )
        if verify.returncode != 0:
            failed.append((name, "verification failed"))
            continue

        success += 1

    print(f"Successfully compiled {success}/{len(BENCHMARKS)} benchmarks")
    if failed:
        print(f"Failed: {[f[0] for f in failed]}")
        for name, err in failed:
            print(f"  {name}: {err[:200]}")
    return success, failed


if __name__ == "__main__":
    create_and_compile()
