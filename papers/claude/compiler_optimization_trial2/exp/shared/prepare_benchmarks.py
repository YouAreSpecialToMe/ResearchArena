#!/usr/bin/env python3
"""Compile PolyBench kernels and small C programs to LLVM IR."""
import subprocess
import json
import os
import glob

WORKSPACE = "/home/nw366/ResearchArena/outputs/claude_t2_compiler_optimization/idea_01"
POLYBENCH_DIR = os.path.join(WORKSPACE, "data/polybench/polybench-c-4.2.1-beta")
UTILITIES_DIR = os.path.join(POLYBENCH_DIR, "utilities")
IR_DIR = os.path.join(WORKSPACE, "data/ir")
CBENCH_DIR = os.path.join(WORKSPACE, "data/cbench")

os.makedirs(IR_DIR, exist_ok=True)
os.makedirs(CBENCH_DIR, exist_ok=True)


def count_ir_instructions(ll_file):
    """Count LLVM IR instructions in a .ll file."""
    count = 0
    with open(ll_file) as f:
        in_function = False
        for line in f:
            stripped = line.strip()
            if stripped.startswith("define "):
                in_function = True
            elif stripped == "}":
                in_function = False
            elif in_function and stripped and not stripped.startswith(";") and not stripped.endswith(":"):
                # Lines that are actual instructions (not labels, not comments)
                if "=" in stripped or stripped.startswith("ret ") or stripped.startswith("br ") or \
                   stripped.startswith("store ") or stripped.startswith("call ") or \
                   stripped.startswith("switch ") or stripped.startswith("unreachable") or \
                   stripped.startswith("invoke ") or stripped.startswith("resume "):
                    count += 1
    return count


def compile_polybench():
    """Compile all PolyBench kernels to LLVM IR."""
    programs = []
    categories = ["datamining", "linear-algebra", "medley", "stencils"]

    for cat in categories:
        cat_dir = os.path.join(POLYBENCH_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        # Walk subdirectories to find .c files
        for root, dirs, files in os.walk(cat_dir):
            c_files = [f for f in files if f.endswith(".c")]
            if not c_files:
                continue
            for cf in c_files:
                name = cf.replace(".c", "")
                src_path = os.path.join(root, cf)
                # Find corresponding .h file
                h_file = os.path.join(root, name + ".h")
                out_path = os.path.join(IR_DIR, f"polybench_{name}.ll")

                # Compile each .c to .bc separately, then link
                poly_bc = os.path.join(IR_DIR, f"_polybench_util.bc")
                kern_bc = os.path.join(IR_DIR, f"_kern_{name}.bc")
                linked_bc = os.path.join(IR_DIR, f"_linked_{name}.bc")

                cmd1 = ["clang", "-O0", "-emit-llvm", "-c",
                        "-DPOLYBENCH_TIME", "-DMINI_DATASET",
                        f"-I{UTILITIES_DIR}", f"-I{root}",
                        os.path.join(UTILITIES_DIR, "polybench.c"),
                        "-o", poly_bc]
                cmd2 = ["clang", "-O0", "-emit-llvm", "-c",
                        "-DPOLYBENCH_TIME", "-DMINI_DATASET",
                        f"-I{UTILITIES_DIR}", f"-I{root}",
                        src_path, "-o", kern_bc]
                cmd3 = ["llvm-link", poly_bc, kern_bc, "-o", linked_bc]
                cmd4 = ["llvm-dis", linked_bc, "-o", out_path]
                try:
                    for c in [cmd1, cmd2, cmd3, cmd4]:
                        r = subprocess.run(c, capture_output=True, text=True, timeout=30)
                        if r.returncode != 0:
                            raise RuntimeError(r.stderr[:200])
                    result_ok = True
                except RuntimeError as e:
                    result_ok = False
                    result_err = str(e)
                except Exception as e:
                    result_ok = False
                    result_err = str(e)

                if result_ok:
                    ic = count_ir_instructions(out_path)
                    programs.append({
                        "name": f"polybench_{name}",
                        "path": out_path,
                        "category": cat,
                        "baseline_instructions": ic
                    })
                    print(f"  OK: polybench_{name} ({ic} instructions)")
                else:
                    print(f"  FAIL: {name}: {result_err}")
                # Clean up temp files
                for tf in [poly_bc, kern_bc, linked_bc]:
                    if os.path.exists(tf):
                        os.remove(tf)
    return programs


# Small C programs for diversity
SMALL_PROGRAMS = {
    "sort_bubble": """
#include <stdlib.h>
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j+1]) {
                int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t;
            }
}
int main() {
    int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = 100 - i;
    bubble_sort(arr, 100);
    return arr[0];
}
""",
    "matrix_multiply": """
#define N 32
int A[N][N], B[N][N], C[N][N];
void matmul() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}
int main() { matmul(); return C[0][0]; }
""",
    "string_ops": """
#include <string.h>
char buf[1024];
int count_chars(const char *s, char c) {
    int cnt = 0;
    while (*s) { if (*s == c) cnt++; s++; }
    return cnt;
}
void reverse_str(char *s) {
    int n = strlen(s);
    for (int i = 0; i < n/2; i++) {
        char t = s[i]; s[i] = s[n-1-i]; s[n-1-i] = t;
    }
}
int main() {
    strcpy(buf, "hello world test string operations");
    reverse_str(buf);
    return count_chars(buf, 'o');
}
""",
    "binary_search": """
int bsearch_arr(int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
int main() {
    int arr[200];
    for (int i = 0; i < 200; i++) arr[i] = i * 2;
    int sum = 0;
    for (int i = 0; i < 100; i++)
        sum += bsearch_arr(arr, 200, i * 3);
    return sum;
}
""",
    "fibonacci_dp": """
int fib[100];
int fibonacci(int n) {
    fib[0] = 0; fib[1] = 1;
    for (int i = 2; i <= n; i++)
        fib[i] = fib[i-1] + fib[i-2];
    return fib[n];
}
int main() {
    int sum = 0;
    for (int i = 1; i < 50; i++)
        sum += fibonacci(i);
    return sum;
}
""",
    "bitwise_ops": """
int popcount(unsigned int x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}
unsigned int reverse_bits(unsigned int x) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}
int main() {
    int sum = 0;
    for (unsigned int i = 0; i < 1000; i++)
        sum += popcount(reverse_bits(i));
    return sum;
}
""",
    "linked_list": """
#include <stdlib.h>
struct Node { int val; struct Node *next; };
struct Node nodes[100];
int node_idx = 0;
struct Node* new_node(int v) {
    struct Node *n = &nodes[node_idx++];
    n->val = v; n->next = 0;
    return n;
}
struct Node* insert_sorted(struct Node *head, int v) {
    struct Node *n = new_node(v);
    if (!head || v < head->val) { n->next = head; return n; }
    struct Node *cur = head;
    while (cur->next && cur->next->val < v) cur = cur->next;
    n->next = cur->next; cur->next = n;
    return head;
}
int sum_list(struct Node *h) {
    int s = 0; while (h) { s += h->val; h = h->next; } return s;
}
int main() {
    struct Node *head = 0;
    for (int i = 50; i >= 0; i--) head = insert_sorted(head, i);
    return sum_list(head);
}
""",
    "hash_table": """
#define TABLE_SIZE 64
#define ENTRIES 200
struct Entry { int key; int val; int used; };
struct Entry table[TABLE_SIZE];
unsigned hash(int key) { return (unsigned)(key * 2654435761u) % TABLE_SIZE; }
void insert(int key, int val) {
    unsigned h = hash(key);
    for (int i = 0; i < TABLE_SIZE; i++) {
        unsigned idx = (h + i) % TABLE_SIZE;
        if (!table[idx].used || table[idx].key == key) {
            table[idx].key = key; table[idx].val = val; table[idx].used = 1;
            return;
        }
    }
}
int lookup(int key) {
    unsigned h = hash(key);
    for (int i = 0; i < TABLE_SIZE; i++) {
        unsigned idx = (h + i) % TABLE_SIZE;
        if (!table[idx].used) return -1;
        if (table[idx].key == key) return table[idx].val;
    }
    return -1;
}
int main() {
    for (int i = 0; i < ENTRIES; i++) insert(i, i * i);
    int sum = 0;
    for (int i = 0; i < ENTRIES; i++) sum += lookup(i);
    return sum % 256;
}
""",
    "recursive_gcd": """
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
int lcm(int a, int b) { return a / gcd(a, b) * b; }
int main() {
    int sum = 0;
    for (int i = 1; i <= 100; i++)
        for (int j = 1; j <= 50; j++)
            sum += gcd(i, j);
    return sum % 256;
}
""",
    "stencil_1d": """
#define N 256
double A[N], B[N];
void jacobi_1d(int steps) {
    for (int t = 0; t < steps; t++) {
        for (int i = 1; i < N-1; i++)
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]);
        for (int i = 1; i < N-1; i++)
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1]);
    }
}
int main() {
    for (int i = 0; i < N; i++) A[i] = (double)i;
    jacobi_1d(10);
    return (int)A[N/2];
}
"""
}


def compile_small_programs():
    """Compile small C benchmark programs to LLVM IR."""
    programs = []
    for name, code in SMALL_PROGRAMS.items():
        src_path = os.path.join(CBENCH_DIR, f"{name}.c")
        out_path = os.path.join(IR_DIR, f"small_{name}.ll")

        with open(src_path, "w") as f:
            f.write(code)

        cmd = ["clang", "-O0", "-emit-llvm", "-S", src_path, "-o", out_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                ic = count_ir_instructions(out_path)
                programs.append({
                    "name": f"small_{name}",
                    "path": out_path,
                    "category": "small_programs",
                    "baseline_instructions": ic
                })
                print(f"  OK: small_{name} ({ic} instructions)")
            else:
                print(f"  FAIL: {name}: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
    return programs


if __name__ == "__main__":
    print("Compiling PolyBench kernels...")
    poly_programs = compile_polybench()
    print(f"\nCompiled {len(poly_programs)} PolyBench programs")

    print("\nCompiling small benchmark programs...")
    small_programs = compile_small_programs()
    print(f"\nCompiled {len(small_programs)} small programs")

    all_programs = poly_programs + small_programs
    manifest = {
        "total_programs": len(all_programs),
        "programs": all_programs
    }
    manifest_path = os.path.join(WORKSPACE, "data/benchmark_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal: {len(all_programs)} programs. Manifest saved to {manifest_path}")
