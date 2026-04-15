"""Compile PolyBench/C 4.2.1 kernels to LLVM IR."""
import subprocess
import os
from pathlib import Path

WORKSPACE = Path("/home/nw366/ResearchArena/outputs/claude_v5_compiler_optimization/idea_01")
POLYBENCH_DIR = WORKSPACE / "data" / "PolyBenchC-4.2.1-master"
BENCHMARK_DIR = WORKSPACE / "experiments" / "benchmarks"
UTILITIES_DIR = POLYBENCH_DIR / "utilities"

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

def find_polybench_sources():
    """Find all PolyBench C source files (excluding utilities and .orig files)."""
    sources = []
    for c_file in sorted(POLYBENCH_DIR.rglob("*.c")):
        if "utilities" in str(c_file):
            continue
        if ".orig" in c_file.name:
            continue
        sources.append(c_file)
    return sources

def compile_to_ir(c_file, output_name):
    """Compile a C file to LLVM IR at -O0."""
    output_path = BENCHMARK_DIR / f"pb_{output_name}.ll"

    # Get the include directory for the benchmark (same dir as source)
    include_dir = c_file.parent

    cmd = [
        "clang", "-O0", "-Xclang", "-disable-O0-optnone",
        "-emit-llvm", "-S",
        f"-I{UTILITIES_DIR}",
        f"-I{include_dir}",
        "-DSMALL_DATASET",  # Use small dataset to keep IR reasonable
        "-DPOLYBENCH_NO_FLUSH_CACHE",
        "-o", str(output_path),
        str(c_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        # Try without polybench.c linked (some benchmarks are self-contained)
        print(f"  WARN: {c_file.name}: {result.stderr[:200]}")
        return None

    # Verify the IR is valid
    verify = subprocess.run(
        ["opt", "--passes=verify", "-S", "-o", "/dev/null", str(output_path)],
        capture_output=True, text=True, timeout=10
    )
    if verify.returncode != 0:
        print(f"  INVALID IR: {output_path.name}")
        output_path.unlink(missing_ok=True)
        return None

    return output_path

def main():
    sources = find_polybench_sources()
    print(f"Found {len(sources)} PolyBench source files")

    compiled = 0
    failed = 0

    for src in sources:
        # Use parent directory name as benchmark name for uniqueness
        name = src.stem.replace("-", "_")
        output = compile_to_ir(src, name)
        if output:
            compiled += 1
            # Count lines
            with open(output) as f:
                lines = len(f.readlines())
            print(f"  OK: pb_{name}.ll ({lines} lines)")
        else:
            failed += 1

    print(f"\nCompiled: {compiled}, Failed: {failed}")

    # List all benchmarks now available
    all_ll = sorted(BENCHMARK_DIR.glob("*.ll"))
    polybench = [f for f in all_ll if f.stem.startswith("pb_")]
    custom = [f for f in all_ll if not f.stem.startswith("pb_")]
    print(f"\nTotal benchmarks: {len(all_ll)} ({len(polybench)} PolyBench, {len(custom)} custom)")

if __name__ == "__main__":
    main()
