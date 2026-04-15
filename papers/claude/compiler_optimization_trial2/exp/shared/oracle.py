#!/usr/bin/env python3
"""Compilation oracle: applies LLVM passes and measures instruction count."""
import subprocess
import json
import os
import hashlib
import tempfile
from pathlib import Path

WORKSPACE = "/home/nw366/ResearchArena/outputs/claude_t2_compiler_optimization/idea_01"

# Curated list of 30 LLVM transform passes (new pass manager syntax)
# Ordered roughly by their position in -O3 pipeline
PASS_CATALOG = [
    {"name": "simplifycfg", "order": 1, "category": "control_flow"},
    {"name": "sroa", "order": 2, "category": "memory"},
    {"name": "early-cse", "order": 3, "category": "peephole"},
    {"name": "mem2reg", "order": 4, "category": "memory"},
    {"name": "instcombine", "order": 5, "category": "peephole"},
    {"name": "ipsccp", "order": 6, "category": "interprocedural"},
    {"name": "globalopt", "order": 7, "category": "interprocedural"},
    {"name": "inline", "order": 8, "category": "interprocedural"},
    {"name": "jump-threading", "order": 9, "category": "control_flow"},
    {"name": "correlated-propagation", "order": 10, "category": "peephole"},
    {"name": "aggressive-instcombine", "order": 11, "category": "peephole"},
    {"name": "tailcallelim", "order": 12, "category": "interprocedural"},
    {"name": "reassociate", "order": 13, "category": "peephole"},
    {"name": "licm", "order": 14, "category": "loop"},
    {"name": "loop-rotate", "order": 15, "category": "loop"},
    {"name": "simple-loop-unswitch", "order": 16, "category": "loop"},
    {"name": "loop-idiom", "order": 17, "category": "loop"},
    {"name": "indvars", "order": 18, "category": "loop"},
    {"name": "loop-deletion", "order": 19, "category": "loop"},
    {"name": "loop-unroll", "order": 20, "category": "loop"},
    {"name": "gvn", "order": 21, "category": "peephole"},
    {"name": "sccp", "order": 22, "category": "peephole"},
    {"name": "bdce", "order": 23, "category": "peephole"},
    {"name": "adce", "order": 24, "category": "peephole"},
    {"name": "memcpyopt", "order": 25, "category": "memory"},
    {"name": "dse", "order": 26, "category": "memory"},
    {"name": "globaldce", "order": 27, "category": "interprocedural"},
    {"name": "constmerge", "order": 28, "category": "interprocedural"},
    {"name": "deadargelim", "order": 29, "category": "interprocedural"},
    {"name": "loop-simplifycfg", "order": 30, "category": "loop"},
]

PASS_NAMES = [p["name"] for p in PASS_CATALOG]
PASS_ORDER = {p["name"]: p["order"] for p in PASS_CATALOG}
N_PASSES = len(PASS_NAMES)


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
                if "=" in stripped or stripped.startswith("ret ") or stripped.startswith("br ") or \
                   stripped.startswith("store ") or stripped.startswith("call ") or \
                   stripped.startswith("switch ") or stripped.startswith("unreachable") or \
                   stripped.startswith("invoke ") or stripped.startswith("resume "):
                    count += 1
    return count


class CompilationOracle:
    """Applies LLVM optimization passes and measures instruction count."""

    def __init__(self, cache_file=None):
        self.cache = {}
        self.cache_file = cache_file or os.path.join(WORKSPACE, "data/oracle_cache.json")
        self.stats = {"calls": 0, "cache_hits": 0}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _cache_key(self, program_name, pass_subset):
        """Create a deterministic cache key."""
        sorted_passes = tuple(sorted(pass_subset))
        return f"{program_name}|{'|'.join(sorted_passes)}"

    def compile_and_measure(self, ll_file, pass_subset, program_name=None):
        """Apply pass_subset to ll_file in canonical order, return instruction count."""
        if program_name is None:
            program_name = os.path.basename(ll_file)

        key = self._cache_key(program_name, pass_subset)
        self.stats["calls"] += 1

        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]

        # Order passes by canonical order
        ordered = sorted(pass_subset, key=lambda p: PASS_ORDER.get(p, 999))

        if not ordered:
            # Empty set: return baseline (unoptimized) count
            ic = count_ir_instructions(ll_file)
            self.cache[key] = ic
            return ic

        # Build pass pipeline string
        # Some passes are module-level, some function-level
        # Use a simple comma-separated pipeline
        pipeline = ",".join(ordered)

        with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = ["opt", f"--passes={pipeline}", ll_file, "-S", "-o", tmp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                # Try wrapping in function() for function passes
                func_passes = []
                mod_passes = []
                module_level = {"globalopt", "globaldce", "constmerge", "deadargelim",
                               "ipsccp", "inline"}
                for p in ordered:
                    if p in module_level:
                        mod_passes.append(p)
                    else:
                        func_passes.append(p)

                parts = []
                if mod_passes:
                    parts.extend(mod_passes)
                if func_passes:
                    parts.append(f"function({','.join(func_passes)})")
                pipeline2 = ",".join(parts)

                cmd2 = ["opt", f"--passes={pipeline2}", ll_file, "-S", "-o", tmp_path]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)

                if result2.returncode != 0:
                    # Fall back: run passes one at a time
                    import shutil
                    shutil.copy(ll_file, tmp_path)
                    for p in ordered:
                        with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp2:
                            tmp2_path = tmp2.name
                        if p in module_level:
                            pp = p
                        else:
                            pp = f"function({p})"
                        cmd3 = ["opt", f"--passes={pp}", tmp_path, "-S", "-o", tmp2_path]
                        r = subprocess.run(cmd3, capture_output=True, text=True, timeout=30)
                        if r.returncode == 0:
                            os.replace(tmp2_path, tmp_path)
                        elif os.path.exists(tmp2_path):
                            os.unlink(tmp2_path)

            ic = count_ir_instructions(tmp_path)
        except subprocess.TimeoutExpired:
            ic = count_ir_instructions(ll_file)  # timeout = no optimization
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        self.cache[key] = ic
        return ic

    def characteristic_value(self, ll_file, pass_subset, baseline_count, program_name=None):
        """Compute v(S) = fractional code size reduction."""
        if not pass_subset:
            return 0.0
        optimized = self.compile_and_measure(ll_file, pass_subset, program_name)
        return (baseline_count - optimized) / baseline_count


def verify_passes(ll_file):
    """Test which passes work with the current LLVM version."""
    working = []
    module_level = {"globalopt", "globaldce", "constmerge", "deadargelim", "ipsccp", "inline"}

    for p in PASS_CATALOG:
        name = p["name"]
        if name in module_level:
            pipeline = name
        else:
            pipeline = f"function({name})"

        cmd = ["opt", f"--passes={pipeline}", ll_file, "-S", "-o", "/dev/null"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            working.append(p)
        else:
            print(f"  SKIP: {name} - {r.stderr[:100]}")
    return working


if __name__ == "__main__":
    # Test the oracle
    import sys
    manifest_path = os.path.join(WORKSPACE, "data/benchmark_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_prog = manifest["programs"][0]
    ll_file = test_prog["path"]
    baseline = test_prog["baseline_instructions"]
    print(f"Testing on: {test_prog['name']} (baseline: {baseline} instructions)")

    # Verify passes
    print("\nVerifying passes...")
    working = verify_passes(ll_file)
    print(f"\n{len(working)} of {len(PASS_CATALOG)} passes work")

    # Test oracle
    oracle = CompilationOracle()

    # Empty set
    v_empty = oracle.characteristic_value(ll_file, set(), baseline, test_prog["name"])
    print(f"\nv(empty) = {v_empty:.4f}")

    # Single passes
    for p in PASS_NAMES[:5]:
        v = oracle.characteristic_value(ll_file, {p}, baseline, test_prog["name"])
        print(f"v({{{p}}}) = {v:.4f}")

    # Full set
    v_all = oracle.characteristic_value(ll_file, set(PASS_NAMES), baseline, test_prog["name"])
    print(f"\nv(all passes) = {v_all:.4f}")

    oracle.save_cache()
    print(f"\nOracle stats: {oracle.stats}")
