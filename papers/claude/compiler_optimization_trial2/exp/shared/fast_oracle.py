#!/usr/bin/env python3
"""Fast compilation oracle using pipe I/O to avoid temp files."""
import subprocess
import json
import os
import re

WORKSPACE = "/home/nw366/ResearchArena/outputs/claude_t2_compiler_optimization/idea_01"

# Top 20 most impactful LLVM passes (ordered by pipeline position)
PASS_CATALOG = [
    {"name": "simplifycfg", "order": 1, "category": "control_flow"},
    {"name": "sroa", "order": 2, "category": "memory"},
    {"name": "early-cse", "order": 3, "category": "peephole"},
    {"name": "mem2reg", "order": 4, "category": "memory"},
    {"name": "instcombine", "order": 5, "category": "peephole"},
    {"name": "ipsccp", "order": 6, "category": "interprocedural"},
    {"name": "globalopt", "order": 7, "category": "interprocedural"},
    {"name": "jump-threading", "order": 8, "category": "control_flow"},
    {"name": "correlated-propagation", "order": 9, "category": "peephole"},
    {"name": "aggressive-instcombine", "order": 10, "category": "peephole"},
    {"name": "reassociate", "order": 11, "category": "peephole"},
    {"name": "licm", "order": 12, "category": "loop"},
    {"name": "loop-rotate", "order": 13, "category": "loop"},
    {"name": "indvars", "order": 14, "category": "loop"},
    {"name": "loop-deletion", "order": 15, "category": "loop"},
    {"name": "gvn", "order": 16, "category": "peephole"},
    {"name": "sccp", "order": 17, "category": "peephole"},
    {"name": "adce", "order": 18, "category": "peephole"},
    {"name": "dse", "order": 19, "category": "memory"},
    {"name": "deadargelim", "order": 20, "category": "interprocedural"},
]

PASS_NAMES = [p["name"] for p in PASS_CATALOG]
PASS_ORDER = {p["name"]: p["order"] for p in PASS_CATALOG}
N_PASSES = len(PASS_NAMES)

MODULE_PASSES = {"globalopt", "deadargelim", "ipsccp"}
# Passes that need loop-mssa() wrapper
LOOP_MSSA_PASSES = {"licm"}
# Passes that need loop() wrapper
LOOP_PASSES = {"loop-rotate", "indvars", "loop-deletion"}


def count_ir_instructions_from_string(ir_text):
    """Fast instruction counting from IR string."""
    count = 0
    in_func = False
    for line in ir_text.split('\n'):
        s = line.strip()
        if s.startswith("define "):
            in_func = True
        elif s == "}":
            in_func = False
        elif in_func and s and not s.startswith(";") and not s.endswith(":"):
            # Quick heuristic: any line with = or starting with known opcodes
            if ("=" in s or s.startswith(("ret ", "br ", "store ", "call ",
                "switch ", "unreachable", "invoke ", "resume "))):
                count += 1
    return count


class FastOracle:
    """Fast compilation oracle using pipe I/O."""

    def __init__(self, cache_file=None):
        self.cache = {}
        self.cache_file = cache_file or os.path.join(WORKSPACE, "data/fast_oracle_cache.json")
        self.stats = {"calls": 0, "cache_hits": 0, "compile_calls": 0}
        self._load_cache()
        # Pre-read IR files
        self._ir_cache = {}

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _get_ir(self, ll_file):
        if ll_file not in self._ir_cache:
            with open(ll_file) as f:
                self._ir_cache[ll_file] = f.read()
        return self._ir_cache[ll_file]

    def _run_opt(self, pipeline, ll_file, timeout=10):
        """Run opt and return instruction count, or None on failure."""
        try:
            proc = subprocess.Popen(
                ["opt", f"--passes={pipeline}", ll_file, "-S"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            if proc.returncode == 0:
                return count_ir_instructions_from_string(stdout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        return None

    def _cache_key(self, program_name, pass_subset):
        sorted_passes = "|".join(sorted(pass_subset))
        return f"{program_name}|{sorted_passes}"

    def compile_and_measure(self, ll_file, pass_subset, program_name=None):
        if program_name is None:
            program_name = os.path.basename(ll_file)

        key = self._cache_key(program_name, pass_subset)
        self.stats["calls"] += 1

        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]

        if not pass_subset:
            ic = count_ir_instructions_from_string(self._get_ir(ll_file))
            self.cache[key] = ic
            return ic

        # Order passes canonically
        ordered = sorted(pass_subset, key=lambda p: PASS_ORDER.get(p, 999))

        # Build pipeline with proper pass manager nesting
        mod_passes = [p for p in ordered if p in MODULE_PASSES]
        loop_mssa = [p for p in ordered if p in LOOP_MSSA_PASSES]
        loop_passes = [p for p in ordered if p in LOOP_PASSES]
        plain_func = [p for p in ordered
                      if p not in MODULE_PASSES and p not in LOOP_MSSA_PASSES and p not in LOOP_PASSES]

        parts = []
        if mod_passes:
            parts.extend(mod_passes)

        # Build function-level pipeline
        func_parts = []
        if plain_func:
            func_parts.extend(plain_func)
        if loop_mssa:
            func_parts.append(f"loop-mssa({','.join(loop_mssa)})")
        if loop_passes:
            func_parts.append(f"loop({','.join(loop_passes)})")

        if func_parts:
            parts.append(f"function({','.join(func_parts)})")
        pipeline = ",".join(parts)

        self.stats["compile_calls"] += 1
        ic = self._run_opt(pipeline, ll_file)
        if ic is None:
            # Fallback: try all passes as flat function pipeline (no loop wrappers)
            all_func = [p for p in ordered if p not in MODULE_PASSES]
            if all_func:
                pipeline2 = f"function({','.join(all_func)})"
                if mod_passes:
                    pipeline2 = ",".join(mod_passes) + "," + pipeline2
            else:
                pipeline2 = ",".join(mod_passes)
            ic = self._run_opt(pipeline2, ll_file)

        if ic is None:
            ic = count_ir_instructions_from_string(self._get_ir(ll_file))

        self.cache[key] = ic
        return ic

    def characteristic_value(self, ll_file, pass_subset, baseline_count, program_name=None):
        if not pass_subset:
            return 0.0
        optimized = self.compile_and_measure(ll_file, pass_subset, program_name)
        return (baseline_count - optimized) / baseline_count


if __name__ == "__main__":
    import time
    import numpy as np

    manifest = json.load(open(os.path.join(WORKSPACE, "data/benchmark_manifest.json")))
    prog = manifest["programs"][0]
    oracle = FastOracle()

    # Benchmark
    rng = np.random.RandomState(42)
    times = []
    for _ in range(50):
        k = rng.randint(1, len(PASS_NAMES))
        subset = set(rng.choice(PASS_NAMES, size=k, replace=False))
        t0 = time.time()
        oracle.compile_and_measure(prog["path"], subset, prog["name"])
        times.append(time.time() - t0)

    print(f"Per-call: mean={np.mean(times):.3f}s, median={np.median(times):.3f}s, p95={np.percentile(times, 95):.3f}s")
    print(f"For 150 perms × 20 passes = 3000 calls: {3000*np.mean(times)/60:.1f} min per program")
    print(f"15 programs × 3 seeds: {15*3*3000*np.mean(times)/3600:.1f} hours (no caching)")

    # Verify values
    bl = prog["baseline_instructions"]
    v = oracle.characteristic_value(prog["path"], set(PASS_NAMES), bl, prog["name"])
    print(f"\nAll passes v={v:.4f}")
    for p in ["mem2reg", "sroa", "gvn", "instcombine", "adce"]:
        v = oracle.characteristic_value(prog["path"], {p}, bl, prog["name"])
        print(f"  {p}: v={v:.4f}")
