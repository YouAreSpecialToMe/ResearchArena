"""
Cooperative game for compiler pass selection.
Maps subsets of LLVM optimization passes to IR instruction count reduction.
"""
import subprocess
import os
import tempfile
import re
import numpy as np
from functools import lru_cache

# The candidate LLVM optimization passes (new pass manager names for LLVM 18)
CANDIDATE_PASSES = [
    "mem2reg",
    "instcombine",
    "simplifycfg",
    "gvn",
    "licm",
    "loop-unroll",
    "loop-rotate",
    "loop-simplify",
    "indvars",
    "sroa",
    "sccp",
    "dce",
    "adce",
    "reassociate",
    "jump-threading",
    "correlated-propagation",
    "early-cse",
    "tailcallelim",
    "dse",
    "bdce",
]

# Canonical ordering (roughly matching O3 pipeline order)
PASS_ORDER = {p: i for i, p in enumerate(CANDIDATE_PASSES)}


def count_ir_instructions(bc_path):
    """Count IR instructions in a bitcode file using llvm-dis."""
    try:
        result = subprocess.run(
            ["llvm-dis", bc_path, "-o", "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        # Count instruction lines
        count = 0
        for line in result.stdout.split('\n'):
            stripped = line.lstrip()
            if stripped and not stripped.startswith(';') and not stripped.startswith('}') \
               and not stripped.startswith('define') and not stripped.startswith('declare') \
               and not stripped.startswith('source_filename') and not stripped.startswith('target') \
               and not stripped.startswith('@') and not stripped.startswith('!') \
               and not stripped.startswith('attributes') and not stripped.startswith('module') \
               and not stripped.endswith(':') and line.startswith('  '):
                count += 1
        return count
    except subprocess.TimeoutExpired:
        return None


def apply_passes(bc_path, passes, output_path=None):
    """Apply LLVM passes. Uses batch mode first, falls back to sequential if batch fails."""
    if not passes:
        return bc_path

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".bc")
        os.close(fd)

    # Try batch mode first (much faster)
    pass_pipeline = ",".join(passes)
    try:
        result = subprocess.run(
            ["opt", f"-passes={pass_pipeline}", bc_path, "-o", output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except subprocess.TimeoutExpired:
        pass

    # Batch failed — apply passes one at a time
    import shutil
    fd2, tmp_a = tempfile.mkstemp(suffix=".bc")
    os.close(fd2)
    fd3, tmp_b = tempfile.mkstemp(suffix=".bc")
    os.close(fd3)

    try:
        shutil.copy2(bc_path, tmp_a)
        for p in passes:
            try:
                result = subprocess.run(
                    ["opt", f"-passes={p}", tmp_a, "-o", tmp_b],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    tmp_a, tmp_b = tmp_b, tmp_a  # swap
            except subprocess.TimeoutExpired:
                pass  # skip this pass
        shutil.copy2(tmp_a, output_path)
    finally:
        for f in [tmp_a, tmp_b]:
            if os.path.exists(f):
                os.remove(f)

    return output_path if os.path.exists(output_path) else None


class CompilerGame:
    """Cooperative game: passes are players, value = IR instruction count reduction."""

    def __init__(self, benchmark_bc_path, passes=None, cache_dir=None):
        self.bc_path = os.path.abspath(benchmark_bc_path)
        self.passes = passes or CANDIDATE_PASSES
        self.n_players = len(self.passes)
        self.cache = {}
        self.cache_dir = cache_dir

        # Compute baseline (unoptimized) instruction count
        self.baseline_count = count_ir_instructions(self.bc_path)
        if self.baseline_count is None or self.baseline_count == 0:
            raise ValueError(f"Could not count instructions in {self.bc_path}")

    def _subset_key(self, binary_vector):
        """Convert binary vector to tuple key for caching."""
        return tuple(int(x) for x in binary_vector)

    def value(self, binary_vector):
        """
        Compute v(S) = (baseline - optimized) / baseline for subset S.
        binary_vector: array of 0/1 indicating which passes are on.
        Returns: fractional reduction in instruction count.
        """
        key = self._subset_key(binary_vector)
        if key in self.cache:
            return self.cache[key]

        # Get selected passes in canonical order
        selected = []
        for i, on in enumerate(binary_vector):
            if on:
                selected.append(self.passes[i])

        if not selected:
            self.cache[key] = 0.0
            return 0.0

        # Apply passes
        fd, tmp_path = tempfile.mkstemp(suffix=".bc")
        os.close(fd)
        try:
            out_path = apply_passes(self.bc_path, selected, tmp_path)
            if out_path is None:
                self.cache[key] = 0.0
                return 0.0

            opt_count = count_ir_instructions(out_path)
            if opt_count is None:
                self.cache[key] = 0.0
                return 0.0

            reduction = (self.baseline_count - opt_count) / self.baseline_count
            self.cache[key] = reduction
            return reduction
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def __call__(self, x):
        """Callable interface for shapiq. Always returns array."""
        x = np.asarray(x)
        if x.ndim == 1:
            return np.array([self.value(x)])
        else:
            return np.array([self.value(row) for row in x])

    def get_full_value(self):
        """Value of the grand coalition (all passes on)."""
        return self.value(np.ones(self.n_players))

    def get_o3_value(self):
        """IR reduction from -O3."""
        fd, tmp_path = tempfile.mkstemp(suffix=".bc")
        os.close(fd)
        try:
            result = subprocess.run(
                ["opt", "-O3", self.bc_path, "-o", tmp_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return None
            count = count_ir_instructions(tmp_path)
            if count is None:
                return None
            return (self.baseline_count - count) / self.baseline_count
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def get_optimization_level_counts(bc_path):
    """Get instruction counts for all LLVM optimization levels."""
    baseline = count_ir_instructions(bc_path)
    results = {"O0": baseline}

    for level in ["O1", "O2", "O3", "Os", "Oz"]:
        fd, tmp_path = tempfile.mkstemp(suffix=".bc")
        os.close(fd)
        try:
            result = subprocess.run(
                ["opt", f"-{level}", bc_path, "-o", tmp_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                count = count_ir_instructions(tmp_path)
                results[level] = count
            else:
                results[level] = None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return results
