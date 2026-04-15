"""Shared utilities for compiler pass algebra experiments."""
import subprocess
import hashlib
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path("/home/nw366/ResearchArena/outputs/claude_v5_compiler_optimization/idea_01")
PASS_LIST_FILE = WORKSPACE / "experiments" / "pass_list.txt"
BENCHMARK_DIR = WORKSPACE / "experiments" / "benchmarks"
RESULTS_DIR = WORKSPACE / "experiments" / "results"
FIGURES_DIR = WORKSPACE / "figures"
TIMEOUT = 30  # seconds per opt invocation


def get_pass_list():
    """Read the list of passes to test."""
    with open(PASS_LIST_FILE) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def get_benchmark_files():
    """Get all .ll benchmark files."""
    return sorted(BENCHMARK_DIR.glob("*.ll"))


def run_opt(input_ir_path, passes, output_path=None, timeout=TIMEOUT):
    """Run opt with given passes on an IR file. Returns output IR as string or writes to output_path."""
    if isinstance(passes, str):
        pass_arg = passes
    elif isinstance(passes, list):
        pass_arg = ",".join(passes)
    else:
        pass_arg = str(passes)

    cmd = ["opt", f"--passes={pass_arg}", "-S"]
    if output_path:
        cmd.extend(["-o", str(output_path)])
    cmd.append(str(input_ir_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        if output_path:
            return output_path
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def run_opt_pipeline(input_ir_path, pipeline_str, timeout=TIMEOUT):
    """Run opt with a standard pipeline like -O2."""
    cmd = ["opt", pipeline_str, "-S", str(input_ir_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def count_instructions(ir_text):
    """Count LLVM IR instructions from IR text."""
    if ir_text is None:
        return None
    count = 0
    for line in ir_text.split('\n'):
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('!'):
            continue
        if line.startswith('define ') or line.startswith('declare '):
            continue
        if line.startswith('@') or line.startswith('target '):
            continue
        if line.startswith('attributes ') or line.startswith('source_filename'):
            continue
        if line == '}' or line.startswith('metadata'):
            continue
        # Count lines that look like instructions (assignments or terminators)
        if '=' in line or line.startswith('ret ') or line.startswith('br ') or \
           line.startswith('switch ') or line.startswith('unreachable') or \
           line.startswith('store ') or line.startswith('call ') or \
           line.startswith('invoke ') or line.startswith('resume '):
            count += 1
    return count


def structural_hash(ir_text):
    """Compute a structural hash of IR text, ignoring variable names and metadata."""
    if ir_text is None:
        return None
    # Normalize: remove comments, metadata, and normalize variable names
    lines = []
    for line in ir_text.split('\n'):
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('!') or \
           line.startswith('source_filename') or line.startswith('target '):
            continue
        # Normalize local variable names (%name -> %v)
        line = re.sub(r'%[a-zA-Z_][a-zA-Z0-9_.]*', '%v', line)
        line = re.sub(r'%\d+', '%v', line)
        # Normalize metadata references
        line = re.sub(r'!dbg !\d+', '', line)
        line = re.sub(r', !tbaa !\d+', '', line)
        line = re.sub(r'!\d+', '!N', line)
        lines.append(line)
    normalized = '\n'.join(lines)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def apply_pass_to_ir(ir_path, pass_name):
    """Apply a single pass and return (ir_text, instcount, hash)."""
    # Handle passes that need wrapping for proper pipeline syntax
    pass_str = get_pass_pipeline_str(pass_name)
    ir_text = run_opt(ir_path, pass_str)
    if ir_text is None:
        return None, None, None
    ic = count_instructions(ir_text)
    h = structural_hash(ir_text)
    return ir_text, ic, h


def apply_pass_to_ir_text(ir_text, pass_name):
    """Apply a pass to IR text (not a file). Returns (ir_text, instcount, hash)."""
    if ir_text is None:
        return None, None, None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir_text)
        tmp_path = f.name
    try:
        result = apply_pass_to_ir(tmp_path, pass_name)
    finally:
        os.unlink(tmp_path)
    return result


def get_pass_pipeline_str(pass_name):
    """Get the proper pipeline string for a pass, wrapping loop passes in proper adaptors."""
    # True loop passes (run inside loop pipeline adaptor)
    loop_passes = {
        'indvars', 'licm', 'loop-deletion', 'loop-idiom', 'loop-reduce',
        'loop-rotate',
    }
    # Module-level passes
    module_passes = {
        'constmerge', 'deadargelim', 'globalopt'
    }
    # Function-level passes (including some that sound like loop passes but aren't)
    # lcssa, loop-simplify, loop-sink, loop-fusion, loop-distribute, loop-unroll
    # are function passes in LLVM 18's new pass manager
    function_passes = {
        'lcssa', 'loop-simplify', 'loop-sink', 'loop-fusion', 'loop-distribute',
    }
    # Special cases
    if pass_name == 'instcombine':
        return "function(instcombine<no-verify-fixpoint>)"
    elif pass_name == 'loop-unroll':
        return "function(loop(loop-unroll-full))"
    elif pass_name in loop_passes:
        return f"function(loop-mssa({pass_name}))"
    elif pass_name in module_passes:
        return pass_name
    elif pass_name in function_passes:
        return f"function({pass_name})"
    else:
        return f"function({pass_name})"


def get_baseline_ir(benchmark_path):
    """Read the unoptimized baseline IR."""
    with open(benchmark_path) as f:
        return f.read()
