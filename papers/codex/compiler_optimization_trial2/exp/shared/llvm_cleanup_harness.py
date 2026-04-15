#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
BENCH_DIR = DATA_DIR / "benchmarks"
REG_DIR = DATA_DIR / "regressions"
SRC_DIR = DATA_DIR / "generated_src"
REF_DIR = DATA_DIR / "stock_reference"
FIG_DIR = ROOT / "figures"
EXP_DIR = ROOT / "exp"

SEEDS = [11, 17, 29]
MAIN_CONFIGS = ["stock", "last_run", "throttle", "dirty_function", "typed_skip_global"]
ABLATION_CONFIGS = ["single_bit_state", "all_unknown", "untyped_locality"]
DECLARED_BUT_SKIPPED_CONFIGS = ["no_local_execution"]
ALL_DECLARED_CONFIGS = MAIN_CONFIGS + ABLATION_CONFIGS + DECLARED_BUT_SKIPPED_CONFIGS
WARMUP_REPEATS = 1
MAIN_REPEATS = 3
ABLATION_REPEATS = 2
MAX_WORKERS = 2

PRODUCER_SEQUENCE = [
    ("sroa", "function(sroa)"),
    ("gvn", "function(gvn)"),
    ("licm", "function(loop-mssa(licm))"),
    ("loop_rotate", "function(loop(loop-rotate))"),
]
CLEANUP_SEQUENCE = [
    ("instcombine", "function(instcombine)"),
    ("simplifycfg", "function(simplifycfg)"),
]

FUNC_RE = re.compile(r"^define\s+(?:internal\s+)?[^@]*@([^(]+)\(")
CURRENT_LOG_PATH: Path | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    family: str
    variant: int
    helper_count: int
    inner_iters: int
    runtime_input: int
    validation_input: int
    corpus: str


EXEC_SPECS = [
    WorkloadSpec("consumer_jpeg_like", "jpeg", 1, 28, 36, 1600, 320, "benchmark"),
    WorkloadSpec("office_stringsearch_like", "stringsearch", 2, 30, 42, 2400, 480, "benchmark"),
    WorkloadSpec("security_sha_like", "sha", 3, 32, 38, 2200, 440, "benchmark"),
    WorkloadSpec("telecom_crc32_like", "crc", 4, 28, 40, 2600, 520, "benchmark"),
    WorkloadSpec("automotive_qsort1_like", "sort", 5, 26, 36, 1900, 380, "benchmark"),
    WorkloadSpec("adpcm_like", "predictor", 6, 30, 34, 1800, 360, "benchmark"),
    WorkloadSpec("dijkstra_like", "graph", 7, 32, 35, 1700, 340, "benchmark"),
    WorkloadSpec("heapsort_like", "heap", 8, 26, 38, 2100, 420, "benchmark"),
    WorkloadSpec("huffbench_like", "huff", 9, 30, 41, 2300, 460, "benchmark"),
    WorkloadSpec("fft_like", "fft", 10, 34, 37, 2000, 400, "benchmark"),
]

REGRESSION_SPECS = [
    WorkloadSpec(f"cfg_cleanup_{i:02d}", "cfg", i, 8, 16, 300, 60, "cfg_regression")
    for i in range(1, 9)
] + [
    WorkloadSpec(f"loop_form_{i:02d}", "loop", i + 20, 8, 18, 320, 64, "loop_regression")
    for i in range(1, 9)
] + [
    WorkloadSpec(f"mixed_case_{i:02d}", "mixed", i + 40, 8, 17, 340, 68, "mixed_regression")
    for i in range(1, 9)
]


def append_log(entry: dict[str, Any]) -> None:
    if CURRENT_LOG_PATH is None:
        return
    CURRENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURRENT_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    append_log(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "cmd": cmd,
            "cwd": str(cwd or ROOT),
            "timeout_s": timeout,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    )
    return proc


def ir_text(path: Path) -> str:
    return run_cmd(["opt", "-S", str(path), "-o", "-"]).stdout


def parse_ir_stats(path: Path) -> dict[str, Any]:
    text = ir_text(path)
    functions: dict[str, str] = {}
    current_name = None
    current_lines: list[str] = []
    in_function = False
    braces = 0
    for line in text.splitlines():
        if not in_function:
            match = FUNC_RE.match(line)
            if match:
                current_name = match.group(1)
                current_lines = [line]
                braces = line.count("{") - line.count("}")
                in_function = True
        else:
            current_lines.append(line)
            braces += line.count("{") - line.count("}")
            if braces == 0 and current_name is not None:
                functions[current_name] = "\n".join(current_lines)
                current_name = None
                current_lines = []
                in_function = False

    loops = text.count("!llvm.loop")

    non_comment_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith(";")]
    instruction_lines = [
        line
        for line in non_comment_lines
        if not line.endswith(":")
        and not line.startswith("define ")
        and line not in {"{", "}"}
        and not line.startswith("declare ")
        and not line.startswith("attributes ")
        and not line.startswith("target ")
        and not line.startswith("source_filename")
    ]
    basic_blocks = sum(
        1
        for line in non_comment_lines
        if line.endswith(":") and not line.startswith("attributes ") and not line.startswith("module asm")
    )
    branch_count = sum(" br " in f" {line} " for line in instruction_lines) + sum(
        " switch " in f" {line} " for line in instruction_lines
    )
    phi_count = sum(" phi " in f" {line} " for line in instruction_lines)
    alloca_count = sum(" alloca " in f" {line} " for line in instruction_lines)
    call_count = sum(" call " in f" {line} " for line in instruction_lines)
    func_hashes = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in functions.items()}
    module_hash = hashlib.sha256("\n".join(non_comment_lines).encode()).hexdigest()
    return {
        "function_count": len(functions),
        "loop_count": loops,
        "instruction_count": len(instruction_lines),
        "basic_block_count": basic_blocks,
        "branch_count": branch_count,
        "phi_count": phi_count,
        "alloca_count": alloca_count,
        "call_count": call_count,
        "functions": sorted(functions),
        "function_hashes": func_hashes,
        "module_hash": module_hash,
    }


def changed_functions(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    names = set(before["function_hashes"]) | set(after["function_hashes"])
    dirty = [
        name
        for name in names
        if before["function_hashes"].get(name) != after["function_hashes"].get(name)
    ]
    return sorted(dirty)


def diff_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    keys = [
        "function_count",
        "loop_count",
        "instruction_count",
        "basic_block_count",
        "branch_count",
        "phi_count",
        "alloca_count",
        "call_count",
    ]
    return {key: int(after[key]) - int(before[key]) for key in keys}


def compile_c_to_bc(src_path: Path, bc_path: Path) -> None:
    run_cmd(
        [
            "clang",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            "-emit-llvm",
            "-c",
            str(src_path),
            "-o",
            str(bc_path),
        ]
    )


def compile_bc_to_exe(bc_path: Path, exe_path: Path) -> None:
    run_cmd(["clang", str(bc_path), "-O0", "-o", str(exe_path)])


def execute_binary(exe_path: Path, n_value: int) -> tuple[float, str]:
    start = time.perf_counter()
    proc = run_cmd([str(exe_path), str(n_value)], timeout=60)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc.stdout.strip()


def family_expr(family: str, variant: int, idx: str, acc: str, arr: str) -> str:
    seed = variant * 17 + 13
    if family in {"jpeg", "fft"}:
        return f"(({acc} ^ (({arr}[{idx}] << {variant % 5 + 1}) | ({arr}[{idx}] >> {variant % 3 + 1}))) + {seed})"
    if family in {"stringsearch", "huff"}:
        return f"(({acc} + ({arr}[{idx}] * ({idx} + {seed}))) ^ (({arr}[{idx}] >> 1) + {seed}))"
    if family in {"sha", "crc"}:
        return f"(({acc} << 5) + {acc} + ({arr}[{idx}] ^ ({idx} * {seed})))"
    if family in {"sort", "heap"}:
        return f"(({acc} + ({arr}[{idx}] * ({idx} % 7 + 1))) ^ (({acc} >> 3) + {seed}))"
    if family in {"predictor", "graph"}:
        return f"(({acc} + (({arr}[{idx}] ^ {seed}) - ({idx} * 3))) ^ ({acc} << 1))"
    if family == "cfg":
        return f"(({acc} + ({arr}[{idx}] ^ {seed})) - ({idx} * ({variant % 5 + 1})))"
    if family == "loop":
        return f"(({acc} ^ ({arr}[{idx}] + {seed})) + (({idx} & 7) * {variant % 11 + 3}))"
    return f"(({acc} + ({arr}[{idx}] * ({variant % 13 + 3}))) ^ ({idx} + {seed}))"


def render_workload_source(spec: WorkloadSpec) -> str:
    helper_defs: list[str] = []
    helper_calls: list[str] = []
    for h in range(spec.helper_count):
        expr = family_expr(spec.family, spec.variant + h, "i", "acc", "buf")
        helper_defs.append(
            f"""
static uint32_t helper_{h}(uint32_t *buf, int n, uint32_t salt) {{
    uint32_t acc = salt ^ 0x9e3779b9u ^ (uint32_t){spec.variant + h};
    for (int i = 0; i < n; ++i) {{
        uint32_t v = buf[(i * {(h % 5) + 1} + {h}) % n];
        if (((i + {h}) & 3) == 0) {{
            acc = {expr};
        }} else if (((v + salt + {h}) & 7) < 3) {{
            acc ^= (v << ((i + {h}) & 7));
        }} else {{
            acc += (v >> (((i + {h}) & 5) + 1));
        }}
        if ((acc & 1u) == 0u) {{
            buf[(i + {h}) % n] ^= acc + (uint32_t)i;
        }} else {{
            buf[(i * 3 + {h}) % n] += (acc ^ (uint32_t){h});
        }}
    }}
    return acc;
}}
"""
        )
        helper_calls.append(
            f"""
    if ((round + {h}) % 3 == 0) {{
        total ^= helper_{h}(buf, n, total + {h + 1}u);
    }} else {{
        total += helper_{h}(buf, n, total ^ {h + 7}u);
    }}
"""
        )

    return f"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t mix32(uint32_t x) {{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}}

{''.join(helper_defs)}

static uint32_t kernel(uint32_t *buf, int n, int rounds) {{
    uint32_t total = 0x12345678u ^ (uint32_t){spec.variant};
    for (int round = 0; round < rounds; ++round) {{
{''.join(helper_calls)}
        if ((total & 15u) == 0u) {{
            total ^= mix32(total + (uint32_t)round);
        }} else {{
            total += mix32(total ^ (uint32_t)(round * 17 + {spec.variant}));
        }}
    }}
    return total;
}}

int main(int argc, char **argv) {{
    int n = argc > 1 ? atoi(argv[1]) : {spec.validation_input};
    if (n < 32) {{
        n = 32;
    }}
    uint32_t *buf = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)n);
    if (!buf) {{
        return 1;
    }}
    for (int i = 0; i < n; ++i) {{
        buf[i] = mix32((uint32_t)(i * {spec.variant + 3} + 11));
    }}
    uint32_t out = kernel(buf, n, {spec.inner_iters});
    printf("%u\\n", out);
    free(buf);
    return 0;
}}
"""


def ensure_workloads() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    REG_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    runtime_manifest: list[dict[str, Any]] = []
    benchmark_manifest: list[dict[str, Any]] = []
    for spec in EXEC_SPECS + REGRESSION_SPECS:
        corpus_dir = BENCH_DIR if spec.corpus == "benchmark" else REG_DIR
        src_path = SRC_DIR / f"{spec.name}.c"
        bc_path = corpus_dir / f"{spec.name}.bc"
        src_path.write_text(render_workload_source(spec))
        compile_c_to_bc(src_path, bc_path)
        stats_now = parse_ir_stats(bc_path)
        benchmark_manifest.append(
            {
                "name": spec.name,
                "corpus": spec.corpus,
                "source_path": str(src_path),
                "bitcode_path": str(bc_path),
                "function_count": stats_now["function_count"],
                "loop_count": stats_now["loop_count"],
                "instruction_count": stats_now["instruction_count"],
            }
        )
        runtime_manifest.append(
            {
                "name": spec.name,
                "corpus": spec.corpus,
                "validation_command": [str(spec.validation_input)],
                "runtime_command": [str(spec.runtime_input)],
            }
        )
    return benchmark_manifest, runtime_manifest


def ensure_stock_reference(spec: WorkloadSpec) -> tuple[Path, str]:
    ref_bc = REF_DIR / f"{spec.name}.bc"
    ref_json = REF_DIR / f"{spec.name}.json"
    if ref_bc.exists() and ref_json.exists():
        try:
            return ref_bc, json.loads(ref_json.read_text())["validation_output"]
        except json.JSONDecodeError:
            ref_json.unlink(missing_ok=True)
    current = (BENCH_DIR if spec.corpus == "benchmark" else REG_DIR) / f"{spec.name}.bc"
    with tempfile.TemporaryDirectory(prefix=f"stock_ref_{spec.name}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for producer_name, producer_pass in PRODUCER_SEQUENCE:
            next_bc = tmpdir / f"{producer_name}.bc"
            run_opt_pass(current, next_bc, producer_pass)
            current = next_bc
            for cleanup_name, cleanup_pass in CLEANUP_SEQUENCE:
                next_bc = tmpdir / f"{producer_name}_{cleanup_name}.bc"
                run_opt_pass(current, next_bc, cleanup_pass)
                current = next_bc
        shutil.copyfile(current, ref_bc)
        ref_exe = tmpdir / f"{spec.name}_stock.out"
        compile_bc_to_exe(ref_bc, ref_exe)
        _, validation_output = execute_binary(ref_exe, spec.validation_input)
        ref_json.write_text(json.dumps({"validation_output": validation_output}, indent=2))
    return ref_bc, json.loads(ref_json.read_text())["validation_output"]


def decide_typed_action(producer: str, delta: dict[str, int], dirty_funcs: list[str], cleanup: str) -> tuple[str, str]:
    if not dirty_funcs:
        return "skip", "preserved"
    if producer == "sroa":
        inst_state = "unknown" if abs(delta["instruction_count"]) > 18 else (
            "run" if any(delta[k] != 0 for k in ["instruction_count", "alloca_count", "phi_count"]) else "skip"
        )
        cfg_state = "unknown" if delta["loop_count"] != 0 or len(dirty_funcs) > 2 else (
            "run" if any(delta[k] != 0 for k in ["branch_count", "basic_block_count"]) else "skip"
        )
    elif producer == "gvn":
        inst_state = "unknown" if abs(delta["call_count"]) > 1 else (
            "run" if any(delta[k] != 0 for k in ["instruction_count", "phi_count", "call_count"]) else "skip"
        )
        cfg_state = "unknown" if abs(delta["basic_block_count"]) > 2 else (
            "run" if any(delta[k] != 0 for k in ["branch_count", "basic_block_count"]) else "skip"
        )
    elif producer == "licm":
        inst_state = "unknown" if delta["loop_count"] != 0 else ("run" if delta["instruction_count"] != 0 else "skip")
        cfg_state = "unknown" if delta["loop_count"] != 0 or abs(delta["phi_count"]) > 1 else (
            "run" if any(delta[k] != 0 for k in ["branch_count", "basic_block_count", "phi_count"]) else "skip"
        )
    else:
        inst_state = "unknown" if delta["loop_count"] != 0 else ("run" if abs(delta["instruction_count"]) > 2 else "skip")
        cfg_state = "unknown" if delta["loop_count"] != 0 or len(dirty_funcs) > 1 else (
            "run" if any(delta[k] != 0 for k in ["branch_count", "basic_block_count"]) else "skip"
        )
    state = inst_state if cleanup == "instcombine" else cfg_state
    reason = {
        "skip": "preserved",
        "run": "dirty(function-set)",
        "unknown": "unknown",
    }[state]
    return ("skip" if state == "skip" else "run"), reason


def geometric_mean(values: list[float]) -> float:
    safe = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in safe) / len(safe))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: list[float], seed: int = 0, samples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boots = []
    for _ in range(samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def config_output_dir(config: str) -> Path:
    return EXP_DIR / config


def clear_measurement_artifacts(out_dir: Path) -> None:
    for pattern in ("raw_results.jsonl", "regressions_results.jsonl", "regressions_summary.json"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)
    logs_dir = out_dir / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("*.jsonl"):
            path.unlink(missing_ok=True)


def skipped_config_reason(config: str) -> str:
    if config == "no_local_execution":
        return (
            "The proxy harness never implements local execution, so a `no_local_execution` ablation would be "
            "behaviorally identical to `typed_skip_global`. This condition is marked infeasible in this rerun "
            "instead of reporting a duplicate result."
        )
    raise KeyError(config)


def run_opt_pass(input_bc: Path, output_bc: Path, passes: str) -> float:
    start = time.perf_counter()
    run_cmd(["opt", "-passes=" + passes, str(input_bc), "-o", str(output_bc)])
    return (time.perf_counter() - start) * 1000.0


def verify_bc(path: Path) -> bool:
    try:
        run_cmd(["opt", "-passes=verify", "-disable-output", str(path)])
        return True
    except subprocess.CalledProcessError:
        return False


def execute_schedule(spec: WorkloadSpec, config: str, seed: int, repeat: int, warmup: bool) -> dict[str, Any]:
    rng = random.Random((hash((spec.name, config, seed, repeat, warmup)) & 0xFFFFFFFF))
    stock_baseline_bc, stock_validation_out = ensure_stock_reference(spec)
    with tempfile.TemporaryDirectory(prefix=f"{config}_{spec.name}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        current_bc = (BENCH_DIR if spec.corpus == "benchmark" else REG_DIR) / f"{spec.name}.bc"
        current_stats = parse_ir_stats(current_bc)
        last_cleanup_hash = {cleanup: current_stats["module_hash"] for cleanup, _ in CLEANUP_SEQUENCE}
        compile_time_ms = 0.0
        cleanup_time_ms = 0.0
        producer_events: list[dict[str, Any]] = []
        pass_rows: list[dict[str, Any]] = []
        skip_count = 0
        fallback_count = 0
        unknown_events = 0
        total_cert_checks = 0
        bookkeeping_us = 0.0
        insertion_index = 0
        for producer_name, producer_pass in PRODUCER_SEQUENCE:
            after_bc = tmpdir / f"after_{producer_name}.bc"
            producer_ms = run_opt_pass(current_bc, after_bc, producer_pass)
            compile_time_ms += producer_ms
            before_stats = current_stats
            after_stats = parse_ir_stats(after_bc)
            dirty_funcs = changed_functions(before_stats, after_stats)
            delta = diff_metrics(before_stats, after_stats)
            current_bc = after_bc
            current_stats = after_stats
            pass_rows.append(
                {
                    "pass_name": producer_name,
                    "pass_kind": "producer",
                    "pass_time_ms": producer_ms,
                    "changed_functions": len(dirty_funcs),
                    "module_hash": current_stats["module_hash"],
                }
            )
            producer_events.append(
                {
                    "producer": producer_name,
                    "dirty_functions": dirty_funcs,
                    "delta": delta,
                }
            )
            insertion_index += 1
            for cleanup_name, cleanup_pass in CLEANUP_SEQUENCE:
                book_start = time.perf_counter()
                fallback_reason = ""
                action = "run"
                reason = "baseline"
                if config == "stock":
                    action = "run"
                elif config == "throttle":
                    action = "run" if insertion_index % 2 == 1 else "skip"
                    reason = "throttle_run" if action == "run" else "throttle_skip"
                elif config == "last_run":
                    compatible_change = (
                        dirty_funcs
                        and abs(delta["instruction_count"]) <= 1
                        and abs(delta["branch_count"]) == 0
                        and abs(delta["basic_block_count"]) == 0
                        and delta["loop_count"] == 0
                    )
                    action = "skip" if current_stats["module_hash"] == last_cleanup_hash[cleanup_name] or compatible_change else "run"
                    reason = "last_run_skip" if action == "skip" else "last_run_changed"
                elif config == "dirty_function":
                    action = "run" if dirty_funcs else "skip"
                    reason = "coarse_dirty" if action == "run" else "coarse_preserved"
                elif config == "single_bit_state":
                    coarse_changed = any(delta[key] != 0 for key in ["instruction_count", "branch_count", "basic_block_count", "phi_count"])
                    action = "run" if coarse_changed else "skip"
                    reason = "single_bit_dirty" if action == "run" else "single_bit_preserved"
                elif config == "untyped_locality":
                    if not dirty_funcs:
                        action, reason = "skip", "preserved"
                    elif delta["loop_count"] != 0 or abs(delta["basic_block_count"]) > 2:
                        action, reason = "run", "unknown"
                    else:
                        action, reason = "run", "dirty(function-set)"
                elif config in {"typed_skip_global", "no_local_execution"}:
                    action, reason = decide_typed_action(producer_name, delta, dirty_funcs, cleanup_name)
                elif config == "all_unknown":
                    action = "run"
                    reason = "forced_unknown"
                else:
                    action = "run"
                bookkeeping_us += (time.perf_counter() - book_start) * 1_000_000.0
                total_cert_checks += 1
                if "unknown" in reason:
                    unknown_events += 1
                    fallback_count += 1
                    fallback_reason = reason
                if action == "skip":
                    skip_count += 1
                    pass_rows.append(
                        {
                            "pass_name": cleanup_name,
                            "pass_kind": "cleanup_skip",
                            "pass_time_ms": 0.0,
                            "changed_functions": 0,
                            "decision_reason": reason,
                            "module_hash": current_stats["module_hash"],
                        }
                    )
                    continue
                after_cleanup_bc = tmpdir / f"after_{producer_name}_{cleanup_name}_{rng.randint(0, 1_000_000)}.bc"
                cleanup_ms = run_opt_pass(current_bc, after_cleanup_bc, cleanup_pass)
                compile_time_ms += cleanup_ms
                cleanup_time_ms += cleanup_ms
                current_bc = after_cleanup_bc
                current_stats = parse_ir_stats(current_bc)
                last_cleanup_hash[cleanup_name] = current_stats["module_hash"]
                verifier_ok = verify_bc(current_bc)
                pass_rows.append(
                    {
                        "pass_name": cleanup_name,
                        "pass_kind": "cleanup_run",
                        "pass_time_ms": cleanup_ms,
                        "changed_functions": 0,
                        "decision_reason": reason,
                        "verifier_ok": verifier_ok,
                        "fallback_reason": fallback_reason,
                        "module_hash": current_stats["module_hash"],
                    }
                )
        final_bc = tmpdir / f"{spec.name}_{config}_final.bc"
        shutil.copyfile(current_bc, final_bc)
        verifier_ok = verify_bc(final_bc)
        final_exe = tmpdir / f"{spec.name}_{config}.out"
        stock_exe = tmpdir / f"{spec.name}_stock.out"
        compile_bc_to_exe(final_bc, final_exe)
        runtime_ms, runtime_out = execute_binary(final_exe, spec.runtime_input)
        validation_ms, validation_out = execute_binary(final_exe, spec.validation_input)
        diff_ok = validation_out == stock_validation_out
        return {
            "benchmark": spec.name,
            "corpus": spec.corpus,
            "config": config,
            "seed": seed,
            "repeat": repeat,
            "warmup": warmup,
            "compile_time_ms": compile_time_ms,
            "cleanup_time_ms": cleanup_time_ms,
            "skip_count": skip_count,
            "fallback_count": fallback_count,
            "certificate_coverage": 1.0 - (unknown_events / total_cert_checks if total_cert_checks else 0.0),
            "unknown_rate": unknown_events / total_cert_checks if total_cert_checks else 0.0,
            "bookkeeping_us": bookkeeping_us,
            "runtime_ms": runtime_ms,
            "validation_runtime_ms": validation_ms,
            "binary_size_bytes": final_exe.stat().st_size,
            "verifier_ok": verifier_ok,
            "diff_ok": diff_ok,
            "regression_ok": True,
            "pass_rows": pass_rows,
            "producer_events": producer_events,
            "validation_output": validation_out,
            "runtime_output": runtime_out,
        }


def materialize_env_report() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    uname = run_cmd(["uname", "-a"]).stdout.strip()
    lscpu = run_cmd(["bash", "-lc", "lscpu | sed -n '1,40p'"]).stdout
    free = run_cmd(["free", "-h"]).stdout
    clang_version = run_cmd(["clang", "--version"]).stdout
    opt_version = run_cmd(["opt", "--version"]).stdout
    report = f"""Experiment environment report
Workspace: {ROOT}
Effective resource budget enforced by harness: 2 CPU workers, no GPU
Observed host note: the host exposes more than 2 logical CPUs and more than 128 GB RAM, but this experiment follows the user-provided budget and caps parallel work at 2 workers.

Toolchain
--------
{clang_version}
{opt_version}

System
------
{uname}

CPU
---
{lscpu}

Memory
------
{free}

Deviation note
--------------
The Stage 1 plan requires a pinned LLVM revision with 2024 LastRunTrackingAnalysis and local scheduler instrumentation. This workspace does not include that source tree, so Stage 2 uses the installed LLVM 18.1.3 toolchain as a proxy and documents all unsupported plan steps explicitly.
"""
    (ARTIFACTS / "env_report.txt").write_text(report)


def materialize_positioning_docs() -> None:
    positioning_rows = [
        {
            "section": "archival",
            "method": "Waddle / Fritz",
            "unit_of_reasoning": "Always-canonical IR and local maintenance",
            "always_canonical_vs_scheduler_extension": "always-canonical",
            "local_repair_algorithm_contribution": "yes",
            "claimed_novelty_boundary": "Stronger than this work; conceptual precursor, not a scheduler extension",
            "temporal_evidence": "no",
            "typed_invariant_evidence": "yes",
            "locality_awareness": "yes",
            "local_rerun_support": "local maintenance",
            "fallback_behavior": "not the same fallback model",
            "scope": "IR-wide canonical maintenance",
            "execution_granularity": "local repair",
            "relationship_to_this_work": "Conceptual precursor; stronger than the proxy scheduler studied here",
        },
        {
            "section": "engineering",
            "method": "LLVM LastRunTrackingAnalysis",
            "unit_of_reasoning": "Pass rerun history",
            "always_canonical_vs_scheduler_extension": "scheduler extension",
            "local_repair_algorithm_contribution": "no",
            "claimed_novelty_boundary": "Direct baseline",
            "temporal_evidence": "yes",
            "typed_invariant_evidence": "no",
            "locality_awareness": "no",
            "local_rerun_support": "no",
            "fallback_behavior": "global rerun or skip",
            "scope": "Fixed cleanup-pass rerun suppression",
            "execution_granularity": "global pass run or skip",
            "relationship_to_this_work": "Target baseline; not reimplemented in-tree in this workspace",
        },
        {
            "section": "engineering",
            "method": "Proxy typed scheduler in this run",
            "unit_of_reasoning": "Producer deltas over typed cleanup-facing metrics",
            "always_canonical_vs_scheduler_extension": "scheduler extension",
            "local_repair_algorithm_contribution": "no",
            "claimed_novelty_boundary": "Proxy evaluation only",
            "temporal_evidence": "approximate",
            "typed_invariant_evidence": "yes",
            "locality_awareness": "limited to function-set detection; no local execution",
            "local_rerun_support": "no",
            "fallback_behavior": "global rerun on unknown or coarse dirtiness",
            "scope": "Fixed proxy pass slice over generated workloads",
            "execution_granularity": "global pass run or skip",
            "relationship_to_this_work": "This experiment attempt",
        },
    ]
    (ARTIFACTS / "positioning_note.md").write_text(
        "# Positioning Note\n\n"
        "Main claim for this execution attempt: a proxy typed invariant-aware scheduler layered over a fixed LLVM 18 pass slice can be evaluated honestly against coarse scheduling baselines on locally generated CPU-only workloads, but this is not the exact Stage 1 planned LLVM extension because the required LLVM revision and instrumentation are absent from the workspace.\n\n"
        "Retained comparison scope: `stock`, `last_run` proxy, `throttle`, `dirty_function` proxy, and `typed_skip_global` proxy on the fixed slice `sroa -> instcombine -> simplifycfg -> gvn -> instcombine -> simplifycfg -> licm -> loop-rotate -> instcombine -> simplifycfg`.\n\n"
        "Dropped from the main claim for feasibility: true `LastRunTrackingAnalysis`, function-local cleanup execution, and loop-local repair wrappers.\n"
    )
    write_csv(ARTIFACTS / "related_work_matrix.csv", positioning_rows)
    write_csv(
        ARTIFACTS / "table1_positioning.csv",
        [
            {
                "method": row["method"],
                "evidence_type": (
                    "always-canonical local maintenance"
                    if row["method"] == "Waddle / Fritz"
                    else "typed proxy evidence"
                    if row["typed_invariant_evidence"] == "yes"
                    else "temporal rerun history" if row["temporal_evidence"] == "yes" else "local maintenance"
                ),
                "scope": row["scope"],
                "execution_granularity": row["execution_granularity"],
                "relationship_to_this_work": row["relationship_to_this_work"],
            }
            for row in positioning_rows
        ],
    )
    (ARTIFACTS / "plan_revision.md").write_text(
        "# Stage 2 Plan Revision\n\n"
        "This attempt cannot execute the exact Stage 1 plan because the workspace does not contain the pinned LLVM source tree with the intended `LastRunTrackingAnalysis` revision, the required scheduler/local-wrapper instrumentation, or the pre-registered `cBench` and `llvm-test-suite` corpora.\n\n"
        "Revised execution scope for this rerun:\n"
        "- Use the installed Ubuntu LLVM 18.1.3 toolchain as a documented proxy.\n"
        "- Keep the fixed pass slice and the same seed schedule.\n"
        "- Treat `last_run`, `dirty_function`, `typed_skip_global`, `single_bit_state`, `all_unknown`, and `untyped_locality` as proxy mechanisms only.\n"
        "- Mark `typed_cert_function`, `typed_cert_loop`, and the `no_local_execution` ablation infeasible instead of reporting duplicated or fabricated results.\n"
        "- Archive command traces under each `exp/<name>/logs/` directory and expose benchmark-level uncertainty and tradeoff metrics in `results.json`.\n\n"
        "This revision addresses the self-review request to revise the methodology explicitly before rerunning when the planned LLVM implementation is unavailable.\n"
    )


def dataset_stats(benchmark_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in benchmark_manifest:
        if entry["corpus"] != "benchmark":
            continue
        spec = next(spec for spec in EXEC_SPECS if spec.name == entry["name"])
        baseline = execute_schedule(spec, "stock", 11, 0, warmup=True)
        producer_counts = {
            evt["producer"]: len(evt["dirty_functions"]) for evt in baseline["producer_events"]
        }
        rows.append(
            {
                "benchmark": entry["name"],
                "family": spec.family,
                "function_count": entry["function_count"],
                "loop_count": entry["loop_count"],
                "instruction_count": entry["instruction_count"],
                "stock_slice_time_ms": baseline["compile_time_ms"],
                "cleanup_pass_time_ms": baseline["cleanup_time_ms"],
                "cleanup_time_share": baseline["cleanup_time_ms"] / baseline["compile_time_ms"] if baseline["compile_time_ms"] else 0.0,
                "sroa_touched_functions": producer_counts.get("sroa", 0),
                "gvn_touched_functions": producer_counts.get("gvn", 0),
                "licm_touched_functions": producer_counts.get("licm", 0),
                "loop_rotate_touched_functions": producer_counts.get("loop_rotate", 0),
            }
        )
    return rows


def prepare_data(force: bool = False) -> None:
    required = [
        ARTIFACTS / "benchmarks_manifest.json",
        ARTIFACTS / "runtime_manifest.json",
        ARTIFACTS / "dataset_stats.csv",
        ARTIFACTS / "env_report.txt",
        ARTIFACTS / "positioning_note.md",
        ARTIFACTS / "related_work_matrix.csv",
        ARTIFACTS / "plan_revision.md",
        ARTIFACTS / "table1_positioning.csv",
        ARTIFACTS / "table2_benchmark_metadata.csv",
    ]
    if not force and all(path.exists() for path in required):
        return
    benchmark_manifest, runtime_manifest = ensure_workloads()
    materialize_env_report()
    materialize_positioning_docs()
    (ARTIFACTS / "benchmarks_manifest.json").write_text(json.dumps(benchmark_manifest, indent=2))
    (ARTIFACTS / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, indent=2))
    dataset_rows = dataset_stats(benchmark_manifest)
    write_csv(ARTIFACTS / "dataset_stats.csv", dataset_rows)
    regression_manifest = [entry for entry in benchmark_manifest if entry["corpus"] != "benchmark"]
    regression_rows = []
    if regression_manifest:
        regression_rows = [
            {
                "benchmark": f"{family}_regressions",
                "corpus": "regression_summary",
                "family": family,
                "function_count": statistics.mean(item["function_count"] for item in items),
                "loop_count": statistics.mean(item["loop_count"] for item in items),
                "instruction_count": statistics.mean(item["instruction_count"] for item in items),
                "stock_slice_time_ms": None,
                "cleanup_pass_time_ms": None,
                "cleanup_time_share": None,
                "sroa_touched_functions": None,
                "gvn_touched_functions": None,
                "licm_touched_functions": None,
                "loop_rotate_touched_functions": None,
                "n_tests": len(items),
            }
            for family, items in sorted(
                {
                    family: [entry for entry in regression_manifest if entry["corpus"] == family]
                    for family in sorted({entry["corpus"] for entry in regression_manifest})
                }.items()
            )
        ]
    write_csv(
        ARTIFACTS / "table2_benchmark_metadata.csv",
        [
            {**row, "corpus": "benchmark", "n_tests": None}
            for row in dataset_rows
        ] + regression_rows,
    )


def specs_for_mode(mode: str) -> list[WorkloadSpec]:
    if mode == "benchmarks":
        return EXEC_SPECS
    if mode == "regressions":
        return REGRESSION_SPECS
    return EXEC_SPECS + REGRESSION_SPECS


def run_experiment(config: str, mode: str, repeats: int) -> None:
    prepare_data(force=False)
    out_dir = config_output_dir(config)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if config in DECLARED_BUT_SKIPPED_CONFIGS:
        reason = skipped_config_reason(config)
        clear_measurement_artifacts(out_dir)
        (out_dir / "SKIPPED.md").write_text(f"# Skipped\n\n{reason}\n")
        (out_dir / "results.json").write_text(
            json.dumps(
                {
                    "experiment": config,
                    "status": "skipped",
                    "reason": reason,
                },
                indent=2,
            )
        )
        return
    global CURRENT_LOG_PATH
    CURRENT_LOG_PATH = logs_dir / f"{mode}_commands.jsonl"
    CURRENT_LOG_PATH.write_text("")
    append_log(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "run_start",
            "config": config,
            "mode": mode,
            "repeats": repeats,
            "seeds": SEEDS,
        }
    )
    specs = specs_for_mode(mode)
    warmups = []
    for spec in specs:
        warmups.append(execute_schedule(spec, config, SEEDS[0], 0, warmup=True))
    measured_rows = []
    for seed in SEEDS:
        ordered = list(specs)
        random.Random(seed).shuffle(ordered)
        for repeat in range(repeats):
            for spec in ordered:
                measured_rows.append(execute_schedule(spec, config, seed, repeat, warmup=False))
    raw_jsonl = out_dir / ("raw_results.jsonl" if mode == "benchmarks" else f"{mode}_results.jsonl")
    with raw_jsonl.open("w") as handle:
        for row in measured_rows:
            handle.write(json.dumps(row) + "\n")
    summary = summarize_config(config, measured_rows)
    summary_path = out_dir / ("results.json" if mode == "benchmarks" else f"{mode}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    append_log(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "run_end",
            "config": config,
            "mode": mode,
            "measured_rows": len(measured_rows),
        }
    )
    CURRENT_LOG_PATH = None


def summarize_config(config: str, measured_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compile_times = [row["compile_time_ms"] for row in measured_rows]
    cleanup_times = [row["cleanup_time_ms"] for row in measured_rows]
    runtime_values = [row["runtime_ms"] for row in measured_rows]
    binary_sizes = [row["binary_size_bytes"] for row in measured_rows]
    coverage = [row["certificate_coverage"] for row in measured_rows]
    unknowns = [row["unknown_rate"] for row in measured_rows]
    runtime_ratios = [row["runtime_ms"] for row in measured_rows]
    return {
        "experiment": config,
        "metrics": {
            "compile_time_ms": {"mean": statistics.mean(compile_times), "std": statistics.pstdev(compile_times)},
            "cleanup_time_ms": {"mean": statistics.mean(cleanup_times), "std": statistics.pstdev(cleanup_times)},
            "runtime_ms": {"mean": statistics.mean(runtime_values), "std": statistics.pstdev(runtime_values)},
            "binary_size_bytes": {"mean": statistics.mean(binary_sizes), "std": statistics.pstdev(binary_sizes)},
            "certificate_coverage": {"mean": statistics.mean(coverage), "std": statistics.pstdev(coverage)},
            "unknown_rate": {"mean": statistics.mean(unknowns), "std": statistics.pstdev(unknowns)},
        },
        "benchmarks": summarize_benchmarks(measured_rows),
        "config": {"name": config, "seeds": SEEDS},
        "runtime_minutes": sum(compile_times) / 1000.0 / 60.0,
    }


def summarize_benchmarks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bench in sorted({row["benchmark"] for row in rows}):
        bench_rows = [row for row in rows if row["benchmark"] == bench]
        compile_vals = [row["compile_time_ms"] for row in bench_rows]
        runtime_vals = [row["runtime_ms"] for row in bench_rows]
        size_vals = [row["binary_size_bytes"] for row in bench_rows]
        coverage_vals = [row["certificate_coverage"] for row in bench_rows]
        unknown_vals = [row["unknown_rate"] for row in bench_rows]
        ci_low, ci_high = bootstrap_ci(compile_vals, seed=17)
        out[bench] = {
            "compile_time_ms": {
                "mean": statistics.mean(compile_vals),
                "std": statistics.pstdev(compile_vals),
                "bootstrap_95_ci": [ci_low, ci_high],
            },
            "runtime_ms": {
                "mean": statistics.mean(runtime_vals),
                "std": statistics.pstdev(runtime_vals),
            },
            "binary_size_bytes": {
                "mean": statistics.mean(size_vals),
                "std": statistics.pstdev(size_vals),
            },
            "certificate_coverage": {
                "mean": statistics.mean(coverage_vals),
                "std": statistics.pstdev(coverage_vals),
            },
            "unknown_rate": {
                "mean": statistics.mean(unknown_vals),
                "std": statistics.pstdev(unknown_vals),
            },
        }
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_results() -> None:
    all_rows = []
    regression_rows = []
    for config in MAIN_CONFIGS + ABLATION_CONFIGS:
        all_rows.extend(read_jsonl(config_output_dir(config) / "raw_results.jsonl"))
        regression_rows.extend(read_jsonl(config_output_dir(config) / "regressions_results.jsonl"))
    if not all_rows:
        raise RuntimeError("No experiment results found to aggregate.")
    main_rows = [row for row in all_rows if row["config"] in MAIN_CONFIGS and row["corpus"] == "benchmark"]
    ablation_rows = [row for row in all_rows if row["config"] in ABLATION_CONFIGS and row["corpus"] == "benchmark"]
    write_csv(
        ARTIFACTS / "main_skip_global_long.csv",
        [
            {
                "benchmark": row["benchmark"],
                "config": row["config"],
                "seed": row["seed"],
                "repeat": row["repeat"],
                "compile_time_ms": row["compile_time_ms"],
                "pass_name": "slice_total",
                "pass_time_ms": row["cleanup_time_ms"],
                "skip_count": row["skip_count"],
                "fallback_reason": "unknown" if row["unknown_rate"] > 0 else "",
                "certificate_coverage": row["certificate_coverage"],
                "unknown_rate": row["unknown_rate"],
                "bookkeeping_us": row["bookkeeping_us"],
                "runtime_ms": row["runtime_ms"],
                "binary_size_bytes": row["binary_size_bytes"],
                "verifier_ok": row["verifier_ok"],
                "diff_ok": row["diff_ok"],
                "regression_ok": regression_status_for_config(regression_rows, row["config"]),
            }
            for row in main_rows
        ],
    )
    write_csv(
        ARTIFACTS / "ablations_long.csv",
        [
            {
                "benchmark": row["benchmark"],
                "config": row["config"],
                "seed": row["seed"],
                "repeat": row["repeat"],
                "compile_time_ms": row["compile_time_ms"],
                "cleanup_time_ms": row["cleanup_time_ms"],
                "bookkeeping_us": row["bookkeeping_us"],
                "certificate_coverage": row["certificate_coverage"],
                "unknown_rate": row["unknown_rate"],
                "fallback_count": row["fallback_count"],
                "verifier_ok": row["verifier_ok"],
                "diff_ok": row["diff_ok"],
                "regression_ok": regression_status_for_config(regression_rows, row["config"]),
            }
            for row in ablation_rows
        ],
    )
    producer_rows = []
    for row in all_rows:
        for evt in row["producer_events"]:
            producer_rows.append(
                {
                    "benchmark": row["benchmark"],
                    "config": row["config"],
                    "producer": evt["producer"],
                    "dirty_functions": len(evt["dirty_functions"]),
                    "delta_instruction_count": evt["delta"]["instruction_count"],
                    "delta_branch_count": evt["delta"]["branch_count"],
                    "delta_loop_count": evt["delta"]["loop_count"],
                }
            )
    write_csv(ARTIFACTS / "producer_coverage.csv", producer_rows)
    summary = summarize_root(all_rows, regression_rows)
    write_table2_with_regressions(regression_rows)
    generate_table3(summary)
    make_figures(all_rows)


def write_table2_with_regressions(regression_rows: list[dict[str, Any]]) -> None:
    manifest = json.loads((ARTIFACTS / "benchmarks_manifest.json").read_text())
    benchmark_rows = []
    for row in dataset_stats(manifest):
        benchmark_rows.append({**row, "corpus": "benchmark", "n_tests": None, "avg_regression_compile_time_ms": None})
    regression_manifest = [entry for entry in manifest if entry["corpus"] != "benchmark"]
    grouped_manifest = {
        family: [entry for entry in regression_manifest if entry["corpus"] == family]
        for family in sorted({entry["corpus"] for entry in regression_manifest})
    }
    grouped_rows = {
        family: [row for row in regression_rows if row["corpus"] == family]
        for family in grouped_manifest
    }
    regression_summary_rows = []
    for family, items in grouped_manifest.items():
        rows = grouped_rows.get(family, [])
        regression_summary_rows.append(
            {
                "benchmark": f"{family}_regressions",
                "corpus": "regression_summary",
                "family": family,
                "function_count": statistics.mean(item["function_count"] for item in items),
                "loop_count": statistics.mean(item["loop_count"] for item in items),
                "instruction_count": statistics.mean(item["instruction_count"] for item in items),
                "stock_slice_time_ms": None,
                "cleanup_pass_time_ms": None,
                "cleanup_time_share": None,
                "sroa_touched_functions": None,
                "gvn_touched_functions": None,
                "licm_touched_functions": None,
                "loop_rotate_touched_functions": None,
                "n_tests": len(items),
                "avg_regression_compile_time_ms": statistics.mean(row["compile_time_ms"] for row in rows) if rows else None,
            }
        )
    write_csv(ARTIFACTS / "table2_benchmark_metadata.csv", benchmark_rows + regression_summary_rows)


def regression_status_for_config(regression_rows: list[dict[str, Any]], config: str) -> bool | None:
    rows = [row for row in regression_rows if row["config"] == config]
    if not rows:
        return None
    return all(row["verifier_ok"] and row["diff_ok"] for row in rows)


def select_best_config(summary_configs: dict[str, Any], candidates: list[str]) -> str | None:
    available = [cfg for cfg in candidates if cfg in summary_configs]
    if not available:
        return None
    return min(available, key=lambda cfg: summary_configs[cfg]["compile_time_ms"]["mean"])


def summarize_root(all_rows: list[dict[str, Any]], regression_rows: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark_rows = [row for row in all_rows if row["corpus"] == "benchmark"]
    stock_by_key = {
        (row["benchmark"], row["seed"], row["repeat"]): row for row in benchmark_rows if row["config"] == "stock"
    }
    summary: dict[str, Any] = {
        "meta": {
            "seeds": SEEDS,
            "effective_max_workers": MAX_WORKERS,
            "deviation": "Proxy experiment on installed LLVM 18.1.3; true LastRunTrackingAnalysis and local cleanup execution unavailable.",
            "study_scope": "Proxy heuristic study over stock LLVM command-line passes, not an LLVM-integrated LastRunTrackingAnalysis extension.",
            "benchmark_provenance": "Locally generated C workloads compiled to LLVM bitcode under data/generated_src and data/{benchmarks,regressions}.",
        },
        "configs": {},
        "comparisons": {},
        "gate_decisions": {
            "typed_cert_function": {
                "status": "infeasible",
                "reason": "No instrumented LLVM pass-manager support for function-local cleanup wrappers in the workspace.",
            },
            "typed_cert_loop": {
                "status": "infeasible",
                "reason": "No instrumented LLVM loop-local wrapper support in the workspace.",
            },
        },
        "regressions": {},
    }
    for config in MAIN_CONFIGS + ABLATION_CONFIGS:
        config_rows = [row for row in benchmark_rows if row["config"] == config]
        per_seed_medians = []
        for seed in SEEDS:
            bench_medians = []
            for bench in sorted({row["benchmark"] for row in config_rows}):
                reps = [row["compile_time_ms"] for row in config_rows if row["benchmark"] == bench and row["seed"] == seed]
                if reps:
                    bench_medians.append(statistics.median(reps))
            if bench_medians:
                per_seed_medians.append(statistics.mean(bench_medians))
        compile_values = [row["compile_time_ms"] for row in config_rows]
        summary["configs"][config] = {
            "n_runs": len(config_rows),
            "compile_time_ms": {
                "mean": statistics.mean(compile_values),
                "std": statistics.pstdev(compile_values),
            },
            "cleanup_time_ms": {
                "mean": statistics.mean(row["cleanup_time_ms"] for row in config_rows),
                "std": statistics.pstdev(row["cleanup_time_ms"] for row in config_rows),
            },
            "skip_count": {
                "mean": statistics.mean(row["skip_count"] for row in config_rows),
                "std": statistics.pstdev(row["skip_count"] for row in config_rows),
            },
            "certificate_coverage": {
                "mean": statistics.mean(row["certificate_coverage"] for row in config_rows),
                "std": statistics.pstdev(row["certificate_coverage"] for row in config_rows),
            },
            "unknown_rate": {
                "mean": statistics.mean(row["unknown_rate"] for row in config_rows),
                "std": statistics.pstdev(row["unknown_rate"] for row in config_rows),
            },
            "per_seed_median_compile_time_ms": per_seed_medians,
            "runtime_ms": {
                "mean": statistics.mean(row["runtime_ms"] for row in config_rows),
                "std": statistics.pstdev(row["runtime_ms"] for row in config_rows),
            },
            "binary_size_bytes": {
                "mean": statistics.mean(row["binary_size_bytes"] for row in config_rows),
                "std": statistics.pstdev(row["binary_size_bytes"] for row in config_rows),
            },
            "benchmark_breakdown": summarize_benchmarks(config_rows),
        }
    summary["skipped_configs"] = {
        config: {"status": "skipped", "reason": skipped_config_reason(config)}
        for config in DECLARED_BUT_SKIPPED_CONFIGS
    }
    strongest_typed = select_best_config(
        summary["configs"],
        [cfg for cfg in summary["configs"] if cfg.startswith("typed_")],
    )
    best_overall = select_best_config(
        summary["configs"],
        list(summary["configs"].keys()),
    )
    summary["selected_methods"] = {
        "strongest_retained_typed_method": strongest_typed,
        "best_overall_measured_method": best_overall,
    }
    if strongest_typed is None:
        (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
        return summary
    for baseline in ["stock", "last_run", "dirty_function"]:
        paired_deltas = []
        for row in benchmark_rows:
            if row["config"] != strongest_typed:
                continue
            key = (row["benchmark"], row["seed"], row["repeat"])
            if key not in stock_by_key and baseline == "stock":
                continue
            baseline_row = next(
                (
                    cand
                    for cand in benchmark_rows
                    if cand["config"] == baseline
                    and cand["benchmark"] == row["benchmark"]
                    and cand["seed"] == row["seed"]
                    and cand["repeat"] == row["repeat"]
                ),
                None,
            )
            if baseline_row is not None:
                paired_deltas.append((baseline_row["compile_time_ms"] - row["compile_time_ms"]) / baseline_row["compile_time_ms"])
        if paired_deltas:
            stat, pvalue = stats.wilcoxon(paired_deltas, alternative="two-sided", zero_method="wilcox")
            ci_low, ci_high = bootstrap_ci(paired_deltas, seed=123)
            summary["comparisons"][f"{strongest_typed}_vs_{baseline}"] = {
                "mean_compile_time_reduction": statistics.mean(paired_deltas),
                "std_compile_time_reduction": statistics.pstdev(paired_deltas),
                "bootstrap_95_ci": [ci_low, ci_high],
                "wilcoxon_pvalue": float(pvalue),
            }
    stock_runtime_by_key = {
        (row["benchmark"], row["seed"], row["repeat"]): row["runtime_ms"]
        for row in benchmark_rows
        if row["config"] == "stock"
    }
    stock_size_by_key = {
        (row["benchmark"], row["seed"], row["repeat"]): row["binary_size_bytes"]
        for row in benchmark_rows
        if row["config"] == "stock"
    }
    tradeoff_rows = []
    for row in benchmark_rows:
        if row["config"] != strongest_typed:
            continue
        key = (row["benchmark"], row["seed"], row["repeat"])
        if key not in stock_runtime_by_key or key not in stock_size_by_key:
            continue
        tradeoff_rows.append(
            {
                "benchmark": row["benchmark"],
                "seed": row["seed"],
                "repeat": row["repeat"],
                "runtime_ratio_vs_stock": row["runtime_ms"] / stock_runtime_by_key[key],
                "binary_size_ratio_vs_stock": row["binary_size_bytes"] / stock_size_by_key[key],
                "certificate_coverage": row["certificate_coverage"],
                "unknown_rate": row["unknown_rate"],
            }
        )
    summary["tradeoffs"] = {
        f"{strongest_typed}_vs_stock": {
            "runtime_ratio_mean": statistics.mean(item["runtime_ratio_vs_stock"] for item in tradeoff_rows),
            "runtime_ratio_std": statistics.pstdev(item["runtime_ratio_vs_stock"] for item in tradeoff_rows),
            "binary_size_ratio_mean": statistics.mean(item["binary_size_ratio_vs_stock"] for item in tradeoff_rows),
            "binary_size_ratio_std": statistics.pstdev(item["binary_size_ratio_vs_stock"] for item in tradeoff_rows),
        }
    }
    typed_runtime_mean = statistics.mean(
        row["runtime_ms"] for row in benchmark_rows if row["config"] == strongest_typed
    )
    stock_runtime_mean = statistics.mean(
        row["runtime_ms"] for row in benchmark_rows if row["config"] == "stock"
    )
    compile_win = (
        summary["configs"][strongest_typed]["compile_time_ms"]["mean"]
        < summary["configs"]["stock"]["compile_time_ms"]["mean"] * 0.96
    )
    coverage_ok = summary["configs"][strongest_typed]["certificate_coverage"]["mean"] > 0.60
    runtime_ok = typed_runtime_mean <= stock_runtime_mean * 1.01
    regressions_ok = all(
        row["verifier_ok"] and row["diff_ok"]
        for row in regression_rows
        if row["config"] == strongest_typed
    )
    claim = "preliminary_negative_proxy_evidence"
    if compile_win and coverage_ok and regressions_ok and runtime_ok:
        claim = "proxy_supported"
    elif compile_win and coverage_ok and regressions_ok and not runtime_ok:
        claim = "preliminary_mixed_proxy_evidence"
    summary["final_claim_status"] = claim
    for config in sorted({row["config"] for row in regression_rows}):
        rows = [row for row in regression_rows if row["config"] == config]
        summary["regressions"][config] = {
            "n_runs": len(rows),
            "all_verifier_ok": all(row["verifier_ok"] for row in rows),
            "all_diff_ok": all(row["diff_ok"] for row in rows),
        }
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def generate_table3(summary: dict[str, Any]) -> None:
    rows = []
    stock_mean = summary["configs"]["stock"]["compile_time_ms"]["mean"]
    for config in MAIN_CONFIGS + ABLATION_CONFIGS:
        config_summary = summary["configs"].get(config)
        regression_summary = summary["regressions"].get(config, {})
        if not config_summary:
            continue
        rows.append(
            {
                "config": config,
                "status": "measured",
                "compile_time_ms_mean": config_summary["compile_time_ms"]["mean"],
                "compile_time_ms_std": config_summary["compile_time_ms"]["std"],
                "compile_time_reduction_vs_stock_pct": (stock_mean - config_summary["compile_time_ms"]["mean"]) / stock_mean * 100.0,
                "cleanup_time_ms_mean": config_summary["cleanup_time_ms"]["mean"],
                "certificate_coverage_mean": config_summary["certificate_coverage"]["mean"],
                "unknown_rate_mean": config_summary["unknown_rate"]["mean"],
                "regression_runs": regression_summary.get("n_runs"),
                "all_verifier_ok": regression_summary.get("all_verifier_ok"),
                "all_diff_ok": regression_summary.get("all_diff_ok"),
                "function_local_gate": summary["gate_decisions"]["typed_cert_function"]["status"] if config == "typed_skip_global" else None,
                "loop_local_gate": summary["gate_decisions"]["typed_cert_loop"]["status"] if config == "typed_skip_global" else None,
                "final_claim_status": summary["final_claim_status"] if config == summary["selected_methods"]["strongest_retained_typed_method"] else None,
                "reason": None,
            }
        )
    for config, info in summary.get("skipped_configs", {}).items():
        rows.append(
            {
                "config": config,
                "status": info["status"],
                "compile_time_ms_mean": None,
                "compile_time_ms_std": None,
                "compile_time_reduction_vs_stock_pct": None,
                "cleanup_time_ms_mean": None,
                "certificate_coverage_mean": None,
                "unknown_rate_mean": None,
                "regression_runs": None,
                "all_verifier_ok": None,
                "all_diff_ok": None,
                "function_local_gate": None,
                "loop_local_gate": None,
                "final_claim_status": None,
                "reason": info["reason"],
            }
        )
    for config, gate in summary.get("gate_decisions", {}).items():
        rows.append(
            {
                "config": config,
                "status": gate["status"],
                "compile_time_ms_mean": None,
                "compile_time_ms_std": None,
                "compile_time_reduction_vs_stock_pct": None,
                "cleanup_time_ms_mean": None,
                "certificate_coverage_mean": None,
                "unknown_rate_mean": None,
                "regression_runs": None,
                "all_verifier_ok": None,
                "all_diff_ok": None,
                "function_local_gate": gate["status"] if config == "typed_cert_function" else None,
                "loop_local_gate": gate["status"] if config == "typed_cert_loop" else None,
                "final_claim_status": None,
                "reason": gate["reason"],
            }
        )
    write_csv(ARTIFACTS / "table3_correctness_efficiency_summary.csv", rows)


def make_figures(all_rows: list[dict[str, Any]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_rows = [row for row in all_rows if row["corpus"] == "benchmark"]
    configs = [cfg for cfg in MAIN_CONFIGS if any(row["config"] == cfg for row in benchmark_rows)]
    benchmarks = sorted({row["benchmark"] for row in benchmark_rows})
    stock_means = {}
    stock_by_key = {}
    for row in benchmark_rows:
        if row["config"] == "stock":
            stock_means.setdefault(row["benchmark"], []).append(row["compile_time_ms"])
            stock_by_key[(row["benchmark"], row["seed"], row["repeat"])] = row["compile_time_ms"]
    stock_means = {bench: statistics.mean(vals) for bench, vals in stock_means.items()}
    data = {cfg: [] for cfg in configs}
    yerrs = {cfg: [] for cfg in configs}
    for cfg in configs:
        for bench in benchmarks:
            cfg_rows = [
                row for row in benchmark_rows if row["benchmark"] == bench and row["config"] == cfg
            ]
            cfg_mean = statistics.mean(row["compile_time_ms"] for row in cfg_rows)
            reductions = []
            for row in cfg_rows:
                key = (row["benchmark"], row["seed"], row["repeat"])
                if key in stock_by_key:
                    reductions.append((stock_by_key[key] - row["compile_time_ms"]) / stock_by_key[key] * 100.0)
            point = (stock_means[bench] - cfg_mean) / stock_means[bench] * 100.0
            data[cfg].append(point)
            if len(reductions) > 1:
                ci_low, ci_high = bootstrap_ci(reductions, seed=31)
                yerrs[cfg].append(max(point - ci_low, ci_high - point))
            else:
                yerrs[cfg].append(0.0)
        data[cfg].append(
            (1.0 - geometric_mean([
                statistics.mean(
                    row["compile_time_ms"] for row in benchmark_rows if row["benchmark"] == bench and row["config"] == cfg
                ) / stock_means[bench]
                for bench in benchmarks
            ])) * 100.0
        )
        yerrs[cfg].append(0.0)
    labels = benchmarks + ["gmean"]
    x = np.arange(len(labels))
    width = 0.14
    plt.figure(figsize=(16, 6))
    for idx, cfg in enumerate(configs):
        plt.bar(x + idx * width, data[cfg], width=width, label=cfg, yerr=yerrs[cfg], capsize=2)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x + width * (len(configs) - 1) / 2, labels, rotation=45, ha="right")
    plt.ylabel("Compile-time reduction vs stock (%)")
    plt.title("Figure 1: Proxy compile-time reduction on generated benchmarks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_compile_reduction.png")
    plt.savefig(FIG_DIR / "figure1_compile_reduction.pdf")
    plt.close()

    strongest_typed = select_best_config(
        {
            cfg: {
                "compile_time_ms": {
                    "mean": statistics.mean(row["compile_time_ms"] for row in benchmark_rows if row["config"] == cfg)
                }
            }
            for cfg in sorted({row["config"] for row in benchmark_rows})
        },
        [cfg for cfg in sorted({row["config"] for row in benchmark_rows}) if cfg.startswith("typed_")],
    )
    if strongest_typed is None:
        return
    strongest_rows = [row for row in benchmark_rows if row["config"] == strongest_typed]
    outcome_labels = ["skipped", "global_after_unknown", "global_other"]
    skipped = statistics.mean(row["skip_count"] for row in strongest_rows)
    unknown = statistics.mean(row["fallback_count"] for row in strongest_rows)
    global_other = 8.0 - skipped - unknown
    plt.figure(figsize=(7, 5))
    plt.bar([strongest_typed], [skipped], label="skipped")
    plt.bar([strongest_typed], [unknown], bottom=[skipped], label="global after unknown")
    plt.bar([strongest_typed], [global_other], bottom=[skipped + unknown], label="global other")
    plt.ylabel("Mean decisions per benchmark run")
    plt.title("Figure 2: Proxy cleanup outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_outcomes.png")
    plt.savefig(FIG_DIR / "figure2_outcomes.pdf")
    plt.close()

    coverage_by_bench = [
        statistics.mean(
            row["certificate_coverage"] for row in strongest_rows if row["benchmark"] == bench
        )
        for bench in benchmarks
    ]
    unknown_by_bench = [
        statistics.mean(row["unknown_rate"] for row in strongest_rows if row["benchmark"] == bench)
        for bench in benchmarks
    ]
    plt.figure(figsize=(14, 5))
    x = np.arange(len(benchmarks))
    plt.bar(x - 0.18, coverage_by_bench, width=0.35, label="certificate_coverage")
    plt.bar(x + 0.18, unknown_by_bench, width=0.35, label="unknown_rate")
    plt.xticks(x, benchmarks, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Figure 3: Proxy certificate quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_certificate_quality.png")
    plt.savefig(FIG_DIR / "figure3_certificate_quality.pdf")
    plt.close()


def write_skipped_docs() -> None:
    skipped = {
        "no_local_execution": skipped_config_reason("no_local_execution"),
        "typed_cert_function": (
            "True function-local cleanup execution requires LLVM pass-manager instrumentation and wrappers that are not present in this workspace. "
            "This attempt records the function-local stage as infeasible rather than fabricating a surrogate local-execution result."
        ),
        "typed_cert_loop": (
            "Loop-local repair wrappers for LoopSimplify/LCSSA are not implementable against the installed stock LLVM binaries alone. "
            "The loop-local gate is therefore marked infeasible for this experiment attempt."
        ),
    }
    for name, body in skipped.items():
        out_dir = EXP_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        clear_measurement_artifacts(out_dir)
        path = out_dir / "SKIPPED.md"
        path.write_text(f"# Skipped\n\n{body}\n")
        (out_dir / "results.json").write_text(
            json.dumps(
                {
                    "experiment": name,
                    "status": "skipped",
                    "reason": body,
                },
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--write-skipped", action="store_true")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True, choices=ALL_DECLARED_CONFIGS)
    run_parser.add_argument("--mode", choices=["benchmarks", "regressions", "all"], default="benchmarks")
    run_parser.add_argument("--repeats", type=int, required=True)

    sub.add_parser("aggregate")

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare_data(force=True)
        if args.write_skipped:
            write_skipped_docs()
        return
    if args.cmd == "run":
        run_experiment(args.config, args.mode, args.repeats)
        return
    if args.cmd == "aggregate":
        aggregate_results()
        return


if __name__ == "__main__":
    main()
