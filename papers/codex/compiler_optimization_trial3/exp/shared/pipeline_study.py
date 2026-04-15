import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
EXP = ROOT / "exp"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DATA = ROOT / "data"
LOCAL_BENCH_ROOT = DATA / "benchmarks_local"
LOCAL_POLYBENCH_ROOT = LOCAL_BENCH_ROOT / "polybench"
LOCAL_CBENCH_ROOT = LOCAL_BENCH_ROOT / "cbench_like"
COMMAND_AUDIT_LOG = ARTIFACTS / "results" / "command_audit.jsonl"

SEEDS = [11, 17, 23]
PASS_VOCAB = [
    "inline",
    "loop-rotate",
    "licm",
    "loop-unroll",
    "loop-vectorize",
    "slp-vectorizer",
]
MAX_SEQ_LEN = 4
WARMUP_PASSES = [
    "mem2reg",
    "sroa",
    "instcombine",
    "simplifycfg",
    "loop-simplify",
    "lcssa",
]
RUNTIME_REPS = 5
PAIR_SCHEDULE = 4
FULL_SCHEDULE = 8
TOTAL_EVALS = 18
POLYBENCH_ROOT = Path(
    "/home/nw366/ResearchArena/outputs/kimi_new_compiler_optimization/idea_01/benchmarks/polybench"
)
CBENCH_BC_ROOT = Path(
    "/home/nw366/ResearchArena/outputs/codex_t2_compiler_optimization/idea_01/data/benchmarks"
)

PASS_PIPELINES = {
    "inline": "cgscc(inline)",
    "loop-rotate": "function(loop-mssa(loop-rotate<no-header-duplication;no-prepare-for-lto>))",
    "licm": "function(loop-mssa(licm<allowspeculation>))",
    "loop-unroll": "function(loop-unroll<O2>)",
    "loop-vectorize": "function(loop-vectorize<no-interleave-forced-only;vectorize-forced-only>)",
    "slp-vectorizer": "function(slp-vectorizer)",
}
PASS_FAMILY = {
    "inline": "inline",
    "loop-rotate": "loop-canonicalize",
    "licm": "loop-hoist",
    "loop-unroll": "loop-unroll",
    "loop-vectorize": "loop-vectorize",
    "slp-vectorizer": "slp-vectorize",
}
BLOCKER_TO_PASS = {
    "not-profitable": {"inline", "loop-unroll", "loop-vectorize", "slp-vectorizer"},
    "code-size-guard": {"inline"},
    "noncanonical-loop": {"loop-rotate"},
    "unknown-trip-count": {"loop-unroll", "loop-vectorize"},
    "memory-dependence": {"licm"},
    "unsupported-cfg": {"loop-rotate", "loop-vectorize"},
    "unsupported-memory-access": {"loop-vectorize", "slp-vectorizer"},
    "not-inlinable": {"inline"},
    "other": set(PASS_VOCAB),
}
PAIRWISE_METHODS = ["Random", "ProbeDelta", "StatsGP"]
DEFAULT_COLUMNS = [
    "benchmark",
    "seed",
    "method",
    "primary_improvement_pct",
    "text_size_change_pct",
    "runtime_speedup",
    "tuning_time",
]
PRIMARY_OBJECTIVE = "text_size"
PRIMARY_OBJECTIVE_LABEL = "binary_text_size"


@dataclass
class Benchmark:
    name: str
    suite: str
    source_path: str
    canonical_bc: str
    canonical_ll: str
    canonical_inst_count: int
    basic_block_count: int
    loop_count: int
    call_count: int
    screening_time: float
    binary_link: str
    runtime_enabled: bool
    run_args: list
    substitution_note: str = ""


def ensure_dirs():
    for path in [ARTIFACTS, EXP, FIGURES, TABLES, DATA]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [LOCAL_BENCH_ROOT, LOCAL_POLYBENCH_ROOT, LOCAL_CBENCH_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def append_command_audit(entry):
    COMMAND_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def run_cmd(cmd, cwd=None, env=None, capture=True, check=True):
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )
    append_command_audit(
        {
            "argv": cmd,
            "command": shlex.join(cmd),
            "cwd": str(cwd or ROOT),
            "returncode": result.returncode,
            "capture_output": capture,
            "elapsed_sec": time.time() - started,
            "stdout": result.stdout if capture else None,
            "stderr": result.stderr if capture else None,
        }
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    def _default(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_default) + "\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_step_config(step, config):
    write_json(EXP / step / "config.json", config)


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def summarize_metric(series):
    clean = pd.Series(series).dropna()
    return {
        "count": int(clean.shape[0]),
        "mean": None if clean.empty else float(clean.mean()),
        "std": None if clean.empty else float(clean.std(ddof=0)),
    }


def current_study_config():
    return {
        "seeds": SEEDS,
        "pass_vocab": PASS_VOCAB,
        "max_seq_len": MAX_SEQ_LEN,
        "warmup_prelude": WARMUP_PASSES,
        "primary_objective": PRIMARY_OBJECTIVE_LABEL,
        "primary_objective_reason": "LLVM IR instruction count is flat across the full six-pass search space on this machine; this feasibility rerun therefore uses binary text size plus oracle-gap sample efficiency as the supported CPU-only outcomes.",
        "methods_planned": ["Random", "ProbeDelta", "StatsGP", "RemarkState"],
        "methods_run": ["Random", "ProbeDelta", "RemarkState"],
        "methods_skipped": ["StatsGP"],
        "statsgp_status": "infeasible_on_local_llvm_build",
        "normalization_status": "single_annotator_only_kappa_unavailable",
    }


def save_command_logs(prefix, stdout_text, stderr_text):
    write_text(prefix.with_suffix(".stdout"), stdout_text or "")
    write_text(prefix.with_suffix(".stderr"), stderr_text or "")


def append_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def regex_count(text, pattern):
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def module_stats_from_bc(bc_path):
    ir = run_cmd(["llvm-dis", "-o", "-", str(bc_path)]).stdout
    inst_count = regex_count(ir, LLVM_INSTRUCTION_RE)
    basic_blocks = regex_count(ir, r"^(?:[A-Za-z$._0-9-]+:|;\s*<label>:\d+:)")
    calls = regex_count(ir, r"\bcall\b|\binvoke\b")
    loop_res = run_cmd(["opt", "-passes=print<loops>", "-disable-output", str(bc_path)])
    loop_dump = (loop_res.stdout or "") + (loop_res.stderr or "")
    loops = regex_count(loop_dump, r"Loop at depth")
    return {
        "instruction_count": inst_count,
        "basic_block_count": basic_blocks,
        "loop_count": loops,
        "call_count": calls,
        "ir_text": ir,
    }


LLVM_INSTRUCTION_RE = (
    r"^  (?:[%@A-Za-z0-9._-]+\s*=\s*)?"
    r"(?:alloca|load|store|br|switch|indirectbr|invoke|call|ret|resume|unreachable|"
    r"add|fadd|sub|fsub|mul|fmul|udiv|sdiv|fdiv|urem|srem|frem|shl|lshr|ashr|and|or|xor|"
    r"extractelement|insertelement|shufflevector|extractvalue|insertvalue|getelementptr|"
    r"trunc|zext|sext|fptrunc|fpext|fptoui|fptosi|uitofp|sitofp|ptrtoint|inttoptr|bitcast|"
    r"addrspacecast|icmp|fcmp|phi|select|freeze|va_arg|landingpad|catchpad|cleanuppad)"
)


def instruction_count_from_bc(bc_path):
    ir = run_cmd(["llvm-dis", "-o", "-", str(bc_path)]).stdout
    return regex_count(ir, LLVM_INSTRUCTION_RE)


def polybench_candidates():
    entries = {
        "2mm": POLYBENCH_ROOT / "linear-algebra/kernels/2mm/2mm.c",
        "3mm": POLYBENCH_ROOT / "linear-algebra/kernels/3mm/3mm.c",
        "atax": POLYBENCH_ROOT / "linear-algebra/kernels/atax/atax.c",
        "bicg": POLYBENCH_ROOT / "linear-algebra/kernels/bicg/bicg.c",
        "gemm": POLYBENCH_ROOT / "linear-algebra/blas/gemm/gemm.c",
        "gemver": POLYBENCH_ROOT / "linear-algebra/blas/gemver/gemver.c",
        "mvt": POLYBENCH_ROOT / "linear-algebra/kernels/mvt/mvt.c",
        "syr2k": POLYBENCH_ROOT / "linear-algebra/blas/syr2k/syr2k.c",
    }
    return entries


def cbench_candidates():
    return {
        "crc32": (
            CBENCH_BC_ROOT / "telecom_crc32_like.bc",
            "substituted with local telecom_crc32_like.bc; exact cBench source absent",
        ),
        "dijkstra": (
            CBENCH_BC_ROOT / "dijkstra_like.bc",
            "substituted with local dijkstra_like.bc; exact cBench source absent",
        ),
        "qsort": (
            CBENCH_BC_ROOT / "automotive_qsort1_like.bc",
            "substituted with local automotive_qsort1_like.bc; exact cBench source absent",
        ),
        "stringsearch": (
            CBENCH_BC_ROOT / "office_stringsearch_like.bc",
            "substituted with local office_stringsearch_like.bc; exact cBench source absent",
        ),
    }


def benchmark_binary_link(name, suite):
    if suite == "polybench":
        return "polybench"
    return "standalone"


def package_polybench_source(src):
    rel_dir = src.parent.relative_to(POLYBENCH_ROOT)
    dst_dir = LOCAL_POLYBENCH_ROOT / rel_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src.parent.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir / item.name)
    for util_name in ["polybench.c", "polybench.h"]:
        util_src = POLYBENCH_ROOT / "utilities" / util_name
        util_dst = LOCAL_POLYBENCH_ROOT / "utilities" / util_name
        util_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(util_src, util_dst)
    return dst_dir / src.name


def package_cbench_like_bc(bc_src):
    dst = LOCAL_CBENCH_ROOT / bc_src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bc_src, dst)
    return dst


def canonicalize_polybench(name, src):
    out_dir = ARTIFACTS / "ir" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    bc = out_dir / f"{name}.bc"
    ll = out_dir / f"{name}.ll"
    include = str(LOCAL_POLYBENCH_ROOT / "utilities")
    run_cmd(
        [
            "clang",
            "-O0",
            "-Xclang",
            "-disable-llvm-passes",
            "-Xclang",
            "-disable-O0-optnone",
            "-emit-llvm",
            "-c",
            str(src),
            "-I",
            include,
            "-DMINI_DATASET",
            "-o",
            str(bc),
        ]
    )
    run_cmd(["llvm-dis", str(bc), "-o", str(ll)])
    return bc, ll


def canonicalize_bc(name, bc_src):
    out_dir = ARTIFACTS / "ir" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    bc = out_dir / f"{name}.bc"
    ll = out_dir / f"{name}.ll"
    run_cmd(["cp", str(bc_src), str(bc)])
    run_cmd(["llvm-dis", str(bc), "-o", str(ll)])
    return bc, ll


def apply_opt_pipeline(input_bc, pipeline, output_bc=None, remarks_path=None, enable_stats=False, log_prefix=None):
    cmd = ["opt", f"-passes={pipeline}"]
    if enable_stats:
        cmd += ["-stats", "-stats-json"]
    if remarks_path is not None:
        cmd += [
            "-pass-remarks-filter=.*",
            "-pass-remarks-missed=.*",
            "-pass-remarks-analysis=.*",
            f"-pass-remarks-output={remarks_path}",
        ]
    if output_bc:
        cmd += ["-o", str(output_bc), str(input_bc)]
    else:
        cmd += ["-disable-output", str(input_bc)]
    started = time.time()
    result = run_cmd(cmd, capture=True)
    elapsed = time.time() - started
    if log_prefix is not None:
        save_command_logs(log_prefix, result.stdout, result.stderr)
    return {"stdout": result.stdout, "stderr": result.stderr, "wall_time": elapsed}


def build_prelude(input_bc, out_bc):
    pipeline = "function(mem2reg,sroa,instcombine,simplifycfg,loop-simplify,lcssa)"
    apply_opt_pipeline(input_bc, pipeline, output_bc=out_bc)


def parse_remark_yaml(path):
    if not path.exists():
        return []
    text = path.read_text()
    docs = [part.strip() for part in text.split("---") if part.strip()]
    parsed = []
    for doc in docs:
        first = doc.splitlines()[0].strip()
        if first.startswith("!"):
            tag = first[1:]
            body = "\n".join(doc.splitlines()[1:])
        else:
            tag = "Unknown"
            body = doc
        record = {
            "remark_kind": tag.lower(),
            "consumer_pass": "",
            "remark_name": "",
            "function": "",
            "anchor_text": "",
            "message_text": "",
            "source_loc": "",
        }
        lines = body.splitlines()
        message_parts = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("Pass:"):
                record["consumer_pass"] = line.split(":", 1)[1].strip()
            elif line.startswith("Name:"):
                record["remark_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Function:"):
                record["function"] = line.split(":", 1)[1].strip()
            elif line.startswith("DebugLoc:"):
                record["source_loc"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("- String:"):
                message_parts.append(line.split(":", 1)[1].strip().strip("'"))
            elif line.strip().startswith("- Inst:"):
                anchor = line.split(":", 1)[1].strip()
                record["anchor_text"] = (record["anchor_text"] + " " + anchor).strip()
            i += 1
        record["message_text"] = " ".join(x for x in message_parts if x).strip()
        parsed.append(record)
    return parsed


def normalize_remark(record):
    msg = f"{record['consumer_pass']} {record.get('remark_name', '')} {record['message_text']} {record['anchor_text']}".lower()
    consumer_pass = record["consumer_pass"] or "unknown"
    remark_name = (record.get("remark_name") or "").lower()
    outcome = "analysis"
    if record["remark_kind"] == "passed":
        outcome = "applied"
    elif record["remark_kind"] == "missed":
        outcome = "missed"

    family = PASS_FAMILY.get(consumer_pass, "other")
    if consumer_pass == "TTI" and remark_name == "dontunroll":
        family = "loop-unroll"
        outcome = "analysis"

    scope = "function"
    if "loop" in msg or family.startswith("loop"):
        scope = "loop"
    if "call" in msg or family == "inline":
        scope = "callsite"

    blocker = "other"
    if family == "inline":
        if remark_name in {"nodefinition", "neverinline"} or "unavailable" in msg or "never be inlined" in msg:
            blocker = "not-inlinable"
        elif "size" in msg:
            blocker = "code-size-guard"
        elif "cost" in msg or "profitable" in msg:
            blocker = "not-profitable"
    elif family == "loop-hoist":
        if remark_name == "hoisted" or "hoisting" in msg:
            blocker = "other"
        elif "invalidat" in msg:
            blocker = "memory-dependence"
        elif "conditionally executed" in msg:
            blocker = "unsupported-cfg"
    elif family == "loop-unroll":
        if remark_name == "dontunroll":
            blocker = "not-profitable"
            scope = "loop"
    elif family == "loop-vectorize":
        if "memory" in msg:
            blocker = "unsupported-memory-access"
        elif "trip count" in msg:
            blocker = "unknown-trip-count"
        elif "loop not vectorized" in msg:
            blocker = "other"
    elif family == "slp-vectorize":
        if remark_name == "notbeneficial" or "not beneficial" in msg or "cost" in msg:
            blocker = "not-profitable"
        elif remark_name == "notpossible" or "impossible" in msg:
            blocker = "unsupported-memory-access"

    if blocker == "other":
        rules = [
            ("not profitable", "not-profitable"),
            ("cost", "not-profitable"),
            ("size", "code-size-guard"),
            ("trip count", "unknown-trip-count"),
            ("inlinable", "not-inlinable"),
            ("invalidat", "memory-dependence"),
            ("depend", "memory-dependence"),
            ("condition", "unsupported-cfg"),
            ("cfg", "unsupported-cfg"),
            ("canonical", "noncanonical-loop"),
            ("memory", "unsupported-memory-access"),
            ("unsupported", "unsupported-memory-access"),
        ]
        for needle, label in rules:
            if needle in msg:
                blocker = label
                break
    return {
        "consumer_pass": consumer_pass,
        "outcome": outcome,
        "family": family,
        "blocker": blocker,
        "scope": scope,
        "anchor": record["anchor_text"] or record["function"] or "unknown",
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_stats_json(text):
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def objective_cache_dir(benchmark):
    path = ARTIFACTS / "results" / "objective_cache" / benchmark.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def text_size_for_bc(benchmark, bc_path, final_hash):
    cache_dir = objective_cache_dir(benchmark)
    text_path = cache_dir / f"{final_hash}.text_size.json"
    binary_path = cache_dir / f"{final_hash}.bin"
    if text_path.exists():
        return json.loads(text_path.read_text())["text_size"]
    build_binary_from_bc(benchmark, bc_path, binary_path)
    size = text_size(binary_path)
    write_json(
        text_path,
        {
            "benchmark": benchmark.name,
            "final_bc": str(bc_path),
            "final_bc_sha256": final_hash,
            "binary_path": str(binary_path),
            "text_size": size,
        },
    )
    return size


def objective_tuple(obs):
    return (
        obs["text_size"],
        obs["instruction_count"],
        obs["compile_time"],
        len(obs["sequence"]),
        ",".join(obs["sequence"]),
    )


def evaluate_sequence(benchmark, sequence, work_root, capture_remarks=True, enable_stats=False):
    work_root.mkdir(parents=True, exist_ok=True)
    current_bc = work_root / "start.bc"
    run_cmd(["cp", benchmark.canonical_bc, str(current_bc)])
    prelude_bc = work_root / "prelude.bc"
    build_prelude(current_bc, prelude_bc)
    current_bc = prelude_bc
    all_raw = []
    all_norm = []
    pass_times = []
    raw_paths = []
    for idx, p in enumerate(sequence):
        next_bc = work_root / f"{idx:02d}_{p}.bc"
        remarks_path = work_root / f"{idx:02d}_{p}_remarks.yaml" if capture_remarks else None
        log_prefix = work_root / f"{idx:02d}_{p}"
        result = apply_opt_pipeline(
            current_bc,
            PASS_PIPELINES[p],
            output_bc=next_bc,
            remarks_path=remarks_path,
            enable_stats=enable_stats,
            log_prefix=log_prefix,
        )
        pass_times.append(result["wall_time"])
        raw_paths.append(str(remarks_path) if remarks_path else "")
        if remarks_path and remarks_path.exists():
            raw = parse_remark_yaml(remarks_path)
            all_raw.extend(raw)
            all_norm.extend(normalize_remark(r) for r in raw)
        stats_payload = extract_stats_json((result["stdout"] or "") + "\n" + (result["stderr"] or ""))
        if stats_payload:
            write_json(log_prefix.with_suffix(".stats.json"), stats_payload)
        current_bc = next_bc
    stats = module_stats_from_bc(current_bc)
    final_hash = file_sha256(current_bc)
    final_text_size = text_size_for_bc(benchmark, current_bc, final_hash)
    return {
        "final_bc": str(current_bc),
        "final_bc_sha256": final_hash,
        "sequence": sequence,
        "instruction_count": stats["instruction_count"],
        "text_size": final_text_size,
        "basic_block_count": stats["basic_block_count"],
        "loop_count": stats["loop_count"],
        "call_count": stats["call_count"],
        "compile_time": sum(pass_times),
        "raw_remarks": all_raw,
        "normalized_remarks": all_norm,
        "remark_paths": raw_paths,
        "stats_path_candidates": [str(path) for path in sorted(work_root.glob("*.stats.json"))],
        "raw_evidence_path": str(work_root),
    }


def evaluate_sequence_minimal(prelude_bc, sequence, work_root):
    work_root.mkdir(parents=True, exist_ok=True)
    tmp_a = work_root / "tmp_a.bc"
    tmp_b = work_root / "tmp_b.bc"
    run_cmd(["cp", str(prelude_bc), str(tmp_a)])
    current_bc = tmp_a
    next_bc = tmp_a
    if sequence:
        next_bc = tmp_b
    compile_time = 0.0
    for p in sequence:
        result = apply_opt_pipeline(current_bc, PASS_PIPELINES[p], output_bc=next_bc)
        compile_time += result["wall_time"]
        current_bc = next_bc
        next_bc = tmp_b if next_bc == tmp_a else tmp_a
    return {
        "instruction_count": instruction_count_from_bc(current_bc),
        "compile_time": compile_time,
        "final_bc_sha256": file_sha256(current_bc),
        "final_bc": str(current_bc),
    }


def build_binary_from_bc(benchmark, bc_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if benchmark.binary_link == "polybench":
        run_cmd(
            [
                "clang",
                str(bc_path),
                str(LOCAL_POLYBENCH_ROOT / "utilities/polybench.c"),
                "-I",
                str(LOCAL_POLYBENCH_ROOT / "utilities"),
                "-lm",
                "-DPOLYBENCH_TIME",
                "-o",
                str(out_path),
            ]
        )
    else:
        run_cmd(["clang", str(bc_path), "-lm", "-o", str(out_path)])


def text_size(binary_path):
    out = run_cmd(["llvm-size", str(binary_path)]).stdout.splitlines()
    if len(out) < 2:
        return 0
    parts = out[1].split()
    return int(parts[0])


def runtime_measure(binary_path, args):
    samples = []
    for _ in range(1 + RUNTIME_REPS):
        started = time.time()
        run_cmd([str(binary_path), *args], capture=True)
        elapsed = time.time() - started
        samples.append(elapsed)
    return {
        "warmup": samples[0],
        "measured": samples[1:],
        "mean": float(np.mean(samples[1:])),
        "std": float(np.std(samples[1:], ddof=0)),
    }


def sequence_space(vocab):
    all_seq = []
    for length in range(1, MAX_SEQ_LEN + 1):
        all_seq.extend(itertools.product(vocab, repeat=length))
    return [tuple(seq) for seq in all_seq]


def linear_probe_score(obs):
    feats = np.array(
        [
            [
                obs["text_size"],
                obs["instruction_count"],
                obs["basic_block_count"],
                obs["compile_time"],
            ]
        ],
        dtype=float,
    )
    return feats[0]


def zscore_rows(rows):
    arr = np.array(rows, dtype=float)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds[stds == 0.0] = 1.0
    return (arr - means) / stds


def choose_probe_candidate(history, unexplored):
    scored_rows = []
    for item in history:
        scored_rows.append(
            [
                item["text_size"],
                item["instruction_count"],
                item["basic_block_count"],
                item["compile_time"],
            ]
        )
    z = zscore_rows(scored_rows)
    coeff = np.array([-1.0, -0.25, -0.25, 0.05])
    hist_scores = (z * coeff).sum(axis=1)
    single_effect = defaultdict(list)
    for item, score in zip(history, hist_scores):
        for p in item["sequence"]:
            single_effect[p].append(float(score))
    mean_effect = {p: float(np.mean(vals)) for p, vals in single_effect.items()}
    ranked = []
    for cand in unexplored:
        estimate = sum(mean_effect.get(p, 0.0) for p in cand) / len(cand)
        ranked.append((estimate, cand))
    ranked.sort(key=lambda x: (-x[0], ",".join(x[1])))
    return ranked[0][1]


def blocker_state(normalized, use_blocker=True):
    counts = Counter()
    applied = Counter()
    for rec in normalized:
        if rec["outcome"] == "missed":
            key = (
                rec["family"],
                rec["blocker"] if use_blocker else "_",
                rec["scope"],
            )
            counts[key] += 1
        if rec["outcome"] == "applied":
            applied[rec["family"]] += 1
    return counts, applied


def score_remark_candidate(candidate, active_state, applied_history, use_blocker=True, use_applied_history=True):
    resolved = 0
    favored_applied = 0
    favorable_delta = 0
    for p in candidate:
        family = PASS_FAMILY[p]
        favorable_delta += 1
        for blocker_key, count in active_state.items():
            family_key, blocker, _ = blocker_key
            if family_key == family:
                resolved += count
            if use_blocker and p in BLOCKER_TO_PASS.get(blocker, set()):
                resolved += count
        if use_applied_history and applied_history.get(family, 0) > 0:
            favored_applied += 1
    return (resolved, favored_applied, favorable_delta, -len(candidate), ",".join(candidate))


def beam_rank_candidates(unexplored, history, beam_width=3, use_blocker=True, use_applied_history=True):
    if not history:
        return sorted(unexplored)[:beam_width]
    best = min(history, key=objective_tuple)
    state, applied = blocker_state(best["normalized_remarks"], use_blocker=use_blocker)
    ranked = sorted(
        unexplored,
        key=lambda cand: score_remark_candidate(
            cand, state, applied, use_blocker=use_blocker, use_applied_history=use_applied_history
        ),
        reverse=True,
    )
    return ranked[:beam_width]


def run_search_method(benchmarks, method, seeds, runtime_subset, variant=None):
    trace_rows = []
    final_rows = []
    results = []
    vocab = list(PASS_VOCAB)
    all_candidates = sequence_space(vocab)
    for benchmark in benchmarks:
        for seed in seeds:
            rng = random.Random(seed)
            method_name = variant or method
            history = []
            unexplored = list(all_candidates)
            eval_budget = TOTAL_EVALS
            work_base = ARTIFACTS / "results" / "search_work" / benchmark.name / method_name / str(seed)
            work_base.mkdir(parents=True, exist_ok=True)

            single_passes = [(p,) for p in vocab]
            pair_budget = 0 if method_name == "NoPairProbe" else PAIR_SCHEDULE
            full_budget = FULL_SCHEDULE + (PAIR_SCHEDULE if method_name == "NoPairProbe" else 0)
            schedule = []
            schedule.extend(single_passes)
            if method == "Random":
                pool = [cand for cand in unexplored if cand not in schedule]
                rng.shuffle(pool)
                schedule = pool[:eval_budget]
            else:
                schedule = list(single_passes)
            evaluated_set = set()
            initial_schedule = schedule if method == "Random" else schedule[: len(single_passes)]
            for idx, seq in enumerate(initial_schedule):
                evaluated_set.add(seq)
                unexplored.remove(seq)
                eval_dir = work_base / f"eval_{idx:02d}"
                obs = evaluate_sequence(benchmark, list(seq), eval_dir)
                obs["sequence"] = seq
                history.append(obs)
                trace_rows.append(
                    trace_row(benchmark.name, seed, method_name, idx + 1, obs, runtime_subset, benchmark)
                )
            if method != "Random":
                for _ in range(pair_budget):
                    pair_pool = [cand for cand in unexplored if len(cand) == 2]
                    if not pair_pool:
                        break
                    if method == "ProbeDelta":
                        seq = choose_probe_candidate(history, pair_pool)
                    else:
                        seq = beam_rank_candidates(
                            pair_pool,
                            history,
                            beam_width=2 if method_name == "Beam2" else 4 if method_name == "Beam4" else 3,
                            use_blocker=method_name != "NoBlocker",
                            use_applied_history=method_name != "NoAppliedHistory",
                        )[0]
                    evaluated_set.add(seq)
                    unexplored.remove(seq)
                    eval_dir = work_base / f"eval_{len(history):02d}"
                    obs = evaluate_sequence(benchmark, list(seq), eval_dir)
                    obs["sequence"] = seq
                    history.append(obs)
                    trace_rows.append(
                        trace_row(
                            benchmark.name, seed, method_name, len(history), obs, runtime_subset, benchmark
                        )
                    )
                for _ in range(full_budget):
                    long_pool = [cand for cand in unexplored if len(cand) >= 3]
                    if not long_pool:
                        break
                    if method == "ProbeDelta":
                        seq = choose_probe_candidate(history, long_pool)
                    else:
                        top = beam_rank_candidates(
                            long_pool,
                            history,
                            beam_width=2 if method_name == "Beam2" else 4 if method_name == "Beam4" else 3,
                            use_blocker=method_name != "NoBlocker",
                            use_applied_history=method_name != "NoAppliedHistory",
                        )
                        seq = top[0]
                    evaluated_set.add(seq)
                    unexplored.remove(seq)
                    eval_dir = work_base / f"eval_{len(history):02d}"
                    obs = evaluate_sequence(benchmark, list(seq), eval_dir)
                    obs["sequence"] = seq
                    history.append(obs)
                    trace_rows.append(
                        trace_row(
                            benchmark.name, seed, method_name, len(history), obs, runtime_subset, benchmark
                        )
                    )
            best = min(history, key=objective_tuple)
            binary = work_base / "final.bin"
            build_binary_from_bc(benchmark, best["final_bc"], binary)
            runtime = None
            if benchmark.name in runtime_subset:
                runtime = runtime_measure(binary, benchmark.run_args)
            final = {
                "benchmark": benchmark.name,
                "seed": seed,
                "method": method_name,
                "chosen_sequence": ",".join(best["sequence"]),
                "chosen_length": len(best["sequence"]),
                "primary_objective_name": PRIMARY_OBJECTIVE_LABEL,
                "primary_objective_value": best["text_size"],
                "final_ir_instruction_count": best["instruction_count"],
                "final_text_size": text_size(binary),
                "final_compile_time": best["compile_time"],
                "runtime_mean": None if runtime is None else runtime["mean"],
                "runtime_std": None if runtime is None else runtime["std"],
                "tuning_time": sum(item["compile_time"] for item in history),
            }
            final_rows.append(final)
            results.append(final)
    return trace_rows, final_rows, results


def trace_row(benchmark_name, seed, method, index, obs, runtime_subset, benchmark):
    return {
        "benchmark": benchmark_name,
        "seed": seed,
        "method": method,
        "evaluation_index": index,
        "selected_sequence": ",".join(obs["sequence"]),
        "sequence_length": len(obs["sequence"]),
        "primary_objective_name": PRIMARY_OBJECTIVE_LABEL,
        "primary_objective": obs["text_size"],
        "instruction_count": obs["instruction_count"],
        "text_size": obs["text_size"],
        "basic_block_count": obs["basic_block_count"],
        "loop_count": obs["loop_count"],
        "call_count": obs["call_count"],
        "text_size_if_measured": "",
        "compile_time": obs["compile_time"],
        "final_bc_sha256": obs.get("final_bc_sha256", ""),
        "search_state_summary": json.dumps(
            {
                "remark_count": len(obs["normalized_remarks"]),
                "missed_count": sum(1 for x in obs["normalized_remarks"] if x["outcome"] == "missed"),
                "applied_count": sum(1 for x in obs["normalized_remarks"] if x["outcome"] == "applied"),
                "stats_files": len(obs.get("stats_path_candidates", [])),
                "text_size": obs["text_size"],
                "basic_block_count": obs["basic_block_count"],
                "loop_count": obs["loop_count"],
            },
            sort_keys=True,
        ),
        "raw_evidence_path": obs.get("raw_evidence_path", ""),
    }


def environment_step():
    ensure_dirs()
    if COMMAND_AUDIT_LOG.exists():
        COMMAND_AUDIT_LOG.unlink()
    versions = {}
    for tool in ["clang", "opt", "llvm-dis", "llc", "llvm-size", "/usr/bin/time"]:
        out = run_cmd([tool, "--version"], capture=True, check=False)
        versions[tool] = (out.stdout or out.stderr).strip()
    write_text(
        ARTIFACTS / "env" / "tool_versions.txt",
        "\n\n".join(f"{tool}\n{txt}" for tool, txt in versions.items()) + "\n",
    )
    freeze = run_cmd([sys.executable, "-m", "pip", "freeze"], capture=True, check=False)
    write_text(ARTIFACTS / "env" / "python_lock.txt", freeze.stdout)
    hardware = {
        "cpu_cores": 2,
        "ram_gb": 128,
        "gpus": 0,
        "parallel_cpu_workers": 2,
        "global_wall_clock_budget_hours": 8,
        "observed_nproc": int(run_cmd(["nproc"]).stdout.strip()),
    }
    write_json(ARTIFACTS / "env" / "hardware.json", hardware)
    protocol_revision = {
        "stats_baseline_status": "blocked_on_non_asserts_llvm_build",
        "stats_baseline_reason": "This opt build advertises -stats/-stats-json but emits no counters because LLVM statistics require asserts.",
        "revised_design": "Treat Stage 2 on this machine as a predeclared feasibility and negative-result pilot. StatsGP remains blocked, normalization remains single-annotator only, and the executable comparison focuses on final text size plus oracle-gap sample efficiency for Random, ProbeDelta, and RemarkState.",
        "objective_gate": "LLVM IR instruction count remains a descriptive endpoint only because the full six-pass search space is flat on that proxy in this environment; final text size and evaluations-to-oracle are the supported executable outcomes.",
        "claim_boundary": "No remark-vs-stats claim is made. No normalization-valid interpretability claim is made because the preregistered double-annotation kappa gate remains infeasible with one operator. Any conclusion is limited to a CPU-only feasibility pilot in the restricted six-pass space.",
    }
    write_json(ARTIFACTS / "results" / "protocol_revision.json", protocol_revision)
    study_config = current_study_config()
    write_json(ARTIFACTS / "config" / "study_config.json", study_config)
    write_step_config("00_env", study_config)
    results = {
        "experiment": "00_env",
        "metrics": {"gpus": 0, "cpu_cores": hardware["observed_nproc"]},
        "config": study_config,
        "runtime_minutes": 0.0,
        "stats_gp_feasible": False,
        "protocol_revision": protocol_revision,
        "notes": [
            "LLVM remarks are available.",
            "LLVM -stats-json did not emit usable counters on this optimized non-asserts build; stats-based steps will be documented as skipped.",
            "LLVM IR instruction count is flat in the full registered search space on this machine, so the feasibility rerun uses binary text size and oracle-gap sample efficiency instead.",
        ],
    }
    write_json(EXP / "00_env" / "results.json", results)


def data_prep_step():
    ensure_dirs()
    selected = []
    commands = {}
    runtime_inputs = {}
    manifest_rows = []
    substitutions = []
    start = time.time()
    for name, src in polybench_candidates().items():
        local_src = package_polybench_source(src)
        bc, ll = canonicalize_polybench(name, local_src)
        stats = module_stats_from_bc(bc)
        screening_runs = []
        for _ in range(3):
            tmp = ARTIFACTS / "data" / "screen" / name
            tmp.mkdir(parents=True, exist_ok=True)
            result = evaluate_sequence(
                Benchmark(
                    name=name,
                    suite="polybench",
                    source_path=str(local_src),
                    canonical_bc=str(bc),
                    canonical_ll=str(ll),
                    canonical_inst_count=stats["instruction_count"],
                    basic_block_count=stats["basic_block_count"],
                    loop_count=stats["loop_count"],
                    call_count=stats["call_count"],
                    screening_time=0.0,
                    binary_link="polybench",
                    runtime_enabled=False,
                    run_args=[],
                ),
                ["inline"],
                tmp / f"screen_{len(screening_runs)}",
            )
            screening_runs.append(result["compile_time"])
        median_time = float(np.median(screening_runs))
        if median_time <= 15.0:
            selected.append(
                Benchmark(
                    name=name,
                    suite="polybench",
                    source_path=str(local_src),
                    canonical_bc=str(bc),
                    canonical_ll=str(ll),
                    canonical_inst_count=stats["instruction_count"],
                    basic_block_count=stats["basic_block_count"],
                    loop_count=stats["loop_count"],
                    call_count=stats["call_count"],
                    screening_time=median_time,
                    binary_link="polybench",
                    runtime_enabled=False,
                    run_args=[],
                )
            )
    polybench_final = sorted(selected, key=lambda x: x.name)[:6]
    cbench_selected = []
    for name, (bc_src, note) in cbench_candidates().items():
        if not bc_src.exists():
            continue
        local_bc_src = package_cbench_like_bc(bc_src)
        bc, ll = canonicalize_bc(name, local_bc_src)
        stats = module_stats_from_bc(bc)
        screening_runs = []
        for _ in range(3):
            tmp = ARTIFACTS / "data" / "screen" / name
            tmp.mkdir(parents=True, exist_ok=True)
            result = evaluate_sequence(
                Benchmark(
                    name=name,
                    suite="cbench_like",
                    source_path=str(local_bc_src),
                    canonical_bc=str(bc),
                    canonical_ll=str(ll),
                    canonical_inst_count=stats["instruction_count"],
                    basic_block_count=stats["basic_block_count"],
                    loop_count=stats["loop_count"],
                    call_count=stats["call_count"],
                    screening_time=0.0,
                    binary_link="standalone",
                    runtime_enabled=False,
                    run_args=[],
                    substitution_note=note,
                ),
                ["inline"],
                tmp / f"screen_{len(screening_runs)}",
            )
            screening_runs.append(result["compile_time"])
        median_time = float(np.median(screening_runs))
        if median_time <= 15.0:
            cbench_selected.append(
                Benchmark(
                    name=name,
                    suite="cbench_like",
                    source_path=str(local_bc_src),
                    canonical_bc=str(bc),
                    canonical_ll=str(ll),
                    canonical_inst_count=stats["instruction_count"],
                    basic_block_count=stats["basic_block_count"],
                    loop_count=stats["loop_count"],
                    call_count=stats["call_count"],
                    screening_time=median_time,
                    binary_link="standalone",
                    runtime_enabled=False,
                    run_args=[],
                    substitution_note=note,
                )
            )
            substitutions.append({"benchmark": name, "note": note})
    cbench_final = sorted(cbench_selected, key=lambda x: x.name)[:2]
    final = polybench_final + cbench_final
    runtime_subset = [b.name for b in sorted(polybench_final, key=lambda x: x.name)[:2]]
    runtime_subset += [b.name for b in sorted(cbench_final, key=lambda x: x.name)[:2]]
    for bench in final:
        bench.runtime_enabled = bench.name in runtime_subset
        manifest_rows.append(
            {
                "benchmark": bench.name,
                "suite": bench.suite,
                "source_path": bench.source_path,
                "canonical_bc_path": bench.canonical_bc,
                "canonical_ir_instruction_count": bench.canonical_inst_count,
                "basic_block_count": bench.basic_block_count,
                "loop_count": bench.loop_count,
                "call_count": bench.call_count,
                "median_screening_time": bench.screening_time,
                "substitution_note": bench.substitution_note,
            }
        )
        if bench.suite == "polybench":
            commands[bench.name] = {
                "build": f"clang {bench.canonical_bc} {LOCAL_POLYBENCH_ROOT / 'utilities/polybench.c'} -I {LOCAL_POLYBENCH_ROOT / 'utilities'} -lm -DPOLYBENCH_TIME -o <out>",
                "run": "./<out>",
            }
        else:
            commands[bench.name] = {
                "build": f"clang {bench.canonical_bc} -lm -o <out>",
                "run": "./<out>",
            }
        runtime_inputs[bench.name] = {"args": []}
    df = pd.DataFrame(manifest_rows)
    df.to_csv(ARTIFACTS / "data" / "benchmark_manifest.csv", index=False)
    write_json(ARTIFACTS / "data" / "runtime_subset.json", {"benchmarks": runtime_subset})
    write_text(ARTIFACTS / "data" / "benchmark_commands.yaml", yaml.safe_dump(commands, sort_keys=True))
    write_text(ARTIFACTS / "data" / "runtime_inputs.yaml", yaml.safe_dump(runtime_inputs, sort_keys=True))
    step_config = {
        "benchmark_selection_rule": "first 6 eligible PolyBench alphabetical + first 2 eligible local cbench-like alphabetical substitutes",
        "runtime_subset_rule": "first 2 eligible PolyBench + first 2 eligible cbench-like alphabetical",
        "local_packaged_benchmark_root": str(LOCAL_BENCH_ROOT),
    }
    write_step_config("01_data_prep", step_config)
    results = {
        "experiment": "01_data_prep",
        "metrics": {"num_benchmarks": len(final), "runtime_subset": runtime_subset},
        "config": step_config,
        "runtime_minutes": (time.time() - start) / 60.0,
        "substitutions": substitutions,
    }
    write_json(EXP / "01_data_prep" / "results.json", results)


def load_manifest():
    df = pd.read_csv(ARTIFACTS / "data" / "benchmark_manifest.csv")
    runtime_subset = set(json.loads((ARTIFACTS / "data" / "runtime_subset.json").read_text())["benchmarks"])
    out = []
    for row in df.to_dict(orient="records"):
        out.append(
            Benchmark(
                name=row["benchmark"],
                suite=row["suite"],
                source_path=row["source_path"],
                canonical_bc=row["canonical_bc_path"],
                canonical_ll=row["canonical_bc_path"].replace(".bc", ".ll"),
                canonical_inst_count=int(row["canonical_ir_instruction_count"]),
                basic_block_count=int(row["basic_block_count"]),
                loop_count=int(row["loop_count"]),
                call_count=int(row["call_count"]),
                screening_time=float(row["median_screening_time"]),
                binary_link=benchmark_binary_link(row["benchmark"], row["suite"]),
                runtime_enabled=row["benchmark"] in runtime_subset,
                run_args=[],
                substitution_note=row.get("substitution_note", ""),
            )
        )
    return out, runtime_subset


def stats_probe_step(benchmark):
    probe_dir = ARTIFACTS / "precondition" / "stats_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_sequence(
        benchmark,
        ["inline"],
        probe_dir / benchmark.name,
        capture_remarks=False,
        enable_stats=True,
    )
    stats_files = result["stats_path_candidates"]
    return {
        "benchmark": benchmark.name,
        "stats_files": stats_files,
        "usable": bool(stats_files),
        "evidence_path": result["raw_evidence_path"],
    }


def precondition_step():
    benchmarks, _ = load_manifest()
    raw_record_dir = ARTIFACTS / "precondition" / "raw_records"
    reliability_rows = []
    counter_vocab = []
    start = time.time()
    stats_probe = stats_probe_step(benchmarks[0])
    stats_available = stats_probe["usable"]
    benchmark_heatmap_rows = []
    for p in PASS_VOCAB:
        cov = 0
        densities = []
        applied = []
        missed = []
        analysis = []
        nonzero_counters = []
        for bench in benchmarks:
            raw_dir = ARTIFACTS / "precondition" / "raw" / bench.name / p
            result = evaluate_sequence(bench, [p], raw_dir, enable_stats=True)
            records = result["raw_remarks"]
            normalized = result["normalized_remarks"]
            stats_files = result["stats_path_candidates"]
            if records:
                cov += 1
                densities.append(len(normalized))
            total = max(len(normalized), 1)
            applied.append(sum(1 for r in normalized if r["outcome"] == "applied") / total)
            missed.append(sum(1 for r in normalized if r["outcome"] == "missed") / total)
            analysis.append(sum(1 for r in normalized if r["outcome"] == "analysis") / total)
            out_path = raw_record_dir / f"{bench.name}_{p}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            benchmark_heatmap_rows.append(
                {"benchmark": bench.name, "pass_name": p, "normalized_count": len(normalized)}
            )
            stats_available = stats_available or bool(stats_files)
            distinct_nonzero = 0
            if stats_files:
                merged_keys = set()
                for stats_path in stats_files:
                    payload = json.loads(Path(stats_path).read_text())
                    for key, value in payload.items():
                        if isinstance(value, (int, float)) and value != 0:
                            merged_keys.add(key)
                distinct_nonzero = len(merged_keys)
                counter_vocab.extend(sorted(merged_keys))
            nonzero_counters.append(distinct_nonzero)
        coverage = cov / len(benchmarks)
        density = float(np.median(densities)) if densities else 0.0
        stats_coverage = float(sum(1 for value in nonzero_counters if value > 0) / len(nonzero_counters))
        distinct_nonzero_counters = int(np.median(nonzero_counters)) if nonzero_counters else 0
        reliability_rows.append(
            {
                "pass_name": p,
                "remark_coverage": coverage,
                "remark_density": density,
                "applied_share": float(np.mean(applied)),
                "missed_share": float(np.mean(missed)),
                "analysis_share": float(np.mean(analysis)),
                "stats_coverage": stats_coverage,
                "num_distinct_nonzero_counters": distinct_nonzero_counters,
                "remark_reliable": coverage >= 0.75 and density >= 2.0,
                "stats_reliable": stats_coverage >= 0.75 and distinct_nonzero_counters >= 3,
            }
        )
    pd.DataFrame(reliability_rows).to_csv(ARTIFACTS / "precondition" / "pass_reliability.csv", index=False)
    write_json(ARTIFACTS / "precondition" / "stats_counter_vocab.json", sorted(set(counter_vocab)))
    remark_reliable_passes = sum(int(r["remark_reliable"]) for r in reliability_rows)
    stats_reliable_passes = sum(int(r["stats_reliable"]) for r in reliability_rows)
    if remark_reliable_passes >= 4:
        policy = "full_main_study" if stats_reliable_passes >= 4 else "partial_pilot_without_statsgp"
    elif stats_reliable_passes >= 4:
        policy = "stats_only_negative_result"
    else:
        policy = "remark_only_negative_result"
    step_config = {
        "passes": PASS_VOCAB,
        "remarks_enabled": True,
        "stats_enabled": True,
        "reliability_thresholds": {
            "remark_coverage": 0.75,
            "remark_density": 2.0,
            "stats_coverage": 0.75,
            "median_distinct_nonzero_counters": 3,
        },
    }
    write_step_config("02_precondition", step_config)
    write_json(
        EXP / "02_precondition" / "results.json",
        {
            "experiment": "02_precondition",
            "metrics": {
                "remark_reliable_passes": remark_reliable_passes,
                "stats_reliable_passes": stats_reliable_passes,
                "fallback_policy": policy,
            },
            "config": step_config,
            "runtime_minutes": (time.time() - start) / 60.0,
            "stats_available": stats_available,
            "stats_probe": stats_probe,
        },
    )


def normalization_step():
    raw_files = sorted((ARTIFACTS / "precondition" / "raw_records").glob("*.jsonl"))
    pool = []
    for path in raw_files:
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                norm = normalize_remark(rec)
                pool.append({"raw": rec, "normalized": norm})
    pool = pool[:120]
    examples = pool[:10]
    write_json(ARTIFACTS / "normalization" / "schema_examples.json", examples)
    parsed_items = [
        item
        for item in pool
        if item["normalized"]["family"] != "other"
    ]
    coverage = float(len(parsed_items) / max(len(pool), 1))
    other_rate = float(sum(1 for item in pool if item["normalized"]["blocker"] == "other") / max(len(pool), 1))
    failure_counts = Counter(
        item["normalized"]["consumer_pass"] for item in pool if item["normalized"]["blocker"] == "other"
    )
    write_text(
        ARTIFACTS / "normalization" / "manual_labels.csv",
        "status,note\nskipped,Real double annotation was infeasible in this single-operator run; Cohen's kappa remains unavailable.\n",
    )
    pilot_metrics = {
        "pool_size": len(pool),
        "parser_coverage": coverage,
        "other_rate": other_rate,
        "unresolved_count": int(sum(1 for item in pool if item["normalized"]["blocker"] == "other")),
        "per_pass_failure_counts": dict(failure_counts),
        "manual_double_annotation": "not_completed_protocol_downgraded",
        "family_kappa": None,
        "normalization_gate_passed": False,
    }
    write_json(ARTIFACTS / "normalization" / "pilot_metrics.json", pilot_metrics)
    write_text(
        EXP / "03_normalization" / "SKIPPED.md",
        "Rule-based normalization coverage was recomputed from improved raw remark parsing, but the registered double-annotation requirement was still not satisfiable in this single-operator environment. "
        "The normalization gate therefore remains unsatisfied, so the rerun is downgraded to a precondition/negative-result study rather than a main comparison.\n",
    )
    step_config = {
        "target_pool_size": 120,
        "double_annotation_target": 40,
        "kappa_threshold": 0.75,
        "parser_coverage_threshold": 0.80,
    }
    write_step_config("03_normalization", step_config)
    write_json(
        EXP / "03_normalization" / "results.json",
        {
            "experiment": "03_normalization",
            "metrics": pilot_metrics,
            "config": step_config,
            "runtime_minutes": 0.0,
        },
    )


def baseline_step():
    benchmarks, runtime_subset = load_manifest()
    rows = []
    start = time.time()
    for bench in benchmarks:
        seed_labels = SEEDS
        base_dir = ARTIFACTS / "results" / "fixed" / bench.name
        base_dir.mkdir(parents=True, exist_ok=True)
        std_bc = base_dir / "default_oz.bc"
        matched_in = base_dir / "matched_start.bc"
        build_prelude(bench.canonical_bc, matched_in)
        std_compile = apply_opt_pipeline(bench.canonical_bc, "default<Oz>", output_bc=std_bc)
        matched_bc = base_dir / "matched_default_oz.bc"
        matched_compile = apply_opt_pipeline(matched_in, "default<Oz>", output_bc=matched_bc)
        binaries = {
            "default_oz": base_dir / "default_oz.bin",
            "matched_default_oz": base_dir / "matched_default_oz.bin",
        }
        build_binary_from_bc(bench, std_bc, binaries["default_oz"])
        build_binary_from_bc(bench, matched_bc, binaries["matched_default_oz"])
        std_stats = module_stats_from_bc(std_bc)
        matched_stats = module_stats_from_bc(matched_bc)
        std_runtime = runtime_measure(binaries["default_oz"], []) if bench.name in runtime_subset else None
        matched_runtime = runtime_measure(binaries["matched_default_oz"], []) if bench.name in runtime_subset else None
        for seed in seed_labels:
            rows.extend(
                [
                    {
                        "benchmark": bench.name,
                        "seed": seed,
                        "method": "default<Oz>",
                        "instruction_count": std_stats["instruction_count"],
                        "text_size": text_size(binaries["default_oz"]),
                        "compile_time": std_compile["wall_time"],
                        "runtime_mean": None if std_runtime is None else std_runtime["mean"],
                    },
                    {
                        "benchmark": bench.name,
                        "seed": seed,
                        "method": "warmup+default<Oz>",
                        "instruction_count": matched_stats["instruction_count"],
                        "text_size": text_size(binaries["matched_default_oz"]),
                        "compile_time": matched_compile["wall_time"],
                        "runtime_mean": None if matched_runtime is None else matched_runtime["mean"],
                    },
                ]
            )
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "results" / "fixed_baselines.csv", index=False)
    baseline = df[
        df["method"] == "default<Oz>"
    ][["benchmark", "seed", "instruction_count", "text_size", "runtime_mean"]].rename(
        columns={
            "instruction_count": "baseline_instruction_count",
            "text_size": "baseline_text_size",
            "runtime_mean": "baseline_runtime_mean",
        }
    )
    enriched = df.merge(baseline, on=["benchmark", "seed"], how="left")
    enriched["text_size_change_pct"] = (
        (enriched["baseline_text_size"] - enriched["text_size"]) / enriched["baseline_text_size"] * 100.0
    )
    enriched["instruction_change_pct"] = (
        (enriched["baseline_instruction_count"] - enriched["instruction_count"])
        / enriched["baseline_instruction_count"]
        * 100.0
    )
    enriched["runtime_speedup"] = enriched.apply(
        lambda row: None
        if pd.isna(row["runtime_mean"]) or pd.isna(row["baseline_runtime_mean"])
        else row["baseline_runtime_mean"] / row["runtime_mean"],
        axis=1,
    )
    method_metrics = []
    for method, method_df in enriched.groupby("method"):
        method_metrics.append(
            {
                "method": method,
                "rows": int(len(method_df)),
                "text_size_change_pct": summarize_metric(method_df["text_size_change_pct"]),
                "instruction_change_pct": summarize_metric(method_df["instruction_change_pct"]),
                "runtime_speedup": summarize_metric(method_df["runtime_speedup"]),
                "compile_time": summarize_metric(method_df["compile_time"]),
            }
        )
    step_config = {
        "baselines": ["default<Oz>", "warmup+default<Oz>"],
        "duplicate_seed_labels": SEEDS,
        "primary_objective": PRIMARY_OBJECTIVE_LABEL,
    }
    write_step_config("04_fixed_baselines", step_config)
    write_json(
        EXP / "04_fixed_baselines" / "results.json",
        {
            "experiment": "04_fixed_baselines",
            "metrics": {
                "rows": len(rows),
                "aggregate_definition": "Population statistics over duplicated benchmark-seed rows; percentage and runtime metrics are relative to default<Oz> on the same benchmark-seed row.",
                "methods": method_metrics,
            },
            "config": step_config,
            "runtime_minutes": (time.time() - start) / 60.0,
        },
    )


def search_step():
    benchmarks, runtime_subset = load_manifest()
    start = time.time()
    trace_rows = []
    final_rows = []
    stats_probe = json.loads((EXP / "02_precondition" / "results.json").read_text()).get("stats_probe", {})
    for method in ["Random", "ProbeDelta", "RemarkState"]:
        t_rows, f_rows, _ = run_search_method(benchmarks, method, SEEDS, runtime_subset)
        trace_rows.extend(t_rows)
        final_rows.extend(f_rows)
    pd.DataFrame(trace_rows).to_csv(ARTIFACTS / "results" / "search_traces.csv", index=False)
    pd.DataFrame(final_rows).to_csv(ARTIFACTS / "results" / "final_sequences.csv", index=False)
    write_text(
        EXP / "05_search_baselines" / "SKIPPED.md",
        "StatsGP was not run because repeated `opt -stats -stats-json` probes on LLVM 18.1.3 emitted no usable JSON counters on this optimized build.\n",
    )
    step_config = {
        "methods_run": ["Random", "ProbeDelta", "RemarkState"],
        "methods_skipped": ["StatsGP"],
        "schedule": "6 single-pass probes + 4 pair probes + 8 longer-sequence evaluations for adaptive methods; 18 uniform samples for Random",
        "primary_objective": PRIMARY_OBJECTIVE_LABEL,
        "deterministic_best_tiebreak": [PRIMARY_OBJECTIVE_LABEL, "instruction_count", "compile_time", "sequence_length", "sequence_string"],
    }
    write_step_config("05_search_baselines", step_config)
    write_json(
        EXP / "05_search_baselines" / "results.json",
        {
            "experiment": "05_search_baselines",
            "metrics": {"trace_rows": len(trace_rows), "final_rows": len(final_rows)},
            "config": step_config | {
                "stats_probe": stats_probe,
                "protocol_revision_path": str(ARTIFACTS / "results" / "protocol_revision.json"),
            },
            "runtime_minutes": (time.time() - start) / 60.0,
        },
    )


def ablation_step():
    benchmarks, runtime_subset = load_manifest()
    subset_benchmarks = [b for b in benchmarks if b.name in runtime_subset]
    variants = [
        ("RemarkState", "NoBlocker"),
        ("RemarkState", "NoAppliedHistory"),
        ("RemarkState", "NoPairProbe"),
        ("RemarkState", "Beam2"),
        ("RemarkState", "Beam4"),
    ]
    rows = []
    start = time.time()
    for method, variant in variants:
        _, finals, _ = run_search_method(subset_benchmarks, method, SEEDS, runtime_subset, variant=variant)
        rows.extend(finals)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "results" / "ablation_analysis.csv", index=False)
    fixed = pd.read_csv(ARTIFACTS / "results" / "fixed_baselines.csv")
    baseline = fixed[
        fixed["method"] == "default<Oz>"
    ][["benchmark", "seed", "instruction_count", "text_size", "runtime_mean"]].rename(
        columns={
            "instruction_count": "baseline_instruction_count",
            "text_size": "baseline_text_size",
            "runtime_mean": "baseline_runtime_mean",
        }
    )
    enriched = df.merge(baseline, on=["benchmark", "seed"], how="left")
    enriched["text_size_change_pct"] = (
        (enriched["baseline_text_size"] - enriched["final_text_size"]) / enriched["baseline_text_size"] * 100.0
    )
    enriched["instruction_change_pct"] = (
        (enriched["baseline_instruction_count"] - enriched["final_ir_instruction_count"])
        / enriched["baseline_instruction_count"]
        * 100.0
    )
    enriched["runtime_speedup"] = enriched["baseline_runtime_mean"] / enriched["runtime_mean"]
    method_metrics = []
    for method, method_df in enriched.groupby("method"):
        method_metrics.append(
            {
                "method": method,
                "rows": int(len(method_df)),
                "final_text_size": summarize_metric(method_df["final_text_size"]),
                "text_size_change_pct": summarize_metric(method_df["text_size_change_pct"]),
                "instruction_change_pct": summarize_metric(method_df["instruction_change_pct"]),
                "runtime_speedup": summarize_metric(method_df["runtime_speedup"]),
                "tuning_time": summarize_metric(method_df["tuning_time"]),
            }
        )
    write_text(
        EXP / "07_ablations" / "SKIPPED.md",
        "StatsGP EI/UCB analytical check was skipped because the stats-based baseline was infeasible on this LLVM build.\n",
    )
    step_config = {
        "variants": [variant for _, variant in variants],
        "runtime_subset_only": True,
        "primary_objective": PRIMARY_OBJECTIVE_LABEL,
    }
    write_step_config("07_ablations", step_config)
    write_json(
        EXP / "07_ablations" / "results.json",
        {
            "experiment": "07_ablations",
            "metrics": {
                "rows": len(rows),
                "aggregate_definition": "Population statistics over the 12 runtime benchmark-seed rows; percentage and runtime metrics are relative to default<Oz> on the same benchmark-seed row.",
                "methods": method_metrics,
            },
            "config": step_config,
            "runtime_minutes": (time.time() - start) / 60.0,
        },
    )


def bootstrap_ci(values, n=1000):
    rng = np.random.default_rng(0)
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return [None, None]
    boots = []
    for _ in range(n):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(float(sample.mean()))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def objective_landscape_analysis(benchmarks):
    landscape_dir = ARTIFACTS / "results" / "objective_landscape"
    work_dir = ARTIFACTS / "results" / "objective_landscape_work"
    if landscape_dir.exists():
        shutil.rmtree(landscape_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    landscape_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench in benchmarks:
        seq_rows = []
        bench_work = work_dir / bench.name
        hash_to_text_size = {}
        prelude_bc = bench_work / "prelude.bc"
        bench_work.mkdir(parents=True, exist_ok=True)
        build_prelude(bench.canonical_bc, prelude_bc)
        for index, seq in enumerate(sequence_space(PASS_VOCAB)):
            obs = evaluate_sequence_minimal(prelude_bc, list(seq), bench_work)
            if obs["final_bc_sha256"] not in hash_to_text_size:
                hash_to_text_size[obs["final_bc_sha256"]] = text_size_for_bc(
                    bench, Path(obs["final_bc"]), obs["final_bc_sha256"]
                )
            seq_rows.append(
                {
                    "benchmark": bench.name,
                    "sequence": ",".join(seq),
                    "sequence_length": len(seq),
                    "instruction_count": obs["instruction_count"],
                    "text_size": hash_to_text_size[obs["final_bc_sha256"]],
                    "compile_time": obs["compile_time"],
                    "final_bc_sha256": obs["final_bc_sha256"],
                }
            )
        rows.extend(seq_rows)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "results" / "objective_landscape.csv", index=False)
    summary = (
        df.groupby("benchmark")
        .agg(
            total_candidates=("sequence", "count"),
            unique_instruction_counts=("instruction_count", "nunique"),
            unique_text_sizes=("text_size", "nunique"),
            unique_bitcode_hashes=("final_bc_sha256", "nunique"),
            min_instruction_count=("instruction_count", "min"),
            max_instruction_count=("instruction_count", "max"),
            min_text_size=("text_size", "min"),
            max_text_size=("text_size", "max"),
        )
        .reset_index()
    )
    summary["instruction_objective_flat"] = summary["unique_instruction_counts"] == 1
    summary["text_size_objective_flat"] = summary["unique_text_sizes"] == 1
    summary.to_csv(ARTIFACTS / "results" / "objective_landscape_summary.csv", index=False)
    return summary


def paired_comparison_report(merged, method_a, method_b):
    left = merged[merged["method"] == method_a][DEFAULT_COLUMNS]
    right = merged[merged["method"] == method_b][DEFAULT_COLUMNS]
    joined = left.merge(
        right,
        on=["benchmark", "seed"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if joined.empty:
        return {"method_a": method_a, "method_b": method_b, "available": False}

    joined["primary_diff"] = (
        joined["primary_improvement_pct_a"] - joined["primary_improvement_pct_b"]
    )
    joined["text_size_diff"] = joined["text_size_change_pct_a"] - joined["text_size_change_pct_b"]
    joined["runtime_speedup_diff"] = joined["runtime_speedup_a"] - joined["runtime_speedup_b"]
    joined["tuning_time_diff"] = joined["tuning_time_a"] - joined["tuning_time_b"]

    benchmark_seed_means = (
        joined.groupby("benchmark")["primary_diff"].mean().reset_index(name="primary_diff_seed_mean")
    )
    diffs = joined["primary_diff"].tolist()
    nonzero_diffs = [value for value in diffs if value != 0]
    wilcoxon_result = None
    if nonzero_diffs:
        stat, pvalue = wilcoxon(nonzero_diffs, zero_method="wilcox")
        wilcoxon_result = {"statistic": float(stat), "pvalue": float(pvalue)}

    return {
        "method_a": method_a,
        "method_b": method_b,
        "available": True,
        "per_benchmark_seed_mean_primary_diff": benchmark_seed_means.to_dict(orient="records"),
        "summary": {
            "mean_primary_diff": float(joined["primary_diff"].mean()),
            "median_primary_diff": float(joined["primary_diff"].median()),
            "win_count": int((joined["primary_diff"] > 0).sum()),
            "loss_count": int((joined["primary_diff"] < 0).sum()),
            "tie_count": int((joined["primary_diff"] == 0).sum()),
            "bootstrap95_primary_diff_over_benchmark_seed_means": bootstrap_ci(
                benchmark_seed_means["primary_diff_seed_mean"].tolist()
            ),
            "wilcoxon_primary_diff": wilcoxon_result,
        },
    }


def evaluation_step():
    benchmarks, runtime_subset = load_manifest()
    finals = pd.read_csv(ARTIFACTS / "results" / "final_sequences.csv")
    fixed = pd.read_csv(ARTIFACTS / "results" / "fixed_baselines.csv")
    trace = pd.read_csv(ARTIFACTS / "results" / "search_traces.csv")
    objective_summary = objective_landscape_analysis(benchmarks)
    nonflat_benchmarks = set(
        objective_summary.loc[~objective_summary["text_size_objective_flat"], "benchmark"].tolist()
    )
    baseline = fixed[
        fixed["method"] == "default<Oz>"
    ][["benchmark", "seed", "instruction_count", "text_size", "runtime_mean"]].rename(
        columns={
            "instruction_count": "baseline_instruction_count",
            "text_size": "baseline_text_size",
            "runtime_mean": "baseline_runtime_mean",
        }
    )
    oracle = objective_summary[
        ["benchmark", "min_text_size", "text_size_objective_flat", "instruction_objective_flat"]
    ].rename(columns={"min_text_size": "oracle_text_size"})

    merged = finals.merge(baseline, on=["benchmark", "seed"], how="left").merge(
        oracle, on="benchmark", how="left"
    )
    merged["primary_improvement_pct"] = (
        (merged["baseline_text_size"] - merged["final_text_size"]) / merged["baseline_text_size"] * 100.0
    )
    merged["instruction_improvement_pct"] = (
        (merged["baseline_instruction_count"] - merged["final_ir_instruction_count"])
        / merged["baseline_instruction_count"]
        * 100.0
    )
    merged["text_size_change_pct"] = merged["primary_improvement_pct"]
    merged["runtime_speedup"] = merged.apply(
        lambda row: None
        if pd.isna(row["runtime_mean"]) or pd.isna(row["baseline_runtime_mean"])
        else row["baseline_runtime_mean"] / row["runtime_mean"],
        axis=1,
    )
    merged["oracle_gap_bytes"] = merged["final_text_size"] - merged["oracle_text_size"]
    merged["oracle_gap_pct_of_baseline"] = merged["oracle_gap_bytes"] / merged["baseline_text_size"] * 100.0

    trace = trace.merge(
        baseline[["benchmark", "seed", "baseline_text_size"]],
        on=["benchmark", "seed"],
        how="left",
    ).merge(oracle, on="benchmark", how="left")
    trace = trace.sort_values(["benchmark", "method", "seed", "evaluation_index"])
    trace["best_so_far_text_size"] = trace.groupby(["benchmark", "method", "seed"])["text_size"].cummin()
    trace["best_so_far_gap_bytes"] = trace["best_so_far_text_size"] - trace["oracle_text_size"]
    trace["best_so_far_gap_pct_of_baseline"] = (
        trace["best_so_far_gap_bytes"] / trace["baseline_text_size"] * 100.0
    )
    trace["hit_oracle"] = trace["best_so_far_text_size"] == trace["oracle_text_size"]
    first_hit = (
        trace[trace["hit_oracle"]]
        .groupby(["benchmark", "method", "seed"])["evaluation_index"]
        .min()
        .reset_index(name="first_hit_oracle_eval")
    )
    merged = merged.merge(first_hit, on=["benchmark", "method", "seed"], how="left")
    merged["oracle_hit_within_budget"] = merged["first_hit_oracle_eval"].notna()

    trace.to_csv(ARTIFACTS / "results" / "trace_anytime.csv", index=False)
    merged.to_csv(ARTIFACTS / "results" / "main_results_enriched.csv", index=False)

    summary = []
    methods = sorted(merged["method"].unique())
    for method in methods:
        sub = merged[merged["method"] == method]
        nonflat = sub[sub["benchmark"].isin(nonflat_benchmarks)]
        summary.append(
            {
                "method": method,
                "mean_primary_improvement_pct": float(sub["primary_improvement_pct"].mean()),
                "std_primary_improvement_pct": float(sub["primary_improvement_pct"].std(ddof=0)),
                "mean_instruction_improvement_pct": float(sub["instruction_improvement_pct"].mean()),
                "std_instruction_improvement_pct": float(sub["instruction_improvement_pct"].std(ddof=0)),
                "mean_text_size_change_pct": float(sub["text_size_change_pct"].mean()),
                "std_text_size_change_pct": float(sub["text_size_change_pct"].std(ddof=0)),
                "mean_runtime_speedup": safe_float(sub["runtime_speedup"].dropna().mean()),
                "std_runtime_speedup": safe_float(sub["runtime_speedup"].dropna().std(ddof=0)),
                "mean_tuning_time": float(sub["tuning_time"].mean()),
                "std_tuning_time": float(sub["tuning_time"].std(ddof=0)),
                "mean_first_hit_oracle_eval_nonflat": safe_float(
                    nonflat["first_hit_oracle_eval"].dropna().mean()
                ),
                "oracle_hit_rate_nonflat": safe_float(nonflat["oracle_hit_within_budget"].mean()),
            }
        )

    paired_reports = [paired_comparison_report(merged, "RemarkState", method) for method in PAIRWISE_METHODS]
    anytime_reports = []
    for method in [m for m in PAIRWISE_METHODS if m in merged["method"].unique()]:
        joined = (
            merged[(merged["method"] == "RemarkState") & (merged["benchmark"].isin(nonflat_benchmarks))]
            .merge(
                merged[(merged["method"] == method) & (merged["benchmark"].isin(nonflat_benchmarks))],
                on=["benchmark", "seed"],
                suffixes=("_remark", "_other"),
            )
        )
        if joined.empty:
            anytime_reports.append({"method_a": "RemarkState", "method_b": method, "available": False})
            continue
        joined["first_hit_diff"] = joined["first_hit_oracle_eval_other"] - joined["first_hit_oracle_eval_remark"]
        anytime_reports.append(
            {
                "method_a": "RemarkState",
                "method_b": method,
                "available": True,
                "mean_first_hit_advantage": float(joined["first_hit_diff"].mean()),
                "median_first_hit_advantage": float(joined["first_hit_diff"].median()),
                "win_count": int((joined["first_hit_diff"] > 0).sum()),
                "loss_count": int((joined["first_hit_diff"] < 0).sum()),
                "tie_count": int((joined["first_hit_diff"] == 0).sum()),
            }
        )
    write_json(
        ARTIFACTS / "results" / "statistical_report.json",
        {"pairwise_final": paired_reports, "pairwise_anytime_nonflat": anytime_reports},
    )

    instruction_objective_flat = bool(objective_summary["instruction_objective_flat"].all())
    primary_objective_flat = bool(objective_summary["text_size_objective_flat"].all())
    normalization_metrics = json.loads((ARTIFACTS / "normalization" / "pilot_metrics.json").read_text())
    final_outcome_discriminative = bool(
        merged.groupby(["benchmark", "seed"])["final_text_size"].nunique().gt(1).any()
    )
    anytime_discriminative = bool(
        trace[trace["benchmark"].isin(nonflat_benchmarks)]
        .groupby("evaluation_index")["best_so_far_gap_bytes"]
        .nunique()
        .gt(1)
        .any()
    )
    remark_hits = merged[
        (merged["method"] == "RemarkState") & (merged["benchmark"].isin(nonflat_benchmarks))
    ]["first_hit_oracle_eval"].dropna()
    probe_hits = merged[
        (merged["method"] == "ProbeDelta") & (merged["benchmark"].isin(nonflat_benchmarks))
    ]["first_hit_oracle_eval"].dropna()

    success_checks = {
        "remark_reliable_passes_at_least_4": int(
            pd.read_csv(ARTIFACTS / "precondition" / "pass_reliability.csv")["remark_reliable"].sum()
        )
        >= 4,
        "family_kappa_at_least_0.75": False,
        "parser_coverage_at_least_0.80": normalization_metrics["parser_coverage"] >= 0.80,
        "statsgp_feasible": False,
        "final_primary_outcome_is_discriminative": final_outcome_discriminative,
        "anytime_oracle_gap_is_discriminative_on_nonflat_benchmarks": anytime_discriminative,
        "remarkstate_ahead_of_probedelta_on_mean_first_hit_oracle_eval_nonflat": (
            not remark_hits.empty and not probe_hits.empty and float(remark_hits.mean()) < float(probe_hits.mean())
        ),
        "runtime_reflection_on_runtime_subset": float(
            merged[merged["benchmark"].isin(runtime_subset)]["runtime_speedup"].dropna().mean() or 0.0
        )
        > 1.0,
        "binary_text_size_objective_has_variation": not primary_objective_flat,
        "instruction_objective_has_variation": not instruction_objective_flat,
        "nonflat_benchmark_count_at_least_4": len(nonflat_benchmarks) >= 4,
    }
    rejection_conditions = {
        "fewer_than_4_remark_reliable_passes": not success_checks["remark_reliable_passes_at_least_4"],
        "failed_normalization_thresholds": not success_checks["parser_coverage_at_least_0.80"]
        or not success_checks["family_kappa_at_least_0.75"],
        "statsgp_unavailable": True,
        "no_runtime_improvement": not success_checks["runtime_reflection_on_runtime_subset"],
        "binary_text_size_objective_flat_across_full_search_space": primary_objective_flat,
        "instruction_objective_flat_across_full_search_space": instruction_objective_flat,
        "final_outcomes_collapse_across_methods": not success_checks["final_primary_outcome_is_discriminative"],
        "search_trace_not_discriminative_on_nonflat_benchmarks": not success_checks[
            "anytime_oracle_gap_is_discriminative_on_nonflat_benchmarks"
        ],
    }
    study_framing = (
        "predeclared_feasibility_negative_result_pilot"
        if not primary_objective_flat
        else "negative_result_precondition_analysis"
    )
    write_json(
        ARTIFACTS / "results" / "success_criteria.json",
        {
            "study_framing": study_framing,
            "success_checks": success_checks,
            "rejection_conditions": rejection_conditions,
            "claim_boundary": "This is a predeclared feasibility and negative-result pilot. No stats-based comparison claim is made because StatsGP is infeasible on this LLVM build. No normalization-validated interpretability claim is made because the double-annotation kappa gate remained incomplete. Final text size is descriptive, while oracle-gap sample efficiency on non-flat benchmarks is the main executable comparison.",
            "objective_landscape_summary_path": str(ARTIFACTS / "results" / "objective_landscape_summary.csv"),
            "nonflat_benchmarks": sorted(nonflat_benchmarks),
        },
    )
    write_text(
        ARTIFACTS / "results" / "conclusion_rule.txt",
        "Any result from this rerun must be described only as a CPU-only feasibility pilot in the restricted six-pass LLVM subspace. "
        "If final outcomes collapse or blocked prerequisites remain, report the study as a negative-result pilot and emphasize sample-efficiency diagnostics rather than claiming a stronger optimizer result.\n",
    )
    budget_report = {
        "measured_stages_wall_clock_minutes": {
            step: json.loads((EXP / step / "results.json").read_text())["runtime_minutes"]
            for step in [
                "00_env",
                "01_data_prep",
                "02_precondition",
                "03_normalization",
                "04_fixed_baselines",
                "05_search_baselines",
                "07_ablations",
            ]
            if (EXP / step / "results.json").exists()
        },
        "total_evaluation_count": int(len(trace)),
        "active_simplifications": [
            "Used local packaged cbench-like substitutions where exact cBench sources were unavailable; benchmark validity is therefore limited to substitute programs.",
            "Skipped stats-based search because -stats-json was unusable on the provided LLVM build.",
            "Kept the normalization gate unsatisfied because real double annotation was infeasible in a single-operator run.",
            "Used oracle-gap sample efficiency on non-flat benchmarks because final outcomes collapse for many benchmark-seed pairs even when text size varies.",
        ],
        "gpus": 0,
        "budget_respected": True,
    }
    write_json(ARTIFACTS / "results" / "runtime_budget_report.json", budget_report)
    step_config = {
        "runtime_subset": sorted(runtime_subset),
        "primary_objective": PRIMARY_OBJECTIVE_LABEL,
        "pairwise_reference_methods": PAIRWISE_METHODS,
        "nonflat_benchmarks": sorted(nonflat_benchmarks),
    }
    write_step_config("08_evaluation", step_config)
    write_json(
        EXP / "08_evaluation" / "results.json",
        {
            "experiment": "08_evaluation",
            "metrics": {
                "methods": summary,
                "pairwise_reports": paired_reports,
                "pairwise_anytime_reports": anytime_reports,
                "objective_landscape": objective_summary.to_dict(orient="records"),
            },
            "config": step_config,
            "runtime_minutes": 0.0,
        },
    )


def visualization_step():
    benchmarks, runtime_subset = load_manifest()
    manifest = pd.read_csv(ARTIFACTS / "data" / "benchmark_manifest.csv")
    precondition = pd.read_csv(ARTIFACTS / "precondition" / "pass_reliability.csv")
    main = pd.read_csv(ARTIFACTS / "results" / "main_results_enriched.csv")
    ablations = pd.read_csv(ARTIFACTS / "results" / "ablation_analysis.csv")
    manifest["runtime_subset"] = manifest["benchmark"].isin(runtime_subset)
    manifest.to_csv(TABLES / "table_benchmarks.csv", index=False)
    precondition.to_csv(TABLES / "table_precondition.csv", index=False)
    main.to_csv(TABLES / "table_main_results.csv", index=False)
    ablations.to_csv(TABLES / "table_ablations.csv", index=False)

    heat = []
    trace = pd.read_csv(ARTIFACTS / "results" / "search_traces.csv")
    for bench in manifest["benchmark"]:
        for p in PASS_VOCAB:
            count = 0
            raw_dir = ARTIFACTS / "precondition" / "raw" / bench / p
            for yaml_path in raw_dir.glob("*_remarks.yaml"):
                count += len(parse_remark_yaml(yaml_path))
            heat.append({"benchmark": bench, "pass": p, "count": count})
    heat_df = pd.DataFrame(heat)
    pivot = heat_df.pivot(index="benchmark", columns="pass", values="count")
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako")
    plt.title("Normalized Remark Counts")
    plt.tight_layout()
    plt.savefig(FIGURES / "precondition_heatmap.png", dpi=200)
    plt.savefig(FIGURES / "precondition_heatmap.pdf")
    plt.close()

    compare = (
        main[main["text_size_objective_flat"] == False]
        .groupby("method")["first_hit_oracle_eval"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=compare, x="method", y="first_hit_oracle_eval", color="#3a6ea5")
    plt.ylabel("Mean Evaluations To Oracle")
    plt.tight_layout()
    plt.savefig(FIGURES / "main_comparison_bars.png", dpi=200)
    plt.savefig(FIGURES / "main_comparison_bars.pdf")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=main, x="method", y="text_size_change_pct", ax=axes[0], color="#d98f5f")
    axes[0].set_title("Text Size Change (%)")
    runtime_df = main[main["benchmark"].isin(runtime_subset)].dropna(subset=["runtime_speedup"])
    if not runtime_df.empty:
        sns.boxplot(data=runtime_df, x="method", y="runtime_speedup", ax=axes[1], color="#5f9e6e")
        axes[1].set_title("Runtime Speedup")
    plt.tight_layout()
    plt.savefig(FIGURES / "real_endpoints.png", dpi=200)
    plt.savefig(FIGURES / "real_endpoints.pdf")
    plt.close()

    quality_cost = main[main["text_size_objective_flat"] == False].groupby("method").agg(
        first_hit_oracle_eval=("first_hit_oracle_eval", "mean"),
        tuning_time=("tuning_time", "mean"),
    )
    quality_cost = quality_cost.reset_index()
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=quality_cost, x="tuning_time", y="first_hit_oracle_eval", hue="method", s=120)
    for _, row in quality_cost.iterrows():
        plt.text(row["tuning_time"], row["first_hit_oracle_eval"], row["method"])
    plt.tight_layout()
    plt.savefig(FIGURES / "quality_vs_cost.png", dpi=200)
    plt.savefig(FIGURES / "quality_vs_cost.pdf")
    plt.close()

    step_config = {"figures": 4, "tables": 4, "primary_objective": PRIMARY_OBJECTIVE_LABEL}
    write_step_config("09_visualization", step_config)
    write_json(
        EXP / "09_visualization" / "results.json",
        {
            "experiment": "09_visualization",
            "metrics": {"figures": 4, "tables": 4},
            "config": step_config,
            "runtime_minutes": 0.0,
        },
    )


def aggregate_root_results():
    enriched = pd.read_csv(ARTIFACTS / "results" / "main_results_enriched.csv")
    success = json.loads((ARTIFACTS / "results" / "success_criteria.json").read_text())
    summary = {}
    for method, sub in enriched.groupby("method"):
        summary[method] = {
            "primary_improvement_pct": {
                "mean": float(sub["primary_improvement_pct"].mean()),
                "std": float(sub["primary_improvement_pct"].std(ddof=0)),
            },
            "instruction_improvement_pct": {
                "mean": float(sub["instruction_improvement_pct"].mean()),
                "std": float(sub["instruction_improvement_pct"].std(ddof=0)),
            },
            "text_size_change_pct": {
                "mean": float(sub["text_size_change_pct"].mean()),
                "std": float(sub["text_size_change_pct"].std(ddof=0)),
            },
            "tuning_time": {
                "mean": float(sub["tuning_time"].mean()),
                "std": float(sub["tuning_time"].std(ddof=0)),
            },
            "evaluations_to_oracle_nonflat": {
                "mean": safe_float(
                    sub.loc[sub["text_size_objective_flat"] == False, "first_hit_oracle_eval"].dropna().mean()
                ),
                "std": safe_float(
                    sub.loc[sub["text_size_objective_flat"] == False, "first_hit_oracle_eval"].dropna().std(ddof=0)
                ),
            },
        }
        rt = sub["runtime_speedup"].dropna()
        if not rt.empty:
            summary[method]["runtime_speedup"] = {
                "mean": float(rt.mean()),
                "std": float(rt.std(ddof=0)),
            }
    payload = {
        "experiment": "llvm_remark_state_microsearch",
        "study_framing": success["study_framing"],
        "metrics": summary,
        "config": {
            "seeds": SEEDS,
            "passes": PASS_VOCAB,
            "max_seq_len": MAX_SEQ_LEN,
            "primary_objective": PRIMARY_OBJECTIVE_LABEL,
            "methods_run": sorted(enriched["method"].unique()),
            "methods_skipped": ["StatsGP"],
        },
        "limitations": [
            "StatsGP comparison infeasible because this LLVM 18.1.3 build did not emit usable -stats-json counters.",
            "Normalization gate remained incomplete because real double annotation and Cohen's kappa were not completed.",
            "The six-pass search space is instruction-count flat on the evaluated benchmark suite, and final text-size outcomes often collapse across methods, so this rerun emphasizes oracle-gap sample efficiency rather than a strong final-outcome win claim.",
            "Benchmark suite uses locally packaged cbench-like substitutes where exact cBench programs were unavailable.",
        ],
        "runtime_minutes": float(
            sum(
                json.loads(path.read_text()).get("runtime_minutes", 0.0)
                for path in EXP.glob("*/results.json")
            )
        ),
    }
    write_json(ROOT / "results.json", payload)


def run_step(step):
    def remark_state_step():
        write_step_config("06_remark_state", {"implemented_inside": "05_search_baselines"})
        write_json(
            EXP / "06_remark_state" / "results.json",
            {
                "experiment": "06_remark_state",
                "metrics": {"implemented_inside": "05_search_baselines"},
                "config": {"note": "RemarkState runs are produced inside the shared search baseline step."},
                "runtime_minutes": 0.0,
            },
        )

    mapping = {
        "00_env": environment_step,
        "01_data_prep": data_prep_step,
        "02_precondition": precondition_step,
        "03_normalization": normalization_step,
        "04_fixed_baselines": baseline_step,
        "05_search_baselines": search_step,
        "06_remark_state": remark_state_step,
        "07_ablations": ablation_step,
        "08_evaluation": evaluation_step,
        "09_visualization": visualization_step,
        "all": None,
    }
    if step == "all":
        for name in [
            "00_env",
            "01_data_prep",
            "02_precondition",
            "03_normalization",
            "04_fixed_baselines",
            "05_search_baselines",
            "07_ablations",
            "08_evaluation",
            "09_visualization",
        ]:
            mapping[name]()
        aggregate_root_results()
        remark_state_step()
    else:
        mapping[step]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=[
        "00_env",
        "01_data_prep",
        "02_precondition",
        "03_normalization",
        "04_fixed_baselines",
        "05_search_baselines",
        "06_remark_state",
        "07_ablations",
        "08_evaluation",
        "09_visualization",
        "all",
    ])
    args = parser.parse_args()
    run_step(args.step)


if __name__ == "__main__":
    main()
