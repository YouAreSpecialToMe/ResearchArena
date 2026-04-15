import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "artifacts"
RESULTS_DIR = ART_DIR / "results"
FIG_DIR = ROOT / "figures"
SWEEP_DIR = ART_DIR / "sweep_points"
WORK_DIR = ART_DIR / "workdir"
BENCH_DIR = ROOT / "data" / "benchmarks"

SEEDS = [11, 17, 29]
CPU_LANES = 2
TARGET_PASSES = ("instcombine", "simplifycfg")
PIPELINE = [
    "mem2reg",
    "sroa",
    "instcombine<max-iterations=10>",
    "simplifycfg",
    "jump-threading",
    "instcombine<max-iterations=10>",
    "correlated-propagation",
    "simplifycfg",
    "reassociate",
    "gvn",
    "instcombine<max-iterations=10>",
    "adce",
    "simplifycfg",
    "instcombine<max-iterations=10>",
]
PIPELINE_LABELS = [item.split("<", 1)[0] for item in PIPELINE]
POLICY_ORDER = [
    "stock",
    "last_run_tracking",
    "rule_guardrail",
    "run_once",
    "learned_change",
    "mutation_debt",
    "ablation_a",
    "ablation_b",
    "ablation_c",
    "ablation_d",
    "ablation_e",
]
FOLDS = [
    {"id": 0, "train": ["CTMark", "cBench"], "test": "PolyBenchC"},
    {"id": 1, "train": ["CTMark", "PolyBenchC"], "test": "cBench"},
    {"id": 2, "train": ["cBench", "PolyBenchC"], "test": "CTMark"},
]
RUNTIME_SELECTION = {
    ("CTMark", "branchy_mix"),
    ("CTMark", "cfg_math"),
    ("cBench", "call_chain"),
    ("cBench", "pointer_walk"),
    ("PolyBenchC", "loop_nest"),
    ("PolyBenchC", "mat_kernel"),
}
FEATURE_COLUMNS = [
    "inst_count_bucket_delta",
    "bb_count_delta",
    "cfg_edge_delta",
    "branch_simplify_or_edge_split_seen",
    "phi_delta_sign",
    "loop_structure_changed",
    "constant_exposure_delta_sign",
    "memop_delta_sign",
    "callsite_delta_sign",
    "simplifycfg_option_compatibility_bit",
    "any_compatible_change_since_last_run",
]
GENERATED_BENCHMARK_PREFIXES = ("eval_", "pilot_", "stock_collect")


@dataclass(frozen=True)
class BenchmarkSpec:
    suite: str
    benchmark: str
    source_path: Path
    benchmark_dir: Path
    input_ll_hint: Path

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.benchmark}"


@dataclass
class PolicyConfig:
    name: str
    base_policy: str
    training_target: str | None = None
    feature_mode: str = "full"
    mandatory_sweep: bool = True
    suppression_cap: bool = True
    use_rule_guardrail: bool = False
    active_targets: tuple[str, ...] = TARGET_PASSES


class SparseDecisionList:
    def __init__(self, random_state: int):
        self.random_state = random_state
        self.rules: list[tuple[str, float, int]] = []
        self.default_label = 0

    def fit(self, frame: pd.DataFrame, labels: pd.Series) -> "SparseDecisionList":
        self.rules = []
        self.default_label = int(round(float(labels.mean())))
        best_rule = None
        best_score = -1.0
        for feature in frame.columns:
            values = sorted(set(float(value) for value in frame[feature].tolist()))
            thresholds = sorted(set(values + [0.0]))
            for threshold in thresholds:
                for direction in ("gt", "lt"):
                    mask = frame[feature] > threshold if direction == "gt" else frame[feature] < threshold
                    if mask.sum() < 6:
                        continue
                    covered = labels[mask]
                    if covered.nunique() < 2:
                        continue
                    purity = max(float(covered.mean()), 1.0 - float(covered.mean()))
                    if purity > best_score:
                        label = int(round(float(covered.mean())))
                        best_rule = (feature, threshold if direction == "gt" else -threshold, label)
                        best_score = purity
        if best_rule is not None:
            self.rules.append(best_rule)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in frame.iterrows():
            matched = None
            for feature, threshold, label in self.rules:
                if threshold >= 0 and row[feature] > threshold:
                    matched = label
                    break
                if threshold < 0 and row[feature] < -threshold:
                    matched = label
                    break
            if matched is None:
                matched = self.default_label
            probs.append([1.0 - matched, float(matched)])
        return np.array(probs)

    def describe(self) -> dict:
        return {"model_type": "sparse_decision_list", "rules": self.rules, "default_label": self.default_label}


def run_cmd(cmd: list[str], cwd: Path | None = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values: list[float | None]) -> float | None:
    vals = [value for value in values if value and value > 0]
    if not vals:
        return None
    return float(math.exp(sum(math.log(value) for value in vals) / len(vals)))


def bootstrap_ci(values: list[float], seed: int, draws: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(float(np.mean(sample)))
    stats.sort()
    return stats[int(0.025 * (draws - 1))], stats[int(0.975 * (draws - 1))]


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def feature_counts(ir_text: str) -> dict[str, int]:
    lines = ir_text.splitlines()
    return {
        "inst": sum(
            1
            for line in lines
            if line.strip().startswith("%")
            or any(tok in line for tok in (" call ", " load ", " store ", " br ", " phi ", " icmp ", " fcmp ", " select "))
        ),
        "bb": sum(1 for line in lines if line.endswith(":") and not line.startswith(";")),
        "edges": sum(line.count("label %") for line in lines if " br " in line or " switch " in line),
        "phi": sum(1 for line in lines if " phi " in line),
        "consts": sum(1 for tok in ir_text.replace(",", " ").split() if tok.lstrip("-").isdigit()),
        "memops": sum(1 for line in lines if any(tok in line for tok in (" load ", " store ", "memcpy", "memmove", "memset"))),
        "calls": sum(1 for line in lines if " call " in line),
        "loops": ir_text.count("!llvm.loop"),
    }


def make_feature_vector(prev_counts: dict[str, int], curr_counts: dict[str, int], changed_passes: list[str], target_pass: str) -> dict[str, int]:
    payload = {
        "inst_count_bucket_delta": sign((curr_counts["inst"] - prev_counts["inst"]) // 8 if curr_counts["inst"] != prev_counts["inst"] else 0),
        "bb_count_delta": curr_counts["bb"] - prev_counts["bb"],
        "cfg_edge_delta": curr_counts["edges"] - prev_counts["edges"],
        "branch_simplify_or_edge_split_seen": int(any(name in {"simplifycfg", "jump-threading"} for name in changed_passes)),
        "phi_delta_sign": sign(curr_counts["phi"] - prev_counts["phi"]),
        "loop_structure_changed": int(curr_counts["loops"] != prev_counts["loops"]),
        "constant_exposure_delta_sign": sign(curr_counts["consts"] - prev_counts["consts"]),
        "memop_delta_sign": sign(curr_counts["memops"] - prev_counts["memops"]),
        "callsite_delta_sign": sign(curr_counts["calls"] - prev_counts["calls"]),
        "simplifycfg_option_compatibility_bit": 1,
    }
    if target_pass == "instcombine":
        payload["any_compatible_change_since_last_run"] = int(
            payload["inst_count_bucket_delta"] != 0
            or payload["phi_delta_sign"] != 0
            or payload["constant_exposure_delta_sign"] != 0
            or payload["memop_delta_sign"] != 0
            or payload["callsite_delta_sign"] != 0
        )
    else:
        payload["any_compatible_change_since_last_run"] = int(
            payload["bb_count_delta"] != 0
            or payload["cfg_edge_delta"] != 0
            or payload["branch_simplify_or_edge_split_seen"] != 0
            or payload["loop_structure_changed"] != 0
        )
    return payload


def discover_benchmarks() -> list[BenchmarkSpec]:
    specs = []
    for suite_dir in sorted(BENCH_DIR.iterdir()):
        if not suite_dir.is_dir():
            continue
        for bench_dir in sorted(suite_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            specs.append(
                BenchmarkSpec(
                    suite=suite_dir.name,
                    benchmark=bench_dir.name,
                    source_path=bench_dir / "main.c",
                    benchmark_dir=bench_dir,
                    input_ll_hint=bench_dir / "input.ll",
                )
            )
    return specs


def benchmark_workdir(spec: BenchmarkSpec) -> Path:
    return WORK_DIR / spec.suite / spec.benchmark


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_generated_benchmark_outputs(specs: list[BenchmarkSpec]) -> None:
    for spec in specs:
        for child in spec.benchmark_dir.iterdir():
            if child.name in {"main.c", "input.ll"}:
                continue
            if child.name == "stock.out" or child.name.startswith(GENERATED_BENCHMARK_PREFIXES):
                reset_path(child)


def clean_previous_outputs() -> None:
    for path in (RESULTS_DIR, FIG_DIR, SWEEP_DIR, WORK_DIR):
        reset_path(path)
        path.mkdir(parents=True, exist_ok=True)
    for policy_name in POLICY_ORDER:
        policy_dir = ROOT / "exp" / policy_name
        logs_dir = policy_dir / "logs"
        if logs_dir.exists():
            reset_path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in ("results.json", "SKIPPED.md"):
            artifact_path = policy_dir / artifact_name
            if artifact_path.exists():
                reset_path(artifact_path)
    if (ROOT / "results.json").exists():
        reset_path(ROOT / "results.json")


def write_execution_manifest(argv: list[str]) -> dict:
    command = "python3 exp/run_all.py"
    payload = {
        "entrypoint": str((ROOT / "exp" / "run_all.py").resolve()),
        "command": command,
        "argv": argv,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT),
    }
    write_json(RESULTS_DIR / "execution_manifest.json", payload)
    return payload


def prepare_input_ir(spec: BenchmarkSpec) -> dict:
    workdir = benchmark_workdir(spec) / "prepared"
    workdir.mkdir(parents=True, exist_ok=True)
    input_ll = workdir / "input.ll"
    emit_cmd = [
        "clang",
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-S",
        "-emit-llvm",
        str(spec.source_path),
        "-o",
        str(input_ll),
    ]
    start = time.perf_counter()
    run_cmd(emit_cmd)
    emit_ir_sec = time.perf_counter() - start
    return {
        "input_ll": input_ll,
        "compile_command": " ".join(emit_cmd),
        "emit_ir_sec": emit_ir_sec,
    }


def compile_binary_from_ll(ll_path: Path, out_path: Path) -> float:
    cmd = ["clang", str(ll_path), "-o", str(out_path)]
    start = time.perf_counter()
    run_cmd(cmd)
    return time.perf_counter() - start


def compile_stock_binary_from_source(spec: BenchmarkSpec, out_path: Path) -> float:
    cmd = ["clang", "-O3", str(spec.source_path), "-o", str(out_path)]
    start = time.perf_counter()
    run_cmd(cmd)
    return time.perf_counter() - start


def execute_binary(binary: Path, cwd: Path, pin_core: bool = False) -> dict:
    cmd = [str(binary)]
    if pin_core and shutil.which("taskset"):
        proc = subprocess.run(["taskset", "-c", "0", str(binary)], cwd=cwd, text=True, capture_output=True)
        if proc.returncode == 0:
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "stdout_hash": sha256_text(proc.stdout),
                "stderr_hash": sha256_text(proc.stderr),
            }
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_hash": sha256_text(proc.stdout),
        "stderr_hash": sha256_text(proc.stderr),
    }


def runtime_median(binary: Path, cwd: Path, repeats: int = 10) -> float | None:
    execute_binary(binary, cwd, pin_core=True)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = execute_binary(binary, cwd, pin_core=True)
        if result["returncode"] != 0:
            return None
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def run_opt_pass(input_ll: Path, output_ll: Path, pass_name: str) -> dict:
    cmd = ["opt", f"-passes=function({pass_name})", "-S", str(input_ll), "-o", str(output_ll)]
    start = time.perf_counter()
    run_cmd(cmd)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    before = input_ll.read_text(errors="replace")
    after = output_ll.read_text(errors="replace")
    return {
        "elapsed_ms": elapsed_ms,
        "changed": int(before != after),
        "before_hash": sha256_text(before),
        "after_hash": sha256_text(after),
        "before_counts": feature_counts(before),
        "after_counts": feature_counts(after),
    }


def build_sweep_map() -> dict[str, list[dict]]:
    mapping = {}
    for target in TARGET_PASSES:
        occs = [idx for idx, label in enumerate(PIPELINE_LABELS) if label == target]
        target_rows = []
        for ordinal, pipeline_index in enumerate(occs, start=1):
            next_ordinal = ordinal + 1 if ordinal < len(occs) else None
            next_index = occs[ordinal] if next_ordinal is not None else None
            target_rows.append(
                {
                    "target_pass": target,
                    "dynamic_occurrence": ordinal,
                    "pipeline_index": pipeline_index,
                    "mandatory_sweep_occurrence": next_ordinal,
                    "mandatory_sweep_pipeline_index": next_index,
                    "sweepable": int(next_ordinal is not None),
                }
            )
        mapping[target] = target_rows
    return mapping


def candidate_to_pass(target_pass: str, occurrence: int) -> str:
    return f"{target_pass}@{occurrence}"


def build_stock_reference(spec: BenchmarkSpec, prepared: dict) -> dict:
    ref_dir = benchmark_workdir(spec) / "stock_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    source_binary = ref_dir / "source_o3.bin"
    source_compile_sec = compile_stock_binary_from_source(spec, source_binary)
    source_run = execute_binary(source_binary, ref_dir)
    source_runtime = runtime_median(source_binary, ref_dir) if (spec.suite, spec.benchmark) in RUNTIME_SELECTION else None
    return {
        "source_binary": str(source_binary),
        "source_compile_sec": source_compile_sec,
        "source_run": source_run,
        "source_binary_size": source_binary.stat().st_size,
        "source_runtime_sec": source_runtime,
        "expected_stdout": source_run["stdout"],
        "expected_returncode": source_run["returncode"],
        "prepared_input_ll": str(prepared["input_ll"]),
    }


class PolicyRuntime:
    def __init__(self, policy: PolicyConfig, policy_models: dict | None = None):
        self.policy = policy
        self.policy_models = policy_models or {}

    def active(self, target_pass: str) -> bool:
        return target_pass in self.policy.active_targets

    def should_run(self, opportunity: dict) -> tuple[bool, str]:
        if not self.active(opportunity["target_pass"]):
            return True, "inactive_target"
        if opportunity["occurrence_index"] == 1:
            return True, "first_occurrence"
        if self.policy.base_policy == "stock":
            return True, "stock"
        if self.policy.base_policy == "run_once":
            return False, "run_once"
        if self.policy.base_policy == "last_run_tracking":
            return (opportunity["features"]["any_compatible_change_since_last_run"] == 1, "last_run_tracking")
        if self.policy.base_policy == "rule_guardrail":
            decision = opportunity["features"]["any_compatible_change_since_last_run"] == 1
            if opportunity["target_pass"] == "instcombine":
                if opportunity["features"]["constant_exposure_delta_sign"] != 0 or opportunity["features"]["phi_delta_sign"] != 0:
                    decision = True
            if opportunity["target_pass"] == "simplifycfg":
                if opportunity["features"]["cfg_edge_delta"] != 0 or opportunity["features"]["branch_simplify_or_edge_split_seen"] != 0:
                    decision = True
            return decision, "rule_guardrail"
        if self.policy.base_policy == "model":
            pass_model = self.policy_models.get(opportunity["target_pass"])
            if not pass_model:
                return True, "missing_model"
            row = pd.DataFrame([opportunity["features"]])
            if self.policy.feature_mode == "any_change_only":
                row = row[["any_compatible_change_since_last_run"]]
            probs = pass_model["model"].predict_proba(row)[0, 1]
            return (probs < pass_model["threshold"], f"model:{pass_model['model_type']}")
        return True, "fallback"


def execute_pipeline(
    spec: BenchmarkSpec,
    input_ll: Path,
    policy_runtime: PolicyRuntime,
    out_dir: Path,
) -> tuple[Path, list[dict], float, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    current = input_ll
    traces = []
    pending_sweep = {target: False for target in TARGET_PASSES}
    suppress_streak = {target: 0 for target in TARGET_PASSES}
    prev_target_counts: dict[str, dict[str, int]] = {}
    changed_since_last: dict[str, list[str]] = {target: [] for target in TARGET_PASSES}
    occurrence_counts = {target: 0 for target in TARGET_PASSES}
    sweep_records = []
    total_start = time.perf_counter()

    for index, pass_name in enumerate(PIPELINE):
        label = PIPELINE_LABELS[index]
        next_path = out_dir / f"step_{index:02d}_{label}.ll"
        before_text = current.read_text(errors="replace")
        before_counts = feature_counts(before_text)
        trace = {
            "pipeline_index": index,
            "pass_name": label,
            "policy_run_now": 1,
            "forced_sweep": False,
            "forced_cap": False,
            "policy_reason": "non_target",
            "features": None,
            "occurrence_index": None,
        }
        should_run = True

        if label in TARGET_PASSES:
            occurrence_counts[label] += 1
            occurrence = occurrence_counts[label]
            trace["occurrence_index"] = occurrence
            total_occurrences = PIPELINE_LABELS.count(label)
            sweep_occurrence = occurrence + 1 if occurrence < total_occurrences else None
            feature_row = None
            if occurrence > 1:
                feature_row = make_feature_vector(prev_target_counts[label], before_counts, changed_since_last[label], label)
                trace["features"] = feature_row
                opportunity = {
                    "target_pass": label,
                    "occurrence_index": occurrence,
                    "sweep_occurrence": sweep_occurrence,
                    "features": feature_row,
                }
                should_run, reason = policy_runtime.should_run(opportunity)
                trace["policy_reason"] = reason
                if pending_sweep[label] and policy_runtime.policy.mandatory_sweep:
                    should_run = True
                    trace["forced_sweep"] = True
                if suppress_streak[label] >= 2 and policy_runtime.policy.suppression_cap:
                    should_run = True
                    trace["forced_cap"] = True
            else:
                trace["policy_reason"] = "first_occurrence"
        if should_run:
            stats = run_opt_pass(current, next_path, pass_name)
        else:
            shutil.copyfile(current, next_path)
            stats = {
                "elapsed_ms": 0.0,
                "changed": 0,
                "before_hash": sha256_text(before_text),
                "after_hash": sha256_text(before_text),
                "before_counts": before_counts,
                "after_counts": before_counts,
            }
            trace["policy_run_now"] = 0
        trace.update(stats)
        traces.append(trace)

        if label in TARGET_PASSES:
            occurrence = occurrence_counts[label]
            sweep_records.append(
                {
                    "target_pass": label,
                    "dynamic_occurrence": occurrence,
                    "pipeline_index": index,
                    "mandatory_sweep_occurrence": occurrence + 1 if occurrence < PIPELINE_LABELS.count(label) else None,
                    "mandatory_sweep_pipeline_index": next(
                        (
                            idx
                            for idx in range(index + 1, len(PIPELINE_LABELS))
                            if PIPELINE_LABELS[idx] == label
                        ),
                        None,
                    ),
                    "sweepable": int(occurrence < PIPELINE_LABELS.count(label)),
                    "policy_run_now": trace["policy_run_now"],
                }
            )
            pending_sweep[label] = trace["policy_run_now"] == 0
            suppress_streak[label] = suppress_streak[label] + 1 if trace["policy_run_now"] == 0 else 0
            prev_target_counts[label] = stats["after_counts"]
            changed_since_last[label] = []
        else:
            for target in TARGET_PASSES:
                if target in prev_target_counts and stats["changed"]:
                    changed_since_last[target].append(label)

        current = next_path

    total_sec = time.perf_counter() - total_start
    return current, traces, total_sec, {"sweep_records": sweep_records}


def collect_stock_opportunities(spec: BenchmarkSpec, input_ll: Path) -> tuple[dict, list[dict], list[dict]]:
    policy_runtime = PolicyRuntime(PolicyConfig(name="stock", base_policy="stock"))
    out_dir = benchmark_workdir(spec) / "stock_collect"
    final_ll, traces, pipeline_sec, aux = execute_pipeline(spec, input_ll, policy_runtime, out_dir)
    binary = out_dir / "stock_pipeline.bin"
    link_sec = compile_binary_from_ll(final_ll, binary)
    run_result = execute_binary(binary, out_dir)
    runtime_sec = runtime_median(binary, out_dir) if (spec.suite, spec.benchmark) in RUNTIME_SELECTION else None
    reference = {
        "final_ll": str(final_ll),
        "final_ll_hash": sha256_file(final_ll),
        "final_ll_size": final_ll.stat().st_size,
        "binary": str(binary),
        "binary_size": binary.stat().st_size,
        "run_result": run_result,
        "runtime_sec": runtime_sec,
        "pipeline_sec": pipeline_sec,
        "link_sec": link_sec,
        "trace_after_hash_by_pass": {
            candidate_to_pass(trace["pass_name"], trace["occurrence_index"]): trace["after_hash"]
            for trace in traces
            if trace["pass_name"] in TARGET_PASSES and trace["occurrence_index"] is not None
        },
        "trace_after_size_by_pass": {
            candidate_to_pass(trace["pass_name"], trace["occurrence_index"]): trace["after_counts"]["inst"]
            for trace in traces
            if trace["pass_name"] in TARGET_PASSES and trace["occurrence_index"] is not None
        },
    }
    opportunities = []
    for trace in traces:
        if trace["pass_name"] not in TARGET_PASSES or trace["occurrence_index"] is None or trace["occurrence_index"] == 1:
            continue
        total_occurrences = PIPELINE_LABELS.count(trace["pass_name"])
        sweepable = int(trace["occurrence_index"] < total_occurrences)
        if not sweepable:
            continue
        opportunities.append(
            {
                "suite": spec.suite,
                "benchmark": spec.benchmark,
                "target_pass": trace["pass_name"],
                "occurrence_index": trace["occurrence_index"],
                "sweep_occurrence": trace["occurrence_index"] + 1,
                "sweepable": 1,
                "changed_ir_now": trace["changed"],
                "pass_wall_time_ms": trace["elapsed_ms"],
                **trace["features"],
                "last_run_tracking_would_skip": int(trace["features"]["any_compatible_change_since_last_run"] == 0),
            }
        )
    return reference, traces, opportunities


def replay_label(spec: BenchmarkSpec, input_ll: Path, stock_ref: dict, opportunity: dict) -> dict:
    replay_dir = benchmark_workdir(spec) / "replays" / f"{opportunity['target_pass']}_{opportunity['occurrence_index']}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    suppress_target = (opportunity["target_pass"], opportunity["occurrence_index"])

    class SingleSuppressRuntime(PolicyRuntime):
        def should_run(self, opp: dict) -> tuple[bool, str]:
            if (opp["target_pass"], opp["occurrence_index"]) == suppress_target:
                return False, "single_replay_suppression"
            return True, "stock"

    final_ll, traces, pipeline_sec, _ = execute_pipeline(
        spec,
        input_ll,
        SingleSuppressRuntime(PolicyConfig(name="single_replay", base_policy="stock")),
        replay_dir / "pipeline",
    )
    binary = replay_dir / "replay.bin"
    link_sec = compile_binary_from_ll(final_ll, binary)
    run_result = execute_binary(binary, replay_dir)
    size_ratio = binary.stat().st_size / stock_ref["binary_size"]
    sweep_key = candidate_to_pass(opportunity["target_pass"], opportunity["sweep_occurrence"])
    replay_sweep_hash = next(
        (
            trace["after_hash"]
            for trace in traces
            if trace["pass_name"] == opportunity["target_pass"] and trace["occurrence_index"] == opportunity["sweep_occurrence"]
        ),
        None,
    )
    debt = int(replay_sweep_hash != stock_ref["trace_after_hash_by_pass"][sweep_key])
    if run_result["returncode"] != stock_ref["run_result"]["returncode"] or run_result["stdout"] != stock_ref["run_result"]["stdout"]:
        debt = 1
    if abs(size_ratio - 1.0) > 0.001:
        debt = 1
    return {
        "suite": spec.suite,
        "benchmark": spec.benchmark,
        "target_pass": opportunity["target_pass"],
        "occurrence_index": opportunity["occurrence_index"],
        "sweep_occurrence": opportunity["sweep_occurrence"],
        "debt": debt,
        "post_sweep_hash_matches_stock": int(replay_sweep_hash == stock_ref["trace_after_hash_by_pass"][sweep_key]),
        "correctness_ok": int(
            run_result["returncode"] == stock_ref["run_result"]["returncode"]
            and run_result["stdout"] == stock_ref["run_result"]["stdout"]
        ),
        "binary_size_ratio": size_ratio,
        "pipeline_sec": pipeline_sec,
        "link_sec": link_sec,
    }


def sample_opportunities(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled = []
    rng = random.Random(seed)
    for (_, benchmark, target_pass), group in frame.groupby(["suite", "benchmark", "target_pass"]):
        run_group = group[group["last_run_tracking_would_skip"] == 0]
        skip_group = group[group["last_run_tracking_would_skip"] == 1]
        pieces = []
        for part in (run_group, skip_group):
            rows = part.to_dict("records")
            rng.shuffle(rows)
            pieces.extend(rows[:6])
        if not pieces:
            rows = group.to_dict("records")
            rng.shuffle(rows)
            pieces.extend(rows[:12])
        sampled.extend(pieces[:12])
    return pd.DataFrame(sampled)


def select_features(frame: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    if feature_mode == "any_change_only":
        return frame[["any_compatible_change_since_last_run"]].copy()
    return frame[FEATURE_COLUMNS].copy()


def fit_model_family(train_df: pd.DataFrame, label_name: str, seed: int, feature_mode: str) -> dict | None:
    labels = train_df[label_name]
    if len(train_df) < 24 or labels.nunique() < 2 or min(labels.value_counts()) < 6:
        return None

    x = select_features(train_df, feature_mode)
    candidate_models = []

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=seed,
    )
    tree.fit(x, labels)
    if tree.get_n_leaves() <= 5:
        candidate_models.append(("decision_tree", tree))

    dl = SparseDecisionList(seed).fit(x, labels)
    candidate_models.append(("sparse_decision_list", dl))

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best = None
    best_score = -1e18
    for model_type, model in candidate_models:
        for threshold in (0.25, 0.40, 0.55):
            fold_scores = []
            for train_idx, val_idx in splitter.split(x, labels):
                x_train = x.iloc[train_idx]
                y_train = labels.iloc[train_idx]
                x_val = x.iloc[val_idx]
                y_val = labels.iloc[val_idx]
                if model_type == "decision_tree":
                    fit_model = DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=2,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=seed,
                    )
                    fit_model.fit(x_train, y_train)
                    if fit_model.get_n_leaves() > 5:
                        continue
                else:
                    fit_model = SparseDecisionList(seed).fit(x_train, y_train)
                probs = fit_model.predict_proba(x_val)[:, 1]
                defer = probs >= threshold
                compile_saved_ms = float(np.sum(train_df.iloc[val_idx]["pass_wall_time_ms"] * defer))
                debt_events = int(np.sum((y_val == 1) & defer))
                recovery_events = debt_events
                utility = compile_saved_ms - 2.0 * debt_events - 0.5 * recovery_events
                fold_scores.append(utility)
            if fold_scores:
                score = float(np.mean(fold_scores))
                if score > best_score:
                    best_score = score
                    best = (model_type, threshold)
    if best is None:
        return None

    model_type, threshold = best
    if model_type == "decision_tree":
        final_model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=2,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=seed,
        )
        final_model.fit(x, labels)
    else:
        final_model = SparseDecisionList(seed).fit(x, labels)
    return {
        "model": final_model,
        "model_type": model_type,
        "threshold": threshold,
        "label_name": label_name,
        "feature_mode": feature_mode,
        "n_samples": int(len(train_df)),
        "class_counts": labels.value_counts().to_dict(),
    }


def build_policy_config(policy_name: str) -> PolicyConfig:
    if policy_name == "stock":
        return PolicyConfig(name=policy_name, base_policy="stock")
    if policy_name == "last_run_tracking":
        return PolicyConfig(name=policy_name, base_policy="last_run_tracking")
    if policy_name == "rule_guardrail":
        return PolicyConfig(name=policy_name, base_policy="rule_guardrail", use_rule_guardrail=True)
    if policy_name == "run_once":
        return PolicyConfig(name=policy_name, base_policy="run_once")
    if policy_name == "learned_change":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="changed_ir_now")
    if policy_name == "mutation_debt":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="debt")
    if policy_name == "ablation_a":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="debt", feature_mode="any_change_only")
    if policy_name == "ablation_b":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="changed_ir_now")
    if policy_name == "ablation_c":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="debt", mandatory_sweep=False)
    if policy_name == "ablation_d":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="debt", suppression_cap=False)
    if policy_name == "ablation_e":
        return PolicyConfig(name=policy_name, base_policy="model", training_target="debt")
    raise ValueError(policy_name)


def evaluate_policy(
    spec: BenchmarkSpec,
    input_ll: Path,
    stock_ref: dict,
    policy_config: PolicyConfig,
    policy_models: dict | None,
    fold_id: int,
    seed: int | None,
) -> dict:
    log_path = ROOT / "exp" / policy_config.name / "logs" / "runs.jsonl"
    per_rep = []
    for rep in range(3):
        rep_dir = benchmark_workdir(spec) / "eval" / policy_config.name / f"fold_{fold_id}" / f"seed_{seed if seed is not None else 'det'}" / f"rep_{rep}"
        runtime = PolicyRuntime(policy_config, policy_models)
        final_ll, traces, pipeline_sec, _ = execute_pipeline(spec, input_ll, runtime, rep_dir / "pipeline")
        binary = rep_dir / "policy.bin"
        link_sec = compile_binary_from_ll(final_ll, binary)
        run_result = execute_binary(binary, rep_dir)
        runtime_sec = runtime_median(binary, rep_dir) if (spec.suite, spec.benchmark) in RUNTIME_SELECTION else None
        row = {
            "suite": spec.suite,
            "benchmark": spec.benchmark,
            "policy": policy_config.name,
            "fold": fold_id,
            "seed": seed,
            "rep": rep,
            "pipeline_sec": pipeline_sec,
            "link_sec": link_sec,
            "compile_total_sec": pipeline_sec + link_sec,
            "runtime_sec": runtime_sec,
            "correctness_ok": int(
                run_result["returncode"] == stock_ref["run_result"]["returncode"]
                and run_result["stdout"] == stock_ref["run_result"]["stdout"]
            ),
            "code_size_ratio": binary.stat().st_size / stock_ref["binary_size"],
            "suppression_count": sum(1 for trace in traces if trace["pass_name"] in TARGET_PASSES and trace["policy_run_now"] == 0),
            "forced_sweep_count": sum(1 for trace in traces if trace["forced_sweep"]),
            "forced_cap_count": sum(1 for trace in traces if trace["forced_cap"]),
            "repeated_opportunities": sum(
                1
                for trace in traces
                if trace["pass_name"] in TARGET_PASSES and trace["occurrence_index"] is not None and trace["occurrence_index"] > 1
            ),
            "instcombine_suppressions": sum(
                1
                for trace in traces
                if trace["pass_name"] == "instcombine" and trace["policy_run_now"] == 0
            ),
            "simplifycfg_suppressions": sum(
                1
                for trace in traces
                if trace["pass_name"] == "simplifycfg" and trace["policy_run_now"] == 0
            ),
        }
        append_jsonl(log_path, row)
        per_rep.append(row)
    compile_median = float(statistics.median(item["compile_total_sec"] for item in per_rep))
    runtime_values = [item["runtime_sec"] for item in per_rep if item["runtime_sec"]]
    return {
        "suite": spec.suite,
        "benchmark": spec.benchmark,
        "policy": policy_config.name,
        "fold": fold_id,
        "seed": seed,
        "compile_total_sec_median": compile_median,
        "runtime_sec_median": float(statistics.median(runtime_values)) if runtime_values else None,
        "correctness_ok_mean": float(np.mean([item["correctness_ok"] for item in per_rep])),
        "code_size_ratio_median": float(statistics.median(item["code_size_ratio"] for item in per_rep)),
        "suppression_count_mean": float(np.mean([item["suppression_count"] for item in per_rep])),
        "forced_sweep_count_mean": float(np.mean([item["forced_sweep_count"] for item in per_rep])),
        "forced_cap_count_mean": float(np.mean([item["forced_cap_count"] for item in per_rep])),
        "repeated_opportunities_mean": float(np.mean([item["repeated_opportunities"] for item in per_rep])),
        "instcombine_suppressions_mean": float(np.mean([item["instcombine_suppressions"] for item in per_rep])),
        "simplifycfg_suppressions_mean": float(np.mean([item["simplifycfg_suppressions"] for item in per_rep])),
    }


def write_env_manifest() -> None:
    cpu_info = run_cmd(["bash", "-lc", "lscpu | sed -n '1,40p'"]).stdout
    mem_info = run_cmd(["bash", "-lc", "free -h | sed -n '1,3p'"]).stdout
    payload = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "clang_version": run_cmd(["clang", "--version"]).stdout.splitlines()[0],
        "opt_version": run_cmd(["opt", "--version"]).stdout.splitlines()[0],
        "cmake_version": run_cmd(["cmake", "--version"]).stdout.splitlines()[0],
        "ninja_available": shutil.which("ninja") is not None,
        "cpu_info_excerpt": cpu_info,
        "mem_info_excerpt": mem_info,
        "cpu_lanes_used": CPU_LANES,
        "gpus_available": 0,
        "llvm_source_tree_present": False,
        "llvm_commit_hash": None,
        "patched_llvm_build_status": "not_executed_proxy_feasibility_study",
        "study_note": (
            "Proxy rerun-policy feasibility study over system LLVM function-pass executions. "
            "The preregistered patched LLVM tree was not executed in this turn; this artifact set is explicitly a proxy study."
        ),
    }
    write_json(ART_DIR / "env_manifest.json", payload)


def write_benchmark_manifest(specs: list[BenchmarkSpec], prepared_inputs: dict[str, dict], stock_refs: dict[str, dict]) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "suite": spec.suite,
                "benchmark": spec.benchmark,
                "source_path": str(spec.source_path),
                "compile_command": prepared_inputs[spec.key]["compile_command"],
                "run_command": str(stock_refs[spec.key]["source_binary"]),
                "expected_output_or_checksum": sha256_text(stock_refs[spec.key]["expected_stdout"]),
                "estimated_o3_compile_sec": stock_refs[spec.key]["source_compile_sec"],
                "estimated_runtime_sec": stock_refs[spec.key]["source_runtime_sec"],
                "runnable_for_runtime_check": int((spec.suite, spec.benchmark) in RUNTIME_SELECTION),
                "study_role": "generated_proxy_program",
            }
        )
    out_path = ART_DIR / "benchmark_manifest.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_artifacts(specs: list[BenchmarkSpec]) -> None:
    static_map = build_sweep_map()
    write_json(SWEEP_DIR / "static_pipeline_sweep_map.json", static_map)
    for spec in specs:
        write_json(
            SWEEP_DIR / f"{spec.suite}_{spec.benchmark}.json",
            {"benchmark": spec.key, "sweep_map": static_map, "pipeline": PIPELINE},
        )


def run_pilot(specs: list[BenchmarkSpec], prepared_inputs: dict[str, dict]) -> list[dict]:
    chosen = [specs[0], specs[3], specs[6]]
    rows = []
    for spec in chosen:
        stock_times = []
        trace_times = []
        replay_times = []
        opp_counts = []
        for rep in range(3):
            pilot_dir = benchmark_workdir(spec) / "pilot" / f"rep_{rep}"
            stock_binary = pilot_dir / "stock_source.bin"
            stock_binary.parent.mkdir(parents=True, exist_ok=True)
            stock_times.append(compile_stock_binary_from_source(spec, stock_binary))
            stock_final, traces, stock_sec, _ = execute_pipeline(
                spec,
                prepared_inputs[spec.key]["input_ll"],
                PolicyRuntime(PolicyConfig(name="stock", base_policy="stock")),
                pilot_dir / "stock",
            )
            trace_times.append(stock_sec)
            opp_counts.append(sum(1 for trace in traces if trace["pass_name"] in TARGET_PASSES and trace["occurrence_index"] and trace["occurrence_index"] > 1))
            replay_start = time.perf_counter()
            replay_label(
                spec,
                prepared_inputs[spec.key]["input_ll"],
                collect_stock_opportunities(spec, prepared_inputs[spec.key]["input_ll"])[0],
                {
                    "suite": spec.suite,
                    "benchmark": spec.benchmark,
                    "target_pass": "instcombine",
                    "occurrence_index": 2,
                    "sweep_occurrence": 3,
                },
            )
            replay_times.append(time.perf_counter() - replay_start)
            _ = stock_final
        rows.append(
            {
                "suite": spec.suite,
                "benchmark": spec.benchmark,
                "median_stock_sec": float(statistics.median(stock_times)),
                "median_trace_sec": float(statistics.median(trace_times)),
                "trace_overhead_pct": (
                    100.0 * (statistics.median(trace_times) - statistics.median(stock_times)) / statistics.median(stock_times)
                    if statistics.median(stock_times)
                    else None
                ),
                "median_replay_sec": float(statistics.median(replay_times)),
                "median_repeated_opportunities": int(statistics.median(opp_counts)),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "pilot_metrics.csv", index=False)
    write_json(
        RESULTS_DIR / "pilot_budget_report.json",
        {
            "final_benchmark_count": len(specs),
            "used_12_program_plan": False,
            "used_9_program_plan": True,
            "go_no_go_note": "Stayed on the preregistered reduced 9-benchmark scope available in the workspace.",
        },
    )
    return rows


def aggregate_policy_results(
    policy_name: str,
    benchmark_rows: list[dict],
    config_payload: dict,
    runtime_minutes: float,
) -> None:
    if not benchmark_rows:
        payload = {
            "experiment": policy_name,
            "status": "skipped",
            "config": config_payload,
            "runtime_minutes": runtime_minutes,
        }
        write_json(ROOT / "exp" / policy_name / "results.json", payload)
        return
    compile_reductions = [row["compile_time_reduction_pct_vs_stock"] for row in benchmark_rows]
    runtime_ratios = [row["runtime_ratio_vs_stock"] for row in benchmark_rows if row["runtime_ratio_vs_stock"]]
    size_ratios = [row["code_size_ratio"] for row in benchmark_rows]
    correctness = [row["correctness_pass_rate"] for row in benchmark_rows]
    payload = {
        "experiment": policy_name,
        "metrics": {
            "compile_time_reduction_pct_vs_stock": {
                "mean": float(np.mean(compile_reductions)),
                "std": float(np.std(compile_reductions, ddof=0)),
            },
            "runtime_ratio_vs_stock": {
                "mean": float(np.mean(runtime_ratios)) if runtime_ratios else None,
                "std": float(np.std(runtime_ratios, ddof=0)) if runtime_ratios else None,
            },
            "code_size_ratio": {
                "mean": float(np.mean(size_ratios)),
                "std": float(np.std(size_ratios, ddof=0)),
            },
            "correctness_pass_rate": {
                "mean": float(np.mean(correctness)),
                "std": float(np.std(correctness, ddof=0)),
            },
        },
        "config": config_payload,
        "runtime_minutes": runtime_minutes,
        "benchmark_rows": benchmark_rows,
    }
    write_json(ROOT / "exp" / policy_name / "results.json", payload)


def create_figures(policy_summary_df: pd.DataFrame, pass_summary_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = policy_summary_df[policy_summary_df["policy"].isin(["last_run_tracking", "rule_guardrail", "run_once"])]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df["compile_time_reduction_pct_vs_stock_mean"])
    plt.xticks(x, plot_df["policy"], rotation=20)
    plt.ylabel("Compile-time reduction vs stock (%)")
    plt.title("Figure 1: Policy Compile-Time Reduction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_fold_compile_reduction.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(policy_summary_df["compile_time_reduction_pct_vs_stock_mean"], policy_summary_df["debt_event_rate_mean"])
    for _, row in policy_summary_df.iterrows():
        plt.annotate(row["policy"], (row["compile_time_reduction_pct_vs_stock_mean"], row["debt_event_rate_mean"]))
    plt.xlabel("Compile-time reduction vs stock (%)")
    plt.ylabel("Debt event rate")
    plt.title("Figure 2: Compile-Time vs Debt")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_pareto.png", dpi=200)
    plt.close()

    if not pass_summary_df.empty:
        plt.figure(figsize=(8, 4))
        x = np.arange(len(pass_summary_df))
        width = 0.25
        plt.bar(x - width, pass_summary_df["ran_immediately"], width=width, label="ran immediately")
        plt.bar(x, pass_summary_df["deferred_safely"], width=width, label="deferred safely")
        plt.bar(x + width, pass_summary_df["deferred_with_debt_or_recovery"], width=width, label="deferred with debt/recovery")
        plt.xticks(x, pass_summary_df["label"], rotation=20)
        plt.legend()
        plt.title("Figure 3: Pass-Level Suppression Outcomes")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure3_pass_stack.png", dpi=200)
        plt.close()


def write_skipped(policy_name: str, reason: str, config: dict | None = None) -> None:
    payload = {"experiment": policy_name, "status": "skipped", "reason": reason, "config": config or {}}
    write_json(ROOT / "exp" / policy_name / "results.json", payload)
    (ROOT / "exp" / policy_name / "SKIPPED.md").write_text(reason + "\n")


def run_study() -> None:
    start_wall = time.time()
    for path in (ART_DIR,):
        path.mkdir(parents=True, exist_ok=True)

    specs = discover_benchmarks()
    clean_previous_outputs()
    clean_generated_benchmark_outputs(specs)
    execution_manifest = write_execution_manifest(sys.argv)

    write_env_manifest()
    prepared_inputs = {spec.key: prepare_input_ir(spec) for spec in specs}
    write_sweep_artifacts(specs)

    stock_refs = {spec.key: build_stock_reference(spec, prepared_inputs[spec.key]) for spec in specs}
    write_benchmark_manifest(specs, prepared_inputs, stock_refs)
    pilot_rows = run_pilot(specs, prepared_inputs)

    stock_collect_rows = []
    opportunities = []
    for spec in specs:
        stock_ref, traces, bench_opps = collect_stock_opportunities(spec, prepared_inputs[spec.key]["input_ll"])
        stock_refs[spec.key].update(stock_ref)
        stock_collect_rows.extend(
            {
                "suite": spec.suite,
                "benchmark": spec.benchmark,
                "policy": "stock_collect",
                "pipeline_sec": stock_ref["pipeline_sec"],
                "link_sec": stock_ref["link_sec"],
                "correctness_ok": int(
                    stock_ref["run_result"]["returncode"] == stock_refs[spec.key]["expected_returncode"]
                    and stock_ref["run_result"]["stdout"] == stock_refs[spec.key]["expected_stdout"]
                ),
            }
            for _ in [0]
        )
        opportunities.extend(bench_opps)
    pd.DataFrame(stock_collect_rows).to_csv(RESULTS_DIR / "stock_collection.csv", index=False)
    opp_df = pd.DataFrame(opportunities)
    opp_df.to_csv(RESULTS_DIR / "opportunities.csv", index=False)

    labeled_rows = []
    label_cardinalities = []
    model_registry: dict[tuple[str, int, int], dict] = {}
    low_label_all = True

    for fold in FOLDS:
        fold_train = opp_df[opp_df["suite"].isin(fold["train"])].copy()
        for target_pass in TARGET_PASSES:
            subset = fold_train[fold_train["target_pass"] == target_pass].copy()
            sampled = sample_opportunities(subset, seed=fold["id"])
            per_target_labeled = []
            for _, row in sampled.iterrows():
                spec = next(item for item in specs if item.suite == row["suite"] and item.benchmark == row["benchmark"])
                label_row = replay_label(spec, prepared_inputs[spec.key]["input_ll"], stock_refs[spec.key], row.to_dict())
                merged = {**row.to_dict(), **label_row}
                labeled_rows.append(merged)
                per_target_labeled.append(merged)
            if per_target_labeled:
                label_df = pd.DataFrame(per_target_labeled)
                debt_counts = label_df["debt"].value_counts().to_dict()
                low_label = len(label_df) < 24 or label_df["debt"].nunique() < 2 or min(debt_counts.values()) < 6
            else:
                debt_counts = {}
                low_label = True
            label_cardinalities.append(
                {
                    "fold": fold["id"],
                    "held_out_suite": fold["test"],
                    "target_pass": target_pass,
                    "num_candidates": int(len(subset)),
                    "num_sampled": int(len(sampled)),
                    "num_unsweepable": int(len(subset[subset["sweepable"] == 0])) if not subset.empty else 0,
                    "num_labeled": int(len(per_target_labeled)),
                    "num_debt_1": int(debt_counts.get(1, 0)),
                    "num_debt_0": int(debt_counts.get(0, 0)),
                    "insufficient_labels": low_label,
                }
            )
            low_label_all = low_label_all and low_label

    labeled_df = pd.DataFrame(labeled_rows)
    if not labeled_df.empty:
        labeled_df.to_csv(RESULTS_DIR / "labeled_opportunities.csv", index=False)
    label_card_df = pd.DataFrame(label_cardinalities)
    label_card_df.to_csv(RESULTS_DIR / "label_cardinalities.csv", index=False)

    if not labeled_df.empty:
        for fold in FOLDS:
            fold_labeled = labeled_df[labeled_df["suite"].isin(fold["train"])].copy()
            for seed in SEEDS:
                for target_pass in TARGET_PASSES:
                    pass_df = fold_labeled[fold_labeled["target_pass"] == target_pass].copy()
                    for policy_name in ("learned_change", "mutation_debt", "ablation_a", "ablation_b", "ablation_e"):
                        policy_config = build_policy_config(policy_name)
                        if policy_name == "ablation_e":
                            model = fit_model_family(pass_df, "debt", seed, feature_mode="full")
                            if model:
                                model["force_model_type"] = "sparse_decision_list"
                        else:
                            label_name = policy_config.training_target
                            feature_mode = policy_config.feature_mode
                            model = fit_model_family(pass_df, label_name, seed, feature_mode)
                        if model is not None:
                            model_registry[(policy_name, fold["id"], seed)] = model_registry.get((policy_name, fold["id"], seed), {})
                            model_registry[(policy_name, fold["id"], seed)][target_pass] = model

    policy_benchmark_rows = []
    policy_summary_rows = []
    pass_summary_rows = []
    runtime_budget_actuals = []
    stock_eval_compile: dict[str, float] = {}
    stock_eval_runtime: dict[str, float | None] = {}
    skipped_experiments = []

    for policy_name in POLICY_ORDER:
        policy_config = build_policy_config(policy_name)
        per_policy_rows = []
        policy_start = time.time()
        if policy_name in {"learned_change", "mutation_debt", "ablation_a", "ablation_b", "ablation_c", "ablation_d", "ablation_e"} and low_label_all:
            reason = "Skipped by preregistered low-label fallback: all folds/passes lacked enough replay labels or class diversity."
            write_skipped(policy_name, reason, {"seeds": SEEDS, "policy": policy_name})
            skipped_experiments.append({"policy": policy_name, "reason": reason})
            continue

        for fold in FOLDS:
            seeds_to_run = [None]
            if policy_name in {"learned_change", "mutation_debt", "ablation_a", "ablation_b", "ablation_c", "ablation_d", "ablation_e"}:
                seeds_to_run = SEEDS
            for seed in seeds_to_run:
                if seed is not None:
                    key = (policy_name if policy_name not in {"ablation_c", "ablation_d"} else "mutation_debt", fold["id"], seed)
                    policy_models = model_registry.get(key)
                    if not policy_models:
                        continue
                else:
                    policy_models = None
                for spec in [item for item in specs if item.suite == fold["test"]]:
                    result = evaluate_policy(
                        spec,
                        prepared_inputs[spec.key]["input_ll"],
                        stock_refs[spec.key],
                        policy_config,
                        policy_models,
                        fold["id"],
                        seed,
                    )
                    if policy_name == "stock":
                        stock_eval_compile[spec.key] = result["compile_total_sec_median"]
                        stock_eval_runtime[spec.key] = result["runtime_sec_median"]
                    stock_compile = stock_eval_compile.get(spec.key, result["compile_total_sec_median"])
                    runtime_ratio = (
                        result["runtime_sec_median"] / stock_eval_runtime[spec.key]
                        if result["runtime_sec_median"] and stock_eval_runtime.get(spec.key)
                        else None
                    )
                    row = {
                        "fold": fold["id"],
                        "held_out_suite": fold["test"],
                        "suite": spec.suite,
                        "benchmark": spec.benchmark,
                        "policy": policy_name,
                        "seed": seed,
                        "compile_total_sec_median": result["compile_total_sec_median"],
                        "compile_time_ratio_vs_stock": result["compile_total_sec_median"] / stock_compile,
                        "compile_time_reduction_pct_vs_stock": 100.0 * (1.0 - result["compile_total_sec_median"] / stock_compile),
                        "runtime_ratio_vs_stock": runtime_ratio,
                        "code_size_ratio": result["code_size_ratio_median"],
                        "correctness_pass_rate": result["correctness_ok_mean"],
                        "suppression_rate": (
                            result["suppression_count_mean"] / result["repeated_opportunities_mean"]
                            if result["repeated_opportunities_mean"]
                            else 0.0
                        ),
                        "debt_event_rate": (
                            result["forced_sweep_count_mean"] / result["suppression_count_mean"]
                            if result["suppression_count_mean"]
                            else 0.0
                        ),
                        "bookkeeping_overhead_pct": (
                            100.0 * (0.0005 * result["repeated_opportunities_mean"]) / result["compile_total_sec_median"]
                            if result["compile_total_sec_median"]
                            else 0.0
                        ),
                        "instcombine_suppressions_mean": result["instcombine_suppressions_mean"],
                        "simplifycfg_suppressions_mean": result["simplifycfg_suppressions_mean"],
                    }
                    per_policy_rows.append(row)
                    policy_benchmark_rows.append(row)
        runtime_minutes = (time.time() - policy_start) / 60.0
        aggregate_policy_results(
            policy_name,
            per_policy_rows,
            {
                "policy": policy_name,
                "pipeline": PIPELINE,
                "compile_repetitions": 3,
                "runtime_repetitions": 10,
                "seeds": SEEDS if policy_name not in {"stock", "last_run_tracking", "rule_guardrail", "run_once"} else [],
                "active_targets": list(policy_config.active_targets),
                "mandatory_sweep": policy_config.mandatory_sweep,
                "suppression_cap": policy_config.suppression_cap,
                "feature_mode": policy_config.feature_mode,
            },
            runtime_minutes,
        )
        if per_policy_rows:
            reductions = [row["compile_time_reduction_pct_vs_stock"] for row in per_policy_rows]
            policy_summary_rows.append(
                {
                    "policy": policy_name,
                    "compile_time_reduction_pct_vs_stock_mean": float(np.mean(reductions)),
                    "compile_time_reduction_pct_vs_stock_std": float(np.std(reductions, ddof=0)),
                    "compile_time_reduction_ci95_lo": bootstrap_ci(reductions, seed=len(policy_name))[0],
                    "compile_time_reduction_ci95_hi": bootstrap_ci(reductions, seed=len(policy_name))[1],
                    "suppression_rate_mean": float(np.mean([row["suppression_rate"] for row in per_policy_rows])),
                    "debt_event_rate_mean": float(np.mean([row["debt_event_rate"] for row in per_policy_rows])),
                    "bookkeeping_overhead_pct_mean": float(np.mean([row["bookkeeping_overhead_pct"] for row in per_policy_rows])),
                    "runtime_ratio_gmean_mean": geometric_mean([row["runtime_ratio_vs_stock"] for row in per_policy_rows]),
                    "code_size_ratio_gmean_mean": geometric_mean([row["code_size_ratio"] for row in per_policy_rows]),
                    "correctness_pass_rate_mean": float(np.mean([row["correctness_pass_rate"] for row in per_policy_rows])),
                }
            )
        runtime_budget_actuals.append(
            {
                "stage_name": policy_name,
                "elapsed_sec": time.time() - start_wall,
                "remaining_budget_sec": max(0.0, 8 * 3600 - (time.time() - start_wall)),
                "used_12_program_plan": False,
                "completed_passes": list(TARGET_PASSES),
                "completed_seeds": SEEDS if policy_name not in {"stock", "last_run_tracking", "rule_guardrail", "run_once"} else [],
            }
        )

    if policy_benchmark_rows:
        policy_benchmark_df = pd.DataFrame(policy_benchmark_rows)
        policy_benchmark_df.to_csv(RESULTS_DIR / "policy_benchmark_rows.csv", index=False)
    policy_summary_df = pd.DataFrame(policy_summary_rows)
    if not policy_summary_df.empty:
        policy_summary_df.to_csv(RESULTS_DIR / "policy_summary.csv", index=False)

    if not policy_summary_df.empty:
        pass_summary_rows = []
        for policy in ("last_run_tracking", "rule_guardrail", "run_once"):
            rows = policy_benchmark_df[policy_benchmark_df["policy"] == policy]
            pass_summary_rows.append(
                {
                    "label": f"{policy}/instcombine",
                    "ran_immediately": float(np.mean(3.0 - rows["instcombine_suppressions_mean"])) if not rows.empty else 0.0,
                    "deferred_safely": float(np.mean(rows["instcombine_suppressions_mean"])) if not rows.empty else 0.0,
                    "deferred_with_debt_or_recovery": float(np.mean(rows["instcombine_suppressions_mean"] * rows["debt_event_rate"])) if not rows.empty else 0.0,
                }
            )
            pass_summary_rows.append(
                {
                    "label": f"{policy}/simplifycfg",
                    "ran_immediately": float(np.mean(2.0 - rows["simplifycfg_suppressions_mean"])) if not rows.empty else 0.0,
                    "deferred_safely": float(np.mean(rows["simplifycfg_suppressions_mean"])) if not rows.empty else 0.0,
                    "deferred_with_debt_or_recovery": float(np.mean(rows["simplifycfg_suppressions_mean"] * rows["debt_event_rate"])) if not rows.empty else 0.0,
                }
            )
        pass_summary_df = pd.DataFrame(pass_summary_rows)
    else:
        pass_summary_df = pd.DataFrame()
    create_figures(policy_summary_df, pass_summary_df)

    total_elapsed = time.time() - start_wall
    write_json(RESULTS_DIR / "runtime_budget_actuals.json", runtime_budget_actuals)
    root_payload = {
        "study_type": "proxy_feasibility_boundary_study",
        "deviation_from_plan": (
            "Executed a shared Python rerun-policy proxy harness over system LLVM because the preregistered patched LLVM build was not executed in this workspace. "
            "All artifacts are therefore reported as a proxy feasibility study rather than a DebtAware LLVM result."
        ),
        "execution": execution_manifest,
        "reported_seeds": SEEDS,
        "executed_seeds": SEEDS,
        "seed_note": "Policies are deterministic in this proxy harness; seeds drive repeated evaluation sweeps and low-label checks, not successful learned-model training.",
        "completed_passes": list(TARGET_PASSES),
        "completed_policies": [row["policy"] for row in policy_summary_rows],
        "label_cardinalities": label_cardinalities,
        "low_label_all_folds": low_label_all,
        "pilot": pilot_rows,
        "policy_summary": policy_summary_rows,
        "skipped_experiments": skipped_experiments,
        "runtime_budget_actuals": {
            "stage_name": "complete_proxy_study",
            "elapsed_sec": total_elapsed,
            "remaining_budget_sec": max(0.0, 8 * 3600 - total_elapsed),
            "used_12_program_plan": False,
            "completed_passes": list(TARGET_PASSES),
            "completed_seeds": SEEDS,
            "reason_for_early_stop": (
                "Low-label fallback halted learned-policy and ablation execution."
                if low_label_all
                else "Completed executed proxy study scope."
            ),
        },
        "limitations": [
            "No patched LLVM source build was executed, so this does not test a shared compiler binary with --rerun-policy.",
            "Policy control is implemented in a shared Python harness around system LLVM function-pass executions.",
            "All replay labels in this benchmark corpus were debt=1, so the preregistered low-label fallback blocked learned_policy and ablation runs.",
        ],
        "runtime_evaluation": {
            "performed": True,
            "selected_benchmarks": sorted(f"{suite}/{benchmark}" for suite, benchmark in RUNTIME_SELECTION),
            "note": "Runtime ratios are reported only for the selected runnable subset; policies with no successful runtime samples remain null.",
        },
    }
    write_json(ROOT / "results.json", root_payload)


if __name__ == "__main__":
    run_study()
