import argparse
import importlib.metadata
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy import linalg, stats
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = ROOT / "figures"
STAGE_TIMING_PATH = DATA_DIR / "stage_timings.json"
DEPENDENCY_MANIFEST_PATH = DATA_DIR / "dependency_manifest.json"
SEEDS = [11, 23, 37]
BUDGETS = [16, 32]
NOMINAL_LEVELS = [0.9, 0.95]
KERNELS = ["rbf", "matern32", "matern52"]
LOG_LENGTHSCALES = np.linspace(-2.0, 1.0, 7)
JITTER_SCHEDULE = [1e-8, 1e-6, 1e-4]
MEASURES = ["uniform", "gaussian"]
DIMS = [1, 2, 3]
DEPENDENCY_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "joblib",
]
TARGET_DEPENDENCY_RANGES = {
    "numpy": "1.26.*",
    "scipy": "1.12.*",
    "scikit-learn": "1.4.*",
    "pandas": "2.2.*",
    "matplotlib": "3.8.*",
    "seaborn": "0.13.*",
    "statsmodels": "0.14.*",
    "joblib": "1.3.*",
}
PLANNED_STAGE_HOURS = {
    "stage_a_reference": 0.9,
    "smoke_test": 0.17,
    "core_fits": 1.4,
    "main_recalibration": 0.5,
    "ablations": 3.5,
    "matched_generator": 0.6,
    "kernel_marginalization": 0.9,
    "aggregate": 0.55,
}


def configure_environment():
    for key in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"


def ensure_dirs():
    for path in [DATA_DIR, RAW_DIR, SUMMARY_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def collect_dependency_manifest():
    deps = {}
    for pkg in DEPENDENCY_PACKAGES:
        deps[pkg] = {
            "version": importlib.metadata.version(pkg),
            "target_range": TARGET_DEPENDENCY_RANGES[pkg],
        }
    return deps


def write_dependency_manifest():
    manifest = {
        "python_executable": os.sys.executable,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "dependencies": collect_dependency_manifest(),
    }
    with open(DEPENDENCY_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_stage_timings():
    if STAGE_TIMING_PATH.exists():
        with open(STAGE_TIMING_PATH) as fh:
            return json.load(fh)
    return {}


def save_stage_timings(stage_timings):
    with open(STAGE_TIMING_PATH, "w") as fh:
        json.dump(stage_timings, fh, indent=2)


def refresh_runtime_and_manifest():
    runtime_df, completed_blocks, skipped_blocks = build_runtime_accounting()
    manifest_path = SUMMARY_DIR / "figure_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    else:
        manifest = {}
    manifest["runtime_accounting"] = {
        "source_csv": str((SUMMARY_DIR / "runtime_accounting.csv").relative_to(ROOT)),
        "completed_blocks": completed_blocks,
        "skipped_blocks": skipped_blocks,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return runtime_df


def timed_stage(stage_name, fn):
    t0 = time.time()
    status = "completed"
    reason = ""
    try:
        return fn()
    except Exception as exc:
        status = "failed"
        reason = str(exc)
        raise
    finally:
        t1 = time.time()
        stage_timings = load_stage_timings()
        stage_timings[stage_name] = {
            "stage_name": stage_name,
            "start_time_unix": t0,
            "end_time_unix": t1,
            "wall_clock_seconds": t1 - t0,
            "status": status,
            "reason": reason,
        }
        save_stage_timings(stage_timings)
        if stage_name == "aggregate" and status == "completed":
            refresh_runtime_and_manifest()


@dataclass
class FunctionSpec:
    kind: str
    params: dict


def kernel_1d(kind, x, z, lengthscale):
    r = np.abs(x[:, None] - z[None, :]) / max(lengthscale, 1e-12)
    if kind == "rbf":
        return np.exp(-0.5 * r**2)
    if kind == "matern32":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    if kind == "matern52":
        a = np.sqrt(5.0) * r
        return (1.0 + a + (a**2) / 3.0) * np.exp(-a)
    raise ValueError(kind)


def kernel_matrix(kind, x, y, lengthscale, amplitude):
    parts = [kernel_1d(kind, x[:, i], y[:, i], lengthscale) for i in range(x.shape[1])]
    out = amplitude * np.ones_like(parts[0])
    for part in parts:
        out *= part
    return out


def quadrature_nodes(measure, order=80):
    if measure == "uniform":
        nodes, weights = leggauss(order)
        return 0.5 * (nodes + 1.0), 0.5 * weights
    if measure == "gaussian":
        nodes, weights = hermgauss(order)
        return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)
    raise ValueError(measure)


def kernel_mean_1d(kind, x, lengthscale, measure):
    nodes, weights = quadrature_nodes(measure)
    vals = kernel_1d(kind, np.asarray(x), nodes, lengthscale)
    return vals @ weights


def kernel_double_mean_1d(kind, lengthscale, measure):
    nodes, weights = quadrature_nodes(measure)
    k = kernel_1d(kind, nodes, nodes, lengthscale)
    return float(weights @ k @ weights)


def kernel_mean_vector(kind, x, lengthscale, amplitude, measure):
    factors = [kernel_mean_1d(kind, x[:, i], lengthscale, measure) for i in range(x.shape[1])]
    out = amplitude * np.ones(x.shape[0])
    for factor in factors:
        out *= factor
    return out


def kernel_double_mean(kind, dim, lengthscale, amplitude, measure):
    val = amplitude
    part = kernel_double_mean_1d(kind, lengthscale, measure)
    for _ in range(dim):
        val *= part
    return float(val)


def design_points(measure, dim, n, seed):
    sob = qmc.Sobol(d=dim, scramble=True, seed=seed)
    x = sob.random(n)
    if measure == "gaussian":
        x = stats.norm.ppf(np.clip(x, 1e-8, 1.0 - 1e-8))
    return x


def polynomial_features(x):
    return np.column_stack([np.ones(len(x)), x])


def basis_integrals(measure, dim):
    if measure == "uniform":
        return np.concatenate([[1.0], np.full(dim, 0.5)])
    if measure == "gaussian":
        return np.concatenate([[1.0], np.zeros(dim)])
    raise ValueError(measure)


def safe_cholesky(k):
    last_err = None
    for jitter in JITTER_SCHEDULE:
        try:
            return linalg.cho_factor(k + jitter * np.eye(k.shape[0]), lower=True), jitter
        except linalg.LinAlgError as err:
            last_err = err
    raise last_err


def solve_gp_bq(x, y, measure, bayes_sard=False):
    best = None
    for kind in KERNELS:
        for log_lengthscale in LOG_LENGTHSCALES:
            lengthscale = float(np.exp(log_lengthscale))
            base = kernel_matrix(kind, x, x, lengthscale, amplitude=1.0)
            try:
                chol0, used_jitter = safe_cholesky(base)
                alpha0 = linalg.cho_solve(chol0, y)
                amplitude = max(float(y @ alpha0) / len(y), 1e-10)
                k = amplitude * base
                chol, used_jitter = safe_cholesky(k)
                alpha = linalg.cho_solve(chol, y)
                logdet = 2.0 * np.log(np.diag(chol[0])).sum()
                lml = -0.5 * (y @ alpha + logdet + len(y) * np.log(2.0 * np.pi))
                q = kernel_mean_vector(kind, x, lengthscale, amplitude, measure)
                qq = kernel_double_mean(kind, x.shape[1], lengthscale, amplitude, measure)
                mean = float(q @ alpha)
                variance = max(float(qq - q @ linalg.cho_solve(chol, q)), 1e-12)
                result = {
                    "kind": kind,
                    "log_lengthscale": log_lengthscale,
                    "lengthscale": lengthscale,
                    "amplitude": amplitude,
                    "chol": chol,
                    "mean": mean,
                    "variance": variance,
                    "q": q,
                    "qq": qq,
                    "lml": lml,
                    "used_jitter": used_jitter,
                    "k": k,
                }
                if bayes_sard:
                    h = polynomial_features(x)
                    k_inv_h = linalg.cho_solve(chol, h)
                    g = h.T @ k_inv_h
                    rhs = h.T @ alpha
                    beta = np.linalg.solve(g + 1e-10 * np.eye(g.shape[0]), rhs)
                    psi = basis_integrals(measure, x.shape[1])
                    resid = y - h @ beta
                    alpha_resid = linalg.cho_solve(chol, resid)
                    mean_bs = float(psi @ beta + q @ alpha_resid)
                    c = psi - h.T @ linalg.cho_solve(chol, q)
                    variance_bs = max(
                        float(qq - q @ linalg.cho_solve(chol, q) + c @ np.linalg.solve(g + 1e-10 * np.eye(g.shape[0]), c)),
                        1e-12,
                    )
                    result["mean"] = mean_bs
                    result["variance"] = variance_bs
                if best is None or result["lml"] > best["lml"]:
                    best = result
            except Exception:
                continue
    if best is None:
        raise RuntimeError("No valid kernel candidate found")
    return best


def solve_kernel_marginalization(x, y, measure):
    means = []
    vars_ = []
    for kind in KERNELS:
        for log_lengthscale in LOG_LENGTHSCALES:
            lengthscale = float(np.exp(log_lengthscale))
            base = kernel_matrix(kind, x, x, lengthscale, amplitude=1.0)
            try:
                chol0, _ = safe_cholesky(base)
                alpha0 = linalg.cho_solve(chol0, y)
                amplitude = max(float(y @ alpha0) / len(y), 1e-10)
                k = amplitude * base
                chol, _ = safe_cholesky(k)
                alpha = linalg.cho_solve(chol, y)
                q = kernel_mean_vector(kind, x, lengthscale, amplitude, measure)
                qq = kernel_double_mean(kind, x.shape[1], lengthscale, amplitude, measure)
                means.append(float(q @ alpha))
                vars_.append(max(float(qq - q @ linalg.cho_solve(chol, q)), 1e-12))
            except Exception:
                continue
    mean = float(np.mean(means))
    variance = float(np.mean(np.array(vars_) + np.array(means) ** 2) - mean**2)
    return {"mean": mean, "variance": max(variance, 1e-12)}


def evaluate_function(spec, x):
    params = spec.params
    if spec.kind == "poly":
        powers = np.asarray(params["powers"])
        coeffs = np.asarray(params["coeffs"])
        shifted = x - np.asarray(params.get("shift", np.zeros(x.shape[1])))
        terms = np.prod(shifted[:, None, :] ** powers[None, :, :], axis=2)
        return terms @ coeffs
    if spec.kind == "osc":
        freq = np.asarray(params["freq"])
        phase = float(params.get("phase", 0.0))
        amp = float(params.get("amp", 1.0))
        shift = np.asarray(params.get("shift", np.zeros(x.shape[1])))
        return amp * np.sin(x @ freq + phase) + float(params.get("bias", 0.0))
    if spec.kind == "bump":
        center = np.asarray(params["center"])
        width = float(params["width"])
        amp = float(params.get("amp", 1.0))
        delta = x - center
        return amp * np.exp(-0.5 * np.sum(delta**2, axis=1) / (width**2))
    if spec.kind == "kink":
        axis = int(params["axis"])
        loc = float(params["loc"])
        power = float(params.get("power", 1.0))
        amp = float(params.get("amp", 1.0))
        return amp * np.abs(x[:, axis] - loc) ** power
    if spec.kind == "mixed":
        y = np.zeros(len(x))
        for weight, child in params["terms"]:
            y += weight * evaluate_function(FunctionSpec(child["kind"], child["params"]), x)
        return y
    raise ValueError(spec.kind)


def integral_1d_poly(power, shift, measure):
    if measure == "uniform":
        total = 0.0
        for j in range(power + 1):
            total += math.comb(power, j) * ((-shift) ** (power - j)) / (j + 1)
        return total
    moments = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0, 5: 0.0, 6: 15.0}
    total = 0.0
    for j in range(power + 1):
        total += math.comb(power, j) * ((-shift) ** (power - j)) * moments.get(j, 0.0)
    return total


def integral_spec(spec, measure, dim):
    params = spec.params
    if spec.kind == "poly":
        powers = np.asarray(params["powers"])
        coeffs = np.asarray(params["coeffs"])
        shift = np.asarray(params.get("shift", np.zeros(dim)))
        vals = []
        for power_vec, coeff in zip(powers, coeffs):
            term = coeff
            for axis in range(dim):
                term *= integral_1d_poly(int(power_vec[axis]), float(shift[axis]), measure)
            vals.append(term)
        return float(np.sum(vals))
    if spec.kind == "osc":
        freq = np.asarray(params["freq"])
        phase = float(params.get("phase", 0.0))
        amp = float(params.get("amp", 1.0))
        shift = np.asarray(params.get("shift", np.zeros(dim)))
        if measure == "uniform":
            comp = []
            for axis in range(dim):
                w = freq[axis]
                if abs(w) < 1e-12:
                    comp.append(1.0)
                else:
                    comp.append((np.exp(1j * w) - 1.0) / (1j * w))
            value = np.prod(comp) * np.exp(1j * (phase + shift @ freq))
            return float(amp * np.imag(value) + params.get("bias", 0.0))
        value = np.exp(1j * phase) * np.exp(-0.5 * np.sum(freq**2))
        return float(amp * np.imag(value) + params.get("bias", 0.0))
    if spec.kind == "bump":
        center = np.asarray(params["center"])
        width = float(params["width"])
        amp = float(params.get("amp", 1.0))
        if measure == "uniform":
            pieces = []
            for c in center:
                lo = (0.0 - c) / (np.sqrt(2.0) * width)
                hi = (1.0 - c) / (np.sqrt(2.0) * width)
                pieces.append(np.sqrt(np.pi / 2.0) * width * (math.erf(hi) - math.erf(lo)))
            return float(amp * np.prod(pieces))
        pieces = []
        for c in center:
            pieces.append(width / np.sqrt(1.0 + width**2) * np.exp(-0.5 * c**2 / (1.0 + width**2)))
        return float(amp * np.prod(pieces))
    if spec.kind == "kink":
        axis = int(params["axis"])
        loc = float(params["loc"])
        power = float(params.get("power", 1.0))
        amp = float(params.get("amp", 1.0))
        if abs(power - 1.0) > 1e-12:
            nodes, weights = quadrature_nodes(measure, order=200)
            base = np.sum(weights * np.abs(nodes - loc) ** power)
        else:
            if measure == "uniform":
                base = 0.5 * (loc**2 + (1.0 - loc) ** 2)
            else:
                a = abs(loc)
                base = 2.0 * stats.norm.pdf(a) + a * (2.0 * stats.norm.cdf(a) - 1.0)
        return float(amp * base)
    if spec.kind == "mixed":
        return float(sum(weight * integral_spec(FunctionSpec(child["kind"], child["params"]), measure, dim) for weight, child in params["terms"]))
    raise ValueError(spec.kind)


def roughness_score(x, y):
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    local = np.abs(y - y[nn]) / np.maximum(dists[np.arange(len(x)), nn], 1e-8)
    return float(np.mean(local) / max(np.std(y), 1e-8))


def fill_distance_proxy(x):
    dists = cdist(x, x)
    np.fill_diagonal(dists, np.inf)
    return float(np.max(np.min(dists, axis=1)))


def wis(y_true, lower, upper, nominal_level):
    alpha = 1.0 - nominal_level
    width = upper - lower
    penalty = 2.0 / alpha * max(lower - y_true, 0.0) + 2.0 / alpha * max(y_true - upper, 0.0)
    return float(width + penalty)


def make_probe_bank(measure):
    records = []
    family_order = ["poly", "osc", "loc"]
    for dim in DIMS:
        dim_index = dim - 1
        for family in family_order:
            for local_idx in range(8):
                rng = np.random.default_rng(1000 + 100 * dim + 10 * family_order.index(family) + local_idx + (0 if measure == "uniform" else 5000))
                if family == "poly":
                    powers = np.zeros((3, dim), dtype=int)
                    powers[0, local_idx % dim] = 1 + (local_idx % 2)
                    powers[1] = (np.arange(dim) + local_idx) % 2
                    powers[2, (local_idx + 1) % dim] = 2
                    coeffs = np.array([0.6, -0.35, 0.2])
                    spec = FunctionSpec("poly", {"powers": powers.tolist(), "coeffs": coeffs.tolist(), "shift": (0.5 * np.ones(dim) if measure == "uniform" else np.zeros(dim)).tolist()})
                    severity = float(np.sum(powers))
                    subtype = "matched"
                elif family == "osc":
                    base_freq = 1.5 + 0.75 * local_idx
                    freq = np.linspace(base_freq, base_freq + 0.3 * (dim - 1), dim)
                    phase = 0.2 * local_idx
                    spec = FunctionSpec("osc", {"freq": freq.tolist(), "phase": phase, "amp": 1.0})
                    severity = float(np.linalg.norm(freq))
                    subtype = "oscillatory"
                else:
                    width = 0.08 + 0.045 * local_idx
                    center = rng.uniform(0.2, 0.8, size=dim) if measure == "uniform" else rng.normal(0.0, 0.8, size=dim)
                    spec = FunctionSpec("bump", {"center": center.tolist(), "width": width, "amp": 1.0})
                    severity = float(1.0 / width)
                    subtype = "bump"
                records.append(
                    {
                        "task_id": f"probe_{measure}_d{dim}_{family}_{local_idx}",
                        "split": "calibration" if local_idx < 5 or (family == "loc" and local_idx == 5) else "test",
                        "measure": measure,
                        "dimension": dim,
                        "family": family,
                        "subtype": subtype,
                        "instance_id": local_idx,
                        "probe_group_id": f"{measure}_d{dim}_{family}_{local_idx}",
                        "severity": severity,
                        "spec": spec,
                    }
                )
    return records


def make_targets():
    records = []
    for measure in MEASURES:
        for dim in DIMS:
            for family in ["matched", "oscillatory", "bump", "kinked"]:
                for instance_id in [0, 1]:
                    rng = np.random.default_rng(20000 + 1000 * (measure == "gaussian") + 100 * dim + 10 * instance_id + len(family))
                    if family == "matched":
                        powers = np.zeros((3, dim), dtype=int)
                        powers[0, 0] = 1
                        powers[1, min(1, dim - 1)] = 2
                        powers[2] = 1
                        poly = {"kind": "poly", "params": {"powers": powers.tolist(), "coeffs": [0.5, -0.2, 0.1], "shift": (0.5 * np.ones(dim) if measure == "uniform" else np.zeros(dim)).tolist()}}
                        osc = {"kind": "osc", "params": {"freq": (0.8 + 0.2 * np.arange(dim)).tolist(), "phase": 0.3 * instance_id, "amp": 0.25}}
                        spec = FunctionSpec("mixed", {"terms": [(1.0, poly), (1.0, osc)]})
                    elif family == "oscillatory":
                        freq = np.linspace(4.5 + 1.5 * instance_id, 5.5 + 1.5 * instance_id + 0.4 * (dim - 1), dim)
                        osc = {"kind": "osc", "params": {"freq": freq.tolist(), "phase": 0.4 + 0.15 * instance_id, "amp": 1.0}}
                        poly = {"kind": "poly", "params": {"powers": np.eye(dim, dtype=int).tolist(), "coeffs": (0.1 * np.ones(dim)).tolist(), "shift": (0.5 * np.ones(dim) if measure == "uniform" else np.zeros(dim)).tolist()}}
                        spec = FunctionSpec("mixed", {"terms": [(1.0, osc), (1.0, poly)]})
                    elif family == "bump":
                        center = rng.uniform(0.1, 0.9, size=dim) if measure == "uniform" else rng.normal(0.0, 1.1, size=dim)
                        width = 0.06 if instance_id == 0 else 0.11
                        bump = {"kind": "bump", "params": {"center": center.tolist(), "width": width, "amp": 1.1}}
                        osc = {"kind": "osc", "params": {"freq": (1.0 + 0.2 * np.arange(dim)).tolist(), "phase": 0.1, "amp": 0.2}}
                        spec = FunctionSpec("mixed", {"terms": [(1.0, bump), (1.0, osc)]})
                    else:
                        axis = instance_id % dim
                        loc = 0.35 if measure == "uniform" else -0.15
                        kink = {"kind": "kink", "params": {"axis": axis, "loc": loc, "power": 1.0 if instance_id == 0 else 0.6, "amp": 1.0}}
                        poly = {"kind": "poly", "params": {"powers": np.zeros((1, dim), dtype=int).tolist(), "coeffs": [0.15], "shift": np.zeros(dim).tolist()}}
                        spec = FunctionSpec("mixed", {"terms": [(1.0, kink), (1.0, poly)]})
                    records.append(
                        {
                            "task_id": f"target_{measure}_d{dim}_{family}_{instance_id}",
                            "split": "target",
                            "measure": measure,
                            "dimension": dim,
                            "family": family,
                            "subtype": family,
                            "instance_id": instance_id,
                            "probe_group_id": "",
                            "severity": None,
                            "spec": spec,
                        }
                    )
    return records


def get_benchmark():
    cache_path = DATA_DIR / "benchmark.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    probes = []
    for measure in MEASURES:
        probes.extend(make_probe_bank(measure))
    targets = make_targets()
    benchmark = {"probes": probes, "targets": targets}
    with open(cache_path, "wb") as fh:
        pickle.dump(benchmark, fh)
    return benchmark


def reference_table():
    ensure_dirs()
    benchmark = get_benchmark()
    rows = []
    for group in [benchmark["probes"], benchmark["targets"]]:
        for rec in group:
            rows.append(
                {
                    "task_id": rec["task_id"],
                    "measure": rec["measure"],
                    "dimension": rec["dimension"],
                    "family": rec["family"],
                    "split": rec["split"],
                    "reference_integral": integral_spec(rec["spec"], rec["measure"], rec["dimension"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "reference_integrals.csv", index=False)
    dependency_manifest = write_dependency_manifest()
    manifest = {
        "available_cpu_cores": 2,
        "available_gpu_count": 0,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "python_executable": os.sys.executable,
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "kernel_families": KERNELS,
        "log_lengthscales": LOG_LENGTHSCALES.tolist(),
        "dependency_manifest_path": str(DEPENDENCY_MANIFEST_PATH.relative_to(ROOT)),
        "dependencies": dependency_manifest["dependencies"],
    }
    with open(DATA_DIR / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return df


def get_reference_lookup():
    path = DATA_DIR / "reference_integrals.csv"
    if not path.exists():
        reference_table()
    df = pd.read_csv(path)
    return dict(zip(df["task_id"], df["reference_integral"]))


def build_instance_row(rec, method, nominal_level, mean_estimate, variance_estimate, inflation_factor, runtime_sec, selected_kernel, selected_log_lengthscale, roughness, fill_distance, cond_proxy, budget_n, seed, fallback_to_global=0):
    truth = get_reference_lookup()[rec["task_id"]]
    z = stats.norm.ppf(0.5 + nominal_level / 2.0)
    std = max(variance_estimate, 1e-12) ** 0.5
    lower = mean_estimate - z * std
    upper = mean_estimate + z * std
    return {
        "task_id": rec["task_id"],
        "split": rec["split"],
        "measure": rec["measure"],
        "dimension": rec["dimension"],
        "family": rec["family"],
        "instance_id": rec["instance_id"],
        "budget_n": budget_n,
        "seed": seed,
        "method": method,
        "nominal_level": nominal_level,
        "mean_estimate": mean_estimate,
        "variance_estimate": variance_estimate,
        "inflation_factor": inflation_factor,
        "ci_lower": lower,
        "ci_upper": upper,
        "abs_error": abs(mean_estimate - truth),
        "covered": int(lower <= truth <= upper),
        "interval_width": upper - lower,
        "wis": wis(truth, lower, upper, nominal_level),
        "runtime_sec": runtime_sec,
        "selected_kernel": selected_kernel,
        "selected_log_lengthscale": selected_log_lengthscale,
        "roughness_score": roughness,
        "fill_distance_proxy": fill_distance,
        "cond_proxy": cond_proxy,
        "probe_group_id": rec["probe_group_id"],
        "fallback_to_global": fallback_to_global,
    }


def run_single_fit(rec, budget_n, seed, method):
    x = design_points(rec["measure"], rec["dimension"], budget_n, seed + 1000 * rec["dimension"] + 7 * rec["instance_id"])
    y = evaluate_function(rec["spec"], x)
    t0 = time.time()
    if method == "uncalibrated":
        fit = solve_gp_bq(x, y, rec["measure"], bayes_sard=False)
    elif method == "bayes_sard":
        fit = solve_gp_bq(x, y, rec["measure"], bayes_sard=True)
    elif method == "kernel_marginalization":
        fit = solve_kernel_marginalization(x, y, rec["measure"])
        fit["kind"] = "mixture21"
        fit["log_lengthscale"] = np.nan
        fit["amplitude"] = np.nan
        fit["q"] = np.nan
        fit["qq"] = np.nan
        fit["lml"] = np.nan
        fit["used_jitter"] = np.nan
        fit["k"] = np.eye(len(x))
    else:
        raise ValueError(method)
    runtime = time.time() - t0
    return {
        "task_id": rec["task_id"],
        "split": rec["split"],
        "measure": rec["measure"],
        "dimension": rec["dimension"],
        "family": rec["family"],
        "instance_id": rec["instance_id"],
        "seed": seed,
        "budget_n": budget_n,
        "method": method,
        "mean_estimate": fit["mean"],
        "base_variance": fit["variance"],
        "runtime_sec": runtime,
        "selected_kernel": fit["kind"],
        "selected_log_lengthscale": fit["log_lengthscale"],
        "roughness_score": roughness_score(x, y),
        "fill_distance_proxy": fill_distance_proxy(x),
        "cond_proxy": float(np.linalg.cond(fit["k"])) if isinstance(fit["k"], np.ndarray) else np.nan,
        "probe_group_id": rec["probe_group_id"],
    }


def fit_rows_to_raw_rows(fit_rows):
    output = []
    for fit in fit_rows:
        rec = {
            "task_id": fit["task_id"],
            "split": fit["split"],
            "measure": fit["measure"],
            "dimension": fit["dimension"],
            "family": fit["family"],
            "instance_id": fit["instance_id"],
            "probe_group_id": fit["probe_group_id"],
        }
        for nominal in NOMINAL_LEVELS:
            output.append(
                build_instance_row(
                    rec=rec,
                    method=fit["method"],
                    nominal_level=nominal,
                    mean_estimate=fit["mean_estimate"],
                    variance_estimate=fit["base_variance"],
                    inflation_factor=1.0,
                    runtime_sec=fit["runtime_sec"],
                    selected_kernel=fit["selected_kernel"],
                    selected_log_lengthscale=fit["selected_log_lengthscale"],
                    roughness=fit["roughness_score"],
                    fill_distance=fit["fill_distance_proxy"],
                    cond_proxy=fit["cond_proxy"],
                    budget_n=fit["budget_n"],
                    seed=fit["seed"],
                )
            )
    return pd.DataFrame(output)


def grouped_probe_records(df, measure, nominal_level, budget_filter=None, constant_roughness=False):
    sub = df[(df["split"] == "calibration") & (df["measure"] == measure) & (df["nominal_level"] == nominal_level) & (df["method"] == "uncalibrated")]
    if budget_filter is not None:
        sub = sub[sub["budget_n"].isin(budget_filter)]
    sub = sub.copy()
    sub["z"] = sub["abs_error"] / np.maximum(np.sqrt(sub["variance_estimate"]), 1e-12)
    grouped = (
        sub.groupby(["probe_group_id", "budget_n"], as_index=False)
        .agg(z=("z", "mean"),
             roughness_score=("roughness_score", "mean"),
             n_records=("seed", "count"),
             family=("family", "first"))
    )
    if constant_roughness:
        grouped["roughness_score"] = grouped["roughness_score"].mean()
    return grouped


def empirical_quantile(values, weights, q):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cum = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cum, q, side="left")])


def apply_calibration_rule(rule, base_var, roughness):
    if rule["type"] == "global":
        return base_var * rule["tau"] ** 2, rule["tau"], 0
    if roughness < rule["min_r"] or roughness > rule["max_r"]:
        return base_var * rule["fallback_tau"] ** 2, rule["fallback_tau"], 1
    if rule["type"] == "isotonic":
        tau = float(np.clip(rule["model"].predict([roughness])[0], 1.0, rule["clip"]))
        return base_var * tau**2, tau, 0
    if rule["type"] == "step":
        idx = np.searchsorted(rule["edges"], roughness, side="right") - 1
        idx = int(np.clip(idx, 0, len(rule["taus"]) - 1))
        tau = float(rule["taus"][idx])
        return base_var * tau**2, tau, 0
    raise ValueError(rule["type"])


def evaluate_rule_on_group(rule, group_df, nominal_level):
    rows = []
    z = stats.norm.ppf(0.5 + nominal_level / 2.0)
    for _, row in group_df.iterrows():
        var, tau, fallback = apply_calibration_rule(rule, 1.0, row["roughness_score"])
        std = math.sqrt(var)
        covered = int(row["z"] <= z * std)
        width = 2.0 * z * std
        penalty = 2.0 / (1.0 - nominal_level) * max(row["z"] - z * std, 0.0)
        rows.append({"covered": covered, "wis": width + penalty, "tau": tau, "fallback": fallback})
    return pd.DataFrame(rows)


def fit_candidate_rules(grouped, nominal_level):
    q = nominal_level
    tau_global = empirical_quantile(grouped["z"], grouped["n_records"], q)
    tau_global = max(1.0, tau_global / stats.norm.ppf(0.5 + nominal_level / 2.0))
    candidates = [{"type": "global", "tau": tau_global, "clip": tau_global, "fallback_tau": tau_global, "min_r": -np.inf, "max_r": np.inf}]
    for clip in [2.0, 2.5, 3.0]:
        iso = IsotonicRegression(y_min=1.0, y_max=clip, increasing=True, out_of_bounds="clip")
        target = np.clip(grouped["z"] / stats.norm.ppf(0.5 + nominal_level / 2.0), 1.0, clip)
        iso.fit(grouped["roughness_score"], target, sample_weight=grouped["n_records"])
        candidates.append(
            {
                "type": "isotonic",
                "model": iso,
                "clip": clip,
                "fallback_tau": tau_global,
                "min_r": float(grouped["roughness_score"].min()),
                "max_r": float(grouped["roughness_score"].max()),
            }
        )
    sorted_grouped = grouped.sort_values("roughness_score").reset_index(drop=True)
    for bins in [2, 3, 4]:
        for clip in [2.0, 2.5, 3.0]:
            quantiles = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(sorted_grouped["roughness_score"], quantiles)
            taus = []
            prev = 1.0
            for i in range(bins):
                lo, hi = edges[i], edges[i + 1] + (1e-12 if i == bins - 1 else 0.0)
                mask = (sorted_grouped["roughness_score"] >= lo) & (sorted_grouped["roughness_score"] < hi)
                subset = sorted_grouped[mask]
                raw_tau = tau_global if len(subset) == 0 else empirical_quantile(subset["z"], subset["n_records"], q) / stats.norm.ppf(0.5 + nominal_level / 2.0)
                tau = float(np.clip(max(prev, raw_tau), 1.0, clip))
                taus.append(tau)
                prev = tau
            candidates.append(
                {
                    "type": "step",
                    "edges": edges,
                    "taus": taus,
                    "clip": clip,
                    "fallback_tau": tau_global,
                    "min_r": float(grouped["roughness_score"].min()),
                    "max_r": float(grouped["roughness_score"].max()),
                }
            )
    return candidates


def select_rule(grouped, nominal_level):
    if len(grouped) == 0:
        raise ValueError("No grouped calibration rows available")
    probe_ids = grouped["probe_group_id"].str.rsplit("_", n=1).str[0].unique()
    scores = []
    for candidate in fit_candidate_rules(grouped, nominal_level):
        fold_rows = []
        for probe in probe_ids:
            train = grouped[~grouped["probe_group_id"].str.startswith(probe)]
            valid = grouped[grouped["probe_group_id"].str.startswith(probe)]
            if len(train) == 0 or len(valid) == 0:
                continue
            retrained = fit_candidate_rules(train, nominal_level)
            matched = None
            for cand in retrained:
                if cand["type"] != candidate["type"]:
                    continue
                if cand["type"] == "global":
                    matched = cand
                elif cand["type"] == "isotonic" and cand["clip"] == candidate["clip"]:
                    matched = cand
                elif cand["type"] == "step" and cand["clip"] == candidate["clip"] and len(cand["taus"]) == len(candidate["taus"]):
                    matched = cand
                if matched is not None:
                    break
            if matched is None:
                continue
            eval_df = evaluate_rule_on_group(matched, valid, nominal_level)
            fold_rows.append({"coverage_gap": abs(eval_df["covered"].mean() - nominal_level), "wis": eval_df["wis"].mean()})
        if not fold_rows:
            continue
        score = pd.DataFrame(fold_rows).mean().to_dict()
        score["candidate"] = candidate
        scores.append(score)
    if not scores:
        fallback = fit_candidate_rules(grouped, nominal_level)
        return fallback[0], pd.DataFrame()
    feasible = [s for s in scores if s["coverage_gap"] <= 0.05]
    pool = feasible if feasible else scores
    best = min(pool, key=lambda s: (s["wis"], s["coverage_gap"]))
    final_candidates = fit_candidate_rules(grouped, nominal_level)
    final = None
    for cand in final_candidates:
        bc = best["candidate"]
        if cand["type"] != bc["type"]:
            continue
        if cand["type"] == "global":
            final = cand
            break
        if cand["type"] == "isotonic" and cand["clip"] == bc["clip"]:
            final = cand
            break
        if cand["type"] == "step" and cand["clip"] == bc["clip"] and len(cand["taus"]) == len(bc["taus"]):
            final = cand
            break
    return final, pd.DataFrame(scores)


def rule_from_filtered_grouped(grouped, nominal_level, force_type=None):
    if len(grouped) == 0:
        return None
    if force_type is None:
        rule, _ = select_rule(grouped, nominal_level)
        return rule
    candidates = [c for c in fit_candidate_rules(grouped, nominal_level) if c["type"] == force_type]
    return candidates[0] if candidates else None


def apply_rule_to_base_rows(base_rows, rule, nominal_level, method_name):
    rows = []
    if rule is None:
        return rows
    for _, base in base_rows.iterrows():
        rec = {
            "task_id": base["task_id"],
            "split": base["split"],
            "measure": base["measure"],
            "dimension": int(base["dimension"]),
            "family": base["family"],
            "instance_id": int(base["instance_id"]),
            "probe_group_id": base["probe_group_id"],
        }
        variance, tau, fallback = apply_calibration_rule(rule, base["variance_estimate"], base["roughness_score"])
        rows.append(
            build_instance_row(
                rec,
                method_name,
                nominal_level,
                base["mean_estimate"],
                variance,
                tau,
                base["runtime_sec"],
                base["selected_kernel"],
                base["selected_log_lengthscale"],
                base["roughness_score"],
                base["fill_distance_proxy"],
                base["cond_proxy"],
                int(base["budget_n"]),
                int(base["seed"]),
                fallback,
            )
        )
    return rows


def get_hard_kernel_subset():
    benchmark = get_benchmark()
    subset = []
    for rec in benchmark["targets"]:
        if rec["dimension"] not in [1, 2]:
            continue
        if rec["family"] not in ["oscillatory", "bump", "kinked"]:
            continue
        if rec["instance_id"] != 0:
            continue
        subset.append(rec)
    return subset


def run_core(smoke=False):
    ensure_dirs()
    benchmark = get_benchmark()
    records = benchmark["targets"] + benchmark["probes"]
    if smoke:
        records = [r for r in records if r["measure"] == "uniform" and r["dimension"] == 1]
        records = [r for r in records if (r["split"] == "target" and r["instance_id"] == 0 and r["family"] in ["matched", "oscillatory"]) or (r["split"] != "target" and r["instance_id"] < 2)]
    tasks = []
    for rec in records:
        for budget in BUDGETS:
            for seed in SEEDS:
                tasks.append((rec, budget, seed, "uncalibrated"))
                if rec["split"] == "target":
                    tasks.append((rec, budget, seed, "bayes_sard"))
    rows = Parallel(n_jobs=2, verbose=0)(delayed(run_single_fit)(*task) for task in tasks)
    df = fit_rows_to_raw_rows(rows)
    block = "smoke" if smoke else "core"
    for seed in SEEDS:
        df[df["seed"] == seed].to_csv(RAW_DIR / f"{block}_{seed}.csv", index=False)
    summary = {"experiment": block, "n_rows": int(len(df)), "runtime_minutes": float(df["runtime_sec"].sum() / 60.0)}
    with open(ROOT / "exp" / ("smoke_test" if smoke else "core_fits") / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return df


def load_core_rows():
    frames = []
    for block in ["core", "smoke"]:
        for seed in SEEDS:
            path = RAW_DIR / f"{block}_{seed}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("Run core fits first")
    df = pd.concat(frames, ignore_index=True)
    return df


def run_main_recalibration():
    ensure_dirs()
    df = load_core_rows()
    rows = []
    selection_rows = []
    for nominal in NOMINAL_LEVELS:
        for measure in MEASURES:
            grouped = grouped_probe_records(df, measure, nominal)
            selected_rule, cv_scores = select_rule(grouped, nominal)
            cv_scores["measure"] = measure
            cv_scores["nominal_level"] = nominal
            selection_rows.append(cv_scores)
            for rule_type in ["global", "isotonic", "step"]:
                if rule_type == "global":
                    candidates = [fit_candidate_rules(grouped, nominal)[0]]
                elif rule_type == "isotonic":
                    candidates = [c for c in fit_candidate_rules(grouped, nominal) if c["type"] == "isotonic"][:1]
                else:
                    candidates = [c for c in fit_candidate_rules(grouped, nominal) if c["type"] == "step"][:1]
                rule = candidates[0]
                method_name = {"global": "global_scaling", "isotonic": "isotonic_inflation", "step": "step_inflation"}[rule_type]
                relevant = df[(df["method"] == "uncalibrated") & (df["measure"] == measure) & (df["nominal_level"] == nominal)]
                for _, base in relevant.iterrows():
                    rec = {
                        "task_id": base["task_id"],
                        "split": base["split"],
                        "measure": base["measure"],
                        "dimension": int(base["dimension"]),
                        "family": base["family"],
                        "instance_id": int(base["instance_id"]),
                        "probe_group_id": base["probe_group_id"],
                    }
                    variance, tau, fallback = apply_calibration_rule(rule, base["variance_estimate"], base["roughness_score"])
                    rows.append(
                        build_instance_row(
                            rec,
                            method_name,
                            nominal,
                            base["mean_estimate"],
                            variance,
                            tau,
                            base["runtime_sec"],
                            base["selected_kernel"],
                            base["selected_log_lengthscale"],
                            base["roughness_score"],
                            base["fill_distance_proxy"],
                            base["cond_proxy"],
                            int(base["budget_n"]),
                            int(base["seed"]),
                            fallback,
                        )
                    )
                if selected_rule["type"] == rule["type"]:
                    relevant_targets = df[(df["method"] == "uncalibrated") & (df["measure"] == measure) & (df["nominal_level"] == nominal)]
                    for _, base in relevant_targets.iterrows():
                        rec = {
                            "task_id": base["task_id"],
                            "split": base["split"],
                            "measure": base["measure"],
                            "dimension": int(base["dimension"]),
                            "family": base["family"],
                            "instance_id": int(base["instance_id"]),
                            "probe_group_id": base["probe_group_id"],
                        }
                        variance, tau, fallback = apply_calibration_rule(selected_rule, base["variance_estimate"], base["roughness_score"])
                        rows.append(
                            build_instance_row(
                                rec,
                                "selected_external_rule",
                                nominal,
                                base["mean_estimate"],
                                variance,
                                tau,
                                base["runtime_sec"],
                                base["selected_kernel"],
                                base["selected_log_lengthscale"],
                                base["roughness_score"],
                                base["fill_distance_proxy"],
                                base["cond_proxy"],
                                int(base["budget_n"]),
                                int(base["seed"]),
                                fallback,
                            )
                        )
    out = pd.DataFrame(rows)
    for seed in SEEDS:
        out[out["seed"] == seed].to_csv(RAW_DIR / f"main_{seed}.csv", index=False)
    selection_df = pd.concat(selection_rows, ignore_index=True)
    selection_df.to_csv(SUMMARY_DIR / "probe_cv_scores.csv", index=False)
    summary = summarize(pd.concat([df, out], ignore_index=True))
    with open(ROOT / "exp" / "main_recalibration" / "results.json", "w") as fh:
        json.dump(
            {
                "experiment": "main_recalibration",
                "n_rows": int(len(out)),
                "source_files": {
                    "raw_rows": [str((RAW_DIR / f"main_{seed}.csv").relative_to(ROOT)) for seed in SEEDS],
                    "cv_scores_csv": str((SUMMARY_DIR / "probe_cv_scores.csv").relative_to(ROOT)),
                },
                "aggregates": {
                    "target_nominal_95": method_metric_snapshot(summary, split="target", nominal_level=0.95),
                    "heldout_probe_nominal_95": method_metric_snapshot(summary, split="test", nominal_level=0.95),
                },
            },
            fh,
            indent=2,
        )
    return out


def get_selected_rules(df, nominal_level=0.95):
    selected = {}
    for measure in MEASURES:
        grouped = grouped_probe_records(df, measure, nominal_level)
        rule, _ = select_rule(grouped, nominal_level)
        selected[measure] = rule
    return selected


def run_ablations():
    ensure_dirs()
    df = load_all_raw()
    ablation_rows = []
    sensitivity_rows = []
    halfbank_summaries = []
    selected_rules_95 = get_selected_rules(df, nominal_level=0.95)
    for nominal in NOMINAL_LEVELS:
        for measure in MEASURES:
            full_grouped = grouped_probe_records(df, measure, nominal)
            reduced_grouped = grouped_probe_records(df, measure, nominal, budget_filter=[32])
            constant_grouped = grouped_probe_records(df, measure, nominal, constant_roughness=True)
            configs = [
                ("ablation_global_only", full_grouped),
                ("ablation_reduced_calibration", reduced_grouped),
                ("ablation_no_roughness", constant_grouped),
            ]
            for name, grouped in configs:
                relevant = df[(df["method"] == "uncalibrated") & (df["measure"] == measure) & (df["nominal_level"] == nominal)]
                rule = rule_from_filtered_grouped(grouped, nominal)
                ablation_rows.extend(apply_rule_to_base_rows(relevant, rule, nominal, name))
            calibration = full_grouped.copy()
            iso = [c for c in fit_candidate_rules(calibration, nominal) if c["type"] == "isotonic"][0]
            step = [c for c in fit_candidate_rules(calibration, nominal) if c["type"] == "step"][0]
            for name, rule in [("ablation_isotonic_only", iso), ("ablation_step_only", step)]:
                relevant = df[(df["method"] == "uncalibrated") & (df["measure"] == measure) & (df["nominal_level"] == nominal)]
                ablation_rows.extend(apply_rule_to_base_rows(relevant, rule, nominal, name))

            if nominal == 0.95:
                base_rows = df[(df["method"] == "uncalibrated") & (df["measure"] == measure) & (df["nominal_level"] == nominal)]
                rng = np.random.default_rng(700 + (0 if measure == "uniform" else 1))
                probe_bases = sorted(full_grouped["probe_group_id"].str.rsplit("_", n=1).str[0].unique())
                for draw in range(20):
                    keep = set(rng.choice(probe_bases, size=min(36, len(probe_bases)), replace=False))
                    sampled = full_grouped[full_grouped["probe_group_id"].str.rsplit("_", n=1).str[0].isin(keep)]
                    method_rules = {
                        "global_scaling": rule_from_filtered_grouped(sampled, nominal, force_type="global"),
                        "selected_external_rule": rule_from_filtered_grouped(sampled, nominal, force_type=selected_rules_95[measure]["type"]),
                    }
                    for base_method_name, rule in method_rules.items():
                        method_name = f"halfbank_{base_method_name}_draw_{draw:02d}"
                        rows = apply_rule_to_base_rows(base_rows[base_rows["split"] == "target"], rule, nominal, method_name)
                        sensitivity_rows.extend(rows)
                        row_df = pd.DataFrame(rows)
                        if len(row_df):
                            halfbank_summaries.append(
                                {
                                    "measure": measure,
                                    "draw": draw,
                                    "method": base_method_name,
                                    "coverage": row_df["covered"].mean(),
                                    "wis": row_df["wis"].mean(),
                                }
                            )
                full_target_rows = base_rows[base_rows["split"] == "target"]
                for base_method_name, rule in {
                    "global_scaling": rule_from_filtered_grouped(full_grouped, nominal, force_type="global"),
                    "selected_external_rule": selected_rules_95[measure],
                }.items():
                    full_rows = apply_rule_to_base_rows(full_target_rows, rule, nominal, f"fullbank_{base_method_name}")
                    full_df = pd.DataFrame(full_rows)
                    if len(full_df):
                        halfbank_summaries.append(
                            {
                                "measure": measure,
                                "draw": -1,
                                "method": base_method_name,
                                "coverage": full_df["covered"].mean(),
                                "wis": full_df["wis"].mean(),
                            }
                        )

                family_map = {"poly": ["matched"], "osc": ["oscillatory"], "loc": ["bump", "kinked"]}
                for removed_family, target_families in family_map.items():
                    filtered = full_grouped[full_grouped["family"] != removed_family]
                    rule = rule_from_filtered_grouped(filtered, nominal)
                    method_name = f"leave_out_{removed_family}"
                    target_subset = base_rows[(base_rows["split"] == "target") & (base_rows["family"].isin(target_families))]
                    sensitivity_rows.extend(apply_rule_to_base_rows(target_subset, rule, nominal, method_name))

                osc_subset = full_grouped[~((full_grouped["family"] == "osc") & (full_grouped["roughness_score"] >= full_grouped[full_grouped["family"] == "osc"]["roughness_score"].quantile(0.75)))]
                bump_subset = full_grouped[~((full_grouped["family"] == "loc") & (full_grouped["roughness_score"] >= full_grouped[full_grouped["family"] == "loc"]["roughness_score"].quantile(0.75)))]
                sensitivity_rows.extend(
                    apply_rule_to_base_rows(
                        base_rows[(base_rows["split"] == "target") & (base_rows["family"] == "oscillatory")],
                        rule_from_filtered_grouped(osc_subset, nominal),
                        nominal,
                        "severity_holdout_osc",
                    )
                )
                sensitivity_rows.extend(
                    apply_rule_to_base_rows(
                        base_rows[(base_rows["split"] == "target") & (base_rows["family"] == "bump")],
                        rule_from_filtered_grouped(bump_subset, nominal),
                        nominal,
                        "severity_holdout_bump",
                    )
                )
    out = pd.DataFrame(ablation_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    for seed in SEEDS:
        out[out["seed"] == seed].to_csv(RAW_DIR / f"ablations_{seed}.csv", index=False)
        if len(sensitivity_df):
            sensitivity_df[sensitivity_df["seed"] == seed].to_csv(RAW_DIR / f"sensitivity_{seed}.csv", index=False)
    if halfbank_summaries:
        pd.DataFrame(halfbank_summaries).to_csv(SUMMARY_DIR / "halfbank_summary.csv", index=False)
    ablation_summary = summarize(pd.concat([df, out, sensitivity_df], ignore_index=True))
    table2 = build_table2(ablation_summary)
    with open(ROOT / "exp" / "ablations" / "results.json", "w") as fh:
        json.dump(
            {
                "experiment": "ablations",
                "n_rows": int(len(out)),
                "sensitivity_rows": int(len(sensitivity_df)),
                "source_files": {
                    "raw_ablation_rows": [str((RAW_DIR / f"ablations_{seed}.csv").relative_to(ROOT)) for seed in SEEDS],
                    "raw_sensitivity_rows": [str((RAW_DIR / f"sensitivity_{seed}.csv").relative_to(ROOT)) for seed in SEEDS],
                    "halfbank_summary_csv": str((SUMMARY_DIR / "halfbank_summary.csv").relative_to(ROOT)),
                    "table2_csv": str((SUMMARY_DIR / "table2_ablations.csv").relative_to(ROOT)),
                },
                "aggregates": {
                    "table2_nominal_95": table2.to_dict(orient="records"),
                },
            },
            fh,
            indent=2,
        )
    return out


def make_matched_generator_records():
    probe_records = []
    target_records = []
    severity_levels = [1, 2, 3, 4]
    osc_bases = [2.0, 3.5, 5.0, 6.5]
    bump_widths = [0.22, 0.15, 0.10, 0.06]
    for measure in MEASURES:
        for dim in [1, 2]:
            for mechanism in ["oscillatory", "bump"]:
                for severity_level in severity_levels:
                    for probe_rep in range(3):
                        if mechanism == "oscillatory":
                            base = osc_bases[severity_level - 1] + 0.25 * probe_rep
                            freq = np.linspace(base, base + 0.35 * (dim - 1), dim)
                            spec = FunctionSpec("osc", {"freq": freq.tolist(), "phase": 0.1 * probe_rep, "amp": 1.0})
                        else:
                            width = bump_widths[severity_level - 1]
                            center = np.full(dim, 0.25 + 0.2 * probe_rep) if measure == "uniform" else np.full(dim, -0.5 + 0.5 * probe_rep)
                            spec = FunctionSpec("bump", {"center": center.tolist(), "width": width, "amp": 1.0})
                        probe_records.append(
                            {
                                "task_id": f"mg_probe_{measure}_d{dim}_{mechanism}_s{severity_level}_r{probe_rep}",
                                "split": "matched_probe",
                                "measure": measure,
                                "dimension": dim,
                                "family": mechanism,
                                "instance_id": probe_rep,
                                "probe_group_id": f"mg_{measure}_d{dim}_{mechanism}_s{severity_level}_r{probe_rep}",
                                "severity_level": severity_level,
                                "spec": spec,
                            }
                        )
                    if mechanism == "oscillatory":
                        base = osc_bases[severity_level - 1] + 0.8
                        freq = np.linspace(base, base + 0.45 * (dim - 1), dim)
                        spec = FunctionSpec(
                            "mixed",
                            {
                                "terms": [
                                    (1.0, {"kind": "osc", "params": {"freq": freq.tolist(), "phase": 0.35, "amp": 1.0}}),
                                    (1.0, {"kind": "poly", "params": {"powers": np.eye(dim, dtype=int).tolist(), "coeffs": (0.08 * np.ones(dim)).tolist(), "shift": (0.5 * np.ones(dim) if measure == "uniform" else np.zeros(dim)).tolist()}}),
                                ]
                            },
                        )
                    else:
                        width = bump_widths[severity_level - 1]
                        center = np.full(dim, 0.35) if measure == "uniform" else np.full(dim, 0.25)
                        spec = FunctionSpec(
                            "mixed",
                            {
                                "terms": [
                                    (1.0, {"kind": "bump", "params": {"center": center.tolist(), "width": width, "amp": 1.1}}),
                                    (1.0, {"kind": "osc", "params": {"freq": (1.0 + 0.2 * np.arange(dim)).tolist(), "phase": 0.15, "amp": 0.15}}),
                                ]
                            },
                        )
                    target_records.append(
                        {
                            "task_id": f"mg_target_{measure}_d{dim}_{mechanism}_s{severity_level}",
                            "split": "matched_target",
                            "measure": measure,
                            "dimension": dim,
                            "family": mechanism,
                            "instance_id": severity_level,
                            "probe_group_id": "",
                            "severity_level": severity_level,
                            "spec": spec,
                        }
                    )
    return probe_records, target_records


def run_matched_generator():
    ensure_dirs()
    probe_records, target_records = make_matched_generator_records()
    tasks = []
    for rec in probe_records + target_records:
        for budget in BUDGETS:
            for seed in SEEDS:
                tasks.append((rec, budget, seed, "uncalibrated"))
    fits = Parallel(n_jobs=2, verbose=0)(delayed(run_single_fit)(*task) for task in tasks)
    enriched = []
    for fit in fits:
        rec = next(r for r in (probe_records + target_records) if r["task_id"] == fit["task_id"])
        truth = integral_spec(rec["spec"], rec["measure"], rec["dimension"])
        enriched.append(
            {
                **fit,
                "severity_level": rec["severity_level"],
                "truth": truth,
                "z": abs(fit["mean_estimate"] - truth) / max(math.sqrt(fit["base_variance"]), 1e-12),
            }
        )
    fit_df = pd.DataFrame(enriched)
    summary_rows = []
    corr_rows = []
    nominal = 0.95
    z_quant = stats.norm.ppf(0.5 + nominal / 2.0)
    for measure in MEASURES:
        for dim in [1, 2]:
            for mechanism in ["oscillatory", "bump"]:
                probe_df = fit_df[
                    (fit_df["split"] == "matched_probe")
                    & (fit_df["measure"] == measure)
                    & (fit_df["dimension"] == dim)
                    & (fit_df["family"] == mechanism)
                ].copy()
                grouped = (
                    probe_df.groupby(["probe_group_id", "budget_n"], as_index=False)
                    .agg(
                        z=("z", "mean"),
                        roughness_score=("roughness_score", "mean"),
                        n_records=("seed", "count"),
                        severity_level=("severity_level", "first"),
                    )
                )
                target_df = fit_df[
                    (fit_df["split"] == "matched_target")
                    & (fit_df["measure"] == measure)
                    & (fit_df["dimension"] == dim)
                    & (fit_df["family"] == mechanism)
                ].copy()
                for severity_level in [1, 2, 3, 4]:
                    supported_grouped = grouped[grouped["severity_level"] <= severity_level]
                    rule, _ = select_rule(supported_grouped, nominal)
                    probe_z_mean = float(supported_grouped[supported_grouped["severity_level"] == severity_level]["z"].mean())
                    sev_targets = target_df[target_df["severity_level"] == severity_level].copy()
                    covereds = []
                    fallbacks = []
                    widths = []
                    wis_vals = []
                    for _, row in sev_targets.iterrows():
                        variance, _, fallback = apply_calibration_rule(rule, row["base_variance"], row["roughness_score"])
                        std = math.sqrt(max(variance, 1e-12))
                        lower = row["mean_estimate"] - z_quant * std
                        upper = row["mean_estimate"] + z_quant * std
                        covereds.append(int(lower <= row["truth"] <= upper))
                        fallbacks.append(fallback)
                        widths.append(upper - lower)
                        wis_vals.append(wis(row["truth"], lower, upper, nominal))
                    summary_rows.append(
                        {
                            "measure": measure,
                            "dimension": dim,
                            "mechanism": mechanism,
                            "severity_level": severity_level,
                            "selected_rule_type": rule["type"],
                            "grouped_probe_z": probe_z_mean,
                            "target_coverage": float(np.mean(covereds)),
                            "target_coverage_gap": abs(float(np.mean(covereds)) - nominal),
                            "fallback_rate": float(np.mean(fallbacks)),
                            "mean_width": float(np.mean(widths)),
                            "wis": float(np.mean(wis_vals)),
                        }
                    )
                block = pd.DataFrame(
                    [r for r in summary_rows if r["measure"] == measure and r["dimension"] == dim and r["mechanism"] == mechanism]
                ).sort_values("severity_level")
                rho, pval = stats.spearmanr(block["grouped_probe_z"], block["target_coverage_gap"])
                corr_rows.append(
                    {
                        "measure": measure,
                        "dimension": dim,
                        "mechanism": mechanism,
                        "spearman_rho": float(rho),
                        "p_value": float(pval),
                    }
                )
    summary_df = pd.DataFrame(summary_rows)
    corr_df = pd.DataFrame(corr_rows)
    summary_df.to_csv(SUMMARY_DIR / "matched_generator_summary.csv", index=False)
    corr_df.to_csv(SUMMARY_DIR / "matched_generator_correlations.csv", index=False)
    with open(ROOT / "exp" / "matched_generator" / "results.json", "w") as fh:
        json.dump(
            {
                "experiment": "matched_generator",
                "summary_rows": int(len(summary_df)),
                "correlation_rows": int(len(corr_df)),
                "source_files": {
                    "summary_csv": str((SUMMARY_DIR / "matched_generator_summary.csv").relative_to(ROOT)),
                    "correlations_csv": str((SUMMARY_DIR / "matched_generator_correlations.csv").relative_to(ROOT)),
                },
                "aggregates": {
                    "summary": summary_df.to_dict(orient="records"),
                    "correlations": corr_df.to_dict(orient="records"),
                },
            },
            fh,
            indent=2,
        )
    return summary_df


def run_kernel_subset():
    ensure_dirs()
    subset = get_hard_kernel_subset()
    tasks = []
    for rec in subset:
        for budget in BUDGETS:
            for seed in SEEDS:
                tasks.append((rec, budget, seed, "kernel_marginalization"))
    rows = Parallel(n_jobs=2, verbose=0)(delayed(run_single_fit)(*task) for task in tasks)
    out = fit_rows_to_raw_rows(rows)
    for seed in SEEDS:
        out[out["seed"] == seed].to_csv(RAW_DIR / f"kernel_{seed}.csv", index=False)
    kernel_summary = summarize(out)
    nominal95 = kernel_summary[kernel_summary["nominal_level"] == 0.95]
    with open(ROOT / "exp" / "kernel_marginalization" / "results.json", "w") as fh:
        json.dump(
            {
                "experiment": "kernel_marginalization",
                "n_rows": int(len(out)),
                "source_files": {
                    "raw_rows": [str((RAW_DIR / f"kernel_{seed}.csv").relative_to(ROOT)) for seed in SEEDS],
                },
                "aggregates": {
                    "hard_case_nominal_95": nominal95.to_dict(orient="records"),
                },
            },
            fh,
            indent=2,
        )
    return out


def load_all_raw():
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No raw result files found")
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    summary = (
        df.groupby(["method", "split", "measure", "dimension", "family", "budget_n", "nominal_level"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            coverage_gap=("covered", lambda s: abs(s.mean() - float(df.loc[s.index[0], "nominal_level"]))),
            mean_width=("interval_width", "mean"),
            wis=("wis", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            fallback_rate=("fallback_to_global", "mean"),
        )
    )
    uncal = summary[summary["method"] == "uncalibrated"][["split", "measure", "dimension", "family", "budget_n", "nominal_level", "mean_width"]].rename(columns={"mean_width": "uncal_width"})
    summary = summary.merge(uncal, on=["split", "measure", "dimension", "family", "budget_n", "nominal_level"], how="left")
    summary["width_inflation_vs_uncalibrated"] = summary["mean_width"] / summary["uncal_width"]
    return summary.drop(columns=["uncal_width"])


def method_metric_snapshot(summary_df, split, nominal_level, family_mode="all"):
    sub = summary_df[(summary_df["split"] == split) & (summary_df["nominal_level"] == nominal_level)].copy()
    if family_mode == "misspecified_only":
        sub = sub[sub["family"] != "matched"]
    elif family_mode == "matched_only":
        sub = sub[sub["family"] == "matched"]
    metrics = ["coverage", "coverage_gap", "wis", "mean_width", "width_inflation_vs_uncalibrated", "fallback_rate", "median_abs_error", "mean_runtime_sec"]
    out = {}
    for method, block in sub.groupby("method"):
        out[method] = {metric: float(block[metric].mean()) for metric in metrics if metric in block}
    return out


def bootstrap_pairs(df, method_a, method_b, nominal_level, split="target", family_scope="all", n_boot=2000):
    sub = df[(df["split"] == split) & (df["nominal_level"] == nominal_level)].copy()
    if family_scope == "misspecified_union":
        sub = sub[sub["family"] != "matched"]
    elif family_scope == "matched_controls":
        sub = sub[sub["family"] == "matched"]
    sub = sub[sub["method"].isin([method_a, method_b])]
    task_metrics = (
        sub.groupby(["task_id", "method"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            wis=("wis", "mean"),
            mean_width=("interval_width", "mean"),
            median_abs_error=("abs_error", "median"),
        )
        .pivot(index="task_id", columns="method")
    )
    if task_metrics.empty:
        return {}
    required = []
    for metric in ["coverage", "wis", "mean_width", "median_abs_error"]:
        for method in [method_a, method_b]:
            required.append((metric, method))
    task_metrics = task_metrics.dropna(subset=required)
    if task_metrics.empty:
        return {}
    coverage_a = task_metrics[("coverage", method_a)].to_numpy()
    coverage_b = task_metrics[("coverage", method_b)].to_numpy()
    wis_a = task_metrics[("wis", method_a)].to_numpy()
    wis_b = task_metrics[("wis", method_b)].to_numpy()
    width_a = task_metrics[("mean_width", method_a)].to_numpy()
    width_b = task_metrics[("mean_width", method_b)].to_numpy()
    error_a = task_metrics[("median_abs_error", method_a)].to_numpy()
    error_b = task_metrics[("median_abs_error", method_b)].to_numpy()
    n_tasks = len(task_metrics)
    rng = np.random.default_rng(123)
    samples = rng.integers(0, n_tasks, size=(n_boot, n_tasks))
    cov_a = coverage_a[samples].mean(axis=1)
    cov_b = coverage_b[samples].mean(axis=1)
    diff_df = pd.DataFrame(
        {
            "coverage_diff": cov_a - cov_b,
            "coverage_gap_diff": np.abs(cov_a - nominal_level) - np.abs(cov_b - nominal_level),
            "wis_diff": wis_a[samples].mean(axis=1) - wis_b[samples].mean(axis=1),
            "mean_width_diff": width_a[samples].mean(axis=1) - width_b[samples].mean(axis=1),
            "median_abs_error_diff": np.median(error_a[samples], axis=1) - np.median(error_b[samples], axis=1),
        }
    )
    out = {}
    for col in ["coverage_diff", "coverage_gap_diff", "wis_diff", "mean_width_diff", "median_abs_error_diff"]:
        out[col] = {
            "mean": float(diff_df[col].mean()),
            "ci_lower": float(diff_df[col].quantile(0.025)),
            "ci_upper": float(diff_df[col].quantile(0.975)),
        }
    return out


def bootstrap_group_metric(df, group_cols, nominal_level, n_boot=2000):
    rows = []
    rng = np.random.default_rng(321)
    for keys, group in df.groupby(group_cols):
        task_ids = sorted(group["task_id"].unique())
        if not task_ids:
            continue
        samples = []
        for _ in range(n_boot):
            sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
            sample_df = pd.concat([group[group["task_id"] == task_id] for task_id in sampled], ignore_index=True)
            samples.append(float(sample_df["covered"].mean()))
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rows.append(
            {
                **key_map,
                "coverage": float(group["covered"].mean()),
                "ci_lower": float(np.quantile(samples, 0.025)),
                "ci_upper": float(np.quantile(samples, 0.975)),
                "nominal_level": nominal_level,
            }
        )
    return pd.DataFrame(rows)


def compute_bootstrap_summary(df):
    method_pairs = [
        ("selected_external_rule", "uncalibrated"),
        ("global_scaling", "uncalibrated"),
        ("isotonic_inflation", "uncalibrated"),
        ("step_inflation", "uncalibrated"),
        ("bayes_sard", "uncalibrated"),
        ("selected_external_rule", "global_scaling"),
        ("bayes_sard", "selected_external_rule"),
    ]
    summary = {}
    for nominal_level in NOMINAL_LEVELS:
        nominal_key = f"nominal_{int(100 * nominal_level)}"
        summary[nominal_key] = {}
        for family_scope in ["misspecified_union", "matched_controls"]:
            summary[nominal_key][family_scope] = {}
            for method_a, method_b in method_pairs:
                summary[nominal_key][family_scope][f"{method_a}_vs_{method_b}"] = bootstrap_pairs(
                    df,
                    method_a,
                    method_b,
                    nominal_level,
                    split="target",
                    family_scope=family_scope,
                )
    return summary


def build_figure1_source(df):
    source = df[
        (df["split"] == "target")
        & (df["nominal_level"] == 0.95)
        & (df["method"].isin(["uncalibrated", "global_scaling", "selected_external_rule", "bayes_sard"]))
    ][["task_id", "measure", "family", "method", "covered"]].copy()
    figure1_source = bootstrap_group_metric(source, ["measure", "family", "method"], nominal_level=0.95)
    figure1_source.to_csv(SUMMARY_DIR / "figure1_coverage_bootstrap.csv", index=False)
    return figure1_source


def build_table2(summary):
    target95 = summary[(summary["split"] == "target") & (summary["nominal_level"] == 0.95)].copy()
    rows = []
    notes = {
        "ablation_global_only": "Global-only recalibration baseline.",
        "ablation_isotonic_only": "Forced isotonic roughness map.",
        "ablation_step_only": "Forced monotone stepwise roughness map.",
        "ablation_reduced_calibration": "Calibration restricted to n=32 probe groups.",
        "ablation_no_roughness": "Roughness replaced by a constant.",
        "leave_out_poly": "Polynomial probes removed; evaluated on matched targets.",
        "leave_out_osc": "Oscillatory probes removed; evaluated on oscillatory targets.",
        "leave_out_loc": "Localized probes removed; evaluated on bump and kinked targets.",
        "severity_holdout_osc": "Roughest oscillatory probes removed before calibration.",
        "severity_holdout_bump": "Narrowest bump probes removed before calibration.",
    }
    methods = list(notes)
    for method in methods:
        block = target95[target95["method"] == method]
        if len(block) == 0:
            continue
        rows.append(
            {
                "row_name": method,
                "coverage_95": float(block["coverage"].mean()),
                "wis_95": float(block["wis"].mean()),
                "width_inflation": float(block["width_inflation_vs_uncalibrated"].mean()),
                "fallback_rate": float(block["fallback_rate"].mean()),
                "notes": notes[method],
            }
        )
    table2 = pd.DataFrame(rows)
    table2.to_csv(SUMMARY_DIR / "table2_ablations.csv", index=False)
    return table2


def build_runtime_accounting():
    stage_timings = load_stage_timings()
    completed_blocks = []
    skipped_blocks = []
    rows = []
    for stage_name, planned_hours in PLANNED_STAGE_HOURS.items():
        stage_info = stage_timings.get(stage_name, {})
        status = stage_info.get("status", "missing")
        if status == "completed":
            completed_blocks.append(stage_name)
        else:
            skipped_blocks.append(stage_name)
        rows.append(
            {
                "block": stage_name,
                "planned_wall_clock_hours": planned_hours,
                "actual_wall_clock_hours": float(stage_info.get("wall_clock_seconds", 0.0) / 3600.0),
                "cpu_cores_used": 2,
                "gpu_count": 0,
                "status": status,
                "reason_for_any_pruning": stage_info.get("reason", ""),
            }
        )
    rows.append(
        {
            "block": "total",
            "planned_wall_clock_hours": float(sum(PLANNED_STAGE_HOURS.values())),
            "actual_wall_clock_hours": float(sum(r["actual_wall_clock_hours"] for r in rows)),
            "cpu_cores_used": 2,
            "gpu_count": 0,
            "status": "completed" if not skipped_blocks else "partial",
            "reason_for_any_pruning": "",
        }
    )
    runtime_df = pd.DataFrame(rows)
    runtime_df.to_csv(SUMMARY_DIR / "runtime_accounting.csv", index=False)
    return runtime_df, completed_blocks, skipped_blocks


def make_figures(summary, figure1_source):
    sns.set_theme(style="whitegrid")
    methods = ["uncalibrated", "global_scaling", "selected_external_rule", "bayes_sard"]
    families = ["matched", "oscillatory", "bump", "kinked"]
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 4), sharey=True)
    width = 0.18
    x = np.arange(len(families))
    for ax, measure in zip(np.atleast_1d(axes), MEASURES):
        panel = figure1_source[figure1_source["measure"] == measure]
        for idx, method in enumerate(methods):
            vals = panel[panel["method"] == method].set_index("family").reindex(families)
            ax.bar(x + (idx - 1.5) * width, vals["coverage"], width=width, label=method)
            ax.errorbar(
                x + (idx - 1.5) * width,
                vals["coverage"],
                yerr=[vals["coverage"] - vals["ci_lower"], vals["ci_upper"] - vals["coverage"]],
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=20)
        ax.set_title(measure)
        ax.set_xlabel("Target family")
    axes[0].set_ylabel("95% coverage")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_coverage.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure1_coverage.pdf", bbox_inches="tight")
    plt.close()

    fig2 = summary[(summary["split"] == "target") & (summary["family"] != "matched") & (summary["nominal_level"] == 0.95)]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=fig2, x="width_inflation_vs_uncalibrated", y="wis", hue="method", size="mean_runtime_sec")
    plt.savefig(FIGURES_DIR / "figure2_wis_vs_width.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure2_wis_vs_width.pdf", bbox_inches="tight")
    plt.close()

    probe = summary[(summary["split"] == "test") & (summary["nominal_level"] == 0.95)].copy()
    targ = summary[(summary["split"] == "target") & (summary["nominal_level"] == 0.95)].copy()
    merged = probe.groupby(["method", "measure"], as_index=False).agg(probe_gap=("coverage_gap", "mean"), probe_wis=("wis", "mean")).merge(
        targ.groupby(["method", "measure"], as_index=False).agg(target_gap=("coverage_gap", "mean"), target_wis=("wis", "mean")),
        on=["method", "measure"],
    )
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=merged, x="probe_gap", y="target_gap", hue="method", style="measure", s=100)
    plt.savefig(FIGURES_DIR / "figure3_transfer.png", dpi=200, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure3_transfer.pdf", bbox_inches="tight")
    plt.close()

    halfbank_path = SUMMARY_DIR / "halfbank_summary.csv"
    if halfbank_path.exists():
        halfbank = pd.read_csv(halfbank_path)
        plot_df = halfbank[halfbank["draw"] >= 0]
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=plot_df, x="measure", y="coverage", hue="method")
        for _, row in halfbank[halfbank["draw"] == -1].iterrows():
            xpos = MEASURES.index(row["measure"])
            plt.hlines(row["coverage"], xpos - 0.35, xpos + 0.35, colors="black", linestyles="--", linewidth=1.0)
        plt.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
        plt.savefig(FIGURES_DIR / "figure4_halfbank.png", dpi=200, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "figure4_halfbank.pdf", bbox_inches="tight")
        plt.close()

    matched_path = SUMMARY_DIR / "matched_generator_summary.csv"
    if matched_path.exists():
        matched = pd.read_csv(matched_path)
        plot_df = matched.melt(
            id_vars=["measure", "dimension", "mechanism", "severity_level"],
            value_vars=["grouped_probe_z", "target_coverage_gap", "fallback_rate"],
            var_name="metric",
            value_name="value",
        )
        g = sns.relplot(
            data=plot_df,
            x="severity_level",
            y="value",
            hue="mechanism",
            style="measure",
            row="dimension",
            col="metric",
            kind="line",
            marker="o",
            facet_kws={"sharex": True, "sharey": False},
            height=3.2,
            aspect=1.05,
        )
        g.set_axis_labels("Severity level", "")
        plt.savefig(FIGURES_DIR / "figure5_severity.png", dpi=200, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "figure5_severity.pdf", bbox_inches="tight")
        plt.close("all")


def save_manifests(summary, completed_blocks, skipped_blocks):
    figure_manifest = {
        "figure1": {
            "source_csv": str((SUMMARY_DIR / "figure1_coverage_bootstrap.csv").relative_to(ROOT)),
            "filters": {"nominal_level": 0.95},
            "bootstrap_settings": {"resamples": 2000, "unit": "target identity"},
        },
        "figure2": {"source_csv": str((SUMMARY_DIR / "main_results.csv").relative_to(ROOT)), "filters": {"split": "target", "nominal_level": 0.95, "family_exclude": "matched"}},
        "figure3": {"source_csv": str((SUMMARY_DIR / "main_results.csv").relative_to(ROOT)), "filters": {"split": ["test", "target"], "nominal_level": 0.95}},
        "figure4": {"source_csv": str((SUMMARY_DIR / "halfbank_summary.csv").relative_to(ROOT)), "filters": {"draw_min": 0}},
        "figure5": {"source_csv": str((SUMMARY_DIR / "matched_generator_summary.csv").relative_to(ROOT)), "filters": {"dimensions": [1, 2], "severity_levels": [1, 2, 3, 4]}},
        "table1": {"source_csv": str((SUMMARY_DIR / "table1_main_benchmark.csv").relative_to(ROOT)), "filters": {"split": "target"}},
        "table2": {"source_csv": str((SUMMARY_DIR / "table2_ablations.csv").relative_to(ROOT)), "filters": {"nominal_level": 0.95}},
        "table3": {"source_csv": str((SUMMARY_DIR / "table3_kernel_subset.csv").relative_to(ROOT)), "filters": {"hard_case_subset": True}},
        "bootstrap_summary": {"source_csv": str((SUMMARY_DIR / "bootstrap_summary.json").relative_to(ROOT)), "filters": {"split": "target"}},
        "runtime_accounting": {
            "source_csv": str((SUMMARY_DIR / "runtime_accounting.csv").relative_to(ROOT)),
            "completed_blocks": completed_blocks,
            "skipped_blocks": skipped_blocks,
        },
    }
    with open(SUMMARY_DIR / "figure_manifest.json", "w") as fh:
        json.dump(figure_manifest, fh, indent=2)


def aggregate_results():
    ensure_dirs()
    df = load_all_raw()
    summary = summarize(df)
    summary.to_csv(SUMMARY_DIR / "main_results.csv", index=False)
    heldout = summary[summary["split"] == "test"]
    heldout.to_csv(SUMMARY_DIR / "heldout_probe_results.csv", index=False)
    table = summary[(summary["split"] == "target") & (summary["family"] != "matched")].copy()
    table.to_csv(SUMMARY_DIR / "table1_main_benchmark.csv", index=False)
    hard_case_ids = {rec["task_id"] for rec in get_hard_kernel_subset()}
    hard_case = df[((df["split"] == "target") & (df["task_id"].isin(hard_case_ids))) | ((df["method"] == "kernel_marginalization") & (df["task_id"].isin(hard_case_ids)))]
    summarize(hard_case).to_csv(SUMMARY_DIR / "table3_kernel_subset.csv", index=False)
    build_table2(summary)
    bootstrap = compute_bootstrap_summary(df)
    with open(SUMMARY_DIR / "bootstrap_summary.json", "w") as fh:
        json.dump(bootstrap, fh, indent=2)
    figure1_source = build_figure1_source(df)
    runtime_df, completed_blocks, skipped_blocks = build_runtime_accounting()
    make_figures(summary, figure1_source)
    save_manifests(summary, completed_blocks, skipped_blocks)
    target95 = summary[(summary["split"] == "target") & (summary["nominal_level"] == 0.95)]
    misspecified = target95[target95["family"] != "matched"]
    matched = target95[target95["family"] == "matched"]
    def metric_block(frame):
        out = {}
        grouped = frame.groupby("method")[["coverage", "coverage_gap", "wis", "mean_width", "width_inflation_vs_uncalibrated", "median_abs_error", "fallback_rate", "mean_runtime_sec"]]
        for method, stats_df in grouped:
            out[method] = {}
            for metric in stats_df.columns:
                out[method][metric] = {
                    "mean": float(stats_df[metric].mean()),
                    "std": float(stats_df[metric].std(ddof=0)),
                }
        return out
    selected_misspecified = misspecified[misspecified["method"] == "selected_external_rule"]
    uncal_misspecified = misspecified[misspecified["method"] == "uncalibrated"]
    selected_matched = matched[matched["method"] == "selected_external_rule"]
    global_misspecified = misspecified[misspecified["method"] == "global_scaling"]
    bayes_misspecified = misspecified[misspecified["method"] == "bayes_sard"]
    coverage_gain = float(selected_misspecified["coverage"].mean() - uncal_misspecified["coverage"].mean())
    wis_gain = float(uncal_misspecified["wis"].mean() - selected_misspecified["wis"].mean())
    matched_width_ratio = float(selected_matched["width_inflation_vs_uncalibrated"].median())
    roughness_beats_global = float(global_misspecified["wis"].mean() - selected_misspecified["wis"].mean()) > 0.0
    halfbank_std = float(pd.read_csv(SUMMARY_DIR / "halfbank_summary.csv").query("draw >= 0 and method == 'selected_external_rule'")["coverage"].std(ddof=0)) if (SUMMARY_DIR / "halfbank_summary.csv").exists() else float("nan")
    practical_recommendation = (
        "Global scaling or Bayes-Sard remain the practical recommendation: the selected external rule improves misspecified-family coverage only by aggressively widening intervals and does not satisfy the preregistered WIS/width success criterion."
        if not (wis_gain > 0.0 and matched_width_ratio <= 1.75)
        else "The selected external rule meets the preregistered efficiency criterion."
    )
    results = {
        "environment": json.load(open(DATA_DIR / "run_manifest.json")),
        "metrics": {
            "misspecified_union": metric_block(misspecified),
            "matched_controls": metric_block(matched),
        },
        "analysis": {
            "primary_success_criteria": {
                "coverage_gain_at_least_0p05": coverage_gain >= 0.05,
                "wis_improves_and_matched_width_leq_1p75x": wis_gain > 0.0 and matched_width_ratio <= 1.75,
                "roughness_aware_beats_global_on_wis": roughness_beats_global,
                "halfbank_coverage_std_below_0p04": halfbank_std < 0.04,
            },
            "selected_external_rule_vs_uncalibrated_95": {
                "coverage_gain": coverage_gain,
                "wis_improvement": wis_gain,
                "matched_control_median_width_inflation": matched_width_ratio,
            },
            "practical_recommendation": practical_recommendation,
        },
        "artifacts": {
            "summary_csv": "results/summary/main_results.csv",
            "heldout_probe_csv": "results/summary/heldout_probe_results.csv",
            "runtime_csv": "results/summary/runtime_accounting.csv",
            "bootstrap_summary_json": "results/summary/bootstrap_summary.json",
            "figure_manifest_json": "results/summary/figure_manifest.json",
            "figures": sorted(str(p.relative_to(ROOT)) for p in FIGURES_DIR.glob("*") if p.is_file()),
        },
    }
    with open(ROOT / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(ROOT / "exp" / "aggregate" / "results.json", "w") as fh:
        json.dump(
            {
                "experiment": "aggregate",
                "summary_rows": int(len(summary)),
                "runtime_rows": int(len(runtime_df)),
                "source_files": {
                    "main_results_csv": str((SUMMARY_DIR / "main_results.csv").relative_to(ROOT)),
                    "heldout_probe_csv": str((SUMMARY_DIR / "heldout_probe_results.csv").relative_to(ROOT)),
                    "table1_csv": str((SUMMARY_DIR / "table1_main_benchmark.csv").relative_to(ROOT)),
                    "table2_csv": str((SUMMARY_DIR / "table2_ablations.csv").relative_to(ROOT)),
                    "table3_csv": str((SUMMARY_DIR / "table3_kernel_subset.csv").relative_to(ROOT)),
                    "bootstrap_json": str((SUMMARY_DIR / "bootstrap_summary.json").relative_to(ROOT)),
                    "figure_manifest_json": str((SUMMARY_DIR / "figure_manifest.json").relative_to(ROOT)),
                },
                "aggregates": {
                    "misspecified_union_nominal_95": method_metric_snapshot(summary, split="target", nominal_level=0.95, family_mode="misspecified_only"),
                    "matched_controls_nominal_95": method_metric_snapshot(summary, split="target", nominal_level=0.95, family_mode="matched_only"),
                },
            },
            fh,
            indent=2,
        )
    return summary


def run_stage(stage):
    configure_environment()
    ensure_dirs()
    if stage in ["reference", "all"]:
        timed_stage("stage_a_reference", reference_table)
    if stage in ["smoke", "all"]:
        timed_stage("smoke_test", lambda: run_core(smoke=True))
    if stage in ["core", "all"]:
        timed_stage("core_fits", lambda: run_core(smoke=False))
    if stage in ["main", "all"]:
        timed_stage("main_recalibration", run_main_recalibration)
    if stage in ["ablations", "all"]:
        timed_stage("ablations", run_ablations)
    if stage in ["matched", "all"]:
        timed_stage("matched_generator", run_matched_generator)
    if stage in ["kernel", "all"]:
        timed_stage("kernel_marginalization", run_kernel_subset)
    if stage in ["aggregate", "all"]:
        timed_stage("aggregate", aggregate_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["reference", "smoke", "core", "main", "ablations", "matched", "kernel", "aggregate", "all"])
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
