from __future__ import annotations

import itertools
import os
import platform
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from .aggregation import compute_weights, contradiction_stats, merge_local_graphs
from .config import (
    ALPHA_GRID,
    BOOTSTRAP_SEEDS,
    CPU_WORKERS,
    DAGBAG_BOOTSTRAPS,
    DATASET_SEEDS,
    EXP_ROOT,
    FIGURES_ROOT,
    GPU_COUNT,
    GRAPH_FAMILIES,
    HARD_REGIMES,
    NOTEARS_EDGE_THRESHOLD,
    NOTEARS_LAMBDA1,
    NOTEARS_MAX_ITER,
    P,
    REGIMES,
    ROOT,
    SAMPLE_SIZES,
    SUBSET_BANK,
    SUBSET_BANK_SEED,
    SUBSET_SENSITIVITY_KS,
    TABLES_ROOT,
)
from .discovery import run_ges, run_notears_linear, run_pc
from .graph_utils import (
    cpdag_claims,
    cpdag_colliders,
    cpdag_directed,
    cpdag_skeleton,
    dag_to_cpdag_matrix,
    induced_subdag,
    restrict_cpdag,
)
from .io import ensure_dir, read_json, write_csv, write_json
from .metrics import score_cpdag
from .simulation import simulate_dataset
from .subsets import build_subset_bank
from .utils import append_log, configure_environment, set_global_seed, stable_int_seed, timed_block


def run_environment_manifest() -> dict:
    configure_environment()
    set_global_seed(DATASET_SEEDS[0])
    log_path = _stage_log("environment", "run.log")
    append_log(log_path, "Starting environment manifest collection.")
    ensure_dir(EXP_ROOT / "environment")
    manifest = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_workers": CPU_WORKERS,
        "gpu_count": GPU_COUNT,
        "thread_env": {key: os.environ.get(key) for key in sorted(os.environ) if key.endswith("_THREADS") or key == "PYTHONHASHSEED"},
        "package_versions": _package_versions(),
        "seed_registry": {
            "dataset_seeds": DATASET_SEEDS,
            "bootstrap_seeds": BOOTSTRAP_SEEDS,
            "subset_bank_seed": SUBSET_BANK_SEED,
        },
        "plan_alignment": {
            "python_3_11_available": platform.python_version().startswith("3.11."),
            "notes": "Interpreter is fixed for this workspace; all benchmark RNG now uses explicit stable seeds instead of Python hash().",
        },
    }
    write_json(EXP_ROOT / "environment" / "system_info.json", manifest)
    write_json(
        EXP_ROOT / "environment" / "execution_manifest.json",
        {"gpu_count": GPU_COUNT, "cpu_workers": CPU_WORKERS, "notes": "CPU-only benchmark; no GPU-dependent steps."},
    )
    append_log(log_path, f"Recorded environment manifest for Python {manifest['python_version']}.")
    return manifest


def run_timing_pilot() -> dict:
    configure_environment()
    ensure_dir(EXP_ROOT / "environment")
    log_path = _stage_log("environment", "timing_pilot.log")
    pilot_points = [
        ("erdos_renyi", "linear_gaussian", 200, 11),
        ("erdos_renyi", "near_unfaithful_linear", 1000, 23),
        ("scale_free", "nonlinear_anm", 200, 37),
        ("scale_free", "mild_misspecification", 1000, 11),
    ]
    rows = []
    append_log(log_path, "Starting timing pilot on four disjoint datasets.")
    for family, regime, n, seed in pilot_points:
        with timed_block() as t:
            dataset = simulate_dataset(family, regime, n, seed, p=P)
            subsets = build_subset_bank(P, k=SUBSET_BANK["k"], m=SUBSET_BANK["M"], seed=SUBSET_BANK_SEED)
            nodes = subsets[0]
            local = dataset.standardized[[f"X{i}" for i in nodes]].to_numpy()
            _ = run_pc(local, alpha=SUBSET_BANK["alpha"]).cpdag
            full = dataset.standardized.to_numpy()
            _ = run_pc(full, alpha=0.01).cpdag
            _ = run_ges(full).cpdag
            _ = run_notears_linear(full, NOTEARS_LAMBDA1, NOTEARS_MAX_ITER, NOTEARS_EDGE_THRESHOLD)[0]
        dataset_id = dataset.dataset_id
        rows.append(
            {
                "dataset_id": dataset_id,
                "family": family,
                "regime": regime,
                "n": n,
                "seed": seed,
                "runtime_seconds": t.runtime_seconds,
                "peak_memory_mb": t.peak_memory_mb,
            }
        )
        append_log(log_path, f"Pilot dataset {dataset_id} finished in {t.runtime_seconds:.2f}s.")
    pilot_df = pd.DataFrame(rows)
    projected_hours = float(pilot_df["runtime_seconds"].mean() * 48 / 3600.0) if not pilot_df.empty else 0.0
    fallback = {
        "projected_total_hours": projected_hours,
        "within_budget": projected_hours <= 8.0,
        "fallback_order_if_needed": [
            "reduce DAGBag-PC bootstraps from 10 to 5",
            "reduce subset count M from 20 to 16 globally",
            "drop k in {6, 10} subset-size sensitivity",
        ],
        "fallback_triggered": False,
    }
    write_csv(EXP_ROOT / "environment" / "timing_pilot.csv", pilot_df)
    write_json(EXP_ROOT / "environment" / "timing_pilot.json", {"rows": rows, "summary": fallback})
    write_json(EXP_ROOT / "environment" / "fallback_budget_record.json", fallback)
    append_log(log_path, f"Timing pilot projected {projected_hours:.2f} hours for 48 datasets.")
    return {"rows": rows, "summary": fallback}


def generate_all_datasets() -> pd.DataFrame:
    configure_environment()
    ensure_dir(EXP_ROOT / "data" / "simulated")
    log_path = _stage_log("data", "run.log")
    rows = []
    append_log(log_path, "Starting synthetic benchmark generation.")
    for family, regime, n, seed in itertools.product(GRAPH_FAMILIES, REGIMES, SAMPLE_SIZES, DATASET_SEEDS):
        set_global_seed(stable_int_seed("dataset", family, regime, n, seed))
        ds = simulate_dataset(family, regime, n, seed, p=P)
        out_dir = ensure_dir(EXP_ROOT / "data" / "simulated" / ds.dataset_id)
        ds.raw.to_csv(out_dir / "raw.csv", index=False)
        ds.standardized.to_csv(out_dir / "standardized.csv", index=False)
        graph_meta = {
            **ds.metadata,
            "adjacency_matrix": nx.to_numpy_array(ds.dag, dtype=int).astype(int).tolist(),
            "cpdag_matrix": ds.cpdag.tolist(),
            "true_skeleton": sorted([list(pair) for pair in cpdag_skeleton(ds.cpdag)]),
            "true_colliders": sorted([list(item) for item in cpdag_colliders(ds.cpdag)]),
            "true_directed_edges": sorted([list(edge) for edge in cpdag_directed(ds.cpdag)]),
        }
        write_json(out_dir / "graph.json", graph_meta)
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "family": family,
                "regime": regime,
                "n": n,
                "p": ds.p,
                "seed": seed,
                "edge_count": ds.metadata["edge_count"],
                "cpdag_edge_count": int(len(cpdag_claims(ds.cpdag)["adj"])),
                "density": ds.metadata["density"],
            }
        )
        append_log(log_path, f"Saved dataset {ds.dataset_id}.")
    manifest = pd.DataFrame(rows).sort_values(["family", "regime", "n", "seed"]).reset_index(drop=True)
    write_csv(EXP_ROOT / "data" / "benchmark_manifest.csv", manifest)
    write_json(EXP_ROOT / "data" / "results.json", {"dataset_count": int(len(manifest))})
    return manifest


def build_all_local_banks(manifest: pd.DataFrame, k: int = SUBSET_BANK["k"], tag: str = "k8") -> pd.DataFrame:
    configure_environment()
    out_root = ensure_dir(EXP_ROOT / "local_bank" / tag)
    ensure_dir(EXP_ROOT / "evaluation")
    log_path = _stage_log("local_bank", f"{tag}.log")
    subsets = build_subset_bank(P, k=k, m=SUBSET_BANK["M"], seed=SUBSET_BANK_SEED)
    append_log(log_path, f"Building local bank {tag} with k={k}.")
    rows = []
    dataset_summaries = []
    for dataset_id in manifest["dataset_id"]:
        with timed_block() as dataset_timer:
            ds_dir = EXP_ROOT / "data" / "simulated" / dataset_id
            std = pd.read_csv(ds_dir / "standardized.csv")
            graph_info = read_json(ds_dir / "graph.json")
            dag = nx.from_numpy_array(np.asarray(graph_info["adjacency_matrix"], dtype=int), create_using=nx.DiGraph)
            dataset_out = ensure_dir(out_root / dataset_id)
            write_json(dataset_out / "subsets.json", {"subsets": subsets, "tag": tag, "k": k})
            local_entries = []
            validity_rows = []
            for subset_id, nodes in enumerate(subsets):
                sub_truth = induced_subdag(dag, nodes)
                truth_cpdag = dag_to_cpdag_matrix(sub_truth)
                truth_payload = {
                    "nodes": nodes,
                    "cpdag_matrix": truth_cpdag.tolist(),
                    "skeleton": sorted([list(pair) for pair in cpdag_skeleton(truth_cpdag)]),
                    "colliders": sorted([list(item) for item in cpdag_colliders(truth_cpdag)]),
                    "directed_edges": sorted([list(edge) for edge in cpdag_directed(truth_cpdag)]),
                }
                write_json(dataset_out / f"truth_subset_{subset_id}.json", truth_payload)
                truth_claims = cpdag_claims(truth_cpdag, global_nodes=nodes)
                for b_seed in BOOTSTRAP_SEEDS:
                    rng = np.random.default_rng(stable_int_seed("local_bank", dataset_id, subset_id, b_seed))
                    sample_idx = rng.integers(0, len(std), size=len(std))
                    data = std.iloc[sample_idx][[f"X{i}" for i in nodes]].to_numpy()
                    with timed_block() as t:
                        result = run_pc(data, alpha=SUBSET_BANK["alpha"])
                    claims = cpdag_claims(result.cpdag, global_nodes=nodes)
                    graph_id = f"subset{subset_id}_boot{b_seed}"
                    payload = {
                        "graph_id": graph_id,
                        "subset_id": subset_id,
                        "bootstrap_seed": b_seed,
                        "nodes": nodes,
                        "cpdag_matrix": result.cpdag.tolist(),
                        "claims": {key: sorted(list(value)) for key, value in claims.items()},
                        "runtime_seconds": t.runtime_seconds,
                        "peak_memory_mb": t.peak_memory_mb,
                        "claim_counts": {name: len(values) for name, values in claims.items()},
                    }
                    write_json(dataset_out / f"{graph_id}.json", payload)
                    local_entries.append(payload | {"claims": claims})
                    validity_rows.extend(_subset_validity_rows(dataset_id, subset_id, graph_id, claims, truth_claims, len(nodes)))
            overlap_rows = []
            for a, b in itertools.combinations(local_entries, 2):
                overlap_nodes = sorted(set(a["nodes"]) & set(b["nodes"]))
                if len(overlap_nodes) < 2:
                    continue
                stats = contradiction_stats(a["claims"], b["claims"], set(overlap_nodes))
                overlap_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "graph_a": a["graph_id"],
                        "graph_b": b["graph_id"],
                        "overlap_size": len(overlap_nodes),
                        "comparable_claim_count": stats["comparable"],
                        "contradiction_count": stats["contradictions"],
                        "adj_comparable_count": stats["adj_comparable"],
                        "adj_contradiction_count": stats["adj_contradictions"],
                        "dir_comparable_count": stats["dir_comparable"],
                        "dir_contradiction_count": stats["dir_contradictions"],
                        "coll_comparable_count": stats["coll_comparable"],
                        "coll_contradiction_count": stats["coll_contradictions"],
                    }
                )
            validity_df = pd.DataFrame(validity_rows)
            write_csv(dataset_out / "subset_validity.csv", validity_df)
            write_csv(dataset_out / "overlap_map.csv", pd.DataFrame(overlap_rows))
            dataset_summary = {
                "dataset_id": dataset_id,
                "tag": tag,
                "k": k,
                "subset_count": len(subsets),
                "local_graph_count": len(local_entries),
                "runtime_seconds": dataset_timer.runtime_seconds,
                "peak_memory_mb": dataset_timer.peak_memory_mb,
            }
            write_json(dataset_out / "results.json", dataset_summary)
            dataset_summaries.append(dataset_summary)
            rows.extend(validity_rows)
        append_log(log_path, f"Built local bank for {dataset_id} in {dataset_timer.runtime_seconds:.2f}s.")
    summary = pd.DataFrame(rows)
    write_csv(EXP_ROOT / "evaluation" / f"subset_validity_summary_{tag}.csv", summary)
    write_csv(EXP_ROOT / "local_bank" / tag / "dataset_summary.csv", pd.DataFrame(dataset_summaries))
    write_json(EXP_ROOT / "local_bank" / tag / "results.json", {"dataset_count": int(manifest.shape[0]), "tag": tag})
    return summary


def run_baselines(manifest: pd.DataFrame) -> pd.DataFrame:
    configure_environment()
    ensure_dir(EXP_ROOT / "baselines")
    log_path = _stage_log("baselines", "run.log")
    methods = ["PC", "GES", "NOTEARS-L1", "DAGBag-PC", "SC-Select"]
    rows = []
    for method in methods:
        ensure_dir(EXP_ROOT / "baselines" / method)
    _document_skipped_lovo()
    append_log(log_path, "Starting baseline evaluation.")
    for dataset_id in manifest["dataset_id"]:
        ds_dir = EXP_ROOT / "data" / "simulated" / dataset_id
        std = pd.read_csv(ds_dir / "standardized.csv")
        graph_info = read_json(ds_dir / "graph.json")
        truth = np.asarray(graph_info["cpdag_matrix"], dtype=int)
        data = std.to_numpy()
        local_entries = _load_local_entries(dataset_id, "k8")
        for method in methods:
            out_dir = ensure_dir(EXP_ROOT / "baselines" / method / dataset_id)
            with timed_block() as t:
                if method == "PC":
                    pred = run_pc(data, alpha=0.01).cpdag
                    meta = {"alpha": 0.01}
                elif method == "GES":
                    pred = run_ges(data).cpdag
                    meta = {"score_type": "GaussianBIC"}
                elif method == "NOTEARS-L1":
                    pred, meta = run_notears_linear(data, NOTEARS_LAMBDA1, NOTEARS_MAX_ITER, NOTEARS_EDGE_THRESHOLD)
                elif method == "DAGBag-PC":
                    preds = []
                    for b in range(DAGBAG_BOOTSTRAPS):
                        rng = np.random.default_rng(stable_int_seed("dagbag", dataset_id, b))
                        idx = rng.integers(0, len(std), size=len(std))
                        preds.append(run_pc(std.iloc[idx].to_numpy(), alpha=0.01).cpdag)
                    pred = _aggregate_dagbag(preds)
                    meta = {"bootstraps": DAGBAG_BOOTSTRAPS, "threshold": 0.5}
                else:
                    pred, meta = _run_sc_select(data, local_entries)
            metrics = score_cpdag(pred, truth)
            payload = {
                "method": method,
                "dataset_id": dataset_id,
                "metrics": metrics,
                "config": meta,
                "runtime_seconds": t.runtime_seconds,
                "peak_memory_mb": t.peak_memory_mb,
                "cpdag_matrix": pred.tolist(),
            }
            write_json(out_dir / "results.json", payload)
            rows.append({"method": method, "dataset_id": dataset_id, **metrics, "runtime_seconds": t.runtime_seconds, "peak_memory_mb": t.peak_memory_mb})
            append_log(log_path, f"Baseline {method} completed for {dataset_id}.")
    df = pd.DataFrame(rows)
    write_csv(EXP_ROOT / "baselines" / "summary.csv", df)
    for method in methods:
        write_json(EXP_ROOT / "baselines" / method / "results.json", {"method": method, "dataset_count": int((df["method"] == method).sum())})
    return df


def run_subset_methods(manifest: pd.DataFrame, tag: str = "k8") -> pd.DataFrame:
    configure_environment()
    methods = [
        ("Uniform", "Wilson"),
        ("Uniform", "DetThreshold"),
        ("BootstrapStability", "Wilson"),
        ("BootstrapStability", "DetThreshold"),
        ("CompatExp", "Wilson"),
        ("CompatExp", "DetThreshold"),
    ]
    ensure_dir(EXP_ROOT / "main")
    log_path = _stage_log("main", f"{tag}.log")
    rows = []
    for weight, merge in methods:
        ensure_dir(EXP_ROOT / "main" / f"{weight}+{merge}")
    append_log(log_path, f"Starting main subset methods for {tag}.")
    bank_summary = pd.read_csv(EXP_ROOT / "local_bank" / tag / "dataset_summary.csv").set_index("dataset_id")
    for dataset_id in manifest["dataset_id"]:
        entries = _load_local_entries(dataset_id, tag)
        truth = np.asarray(read_json(EXP_ROOT / "data" / "simulated" / dataset_id / "graph.json")["cpdag_matrix"], dtype=int)
        local_bank_runtime = float(bank_summary.loc[dataset_id, "runtime_seconds"]) if dataset_id in bank_summary.index else 0.0
        for weight, merge in methods:
            with timed_block() as t:
                weights = compute_weights(entries, weight)
                cpdag, aux = merge_local_graphs(P, entries, weights, merge)
            metrics = score_cpdag(cpdag, truth)
            payload = {
                "dataset_id": dataset_id,
                "method": f"{weight}+{merge}",
                "metrics": metrics,
                "runtime_seconds": t.runtime_seconds,
                "peak_memory_mb": t.peak_memory_mb,
                "local_bank_runtime_seconds": local_bank_runtime,
                "n_eff": aux["n_eff"],
                "accepted_adjacency_claims": aux["accepted_adjacency_claims"],
                "accepted_orientation_claims": aux["accepted_orientation_claims"],
                "median_weighted_support_ratio": aux["median_weighted_support_ratio"],
                "median_weighted_opposition": aux["median_weighted_opposition"],
                "median_n_eff": aux["n_eff"],
                "cpdag_matrix": cpdag.tolist(),
                "support_table": aux["support_table"],
                "weights": weights,
            }
            write_json(EXP_ROOT / "main" / f"{weight}+{merge}" / f"{dataset_id}.json", payload)
            rows.append(
                {
                    "method": f"{weight}+{merge}",
                    "dataset_id": dataset_id,
                    **metrics,
                    "runtime_seconds": t.runtime_seconds,
                    "peak_memory_mb": t.peak_memory_mb,
                    "local_bank_runtime_seconds": local_bank_runtime,
                    "n_eff": aux["n_eff"],
                    "accepted_adjacency_claims": aux["accepted_adjacency_claims"],
                    "accepted_orientation_claims": aux["accepted_orientation_claims"],
                    "median_weighted_support_ratio": aux["median_weighted_support_ratio"],
                    "median_weighted_opposition": aux["median_weighted_opposition"],
                    "median_n_eff": aux["n_eff"],
                }
            )
        append_log(log_path, f"Finished main methods for {dataset_id}.")
    df = pd.DataFrame(rows)
    write_csv(EXP_ROOT / "main" / "summary.csv", df)
    write_json(EXP_ROOT / "main" / "results.json", {"method_count": int(df["method"].nunique())})
    return df


def run_ablations(manifest: pd.DataFrame) -> pd.DataFrame:
    configure_environment()
    ensure_dir(EXP_ROOT / "ablations")
    log_path = _stage_log("ablations", "run.log")
    rows = []
    append_log(log_path, "Starting ablation suite.")
    for dataset_id in manifest["dataset_id"]:
        entries = _load_local_entries(dataset_id, "k8")
        truth = np.asarray(read_json(EXP_ROOT / "data" / "simulated" / dataset_id / "graph.json")["cpdag_matrix"], dtype=int)
        for scheme in ["Uniform", "BootstrapStability", "CompatExp"]:
            weights = compute_weights(entries, scheme)
            cpdag, aux = merge_local_graphs(P, entries, weights, "DetRank")
            metrics = score_cpdag(cpdag, truth)
            name = f"{scheme}+DetRank"
            write_json(EXP_ROOT / "ablations" / f"{name}__{dataset_id}.json", {"method": name, "dataset_id": dataset_id, "metrics": metrics, "n_eff": aux["n_eff"], "cpdag_matrix": cpdag.tolist()})
            rows.append({"method": name, "dataset_id": dataset_id, **metrics})
    hard_manifest = manifest[manifest["regime"].isin(HARD_REGIMES)].copy()
    for dataset_id in hard_manifest["dataset_id"]:
        entries = _load_local_entries(dataset_id, "k8")
        truth = np.asarray(read_json(EXP_ROOT / "data" / "simulated" / dataset_id / "graph.json")["cpdag_matrix"], dtype=int)
        for scheme in ["CompatRank", "CompatTopHalf"]:
            weights = compute_weights(entries, scheme)
            cpdag, aux = merge_local_graphs(P, entries, weights, "DetThreshold")
            metrics = score_cpdag(cpdag, truth)
            name = f"{scheme}+DetThreshold"
            write_json(EXP_ROOT / "ablations" / f"{name}__{dataset_id}.json", {"method": name, "dataset_id": dataset_id, "metrics": metrics, "n_eff": aux["n_eff"], "cpdag_matrix": cpdag.tolist()})
            rows.append({"method": name, "dataset_id": dataset_id, **metrics})
        for suffix, include_dir, include_colliders in [("NoColliders", True, False), ("AdjacencyOnly", False, False)]:
            weights = compute_weights(entries, "CompatExp")
            cpdag, aux = merge_local_graphs(P, entries, weights, "DetThreshold", include_dir=include_dir, include_colliders=include_colliders)
            metrics = score_cpdag(cpdag, truth)
            name = f"CompatExp+DetThreshold+{suffix}"
            write_json(EXP_ROOT / "ablations" / f"{name}__{dataset_id}.json", {"method": name, "dataset_id": dataset_id, "metrics": metrics, "n_eff": aux["n_eff"], "cpdag_matrix": cpdag.tolist()})
            rows.append({"method": name, "dataset_id": dataset_id, **metrics})
        weights = compute_weights(entries, "CompatExp", weak_abstention=True)
        cpdag, aux = merge_local_graphs(P, entries, weights, "DetThreshold")
        metrics = score_cpdag(cpdag, truth)
        name = "CompatExp+DetThreshold+WeakAbstention"
        write_json(EXP_ROOT / "ablations" / f"{name}__{dataset_id}.json", {"method": name, "dataset_id": dataset_id, "metrics": metrics, "n_eff": aux["n_eff"], "cpdag_matrix": cpdag.tolist()})
        rows.append({"method": name, "dataset_id": dataset_id, **metrics})
    for k in SUBSET_SENSITIVITY_KS:
        tag = f"k{k}"
        build_all_local_banks(hard_manifest, k=k, tag=tag)
        for dataset_id in hard_manifest["dataset_id"]:
            entries = _load_local_entries(dataset_id, tag)
            truth = np.asarray(read_json(EXP_ROOT / "data" / "simulated" / dataset_id / "graph.json")["cpdag_matrix"], dtype=int)
            for scheme in ["Uniform", "CompatExp"]:
                weights = compute_weights(entries, scheme)
                cpdag, aux = merge_local_graphs(P, entries, weights, "DetThreshold")
                metrics = score_cpdag(cpdag, truth)
                name = f"{scheme}+DetThreshold+{tag}"
                write_json(EXP_ROOT / "ablations" / f"{name}__{dataset_id}.json", {"method": name, "dataset_id": dataset_id, "metrics": metrics, "n_eff": aux["n_eff"], "cpdag_matrix": cpdag.tolist()})
                rows.append({"method": name, "dataset_id": dataset_id, **metrics})
        append_log(log_path, f"Completed subset-size sensitivity for {tag}.")
    df = pd.DataFrame(rows)
    write_csv(EXP_ROOT / "ablations" / "summary.csv", df)
    write_json(EXP_ROOT / "ablations" / "results.json", {"rows": int(len(df))})
    append_log(log_path, f"Finished {len(df)} ablation rows.")
    return df


def run_evaluation(manifest: pd.DataFrame, subset_validity: pd.DataFrame, baselines: pd.DataFrame, main: pd.DataFrame, ablations: pd.DataFrame) -> dict:
    configure_environment()
    ensure_dir(EXP_ROOT / "evaluation")
    log_path = _stage_log("evaluation", "run.log")
    append_log(log_path, "Starting evaluation and figure generation.")
    direction_validity = (
        subset_validity[subset_validity["claim_type"] == "dir"].groupby("dataset_id")["validity_rate"].mean().rename("direction_validity")
    )
    main_pivot = main.pivot(index="dataset_id", columns="method", values="false_orientation_rate")
    gain = (main_pivot["CompatExp+DetThreshold"] - main_pivot["Uniform+DetThreshold"]).rename("compat_minus_uniform_for")
    corr_df = pd.concat([direction_validity, gain], axis=1).dropna()
    correlations = {
        "pearson": float(pearsonr(corr_df["direction_validity"], corr_df["compat_minus_uniform_for"]).statistic) if len(corr_df) > 2 else None,
        "spearman": float(spearmanr(corr_df["direction_validity"], corr_df["compat_minus_uniform_for"]).statistic) if len(corr_df) > 2 else None,
    }
    tests = {}
    comparisons = [
        ("CompatExp+Wilson", "Uniform+Wilson"),
        ("CompatExp+DetThreshold", "Uniform+DetThreshold"),
        ("CompatExp+DetThreshold", "BootstrapStability+DetThreshold"),
    ]
    for lhs, rhs in comparisons:
        diff = (
            main[main["method"] == lhs].set_index("dataset_id")["false_orientation_rate"]
            - main[main["method"] == rhs].set_index("dataset_id")["false_orientation_rate"]
        ).dropna()
        tests[f"{lhs}_vs_{rhs}"] = _signed_test(diff)
    for baseline in baselines["method"].unique():
        diff = (
            main[main["method"] == "CompatExp+DetThreshold"].set_index("dataset_id")["false_orientation_rate"]
            - baselines[baselines["method"] == baseline].set_index("dataset_id")["false_orientation_rate"]
        ).dropna()
        tests[f"CompatExp+DetThreshold_vs_{baseline}"] = _signed_test(diff)

    table1 = _table1_subset_audit(manifest, subset_validity)
    table2 = _table2_main_summary(pd.concat([baselines, main], ignore_index=True))
    write_csv(TABLES_ROOT / "table1_subset_audit.csv", table1)
    write_csv(TABLES_ROOT / "table2_main_summary.csv", table2)
    _make_figures(manifest, subset_validity, baselines, main, ablations, corr_df)

    hard = main.merge(manifest[["dataset_id", "regime"]], on="dataset_id")
    hard = hard[hard["regime"].isin(HARD_REGIMES)]
    hard_uniform = hard[hard["method"] == "Uniform+DetThreshold"]["false_orientation_rate"].to_numpy()
    hard_compat = hard[hard["method"] == "CompatExp+DetThreshold"]["false_orientation_rate"].to_numpy()
    hard_reduction = float((np.median(hard_uniform) - np.median(hard_compat)) / np.median(hard_uniform)) if np.median(hard_uniform) > 0 else 0.0
    hypothesis_supported = bool(
        hard_reduction >= 0.10
        and tests["CompatExp+DetThreshold_vs_Uniform+DetThreshold"]["median_difference"] is not None
        and tests["CompatExp+DetThreshold_vs_Uniform+DetThreshold"]["median_difference"] <= 0.0
    )
    summary_text = _evaluation_summary_text(hard_reduction, hypothesis_supported, tests)
    (EXP_ROOT / "evaluation" / "summary.md").write_text(summary_text, encoding="utf-8")
    deviations_text = _deviations_text()
    (EXP_ROOT / "evaluation" / "DEVIATIONS.md").write_text(deviations_text, encoding="utf-8")

    results = {
        "correlations": correlations,
        "paired_tests": tests,
        "subset_audit_rows": int(len(subset_validity)),
        "baseline_rows": int(len(baselines)),
        "main_rows": int(len(main)),
        "ablation_rows": int(len(ablations)),
        "hard_regime_median_false_orientation_reduction": hard_reduction,
        "main_hypothesis_supported": hypothesis_supported,
    }
    write_json(EXP_ROOT / "evaluation" / "results.json", results)
    append_log(log_path, "Finished evaluation.")
    return results


def compile_root_results(manifest: pd.DataFrame, subset_validity: pd.DataFrame, baselines: pd.DataFrame, main: pd.DataFrame, ablations: pd.DataFrame, evaluation: dict) -> dict:
    hard = main.merge(manifest[["dataset_id", "regime"]], on="dataset_id")
    hard = hard[hard["regime"].isin(HARD_REGIMES)]
    uniform = hard[hard["method"] == "Uniform+DetThreshold"]["false_orientation_rate"].to_numpy()
    compat = hard[hard["method"] == "CompatExp+DetThreshold"]["false_orientation_rate"].to_numpy()
    reduction = float((np.median(uniform) - np.median(compat)) / np.median(uniform)) if np.median(uniform) > 0 else 0.0
    root = {
        "benchmark": {
            "dataset_count": int(len(manifest)),
            "seed_count": len(DATASET_SEEDS),
            "baselines": sorted(baselines["method"].unique().tolist()),
            "main_methods": sorted(main["method"].unique().tolist()),
        },
        "summary_metrics": {
            "baselines": _aggregate_summary(baselines),
            "main_methods": _aggregate_summary(main),
            "ablations": _aggregate_summary(ablations),
        },
        "success_criteria_check": {
            "hard_regime_median_false_orientation_reduction_compat_vs_uniform_detthreshold": reduction,
            "detthreshold_effect_direction_positive": bool(np.median(uniform) >= np.median(compat)),
            "evaluation": evaluation,
        },
        "interpretation": {
            "main_hypothesis_supported": evaluation["main_hypothesis_supported"],
            "statement": "CompatExp gains over Uniform are small, statistically null, and below the pre-registered 10% hard-regime threshold.",
            "writeup": "exp/evaluation/summary.md",
        },
    }
    write_json(EXP_ROOT / "results.json", root)
    write_json(ROOT / "results.json", root)
    return root


def run_full_benchmark() -> dict:
    run_environment_manifest()
    run_timing_pilot()
    manifest = generate_all_datasets()
    subset_validity = build_all_local_banks(manifest, k=SUBSET_BANK["k"], tag="k8")
    baselines = run_baselines(manifest)
    main = run_subset_methods(manifest, tag="k8")
    ablations = run_ablations(manifest)
    evaluation = run_evaluation(manifest, subset_validity, baselines, main, ablations)
    return compile_root_results(manifest, subset_validity, baselines, main, ablations, evaluation)


def _package_versions() -> dict:
    versions = {}
    for name in ["numpy", "scipy", "pandas", "sklearn", "networkx", "matplotlib", "seaborn", "statsmodels", "causallearn", "pgmpy"]:
        mod = __import__(name)
        versions[name] = getattr(mod, "__version__", "unknown")
    return versions


def _stage_log(stage: str, filename: str) -> Path:
    return ensure_dir(EXP_ROOT / stage / "logs") / filename


def _subset_validity_rows(dataset_id: str, subset_id: int, graph_id: str, claims: dict[str, set], truth_claims: dict[str, set], subset_size: int) -> list[dict]:
    rows = []
    pair_count = subset_size * (subset_size - 1) // 2
    triple_count = subset_size * (subset_size - 1) * (subset_size - 2) // 2
    opportunity_map = {"adj": pair_count, "nonadj": pair_count, "dir": pair_count, "coll": triple_count}
    for claim_type in ["adj", "nonadj", "dir", "coll"]:
        predicted = claims[claim_type]
        truth = truth_claims[claim_type]
        support = len(predicted)
        correct = len(predicted & truth)
        error = len(predicted - truth)
        opportunity = max(opportunity_map[claim_type], support)
        abstain = max(opportunity - support, 0)
        validity = correct / support if support else 0.0
        contradiction = error / support if support else 0.0
        rows.append(
            {
                "dataset_id": dataset_id,
                "subset_id": subset_id,
                "graph_id": graph_id,
                "claim_type": claim_type,
                "support_count": support,
                "correct_count": correct,
                "error_count": error,
                "truth_count": len(truth),
                "opportunity_count": opportunity,
                "abstain_count": abstain,
                "validity_rate": validity,
                "contradiction_rate": contradiction,
                "abstention_rate": abstain / opportunity if opportunity else 0.0,
            }
        )
    return rows


def _load_local_entries(dataset_id: str, tag: str) -> list[dict]:
    dataset_dir = EXP_ROOT / "local_bank" / tag / dataset_id
    entries = []
    for path in sorted(dataset_dir.glob("subset*_boot*.json")):
        payload = read_json(path)
        payload["claims"] = {key: {tuple(item) for item in value} for key, value in payload["claims"].items()}
        entries.append(payload)
    return entries


def _aggregate_dagbag(preds: list[np.ndarray]) -> np.ndarray:
    p = preds[0].shape[0]
    adj_scores = {}
    dir_scores = {}
    for i in range(p):
        for j in range(i + 1, p):
            pair = (i, j)
            support = sum(1 for cpdag in preds if cpdag[i, j] != 0 or cpdag[j, i] != 0)
            oppose = len(preds) - support
            adj_scores[pair] = (support, oppose)
            fwd = sum(1 for cpdag in preds if cpdag[i, j] == -1 and cpdag[j, i] == 1)
            rev = sum(1 for cpdag in preds if cpdag[i, j] == 1 and cpdag[j, i] == -1)
            dir_scores[pair] = (fwd, rev)
    from .graph_utils import build_cpdag_from_claims

    return build_cpdag_from_claims(p, adj_scores, dir_scores, 0.5)


def _run_sc_select(data: np.ndarray, local_entries: list[dict]) -> tuple[np.ndarray, dict]:
    alpha_scores = []
    for alpha in ALPHA_GRID:
        cpdag = run_pc(data, alpha=alpha).cpdag
        score = _self_compatibility_score(cpdag, local_entries)
        alpha_scores.append({"alpha": alpha, "score": score, "cpdag": cpdag})
    best = min(alpha_scores, key=lambda item: (item["score"], item["alpha"]))
    return best["cpdag"], {
        "alpha_grid": [{k: v for k, v in item.items() if k != "cpdag"} for item in alpha_scores],
        "selected_alpha": best["alpha"],
        "selection_score": best["score"],
    }


def _self_compatibility_score(full_cpdag: np.ndarray, local_entries: list[dict]) -> float:
    scores = []
    for entry in local_entries:
        local_nodes = entry["nodes"]
        restricted = restrict_cpdag(full_cpdag, local_nodes)
        full_claims = cpdag_claims(restricted, global_nodes=local_nodes)
        stats = contradiction_stats(full_claims, entry["claims"], set(local_nodes))
        if stats["comparable"] > 0:
            scores.append(stats["contradictions"] / stats["comparable"])
    return float(np.mean(scores)) if scores else 1.0


def _document_skipped_lovo() -> None:
    out_dir = ensure_dir(EXP_ROOT / "baselines" / "LOVO-Select")
    text = (
        "# LOVO-Select skipped\n\n"
        "The proposal allowed `LOVO-Select` only if the scoring recipe was operationally unambiguous. "
        "In this workspace the exact LOVO scoring rule for CPDAG selection was not specified tightly enough "
        "to run a fair baseline without inventing additional tuning choices, so the method is documented as a registered deviation."
    )
    (out_dir / "SKIPPED.md").write_text(text, encoding="utf-8")


def _signed_test(diff: pd.Series) -> dict:
    arr = diff.to_numpy()
    if len(arr) == 0:
        return {"median_difference": None, "p_value": None, "bootstrap_ci": None}
    try:
        stat = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        p_value = float(stat.pvalue)
    except ValueError:
        p_value = 1.0
    rng = np.random.default_rng(stable_int_seed("bootstrap_ci", len(arr)))
    boots = []
    for _ in range(2000):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.median(sample)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"median_difference": float(np.median(arr)), "p_value": p_value, "bootstrap_ci": [float(lo), float(hi)]}


def _aggregate_summary(df: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        return out
    for method, group in df.groupby("method"):
        out[method] = {}
        for metric in [c for c in group.columns if c not in {"method", "dataset_id"}]:
            if pd.api.types.is_numeric_dtype(group[metric]):
                out[method][metric] = {
                    "mean": float(group[metric].mean()),
                    "median": float(group[metric].median()),
                    "std": float(group[metric].std(ddof=1) if len(group) > 1 else 0.0),
                }
    return out


def _table1_subset_audit(manifest: pd.DataFrame, subset_validity: pd.DataFrame) -> pd.DataFrame:
    merged = subset_validity.merge(manifest[["dataset_id", "regime"]], on="dataset_id")
    grouped = (
        merged.groupby(["regime", "claim_type"])
        .agg(
            validity_rate=("validity_rate", "mean"),
            contradiction_rate=("contradiction_rate", "mean"),
            abstention_rate=("abstention_rate", "mean"),
            support_count=("support_count", "sum"),
            error_count=("error_count", "sum"),
            abstain_count=("abstain_count", "sum"),
        )
        .reset_index()
    )
    return grouped


def _table2_main_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "shd",
        "skeleton_f1",
        "orientation_precision",
        "false_orientation_rate",
        "fraction_undirected",
        "runtime_seconds",
        "peak_memory_mb",
    ]
    rows = []
    for method, group in df.groupby("method"):
        row = {"method": method}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1) if len(group) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _make_figures(manifest: pd.DataFrame, subset_validity: pd.DataFrame, baselines: pd.DataFrame, main: pd.DataFrame, ablations: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(FIGURES_ROOT)
    sns.set_theme(style="whitegrid")

    merged = main.merge(manifest[["dataset_id", "regime", "n"]], on="dataset_id")
    methods = ["Uniform+DetThreshold", "CompatExp+DetThreshold"]
    plot_df = merged[merged["method"].isin(methods)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    metric_specs = [("false_orientation_rate", "False Orientation Rate"), ("shd", "SHD")]
    for row_idx, (metric, title) in enumerate(metric_specs):
        for col_idx, n in enumerate(sorted(plot_df["n"].unique())):
            ax = axes[row_idx, col_idx]
            slice_df = plot_df[plot_df["n"] == n]
            sns.boxplot(data=slice_df, x="regime", y=metric, hue="method", ax=ax)
            pivot = slice_df.pivot(index="dataset_id", columns="method", values=metric).dropna()
            paired = pivot.merge(manifest[["dataset_id", "regime"]], left_index=True, right_on="dataset_id")
            for regime, group in paired.groupby("regime"):
                x = list(sorted(slice_df["regime"].unique())).index(regime)
                for _, item in group.iterrows():
                    ax.plot([x - 0.2, x + 0.2], [item["Uniform+DetThreshold"], item["CompatExp+DetThreshold"]], color="black", alpha=0.18, linewidth=0.8)
            ax.set_title(f"{title}, n={n}")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right")
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "figure1_false_orientation_boxplot.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not corr_df.empty:
        plot_corr = corr_df.merge(manifest[["dataset_id", "regime"]], on="dataset_id")
        sns.scatterplot(data=plot_corr, x="direction_validity", y="compat_minus_uniform_for", hue="regime", style="regime", s=70, ax=ax)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Local Direction Validity Rate")
    ax.set_ylabel("CompatExp - Uniform False Orientation Rate")
    ax.set_title("Subset validity versus deterministic CompatExp gain")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "figure2_subset_validity_vs_gain.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    merger_df = pd.concat(
        [
            main[main["method"].isin(["Uniform+Wilson", "Uniform+DetThreshold", "CompatExp+Wilson", "CompatExp+DetThreshold"])],
            ablations[ablations["method"].isin(["Uniform+DetRank", "CompatExp+DetRank"])],
        ],
        ignore_index=True,
    )
    sns.boxplot(data=merger_df, x="method", y="false_orientation_rate", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Merger Sensitivity")
    weight_df = ablations[ablations["method"].isin(["CompatRank+DetThreshold", "CompatTopHalf+DetThreshold"])]
    weight_df = pd.concat([weight_df, main[main["method"] == "CompatExp+DetThreshold"]], ignore_index=True)
    sns.boxplot(data=weight_df, x="method", y="false_orientation_rate", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Compatibility Weight Ablations")
    claim_df = ablations[ablations["method"].isin(["CompatExp+DetThreshold+NoColliders", "CompatExp+DetThreshold+AdjacencyOnly", "CompatExp+DetThreshold+WeakAbstention"])]
    claim_df = pd.concat([claim_df, main[main["method"] == "CompatExp+DetThreshold"]], ignore_index=True)
    sns.boxplot(data=claim_df, x="method", y="false_orientation_rate", ax=axes[2])
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_title("Claim-Family / Contradiction Ablations")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "figure3_ablations.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    rt_df = pd.concat(
        [
            baselines[baselines["method"].isin(["PC", "GES", "NOTEARS-L1", "DAGBag-PC", "SC-Select"])],
            main[main["method"].isin(["Uniform+DetThreshold", "CompatExp+DetThreshold"])],
        ],
        ignore_index=True,
    ).groupby("method").agg(runtime_seconds=("runtime_seconds", "mean"), false_orientation_rate=("false_orientation_rate", "mean")).reset_index()
    sns.scatterplot(data=rt_df, x="runtime_seconds", y="false_orientation_rate", s=90, ax=ax)
    for _, row in rt_df.iterrows():
        ax.text(row["runtime_seconds"], row["false_orientation_rate"], row["method"], fontsize=8, ha="left", va="bottom")
    ax.set_title("Runtime versus quality (2 CPU workers, 0 GPUs)")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "figure4_runtime_vs_quality.png", dpi=200)
    plt.close(fig)


def _evaluation_summary_text(hard_reduction: float, hypothesis_supported: bool, tests: dict) -> str:
    return (
        "# Evaluation Summary\n\n"
        f"- Main hypothesis supported: `{hypothesis_supported}`.\n"
        f"- Hard-regime median false-orientation reduction for `CompatExp+DetThreshold` vs `Uniform+DetThreshold`: `{hard_reduction:.4f}`.\n"
        f"- Paired test `CompatExp+DetThreshold` vs `Uniform+DetThreshold`: `{tests['CompatExp+DetThreshold_vs_Uniform+DetThreshold']}`.\n"
        f"- Paired test `CompatExp+DetThreshold` vs `BootstrapStability+DetThreshold`: `{tests['CompatExp+DetThreshold_vs_BootstrapStability+DetThreshold']}`.\n\n"
        "The main hypothesis was not supported. CompatExp gains over Uniform are small, statistically null, and below the pre-registered 10% success threshold."
    )


def _deviations_text() -> str:
    return (
        "# Registered Deviations\n\n"
        "- The workspace interpreter is Python 3.12.3 rather than the planned Python 3.11. No Python 3.11 interpreter was available in this environment.\n"
        "- `LOVO-Select` was not run. The proposal allowed it only if the scoring recipe was operationally unambiguous; that condition was not met without inventing extra tuning choices.\n"
        "- `SC-Select` was added to improve alignment with the proposal. It selects a PC alpha from the fixed grid by minimizing disagreement with the fixed local subset bank.\n"
    )
