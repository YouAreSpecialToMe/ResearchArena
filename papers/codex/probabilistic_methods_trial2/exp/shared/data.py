from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DATA_PROCESSED, DATA_RAW, RESULTS_DIR, SEEDS
from .io import write_json

ANURAN_URL = "https://archive.ics.uci.edu/static/public/406/anuran+calls+mfccs.zip"


@dataclass
class DatasetBundle:
    name: str
    seed: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    eval_meta_cal: dict[str, np.ndarray]
    eval_meta_test: dict[str, np.ndarray]
    candidate_group_cal: np.ndarray
    candidate_group_test: np.ndarray
    candidate_group_names: list[str]
    group_memberships_test: list[str]
    oracle_membership_cal: np.ndarray | None = None
    oracle_membership_test: np.ndarray | None = None


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_csv_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle)


def _quantile_stats(X_train: np.ndarray) -> dict[str, float]:
    return {
        "train_feature_mean_abs": float(np.mean(np.abs(X_train))),
        "train_feature_std_mean": float(np.mean(np.std(X_train, axis=0))),
        "train_feature_std_min": float(np.min(np.std(X_train, axis=0))),
        "train_feature_std_max": float(np.max(np.std(X_train, axis=0))),
    }


def prepare_synthetic(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_train, n_cal, n_test = 4000, 2000, 2000
    d, n_classes, n_coarse, n_fine_per_coarse = 12, 4, 4, 2
    total = n_train + n_cal + n_test
    z_coarse = rng.integers(0, n_coarse, size=total)
    z_fine = np.array([rng.integers(0, n_fine_per_coarse) for _ in range(total)])
    X = np.zeros((total, d))
    y = np.zeros(total, dtype=int)
    coarse_shifts = rng.normal(0.0, 2.0, size=(n_coarse, d))
    fine_shifts = rng.normal(0.0, 0.8, size=(n_coarse, n_fine_per_coarse, d))
    class_logits = rng.normal(0.0, 0.3, size=(n_coarse, n_fine_per_coarse, n_classes))
    for i in range(total):
        c, f = z_coarse[i], z_fine[i]
        cov_scale = 0.6 + 0.4 * c
        X[i] = rng.normal(coarse_shifts[c] + fine_shifts[c, f], cov_scale, size=d)
        logits = class_logits[c, f].copy()
        logits[(c + f) % n_classes] += 1.2
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        y[i] = rng.choice(n_classes, p=probs)

    X_train, X_rest, y_train, y_rest, c_train, c_rest, f_train, f_rest = train_test_split(
        X, y, z_coarse, z_fine, train_size=n_train, random_state=seed, stratify=y
    )
    X_cal, X_test, y_cal, y_test, c_cal, c_test, f_cal, f_test = train_test_split(
        X_rest, y_rest, c_rest, f_rest, train_size=n_cal, random_state=seed + 1, stratify=y_rest
    )

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    safe_std = np.where(train_std > 0, train_std, 1.0)
    payload = {
        "X_train": (X_train - train_mean) / safe_std,
        "X_cal": (X_cal - train_mean) / safe_std,
        "X_test": (X_test - train_mean) / safe_std,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
        "label_names": np.array([f"class_{i}" for i in range(n_classes)], dtype=object),
        "coarse_cal": c_cal,
        "coarse_test": c_test,
        "fine_cal": c_cal * n_fine_per_coarse + f_cal,
        "fine_test": c_test * n_fine_per_coarse + f_test,
        "coarse_by_class_cal": np.array([f"{c}|{yy}" for c, yy in zip(c_cal, y_cal)], dtype=object),
        "coarse_by_class_test": np.array([f"{c}|{yy}" for c, yy in zip(c_test, y_test)], dtype=object),
    }
    _save_npz(DATA_PROCESSED / f"synthetic_seed{seed}.npz", **payload)
    return {
        "n_raw_missing": 0,
        "n_dropped_all_missing_columns": 0,
        "imputation": "none",
        "scaling": "train mean/std",
        **_quantile_stats(payload["X_train"]),
    }


def prepare_anuran(seed: int) -> dict[str, Any]:
    zip_path = DATA_RAW / "anuran_calls_mfccs.zip"
    if not zip_path.exists():
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        urlretrieve(ANURAN_URL, zip_path)
    df = _load_csv_from_zip(zip_path, "Frogs_MFCCs.csv")
    feature_cols = [c for c in df.columns if c.startswith("MFCCs_")]
    X = df[feature_cols].to_numpy(dtype=float)
    y_raw = df["Species"].astype(str).to_numpy()
    family = df["Family"].astype(str).to_numpy()
    genus = df["Genus"].astype(str).to_numpy()
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, train_size=0.5, random_state=seed, stratify=y)
    cal_idx, test_idx = train_test_split(rest_idx, train_size=0.5, random_state=seed + 1, stratify=y[rest_idx])
    scaler = StandardScaler().fit(X[train_idx])
    payload = {
        "X_train": scaler.transform(X[train_idx]),
        "X_cal": scaler.transform(X[cal_idx]),
        "X_test": scaler.transform(X[test_idx]),
        "y_train": y[train_idx],
        "y_cal": y[cal_idx],
        "y_test": y[test_idx],
        "label_names": le.classes_.astype(object),
        "family_cal": family[cal_idx].astype(object),
        "family_test": family[test_idx].astype(object),
        "genus_cal": genus[cal_idx].astype(object),
        "genus_test": genus[test_idx].astype(object),
    }
    _save_npz(DATA_PROCESSED / f"anuran_seed{seed}.npz", **payload)
    return {
        "n_raw_missing": int(df[feature_cols].isna().sum().sum()),
        "n_dropped_all_missing_columns": 0,
        "imputation": "none",
        "scaling": "train mean/std",
        **_quantile_stats(payload["X_train"]),
    }


def prepare_mice(seed: int) -> dict[str, Any]:
    ds = fetch_openml(name="MiceProtein", as_frame=True, parser="auto")
    df = ds.data.copy()
    y_raw = ds.target.astype(str)
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    raw_missing = int(df.isna().sum().sum())
    df = df.drop(columns=drop_cols)
    train_idx, rest_idx = train_test_split(
        np.arange(len(df)), train_size=0.5, random_state=seed, stratify=y_raw
    )
    cal_idx, test_idx = train_test_split(
        rest_idx, train_size=0.5, random_state=seed + 1, stratify=y_raw.iloc[rest_idx]
    )
    train_medians = df.iloc[train_idx].median(numeric_only=True)
    df = df.fillna(train_medians)
    scaler = StandardScaler().fit(df.iloc[train_idx])
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    def parse_tokens(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        toks = values.str.split("-", expand=True)
        return toks[0].astype(str).to_numpy(), toks[1].astype(str).to_numpy(), toks[2].astype(str).to_numpy()

    genotype, treatment, behavior = parse_tokens(y_raw)
    payload = {
        "X_train": scaler.transform(df.iloc[train_idx]),
        "X_cal": scaler.transform(df.iloc[cal_idx]),
        "X_test": scaler.transform(df.iloc[test_idx]),
        "y_train": y[train_idx],
        "y_cal": y[cal_idx],
        "y_test": y[test_idx],
        "label_names": le.classes_.astype(object),
        "genotype_cal": genotype[cal_idx].astype(object),
        "genotype_test": genotype[test_idx].astype(object),
        "treatment_cal": treatment[cal_idx].astype(object),
        "treatment_test": treatment[test_idx].astype(object),
        "behavior_cal": behavior[cal_idx].astype(object),
        "behavior_test": behavior[test_idx].astype(object),
    }
    _save_npz(DATA_PROCESSED / f"mice_seed{seed}.npz", **payload)
    return {
        "n_raw_missing": raw_missing,
        "n_dropped_all_missing_columns": len(drop_cols),
        "imputation": "train median",
        "scaling": "train mean/std",
        **_quantile_stats(payload["X_train"]),
    }


def _one_hot(values: np.ndarray, ordered_labels: list[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(ordered_labels)}
    out = np.zeros((len(values), len(ordered_labels)), dtype=int)
    for row_idx, value in enumerate(values.astype(str)):
        out[row_idx, mapping[value]] = 1
    return out


def _serialize_memberships(eval_meta: dict[str, np.ndarray], idx: int) -> str:
    parts = []
    for family, groups in sorted(eval_meta.items()):
        parts.append(f"{family}={groups[idx]}")
    return "; ".join(parts)


def prepare_all_datasets() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    group_rules: dict[str, dict[str, Any]] = {}
    prep_meta_by_dataset: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        prep_meta_by_dataset[f"synthetic_seed{seed}"] = prepare_synthetic(seed)
        prep_meta_by_dataset[f"anuran_seed{seed}"] = prepare_anuran(seed)
        prep_meta_by_dataset[f"mice_seed{seed}"] = prepare_mice(seed)
        for dataset in ["synthetic", "anuran", "mice"]:
            bundle = load_dataset(dataset, seed)
            key = f"{dataset}_seed{seed}"
            group_counts = {name: int(len(np.unique(values))) for name, values in bundle.eval_meta_test.items()}
            summaries[key] = {
                "n_train": int(bundle.X_train.shape[0]),
                "n_cal": int(bundle.X_cal.shape[0]),
                "n_test": int(bundle.X_test.shape[0]),
                "n_features": int(bundle.X_train.shape[1]),
                "n_classes": int(len(bundle.label_names)),
                "group_count_by_family": group_counts,
                "n_candidate_groups": int(bundle.candidate_group_cal.shape[1]),
                "class_frequencies_train": {
                    str(lbl): int((bundle.y_train == idx).sum()) for idx, lbl in enumerate(bundle.label_names)
                },
                **prep_meta_by_dataset[key],
            }
            group_rules[key] = {
                "candidate_groups": bundle.candidate_group_names,
                "evaluation_group_families": {
                    name: sorted(np.unique(values.astype(str)).tolist()) for name, values in bundle.eval_meta_test.items()
                },
            }
    write_json(RESULTS_DIR / "dataset_summary.json", summaries)
    write_json(RESULTS_DIR / "group_definitions.json", group_rules)


def load_dataset(name: str, seed: int) -> DatasetBundle:
    payload = np.load(DATA_PROCESSED / f"{name}_seed{seed}.npz", allow_pickle=True)
    label_names = [str(x) for x in payload["label_names"].tolist()]

    if name == "synthetic":
        eval_meta_cal = {
            "coarse": payload["coarse_cal"].astype(object),
            "fine": payload["fine_cal"].astype(object),
            "coarse_by_class": payload["coarse_by_class_cal"].astype(object),
        }
        eval_meta_test = {
            "coarse": payload["coarse_test"].astype(object),
            "fine": payload["fine_test"].astype(object),
            "coarse_by_class": payload["coarse_by_class_test"].astype(object),
        }
        coarse_labels = [str(i) for i in range(4)]
        fine_labels = [str(i) for i in range(8)]
        candidate_group_names = [f"coarse={i}" for i in coarse_labels] + [f"fine={i}" for i in fine_labels]
        candidate_group_cal = np.concatenate(
            [
                _one_hot(payload["coarse_cal"].astype(str), coarse_labels),
                _one_hot(payload["fine_cal"].astype(str), fine_labels),
            ],
            axis=1,
        )
        candidate_group_test = np.concatenate(
            [
                _one_hot(payload["coarse_test"].astype(str), coarse_labels),
                _one_hot(payload["fine_test"].astype(str), fine_labels),
            ],
            axis=1,
        )
        oracle_membership_cal = np.eye(8, dtype=float)[payload["fine_cal"].astype(int)]
        oracle_membership_test = np.eye(8, dtype=float)[payload["fine_test"].astype(int)]
    elif name == "anuran":
        family_cal = payload["family_cal"].astype(str)
        family_test = payload["family_test"].astype(str)
        genus_cal = payload["genus_cal"].astype(str)
        genus_test = payload["genus_test"].astype(str)
        family_class_cal = np.array([f"{fam}|{label_names[int(y)]}" for fam, y in zip(family_cal, payload["y_cal"])], dtype=object)
        family_class_test = np.array([f"{fam}|{label_names[int(y)]}" for fam, y in zip(family_test, payload["y_test"])], dtype=object)
        eval_meta_cal = {
            "Family": family_cal.astype(object),
            "Genus": genus_cal.astype(object),
            "Family x true class": family_class_cal,
        }
        eval_meta_test = {
            "Family": family_test.astype(object),
            "Genus": genus_test.astype(object),
            "Family x true class": family_class_test,
        }
        families = sorted(np.unique(family_test).tolist())
        genera = sorted(np.unique(genus_test).tolist())
        family_classes = sorted(np.unique(family_class_test.astype(str)).tolist())
        candidate_group_names = (
            [f"Family={x}" for x in families]
            + [f"Genus={x}" for x in genera]
            + [f"FamilyClass={x}" for x in family_classes]
        )
        candidate_group_cal = np.concatenate(
            [_one_hot(family_cal, families), _one_hot(genus_cal, genera), _one_hot(family_class_cal.astype(str), family_classes)],
            axis=1,
        )
        candidate_group_test = np.concatenate(
            [_one_hot(family_test, families), _one_hot(genus_test, genera), _one_hot(family_class_test.astype(str), family_classes)],
            axis=1,
        )
        oracle_membership_cal = None
        oracle_membership_test = None
    else:
        genotype_cal = payload["genotype_cal"].astype(str)
        genotype_test = payload["genotype_test"].astype(str)
        treatment_cal = payload["treatment_cal"].astype(str)
        treatment_test = payload["treatment_test"].astype(str)
        behavior_cal = payload["behavior_cal"].astype(str)
        behavior_test = payload["behavior_test"].astype(str)
        genotype_treatment_cal = np.array([f"{g}|{t}" for g, t in zip(genotype_cal, treatment_cal)], dtype=object)
        genotype_treatment_test = np.array([f"{g}|{t}" for g, t in zip(genotype_test, treatment_test)], dtype=object)
        eval_meta_cal = {
            "Genotype": genotype_cal.astype(object),
            "Treatment": treatment_cal.astype(object),
            "Behavior": behavior_cal.astype(object),
            "Genotype x Treatment": genotype_treatment_cal,
        }
        eval_meta_test = {
            "Genotype": genotype_test.astype(object),
            "Treatment": treatment_test.astype(object),
            "Behavior": behavior_test.astype(object),
            "Genotype x Treatment": genotype_treatment_test,
        }
        all_g = sorted(np.unique(genotype_test).tolist())
        all_t = sorted(np.unique(treatment_test).tolist())
        all_b = sorted(np.unique(behavior_test).tolist())
        all_gt = sorted(np.unique(genotype_treatment_test.astype(str)).tolist())
        candidate_group_names = (
            [f"Genotype={x}" for x in all_g]
            + [f"Treatment={x}" for x in all_t]
            + [f"Behavior={x}" for x in all_b]
            + [f"GenotypeTreatment={x}" for x in all_gt]
        )
        candidate_group_cal = np.concatenate(
            [
                _one_hot(genotype_cal, all_g),
                _one_hot(treatment_cal, all_t),
                _one_hot(behavior_cal, all_b),
                _one_hot(genotype_treatment_cal.astype(str), all_gt),
            ],
            axis=1,
        )
        candidate_group_test = np.concatenate(
            [
                _one_hot(genotype_test, all_g),
                _one_hot(treatment_test, all_t),
                _one_hot(behavior_test, all_b),
                _one_hot(genotype_treatment_test.astype(str), all_gt),
            ],
            axis=1,
        )
        oracle_membership_cal = None
        oracle_membership_test = None

    group_memberships_test = [_serialize_memberships(eval_meta_test, i) for i in range(len(payload["y_test"]))]
    return DatasetBundle(
        name=name,
        seed=seed,
        X_train=payload["X_train"].astype(float),
        y_train=payload["y_train"].astype(int),
        X_cal=payload["X_cal"].astype(float),
        y_cal=payload["y_cal"].astype(int),
        X_test=payload["X_test"].astype(float),
        y_test=payload["y_test"].astype(int),
        label_names=label_names,
        eval_meta_cal=eval_meta_cal,
        eval_meta_test=eval_meta_test,
        candidate_group_cal=candidate_group_cal.astype(int),
        candidate_group_test=candidate_group_test.astype(int),
        candidate_group_names=candidate_group_names,
        group_memberships_test=group_memberships_test,
        oracle_membership_cal=oracle_membership_cal,
        oracle_membership_test=oracle_membership_test,
    )
