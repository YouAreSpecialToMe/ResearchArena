import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.data import build_cifar_loaders
from exp.shared.metrics import (
    accuracy,
    adjusted_mi,
    average_effective_rank,
    cluster_subclasses,
    cosine_spread,
    fit_knn,
    fit_linear_probe,
    principal_angle_error,
    subclass_accuracy,
    weighted_supcon_loss,
)
from exp.shared.models import (
    EncoderWithHead,
    LinearClassifier,
    MLPEncoder,
    PointRepresentativeHead,
    SpanRepresentativeHead,
)
from exp.shared.synthetic_data import generate_synthetic_dataset
from exp.shared.utils import (
    AverageMeter,
    Timer,
    count_parameters,
    cosine_lr,
    ensure_dir,
    gpu_memory_mb,
    json_load,
    json_dump,
    set_seed,
    to_python,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_scalar_event(output_dir, stage, payload):
    record = {"stage": stage}
    record.update(to_python(payload))
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    log_path = Path(output_dir) / "logs" / "metrics.jsonl"
    ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_method_head(method, num_classes, embedding_dim, tau_c, rank=2):
    if method in {"cross_entropy", "supcon"}:
        return LinearClassifier(embedding_dim, num_classes)
    if method == "psc":
        return PointRepresentativeHead(num_classes, embedding_dim, num_reps=1, tau_c=tau_c)
    if method == "mpsc":
        return PointRepresentativeHead(num_classes, embedding_dim, num_reps=2, tau_c=tau_c)
    if method == "clop_style":
        return SpanRepresentativeHead(num_classes, embedding_dim, num_reps=1, rank=rank, tau_c=tau_c)
    if method.startswith("span"):
        return SpanRepresentativeHead(num_classes, embedding_dim, num_reps=2, rank=rank, tau_c=tau_c)
    raise ValueError(method)


def forward_head(head, method, z):
    if method in {"cross_entropy", "supcon"}:
        logits = head(z)
        return logits, None, None
    if isinstance(head, SpanRepresentativeHead):
        logits, scores, bases = head(z)
        return logits, scores, bases
    logits, scores = head(z)
    return logits, scores, None


def compute_assignment_weights(method, scores, labels, tau_a):
    if method in {"cross_entropy", "supcon"}:
        return None, None
    device = labels.device
    rows = torch.arange(len(labels), device=device)
    true_scores = scores[rows, labels]
    q = F.softmax(true_scores / tau_a, dim=-1)
    weights = q @ q.T
    return q, weights


def span_regularizers(bases, q, labels, num_classes):
    if bases is None:
        zero = labels.new_tensor(0.0, dtype=torch.float32)
        return zero, zero, 0.0, 0.0
    div = bases.new_tensor(0.0)
    if bases.shape[1] > 1:
        for cls in range(num_classes):
            base_cls = bases[cls]
            for k in range(base_cls.shape[0]):
                for kp in range(k + 1, base_cls.shape[0]):
                    div = div + torch.norm(base_cls[k].T @ base_cls[kp], p="fro") ** 2
    if q is None:
        cov = bases.new_tensor(0.0)
        entropy = 0.0
        dormant = 0.0
    else:
        entropy_values = []
        dormant_values = []
        cov = bases.new_tensor(0.0)
        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() == 0:
                continue
            occ = q[mask].mean(dim=0)
            if q.shape[1] == 1:
                entropy = occ.new_tensor(1.0)
            else:
                entropy = -(occ * torch.log(occ + 1e-8)).sum() / np.log(q.shape[1])
            entropy_values.append(entropy)
            dormant_values.append((occ < 0.05).float().mean())
            cov = cov + (1.0 - entropy)
        entropy = torch.stack(entropy_values).mean().item() if entropy_values else 0.0
        dormant = torch.stack(dormant_values).mean().item() if dormant_values else 0.0
    overlap = div.item() / max(num_classes, 1)
    return div, cov, entropy, dormant


def extract_features(encoder, loader, device):
    encoder.eval()
    feats = []
    proj = []
    fine = []
    coarse = []
    with torch.no_grad():
        for images, fine_labels, coarse_labels, _ in loader:
            images = images.to(device, non_blocking=True)
            feat, z = encoder(images)
            feats.append(feat.cpu().numpy())
            proj.append(z.cpu().numpy())
            fine.append(fine_labels.numpy())
            coarse.append(coarse_labels.numpy())
    return (
        np.concatenate(feats),
        np.concatenate(proj),
        np.concatenate(fine),
        np.concatenate(coarse),
    )


def evaluate_transfer(encoder, train_loader, test_loader, device, diagnostics):
    _, train_proj, train_fine, _ = extract_features(encoder, train_loader, device)
    _, test_proj, test_fine, _ = extract_features(encoder, test_loader, device)
    num_fine_classes = int(max(train_fine.max(), test_fine.max()) + 1)
    lp_acc, lp_f1, _, _ = fit_linear_probe(
        train_proj,
        train_fine,
        test_proj,
        test_fine,
        num_classes=num_fine_classes,
        device=device,
        epochs=30,
    )
    knn_acc, knn_f1, _ = fit_knn(
        train_proj, train_fine, test_proj, test_fine, n_neighbors=20, device=device
    )
    result = {
        "transfer_representation": "normalized_projection_128d",
        "linear_probe_top1": lp_acc,
        "linear_probe_macro_f1": lp_f1,
        "knn_top1": knn_acc,
        "knn_macro_f1": knn_f1,
        "effective_rank_fine": average_effective_rank(test_proj, test_fine),
        "same_class_cosine_spread": cosine_spread(test_proj, test_fine),
    }
    result.update(diagnostics)
    return result


def final_diagnostics_from_curves(method, curves):
    if not curves:
        return {
            "final_assignment_entropy": 1.0 if method == "clop_style" else 0.0,
            "final_dormant_span_fraction": 0.0,
            "final_span_overlap": 0.0,
        }
    last = curves[-1]
    entropy = last.get("assignment_entropy", 0.0)
    if not np.isfinite(entropy):
        entropy = 1.0 if method == "clop_style" else 0.0
    dormant = last.get("dormant_span_fraction", 0.0)
    if not np.isfinite(dormant):
        dormant = 0.0
    overlap = last.get("span_overlap", 0.0)
    if not np.isfinite(overlap):
        overlap = 0.0
    return {
        "final_assignment_entropy": float(entropy),
        "final_dormant_span_fraction": float(dormant),
        "final_span_overlap": float(overlap),
    }


def train_epoch(
    encoder,
    head,
    loader,
    optimizer,
    scaler,
    device,
    config,
    method,
    num_classes,
):
    encoder.train()
    head.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    supcon_meter = AverageMeter()
    acc_meter = AverageMeter()
    ent_meter = AverageMeter()
    dorm_meter = AverageMeter()
    overlap_meter = AverageMeter()
    for view1, view2, labels, _, _, _ in tqdm(loader, leave=False):
        x = torch.cat([view1, view2], dim=0).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_all = labels.repeat(2)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, z = encoder(x)
            logits, scores, bases = forward_head(head, method, z)
            ce = F.cross_entropy(logits, labels_all)
            q, weights = compute_assignment_weights(method, scores, labels_all, config["tau_a"])
            if method == "cross_entropy":
                supcon = z.new_tensor(0.0)
            else:
                supcon = weighted_supcon_loss(
                    z, labels_all, temperature=config["supcon_temp"], positive_weights=weights
                )
            div, cov, entropy, dormant = span_regularizers(bases, q, labels_all, num_classes)
            if method == "span_rank1":
                div_w = config["lambda_div"]
                cov_w = config["lambda_cov"]
            elif method == "span_no_div":
                div_w = 0.0
                cov_w = config["lambda_cov"]
            elif method == "span_no_cov":
                div_w = config["lambda_div"]
                cov_w = 0.0
            elif method.startswith("span"):
                div_w = config["lambda_div"]
                cov_w = config["lambda_cov"]
            else:
                div_w = 0.0
                cov_w = 0.0
            loss = ce + config["lambda_supcon"] * supcon + div_w * div + cov_w * cov
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if isinstance(head, SpanRepresentativeHead):
            with torch.no_grad():
                flat = head.bases.view(-1, head.bases.shape[-2], head.bases.shape[-1])
                qmat, _ = torch.linalg.qr(flat, mode="reduced")
                head.bases.copy_(qmat.view_as(head.bases))
        loss_meter.update(loss.item(), labels.shape[0])
        ce_meter.update(ce.item(), labels.shape[0])
        supcon_meter.update(supcon.item(), labels.shape[0])
        acc_meter.update(accuracy(logits[: labels.shape[0]], labels), labels.shape[0])
        ent_meter.update(entropy, labels.shape[0])
        dorm_meter.update(dormant, labels.shape[0])
        overlap_meter.update(div.item() / max(num_classes, 1), labels.shape[0])
    return {
        "loss": loss_meter.avg,
        "ce": ce_meter.avg,
        "supcon": supcon_meter.avg,
        "train_top1": acc_meter.avg * 100.0,
        "assignment_entropy": ent_meter.avg,
        "dormant_span_fraction": dorm_meter.avg,
        "span_overlap": overlap_meter.avg,
    }


def eval_classifier(encoder, head, loader, device, label_type):
    encoder.eval()
    head.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, fine_labels, coarse_labels, _ in loader:
            labels = coarse_labels if label_type == "coarse" else fine_labels
            images = images.to(device, non_blocking=True)
            _, z = encoder(images)
            logits, _, _ = forward_head(head, "supcon" if isinstance(head, LinearClassifier) else "span", z)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def run_cifar(config, output_dir):
    set_seed(config["seed"])
    device = get_device()
    ensure_dir(output_dir)
    train_loader, train_eval_loader, val_loader, test_loader, _ = build_cifar_loaders(
        config["data_root"],
        config["results_root"],
        config["batch_size"],
        config["num_workers"],
        config["label_type"],
    )
    num_classes = 20 if config["label_type"] == "coarse" else 100
    encoder = EncoderWithHead(config["embedding_dim"]).to(device)
    head = build_method_head(
        config["method"], num_classes, config["embedding_dim"], config["tau_c"], config["rank"]
    ).to(device)
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(head.parameters()),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    timer = Timer()
    curves = []
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(config["epochs"]):
        lr = cosine_lr(optimizer, config["lr"], epoch, config["epochs"])
        train_stats = train_epoch(
            encoder, head, train_loader, optimizer, scaler, device, config, config["method"], num_classes
        )
        val_top1 = eval_classifier(encoder, head, val_loader, device, config["label_type"])
        train_stats["epoch"] = epoch + 1
        train_stats["lr"] = lr
        train_stats["val_top1"] = val_top1
        curves.append(train_stats)
        log_scalar_event(output_dir, "epoch", train_stats)
    final_diag = final_diagnostics_from_curves(config["method"], curves)
    if config["label_type"] == "coarse":
        transfer = evaluate_transfer(encoder, train_eval_loader, test_loader, device, final_diag)
    else:
        fine_top1 = eval_classifier(encoder, head, test_loader, device, "fine")
        transfer = {"fine_label_test_top1": fine_top1}
    result = {
        "experiment": config["method"],
        "setting": config["setting"],
        "seed": config["seed"],
        "config": to_python(config),
        "training_curves": curves,
        "metrics": transfer,
        "runtime_minutes": timer.minutes(),
        "peak_gpu_memory_mb": gpu_memory_mb(),
        "parameter_count": count_parameters(encoder) + count_parameters(head),
    }
    log_scalar_event(
        output_dir,
        "final",
        {
            "experiment": config["method"],
            "setting": config["setting"],
            "seed": config["seed"],
            "metrics": result["metrics"],
            "runtime_minutes": result["runtime_minutes"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "parameter_count": result["parameter_count"],
        },
    )
    ckpt = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "config": config,
        "metrics": result["metrics"],
    }
    torch.save(ckpt, Path(output_dir) / "checkpoint.pt")
    json_dump(result, Path(output_dir) / "results.json")
    return result


def build_synth_loaders(data, batch_size):
    train_ds = TensorDataset(torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"]), torch.from_numpy(data["train_sub"]))
    test_ds = TensorDataset(torch.from_numpy(data["test_x"]), torch.from_numpy(data["test_y"]), torch.from_numpy(data["test_sub"]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    return train_loader, test_loader


def infer_synthetic_subclasses(encoder, head, test_loader, device, method, num_classes):
    encoder.eval()
    head.eval()
    with torch.no_grad():
        all_feat = []
        all_class = []
        all_true_sub = []
        for x, y, sub in test_loader:
            x = x.to(device)
            _, z = encoder(x)
            all_feat.append(z.cpu().numpy())
            all_class.append(y.numpy())
            all_true_sub.append(sub.numpy())
    features = np.concatenate(all_feat)
    class_labels = np.concatenate(all_class)
    true_sub = np.concatenate(all_true_sub)
    pred_labels, pred_bases = cluster_subclasses(features, class_labels, num_subclasses=2)
    return features, class_labels, true_sub, pred_labels, pred_bases


def run_synthetic(config, output_dir):
    set_seed(config["seed"])
    device = get_device()
    ensure_dir(output_dir)
    data = generate_synthetic_dataset(
        config["synthetic_root"], config["seed"], config["regime"], rank=2
    )
    train_loader, test_loader = build_synth_loaders(data, config["batch_size"])
    encoder = MLPEncoder(32, 128, 32).to(device)
    head = build_method_head(config["method"], 10, 32, config["tau_c"], config["rank"]).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=config["lr"]
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    timer = Timer()
    curves = []
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(config["epochs"]):
        encoder.train()
        head.train()
        meter = AverageMeter()
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                _, z = encoder(x)
                logits, scores, bases = forward_head(head, config["method"], z)
                ce = F.cross_entropy(logits, y)
                q, weights = compute_assignment_weights(config["method"], scores, y, config["tau_a"])
                supcon = weighted_supcon_loss(z, y, temperature=config["supcon_temp"], positive_weights=weights)
                div, cov, entropy, dormant = span_regularizers(bases, q, y, 10)
                div_w = config["lambda_div"] if config["method"].startswith("span") else 0.0
                cov_w = config["lambda_cov"] if config["method"].startswith("span") else 0.0
                loss = ce + config["lambda_supcon"] * supcon + div_w * div + cov_w * cov
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item(), len(y))
        epoch_stats = {"epoch": epoch + 1, "loss": meter.avg}
        curves.append(epoch_stats)
        log_scalar_event(output_dir, "epoch", epoch_stats)
    features, class_labels, true_sub, pred_sub, pred_bases = infer_synthetic_subclasses(
        encoder, head, test_loader, device, config["method"], 10
    )
    angle = principal_angle_error(pred_bases, data["true_bases"])
    result = {
        "experiment": f"synthetic_{config['method']}",
        "setting": config["regime"],
        "seed": config["seed"],
        "config": to_python(config),
        "training_curves": curves,
        "metrics": {
            "ami": adjusted_mi(true_sub, pred_sub),
            "subclass_recovery_top1": subclass_accuracy(true_sub, pred_sub),
            "principal_angle_error_deg": angle,
        },
        "runtime_minutes": timer.minutes(),
        "peak_gpu_memory_mb": gpu_memory_mb(),
        "parameter_count": count_parameters(encoder) + count_parameters(head),
    }
    log_scalar_event(
        output_dir,
        "final",
        {
            "experiment": f"synthetic_{config['method']}",
            "setting": config["regime"],
            "seed": config["seed"],
            "metrics": result["metrics"],
            "runtime_minutes": result["runtime_minutes"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "parameter_count": result["parameter_count"],
        },
    )
    ckpt = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "config": config,
        "metrics": result["metrics"],
    }
    torch.save(ckpt, Path(output_dir) / "checkpoint.pt")
    json_dump(result, Path(output_dir) / "results.json")
    return result


def summarize_results(run_results):
    grouped = {}
    for item in run_results:
        key = (item["experiment"], item["setting"])
        grouped.setdefault(key, []).append(item)
    summary = []
    for (exp_name, setting), items in sorted(grouped.items()):
        metrics = {}
        metric_names = items[0]["metrics"].keys()
        for name in metric_names:
            values = [run["metrics"][name] for run in items if name in run["metrics"]]
            if not values or not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                continue
            values = [float(v) for v in values if np.isfinite(v)]
            if not values:
                continue
            metrics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "values": values,
            }
        summary.append(
            {
                "experiment": exp_name,
                "setting": setting,
                "num_seeds": len(items),
                "metrics": metrics,
                "runtime_minutes": {
                    "mean": float(np.mean([r["runtime_minutes"] for r in items])),
                    "std": float(np.std([r["runtime_minutes"] for r in items], ddof=0)),
                },
                "peak_gpu_memory_mb": {
                    "mean": float(np.mean([r["peak_gpu_memory_mb"] for r in items])),
                    "std": float(np.std([r["peak_gpu_memory_mb"] for r in items], ddof=0)),
                },
            }
        )
    return summary


def collect_results(exp_root):
    results = []
    for path in Path(exp_root).glob("**/results.json"):
        if path.name == "results.json" and "logs" not in path.parts:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
    return results


def save_csv(rows, path):
    if not rows:
        return
    ensure_dir(Path(path).parent)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(exp_root, output_path):
    runs = collect_results(exp_root)
    summary = summarize_results(runs)
    complete_runs = [
        run
        for run in runs
        if run.get("setting") != "smoke"
        and run.get("config", {}).get("status", run.get("status", "completed")) == "completed"
    ]
    complete_summary = summarize_results(complete_runs)
    json_dump(
        {
            "runs": runs,
            "summary": summary,
            "completed_runs": complete_runs,
            "completed_summary": complete_summary,
        },
        output_path,
    )
    rows = []
    for item in complete_summary:
        row = {"experiment": item["experiment"], "setting": item["setting"]}
        for name, values in item["metrics"].items():
            row[f"{name}_mean"] = values["mean"]
            row[f"{name}_std"] = values["std"]
        row["runtime_mean"] = item["runtime_minutes"]["mean"]
        row["memory_mean"] = item["peak_gpu_memory_mb"]["mean"]
        rows.append(row)
    save_csv(rows, Path(output_path).parent / "summary.csv")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_models(config, checkpoint, device):
    if config["dataset"] == "cifar100":
        num_classes = 20 if config["label_type"] == "coarse" else 100
        encoder = EncoderWithHead(config["embedding_dim"]).to(device)
        head = build_method_head(
            config["method"], num_classes, config["embedding_dim"], config["tau_c"], config["rank"]
        ).to(device)
    elif config["dataset"] == "synthetic":
        encoder = MLPEncoder(32, 128, 32).to(device)
        head = build_method_head(config["method"], 10, 32, config["tau_c"], config["rank"]).to(device)
    else:
        raise ValueError(config["dataset"])
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    head.eval()
    return encoder, head


def rewrite_audit_log(output_dir, result):
    log_path = Path(output_dir) / "logs" / "metrics.jsonl"
    if log_path.exists():
        log_path.unlink()
    for epoch_stats in result.get("training_curves", []):
        log_scalar_event(output_dir, "epoch", epoch_stats)
    log_scalar_event(
        output_dir,
        "final",
        {
            "experiment": result["experiment"],
            "setting": result["setting"],
            "seed": result["seed"],
            "metrics": result["metrics"],
            "runtime_minutes": result["runtime_minutes"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "parameter_count": result["parameter_count"],
        },
    )


def reevaluate_checkpoint(config, output_dir):
    output_dir = Path(output_dir)
    result_path = output_dir / "results.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing results file: {result_path}")
    old_result = json_load(result_path)
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder, head = load_checkpoint_models(config, checkpoint, device)

    if config["dataset"] == "cifar100":
        _, train_eval_loader, _, test_loader, _ = build_cifar_loaders(
            config["data_root"],
            config["results_root"],
            config["batch_size"],
            int(config.get("num_workers", 4)),
            config["label_type"],
        )
        if config["label_type"] == "coarse":
            curves = old_result.get("training_curves", [])
            final_diag = final_diagnostics_from_curves(config["method"], curves)
            metrics = evaluate_transfer(encoder, train_eval_loader, test_loader, device, final_diag)
        else:
            metrics = {"fine_label_test_top1": eval_classifier(encoder, head, test_loader, device, "fine")}
    elif config["dataset"] == "synthetic":
        data = generate_synthetic_dataset(config["synthetic_root"], config["seed"], config["regime"], rank=2)
        _, test_loader = build_synth_loaders(data, config["batch_size"])
        _, _, true_sub, pred_sub, pred_bases = infer_synthetic_subclasses(
            encoder, head, test_loader, device, config["method"], 10
        )
        metrics = {
            "ami": adjusted_mi(true_sub, pred_sub),
            "subclass_recovery_top1": subclass_accuracy(true_sub, pred_sub),
            "principal_angle_error_deg": principal_angle_error(pred_bases, data["true_bases"]),
        }
    else:
        raise ValueError(config["dataset"])

    old_result["metrics"] = to_python(metrics)
    json_dump(old_result, result_path)
    rewrite_audit_log(output_dir, old_result)
    return old_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate("exp", args.output_dir)
        return
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    json_dump(config, output_dir / "config.json")
    if config["dataset"] == "cifar100":
        config["num_workers"] = int(config.get("num_workers", 4))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.reevaluate:
        reevaluate_checkpoint(config, output_dir)
        return
    if config["dataset"] == "cifar100":
        run_cifar(config, output_dir)
    elif config["dataset"] == "synthetic":
        run_synthetic(config, output_dir)
    else:
        raise ValueError(config["dataset"])


if __name__ == "__main__":
    main()
