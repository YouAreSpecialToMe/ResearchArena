#!/usr/bin/env python3
"""Measure wall-clock training time overhead for each method.

Runs 15 epochs (discard first 5 as warmup) and measures time per epoch.
"""

import time
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.shared.models import SupConModel, CEModel
from exp.shared.losses import SupConLoss, HardNegLoss, TCLLoss, ReweightedSupConLoss, VarConTLoss, CGALoss
from exp.shared.utils import seed_everything, ConfusionTracker, PrototypeTracker
from exp.shared.data_loader import get_dataloaders

WORKSPACE = '/home/zz865/pythonProject/autoresearch/outputs/claude/run_1/supervised_representation_learning/idea_01'


def time_method(method_name, num_epochs=15, warmup=5):
    """Time training for a given method."""
    device = 'cuda'
    seed_everything(42)
    num_classes = 100

    is_ce = (method_name == 'ce')
    train_loader, _, _ = get_dataloaders('cifar100', batch_size=512, num_workers=2,
                                          two_crop=not is_ce, data_root='./data')

    if is_ce:
        model = CEModel('resnet18', num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = SupConModel('resnet18', proj_dim=128).to(device)

    if method_name == 'supcon':
        criterion = SupConLoss(0.07)
    elif method_name == 'hardneg':
        criterion = HardNegLoss(0.07, beta=1.0)
    elif method_name == 'tcl':
        criterion = TCLLoss(0.07)
    elif method_name == 'cga_only':
        criterion = SupConLoss(0.07)
        cga_fn = CGALoss(num_classes, alpha=0.3).to(device)
    elif method_name == 'reweight':
        criterion = ReweightedSupConLoss(0.07, beta_rw=2.0)

    if not is_ce:
        criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)

    # Trackers for methods that need them
    proto_tracker = None
    conf_tracker = None
    if method_name in ['cga_only', 'reweight']:
        proto_tracker = PrototypeTracker(num_classes, 128, nu=0.99, device=device)
        conf_tracker = ConfusionTracker(num_classes, mu=0.99, device=device)

    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.synchronize()
        start = time.time()

        for images, labels in train_loader:
            if is_ce:
                images, labels = images.to(device), labels.to(device)
                feat, logits = model(images)
                loss = criterion(logits, labels)
            else:
                images = torch.cat([images[0], images[1]], dim=0).to(device)
                labels_rep = labels.repeat(2).to(device)
                bsz = labels.size(0)
                feat, z = model(images)

                if method_name == 'reweight':
                    conf_matrix = conf_tracker.get_normalized() if conf_tracker.initialized else None
                    loss = criterion(z, labels_rep, confusion_matrix=conf_matrix)
                else:
                    loss = criterion(z, labels_rep)

                if method_name == 'cga_only' and conf_tracker.initialized:
                    protos = proto_tracker.get_prototypes().detach()
                    conf_norm = conf_tracker.get_normalized().detach()
                    cga_loss = cga_fn(z[:bsz], labels_rep[:bsz], protos, conf_norm)
                    loss = loss + 0.5 * cga_loss

                # Update trackers
                if proto_tracker is not None:
                    proto_tracker.update(z[:bsz].detach(), labels_rep[:bsz])
                if conf_tracker is not None and proto_tracker is not None:
                    conf_tracker.update(z[:bsz].detach(), labels_rep[:bsz],
                                       proto_tracker.get_prototypes())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        epoch_times.append(elapsed)

    # Discard warmup
    measured_times = epoch_times[warmup:]
    return {
        'method': method_name,
        'mean_epoch_time': float(np.mean(measured_times)),
        'std_epoch_time': float(np.std(measured_times)),
        'all_times': [float(t) for t in epoch_times],
    }


if __name__ == '__main__':
    import numpy as np

    methods = ['ce', 'supcon', 'hardneg', 'tcl', 'cga_only', 'reweight']
    results = {}

    for method in methods:
        print(f"Timing {method}...")
        r = time_method(method)
        results[method] = r
        print(f"  {method}: {r['mean_epoch_time']:.2f} ± {r['std_epoch_time']:.2f} s/epoch")

    # Compute overheads relative to SupCon
    supcon_time = results['supcon']['mean_epoch_time']
    print(f"\nOverhead relative to SupCon ({supcon_time:.2f} s/epoch):")
    for method, r in results.items():
        if method != 'supcon':
            overhead = (r['mean_epoch_time'] - supcon_time) / supcon_time * 100
            print(f"  {method}: {overhead:+.1f}%")

    # Save
    out_path = os.path.join(WORKSPACE, 'exp', 'overhead', 'timing_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
