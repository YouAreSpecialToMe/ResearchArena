#!/usr/bin/env python3
"""Quick calibration: find GA/SCRUB/NegGrad hyperparams that achieve meaningful forgetting
while preserving utility. Test on CIFAR-10 seed 42 only."""

import os, sys, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import set_seed, load_dataset, create_splits, get_loader, evaluate_model

set_seed(42)
device = DEVICE

# Load CIFAR-10
train_ds, test_ds, train_eval_ds = load_dataset('cifar10')
splits = create_splits('cifar10', train_ds, 42)
forget_indices = splits['forget_indices']
retain_indices = splits['retain_indices']

forget_loader = get_loader(train_ds, forget_indices, batch_size=512, shuffle=True)
retain_loader = get_loader(train_ds, retain_indices, batch_size=512, shuffle=True)
forget_eval = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)
retain_eval = get_loader(train_eval_ds, retain_indices, batch_size=512, shuffle=False)
test_loader = get_loader(test_ds, None, batch_size=512, shuffle=False)

# Load original model
original = get_model('cifar10', device)
original.load_state_dict(torch.load('exp/models/original/cifar10_seed42.pt', map_location=device, weights_only=True))

def eval_quick(model):
    fa, _ = evaluate_model(model, forget_eval)
    ra, _ = evaluate_model(model, retain_eval)
    ta, _ = evaluate_model(model, test_loader)
    return fa, ra, ta

print(f"Original model:")
fa, ra, ta = eval_quick(original)
print(f"  forget_acc={fa:.4f} retain_acc={ra:.4f} test_acc={ta:.4f}\n")

# ============ GA calibration ============
print("=" * 60)
print("GRADIENT ASCENT calibration")
print("=" * 60)

for lr in [0.005, 0.01, 0.02, 0.05]:
    for epochs in [10, 20]:
        for retain_weight in [0.0, 0.5, 1.0]:
            model = copy.deepcopy(original)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            model.train()

            for epoch in range(epochs):
                # Forget step
                for batch in forget_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = -criterion(logits, y)
                    # Also push toward uniform
                    n_classes = logits.shape[1]
                    uniform = torch.full_like(logits, 1.0 / n_classes)
                    kl = F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='batchmean')
                    loss = loss - 0.5 * kl
                    loss.backward()
                    optimizer.step()

                # Retain step
                if retain_weight > 0:
                    for i, batch in enumerate(retain_loader):
                        if i >= 3:
                            break
                        x, y = batch[0].to(device), batch[1].to(device)
                        optimizer.zero_grad()
                        loss = retain_weight * criterion(model(x), y)
                        loss.backward()
                        optimizer.step()

            fa, ra, ta = eval_quick(model)
            status = "GOOD" if fa < 0.3 and ra > 0.85 else ("FA_OK" if fa < 0.3 else "BAD")
            print(f"  GA lr={lr} epochs={epochs} rw={retain_weight}: "
                  f"FA={fa:.3f} RA={ra:.3f} TA={ta:.3f}  [{status}]")
            del model
            torch.cuda.empty_cache()

# ============ SCRUB calibration ============
print("\n" + "=" * 60)
print("SCRUB calibration")
print("=" * 60)

for forget_lr in [0.001, 0.005, 0.01]:
    for passes in [10, 15, 20]:
        teacher = copy.deepcopy(original)
        teacher.eval()
        student = copy.deepcopy(original)
        opt_f = optim.SGD(student.parameters(), lr=forget_lr, momentum=0.9, weight_decay=5e-4)
        opt_r = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        student.train()

        for p in range(passes):
            # Forget: gradient ascent on CE
            for batch in forget_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                opt_f.zero_grad()
                loss = -criterion(student(x), y)
                loss.backward()
                opt_f.step()

            # Retain: KL to teacher
            retain_iter = iter(retain_loader)
            for i in range(5):
                try:
                    batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    batch = next(retain_iter)
                x, y = batch[0].to(device), batch[1].to(device)
                opt_r.zero_grad()
                with torch.no_grad():
                    t_logits = teacher(x)
                s_logits = student(x)
                loss = F.kl_div(F.log_softmax(s_logits, dim=1),
                               F.softmax(t_logits, dim=1), reduction='batchmean')
                loss.backward()
                opt_r.step()

        fa, ra, ta = eval_quick(student)
        status = "GOOD" if fa < 0.3 and ra > 0.85 else ("FA_OK" if fa < 0.3 else "BAD")
        print(f"  SCRUB forget_lr={forget_lr} passes={passes}: "
              f"FA={fa:.3f} RA={ra:.3f} TA={ta:.3f}  [{status}]")
        del student, teacher
        torch.cuda.empty_cache()

# ============ NegGrad calibration ============
print("\n" + "=" * 60)
print("NEGGRAD+KD calibration")
print("=" * 60)

for lr in [0.005, 0.01, 0.02]:
    for epochs in [10, 15, 20]:
        for forget_w in [0.5, 1.0, 2.0]:
            teacher = copy.deepcopy(original)
            teacher.eval()
            student = copy.deepcopy(original)
            optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            student.train()

            for epoch in range(epochs):
                retain_iter = iter(retain_loader)
                for batch in forget_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    loss_f = -forget_w * criterion(student(x), y)

                    try:
                        r_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        r_batch = next(retain_iter)
                    rx, ry = r_batch[0].to(device), r_batch[1].to(device)
                    with torch.no_grad():
                        t_logits = teacher(rx)
                    s_logits = student(rx)
                    loss_r = F.kl_div(F.log_softmax(s_logits, dim=1),
                                     F.softmax(t_logits, dim=1), reduction='batchmean')
                    loss = loss_f + loss_r
                    loss.backward()
                    optimizer.step()

            fa, ra, ta = eval_quick(student)
            status = "GOOD" if fa < 0.3 and ra > 0.85 else ("FA_OK" if fa < 0.3 else "BAD")
            print(f"  NG lr={lr} epochs={epochs} fw={forget_w}: "
                  f"FA={fa:.3f} RA={ra:.3f} TA={ta:.3f}  [{status}]")
            del student, teacher
            torch.cuda.empty_cache()

print("\nDone!")
