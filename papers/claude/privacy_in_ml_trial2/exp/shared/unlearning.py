import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .config import DEVICE, UNLEARN_EPOCHS, SCRUB_PASSES


def _get_optimizer(model, dataset, lr=None):
    if dataset in ('cifar10', 'cifar100'):
        return optim.SGD(model.parameters(), lr=lr or 0.001, momentum=0.9, weight_decay=5e-4)
    else:
        return optim.Adam(model.parameters(), lr=lr or 0.001, weight_decay=1e-4)


def finetune(model, forget_loader, retain_loader, dataset, epochs=UNLEARN_EPOCHS,
             sample_weights=None, device=DEVICE):
    """Fine-tune on retain set only. When sample_weights is provided (DAU),
    upweight retain samples in same classes as hard forget samples."""
    model = copy.deepcopy(model)
    optimizer = _get_optimizer(model, dataset)
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(retain_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            losses = criterion(model(x), y)
            if sample_weights is not None:
                w = sample_weights[batch_idx * x.size(0): (batch_idx + 1) * x.size(0)]
                if len(w) == len(losses):
                    losses = losses * w.to(device)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
    return model


def gradient_ascent(model, forget_loader, retain_loader, dataset, epochs=UNLEARN_EPOCHS,
                    sample_weights=None, device=DEVICE):
    """Maximize loss on forget set."""
    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(forget_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            losses = criterion(model(x), y)
            if sample_weights is not None:
                # batch indices from the loader - use global indices stored in loader
                w = sample_weights[batch_idx * x.size(0): (batch_idx + 1) * x.size(0)]
                if len(w) == len(losses):
                    losses = losses * w.to(device)
            loss = -losses.mean()  # Negate for ascent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def random_labels(model, forget_loader, retain_loader, dataset, epochs=UNLEARN_EPOCHS,
                  num_classes=10, sample_weights=None, device=DEVICE):
    """Train on forget set with random labels + retain set with true labels."""
    model = copy.deepcopy(model)
    optimizer = _get_optimizer(model, dataset)
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        # Forget set with random labels
        for batch_idx, batch in enumerate(forget_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            fake_y = torch.randint(0, num_classes, (x.size(0),), device=device)
            optimizer.zero_grad()
            losses = criterion_none(model(x), fake_y)
            if sample_weights is not None:
                w = sample_weights[batch_idx * x.size(0): (batch_idx + 1) * x.size(0)]
                if len(w) == len(losses):
                    losses = losses * w.to(device)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

        # Retain set with true labels
        for batch in retain_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def scrub(model, forget_loader, retain_loader, dataset, passes=SCRUB_PASSES,
          sample_weights=None, device=DEVICE):
    """SCRUB (Kurmanji et al., 2023): teacher-student with KL divergence."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)

    if dataset in ('cifar10', 'cifar100'):
        opt_forget = optim.SGD(student.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        opt_retain = optim.SGD(student.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    else:
        opt_forget = optim.Adam(student.parameters(), lr=0.0001, weight_decay=1e-4)
        opt_retain = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)

    student.train()
    kl_loss = nn.KLDivLoss(reduction='none')

    for pass_idx in range(passes):
        # 1 step: maximize KL(student || uniform) on forget set
        for batch_idx, batch in enumerate(forget_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            opt_forget.zero_grad()
            logits = student(x)
            n_classes = logits.shape[1]
            uniform = torch.full_like(logits, 1.0 / n_classes)
            per_sample = kl_loss(F.log_softmax(logits, dim=1), uniform).sum(dim=1)
            if sample_weights is not None:
                w = sample_weights[batch_idx * x.size(0): (batch_idx + 1) * x.size(0)]
                if len(w) == len(per_sample):
                    per_sample = per_sample * w.to(device)
            loss = -per_sample.mean()  # Maximize divergence from uniform
            loss.backward()
            opt_forget.step()
            break  # 1 step only

        # 3 steps: minimize KL(student || teacher) on retain set
        retain_iter = iter(retain_loader)
        for _ in range(3):
            try:
                batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch = next(retain_iter)
            x, y = batch[0].to(device), batch[1].to(device)
            opt_retain.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction='batchmean'
            )
            loss.backward()
            opt_retain.step()

    return student


def neggrad_kd(model, forget_loader, retain_loader, dataset, epochs=UNLEARN_EPOCHS,
               sample_weights=None, device=DEVICE):
    """NegGrad + Knowledge Distillation."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)
    optimizer = _get_optimizer(student, dataset)
    criterion = nn.CrossEntropyLoss(reduction='none')
    student.train()

    for epoch in range(epochs):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        # Process forget batches
        for batch_idx, batch in enumerate(forget_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            losses = criterion(student(x), y)
            if sample_weights is not None:
                w = sample_weights[batch_idx * x.size(0): (batch_idx + 1) * x.size(0)]
                if len(w) == len(losses):
                    losses = losses * w.to(device)
            loss_forget = -0.5 * losses.mean()

            # Get a retain batch
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            rx, ry = r_batch[0].to(device), r_batch[1].to(device)
            with torch.no_grad():
                teacher_logits = teacher(rx)
            student_logits = student(rx)
            loss_retain = 0.5 * F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction='batchmean'
            )

            loss = loss_forget + loss_retain
            loss.backward()
            optimizer.step()

    return student


UNLEARN_FN = {
    'ft': finetune,
    'ga': gradient_ascent,
    'rl': random_labels,
    'scrub': scrub,
    'neggrad': neggrad_kd,
}


def run_unlearning(model, forget_loader, retain_loader, method, dataset,
                   sample_weights=None, num_classes=10, **kwargs):
    """Run unlearning with the specified method."""
    fn = UNLEARN_FN[method]
    extra = {}
    if method == 'rl':
        extra['num_classes'] = num_classes
    if sample_weights is not None:
        extra['sample_weights'] = sample_weights
    # Map 'epochs' to 'passes' for scrub
    if method == 'scrub' and 'epochs' in kwargs:
        kwargs['passes'] = kwargs.pop('epochs')
    return fn(model, forget_loader, retain_loader, dataset, **extra, **kwargs)
