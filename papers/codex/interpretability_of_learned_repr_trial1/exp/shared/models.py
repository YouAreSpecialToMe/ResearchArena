from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import CSVLogger, count_parameters, device, elapsed_minutes, get_peak_memory_mb, now, reset_peak_memory


def decode(decoder: nn.Linear, code: torch.Tensor) -> torch.Tensor:
    return decoder(code)


class Encoder(nn.Module):
    def __init__(self, signed: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out if self.signed else torch.relu(out)


class SAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(signed=False)
        self.decoder = nn.Linear(1024, 768, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class SSAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(signed=True)
        self.decoder = nn.Linear(1024, 768, bias=False)

    def forward(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.encoder(delta)
        recon = self.decoder(s)
        return s, recon


class ASDModel(nn.Module):
    def __init__(self, shared_decoder: bool = True) -> None:
        super().__init__()
        self.anchor_encoder = Encoder(signed=False)
        self.shift_encoder = Encoder(signed=True)
        self.shared_decoder = shared_decoder
        self.anchor_decoder = nn.Linear(1024, 768, bias=False)
        self.shift_decoder = self.anchor_decoder if shared_decoder else nn.Linear(1024, 768, bias=False)

    def forward_anchor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.anchor_encoder(x)
        return z, self.anchor_decoder(z)

    def forward_shift(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.shift_encoder(delta)
        return s, self.shift_decoder(s)


@dataclass
class TrainResult:
    val_score: float
    runtime_minutes: float
    peak_memory_mb: float
    parameter_count: int


def _loader(x: torch.Tensor, y: Optional[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x) if y is None else TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def train_sae(
    model: SAEModel,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    lambda_l1: float,
    seed: int,
    log_path,
    max_epochs: int = 20,
    batch_size: int = 1024,
) -> TrainResult:
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_l1"])
    best_state = None
    best_score = float("inf")
    patience = 0
    start = now()
    reset_peak_memory()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for (xb,) in _loader(train_x, None, batch_size, True):
            xb = xb.to(dev)
            z, recon = model(xb)
            loss = torch.mean((recon - xb) ** 2) + lambda_l1 * torch.mean(torch.abs(z))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z, recon = model(val_x.to(dev))
            val_rec = torch.mean((recon - val_x.to(dev)) ** 2).item()
            val_l1 = torch.mean(torch.abs(z)).item()
            val_score = val_rec + lambda_l1 * val_l1
        logger.log({"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_rec, "val_l1": val_l1})
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return TrainResult(best_score, elapsed_minutes(start), get_peak_memory_mb(), count_parameters(model))


def train_ssae(
    model: SSAEModel,
    train_delta: torch.Tensor,
    val_delta: torch.Tensor,
    lambda_l1: float,
    seed: int,
    log_path,
    max_epochs: int = 20,
    batch_size: int = 1024,
) -> TrainResult:
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_loss", "val_l1"])
    best_state = None
    best_score = float("inf")
    patience = 0
    start = now()
    reset_peak_memory()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for (xb,) in _loader(train_delta, None, batch_size, True):
            xb = xb.to(dev)
            s, recon = model(xb)
            loss = torch.mean((recon - xb) ** 2) + lambda_l1 * torch.mean(torch.abs(s))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            s, recon = model(val_delta.to(dev))
            val_rec = torch.mean((recon - val_delta.to(dev)) ** 2).item()
            val_l1 = torch.mean(torch.abs(s)).item()
            val_score = val_rec + lambda_l1 * val_l1
        logger.log({"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_rec, "val_l1": val_l1})
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return TrainResult(best_score, elapsed_minutes(start), get_peak_memory_mb(), count_parameters(model))


def train_asd(
    model: ASDModel,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_pair_src: torch.Tensor,
    train_pair_tgt: torch.Tensor,
    val_pair_src: torch.Tensor,
    val_pair_tgt: torch.Tensor,
    lambda_sparse: float,
    lambda_tie: float,
    log_path,
    max_epochs: int = 20,
    batch_size: int = 1024,
) -> TrainResult:
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    logger = CSVLogger(log_path, ["epoch", "train_loss", "val_anchor", "val_shift", "val_tie"])
    best_state = None
    best_score = float("inf")
    patience = 0
    start = now()
    reset_peak_memory()
    anchor_loader = _loader(train_x, None, batch_size, True)
    shift_loader = _loader(train_pair_src, train_pair_tgt, batch_size, True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        shift_iter = iter(shift_loader)
        for (xb,) in anchor_loader:
            try:
                sb, tb = next(shift_iter)
            except StopIteration:
                shift_iter = iter(shift_loader)
                sb, tb = next(shift_iter)
            xb = xb.to(dev)
            sb = sb.to(dev)
            tb = tb.to(dev)
            db = tb - sb
            z, recon_x = model.forward_anchor(xb)
            s, recon_d = model.forward_shift(db)
            z_src = model.anchor_encoder(sb)
            z_tgt = model.anchor_encoder(tb)
            tie = torch.mean(torch.abs(s - (z_tgt - z_src).detach()))
            loss = (
                torch.mean((recon_x - xb) ** 2)
                + torch.mean((recon_d - db) ** 2)
                + lambda_sparse * (torch.mean(torch.abs(z)) + torch.mean(torch.abs(s)))
                + lambda_tie * tie
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z, recon_x = model.forward_anchor(val_x.to(dev))
            val_src = val_pair_src.to(dev)
            val_tgt = val_pair_tgt.to(dev)
            val_delta = val_tgt - val_src
            s, recon_d = model.forward_shift(val_delta)
            val_anchor = torch.mean((recon_x - val_x.to(dev)) ** 2).item()
            val_shift = torch.mean((recon_d - val_delta) ** 2).item()
            val_tie = torch.mean(torch.abs(s - (model.anchor_encoder(val_tgt) - model.anchor_encoder(val_src)))).item()
            val_score = val_anchor + val_shift + lambda_sparse * (torch.mean(torch.abs(z)).item() + torch.mean(torch.abs(s)).item()) + lambda_tie * val_tie
        logger.log({"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_anchor": val_anchor, "val_shift": val_shift, "val_tie": val_tie})
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return TrainResult(best_score, elapsed_minutes(start), get_peak_memory_mb(), count_parameters(model))
