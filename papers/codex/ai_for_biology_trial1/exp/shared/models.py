from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config


@dataclass
class PCAWrapper:
    model: object | None
    n_components: int
    active_components: int
    mean_: np.ndarray | None = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.model is None:
            out = x
        else:
            out = self.model.transform(x).astype(np.float32, copy=False)
        if out.shape[1] >= self.n_components:
            return out[:, : self.n_components].astype(np.float32, copy=False)
        padded = np.zeros((out.shape[0], self.n_components), dtype=np.float32)
        padded[:, : out.shape[1]] = out
        return padded

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x_active = x[:, : self.active_components] if x.ndim == 2 else x[: self.active_components]
        if self.model is None:
            return x_active.astype(np.float32, copy=False)
        return self.model.inverse_transform(x_active).astype(np.float32, copy=False)


def fit_pca(x: np.ndarray, max_components: int) -> PCAWrapper:
    n = min(max_components, x.shape[0], x.shape[1])
    if n < 1:
        return PCAWrapper(None, max_components, min(max_components, x.shape[1]), None)
    model = PCA(n_components=n, random_state=0)
    model.fit(x)
    return PCAWrapper(
        model=model,
        n_components=max_components,
        active_components=n,
        mean_=getattr(model, "mean_", None),
    )


def fit_truncated_svd(x: np.ndarray, max_components: int) -> PCAWrapper:
    n = min(max_components, x.shape[0] - 1, x.shape[1] - 1)
    if n < 2:
        return PCAWrapper(None, max_components, min(max_components, x.shape[1]), None)
    model = TruncatedSVD(n_components=n, random_state=0)
    model.fit(x)
    return PCAWrapper(model=model, n_components=max_components, active_components=n)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.MLP_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.MLP_HIDDEN_DIM, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def fit_pls(x_train: np.ndarray, y_train: np.ndarray, n_components: int) -> PLSRegression:
    model = PLSRegression(n_components=n_components)
    model.fit(x_train, y_train)
    return model


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    device: str,
    eval_fn,
    log_fn=None,
) -> tuple[MLP, dict[str, float]]:
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=min(config.BATCH_SIZE, len(train_ds)), shuffle=True)

    best_state = None
    best_score = float("-inf")
    best_rmse = float("inf")
    best_epoch = 0
    patience_left = config.EARLY_STOPPING_PATIENCE

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_items = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(xb)
            n_items += len(xb)

        model.eval()
        with torch.no_grad():
            val_score, val_rmse = eval_fn(model)
        if log_fn is not None:
            log_fn(
                {
                    "epoch": epoch,
                    "train_mse": epoch_loss / max(1, n_items),
                    "val_pearson": val_score,
                    "val_rmse": val_rmse,
                }
            )
        if (val_score > best_score + 1e-8) or (
            abs(val_score - best_score) <= 1e-8 and val_rmse < best_rmse - 1e-8
        ):
            best_score = val_score
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_pearson": float(best_score),
        "best_val_rmse": float(best_rmse),
        "best_epoch": float(best_epoch),
    }
