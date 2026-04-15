from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    stable = a - a_max
    return np.squeeze(a_max, axis=axis) + np.log(np.sum(np.exp(stable), axis=axis))


def _diag_logpdf(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    n, d = X.shape
    k = means.shape[0]
    diff = X[:, None, :] - means[None, :, :]
    inv_var = 1.0 / variances[None, :, :]
    quad = np.sum(diff * diff * inv_var, axis=2)
    log_det = np.sum(np.log(variances), axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det[None, :] + quad)


@dataclass
class HierarchicalMixtureClassifier:
    n_coarse: int = 4
    n_fine_per_coarse: int = 2
    covariance_floor: float = 1e-3
    random_state: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalMixtureClassifier":
        start = time.time()
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        coarse = GaussianMixture(
            n_components=self.n_coarse,
            covariance_type="diag",
            reg_covar=self.covariance_floor,
            random_state=self.random_state,
            n_init=3,
            max_iter=200,
        )
        coarse.fit(X)
        coarse_resp = coarse.predict_proba(X)

        self.coarse_weights_ = coarse.weights_
        self.coarse_means_ = coarse.means_
        self.coarse_vars_ = coarse.covariances_ + self.covariance_floor
        self.fine_weights_ = np.zeros((self.n_coarse, self.n_fine_per_coarse))
        self.fine_means_ = np.zeros((self.n_coarse, self.n_fine_per_coarse, self.n_features_))
        self.fine_vars_ = np.zeros((self.n_coarse, self.n_fine_per_coarse, self.n_features_))
        self.class_probs_ = np.zeros((self.n_coarse, self.n_fine_per_coarse, self.n_classes_))

        hard_coarse = coarse_resp.argmax(axis=1)
        for c in range(self.n_coarse):
            idx = np.where(hard_coarse == c)[0]
            if idx.size < self.n_fine_per_coarse * 10:
                idx = np.argsort(coarse_resp[:, c])[-max(self.n_fine_per_coarse * 20, idx.size):]
            Xc = X[idx]
            yc = y_idx[idx]

            if len(Xc) < self.n_fine_per_coarse:
                reps = np.repeat(X[np.newaxis, 0, :], self.n_fine_per_coarse, axis=0)
                self.fine_weights_[c] = 1.0 / self.n_fine_per_coarse
                self.fine_means_[c] = reps
                self.fine_vars_[c] = np.ones_like(reps)
                self.class_probs_[c] = 1.0 / self.n_classes_
                continue

            fine = GaussianMixture(
                n_components=self.n_fine_per_coarse,
                covariance_type="diag",
                reg_covar=self.covariance_floor,
                random_state=self.random_state + c + 1,
                n_init=3,
                max_iter=200,
            )
            fine.fit(Xc)
            fine_resp = fine.predict_proba(Xc)
            self.fine_weights_[c] = fine.weights_
            self.fine_means_[c] = fine.means_
            self.fine_vars_[c] = fine.covariances_ + self.covariance_floor
            hard_fine = fine_resp.argmax(axis=1)
            for f in range(self.n_fine_per_coarse):
                mask = hard_fine == f
                counts = np.bincount(yc[mask], minlength=self.n_classes_) + 1.0
                self.class_probs_[c, f] = counts / counts.sum()

        self.fit_time_sec_ = time.time() - start
        self.node_count_ = 1 + self.n_coarse + self.n_coarse * self.n_fine_per_coarse
        self.sum_node_count_ = 1 + self.n_coarse
        self.max_depth_ = 2
        return self

    def _fine_log_joint_x(self, X: np.ndarray) -> np.ndarray:
        log_coarse = np.log(self.coarse_weights_ + 1e-12) + _diag_logpdf(X, self.coarse_means_, self.coarse_vars_)
        out = np.zeros((X.shape[0], self.n_coarse, self.n_fine_per_coarse))
        for c in range(self.n_coarse):
            out[:, c, :] = (
                log_coarse[:, [c]]
                + np.log(self.fine_weights_[c] + 1e-12)[None, :]
                + _diag_logpdf(X, self.fine_means_[c], self.fine_vars_[c])
            )
        return out

    def posterior_memberships(self, X: np.ndarray) -> dict[str, np.ndarray]:
        fine_log_joint = self._fine_log_joint_x(X)
        flat = fine_log_joint.reshape(X.shape[0], -1)
        norm = _logsumexp(flat, axis=1)[:, None]
        fine = np.exp(flat - norm).reshape(X.shape[0], self.n_coarse, self.n_fine_per_coarse)
        coarse = fine.sum(axis=2)
        return {"coarse": coarse, "fine": fine.reshape(X.shape[0], -1)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        fine_log_joint_x = self._fine_log_joint_x(X).reshape(X.shape[0], -1)
        fine_class_log = np.log(self.class_probs_.reshape(-1, self.n_classes_) + 1e-12)
        log_joint_xy = fine_log_joint_x[:, :, None] + fine_class_log[None, :, :]
        log_py_x = _logsumexp(log_joint_xy, axis=1)
        norm = _logsumexp(log_py_x, axis=1)[:, None]
        return np.exp(log_py_x - norm)

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - self.predict_proba(X)

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "HierarchicalMixtureClassifier":
        with path.open("rb") as handle:
            return pickle.load(handle)
