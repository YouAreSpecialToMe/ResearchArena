"""Assumption diagnostic module for AACD (Stage 1).

Computes 5 diagnostics per variable pair:
D1: Linearity score
D2: Non-Gaussianity score
D3: ANM asymmetry score
D4: Faithfulness proximity score
D5: Homoscedasticity score
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


def compute_diagnostics(data, max_cond_set_size=2, n_star=None):
    """Compute all 5 diagnostics on data (n x p array).

    Returns dict with D1-D5 matrices, marginal scores, global summary, confidence weights.
    """
    if n_star is None:
        n_star = {'D1': 300, 'D2': 200, 'D3': 500, 'D4': 500, 'D5': 300}

    n, p = data.shape

    # Compute shared regression fits
    D1, D2, D5, fits = _compute_d1_d2_d5(data)
    D3 = _compute_d3(data)
    D4 = _compute_d4(data, max_cond_set_size)

    # Marginal non-Gaussianity
    marginal_ng = np.zeros(p)
    for i in range(p):
        kurt = abs(stats.kurtosis(data[:, i]))
        skew = abs(stats.skew(data[:, i]))
        marginal_ng[i] = min(1.0, (kurt / 3 + skew / 2) / 2)

    # Confidence weights
    confidence = {k: min(1.0, n / v) for k, v in n_star.items()}

    # Global summary
    mask = ~np.eye(p, dtype=bool)
    global_summary = {
        'avg_linearity': float(np.mean(D1[mask])),
        'avg_nongaussianity': float(np.mean(D2[mask])),
        'avg_anm_score': float(np.mean(D3[mask])),
        'avg_faithfulness_proximity': float(np.mean(D4[mask])),
        'avg_homoscedasticity': float(np.mean(D5[mask])),
        'avg_marginal_nongaussianity': float(np.mean(marginal_ng)),
    }

    return {
        'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4, 'D5': D5,
        'marginal_nongaussianity': marginal_ng,
        'global_summary': global_summary,
        'confidence_weights': confidence,
    }


def _compute_d1_d2_d5(data):
    """Compute linearity (D1), non-Gaussianity (D2), and homoscedasticity (D5) scores."""
    n, p = data.shape
    D1 = np.zeros((p, p))
    D2 = np.zeros((p, p))
    D5 = np.zeros((p, p))
    fits = {}

    # Subsample for speed
    max_n = min(n, 2000)
    idx = np.random.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    sub_data = data[idx]

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            x = sub_data[:, i].reshape(-1, 1)
            y = sub_data[:, j]

            # Linear fit
            lr = LinearRegression().fit(x, y)
            r2_lin = max(0, lr.score(x, y))
            resid_lin = y - lr.predict(x)

            # Polynomial fit (degree 3)
            poly = PolynomialFeatures(degree=3, include_bias=False)
            x_poly = poly.fit_transform(x)
            pr = LinearRegression().fit(x_poly, y)
            r2_poly = max(0, pr.score(x_poly, y))
            resid_poly = y - pr.predict(x_poly)

            # D1: Linearity score
            improvement = max(0, r2_poly - r2_lin) / max(1 - r2_lin, 0.01)
            D1[i, j] = np.clip(1 - improvement, 0, 1)

            # Choose best residuals
            if r2_poly > r2_lin + 0.02:
                resid = resid_poly
                fits[(i, j)] = 'poly'
            else:
                resid = resid_lin
                fits[(i, j)] = 'linear'

            # D2: Non-Gaussianity score (Shapiro-Wilk on residuals)
            n_sw = min(len(resid), 5000)
            try:
                _, p_val = stats.shapiro(resid[:n_sw])
                D2[i, j] = 1 - p_val
            except Exception:
                D2[i, j] = 0.5

            # D5: Homoscedasticity (Breusch-Pagan)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                import statsmodels.api as sm
                x_const = sm.add_constant(sub_data[:, i])
                model = sm.OLS(y, x_const).fit()
                _, bp_pval, _, _ = het_breuschpagan(model.resid, x_const)
                D5[i, j] = 1 - bp_pval
            except Exception:
                D5[i, j] = 0.5

    return D1, D2, D5, fits


def _compute_d3(data):
    """Compute ANM asymmetry score (D3) using fast Spearman-based independence test."""
    from scipy.stats import spearmanr
    n, p = data.shape
    D3 = np.zeros((p, p))

    max_n = min(n, 500)
    idx = np.random.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    sub_data = data[idx]

    for i in range(p):
        for j in range(i + 1, p):
            # Forward: X_j = f(X_i) + N
            p_fwd = _fast_residual_independence_test(sub_data[:, i], sub_data[:, j])
            # Reverse: X_i = f(X_j) + N
            p_rev = _fast_residual_independence_test(sub_data[:, j], sub_data[:, i])

            asym = abs(p_fwd - p_rev)
            D3[i, j] = asym
            D3[j, i] = asym

    return D3


def _fast_residual_independence_test(x, y):
    """Fast residual independence test using Spearman correlation.

    Fits polynomial regression y = f(x) + N, then tests independence
    of residuals from x using Spearman correlation (linear and nonlinear dependence).

    Returns p-value (high = independent residuals = good ANM fit).
    """
    from scipy.stats import spearmanr

    x_r = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(x_r)
    lr = LinearRegression().fit(x_poly, y)
    resid = y - lr.predict(x_poly)

    # Multiple independence tests
    _, p1 = spearmanr(x, resid)
    _, p2 = spearmanr(x, np.abs(resid))
    _, p3 = spearmanr(x, resid**2)

    return min(p1, p2, p3)


def _compute_d4(data, max_k=2):
    """Compute faithfulness proximity score (D4)."""
    n, p = data.shape
    D4 = np.zeros((p, p))

    if p > 30:
        max_k = min(max_k, 1)

    # Correlation matrix
    corr = np.corrcoef(data.T)

    for i in range(p):
        for j in range(i + 1, p):
            max_proximity = 0

            # Zero-order
            r = abs(corr[i, j])
            if r < 0.05:
                max_proximity = 1.0

            # First-order partial correlations
            if max_k >= 1:
                for k in range(p):
                    if k == i or k == j:
                        continue
                    pr = _partial_corr(corr, i, j, [k])
                    if abs(pr) < 0.05:
                        max_proximity = max(max_proximity, 1.0)
                        break

            # Second-order partial correlations
            if max_k >= 2 and max_proximity < 1.0:
                for k1 in range(p):
                    if k1 == i or k1 == j:
                        continue
                    for k2 in range(k1 + 1, p):
                        if k2 == i or k2 == j:
                            continue
                        pr = _partial_corr(corr, i, j, [k1, k2])
                        if abs(pr) < 0.05:
                            max_proximity = max(max_proximity, 1.0)
                            break
                    if max_proximity >= 1.0:
                        break

            D4[i, j] = max_proximity
            D4[j, i] = max_proximity

    return D4


def _partial_corr(corr, i, j, cond_set):
    """Compute partial correlation of i,j given cond_set from correlation matrix."""
    if len(cond_set) == 0:
        return corr[i, j]

    idx = [i, j] + list(cond_set)
    sub = corr[np.ix_(idx, idx)]
    try:
        inv = np.linalg.inv(sub)
        denom = np.sqrt(abs(inv[0, 0] * inv[1, 1]))
        if denom < 1e-10:
            return 0.0
        return -inv[0, 1] / denom
    except np.linalg.LinAlgError:
        return corr[i, j]
