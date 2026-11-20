"""Recherche d'une vraie transition de phase (après la chaleur spécifique).

Test 1 — « trouve / gaffe » (analogue du 1er ordre, coexistence de deux phases) :
  perte du coup joué ΔE_p classée en bon (< 50 cp), moyen (50-300), gaffe (>= 300) ;
  comparaison observé / prédit par la loi étirée de la case, par tranche d'écart 1er-2e coup.
  Raideur : régression logistique  logit P(trouve) = a + b * logit(p_modèle) ; b > 1 = bascule plus
  brutale que ne le veut la loi continue (tout-ou-rien), b = 1 = le modèle suffit.
Test 2 — effet de taille (finite-size scaling) : courbe C(s) par quartile du nombre de coups légaux
  (hors échec), repondérée à la même distribution d'écart ; pic s*, hauteur C_max, largeur à mi-hauteur
  (en décades de s). Une transition : C_max croît et la largeur diminue quand N augmente.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

GOOD, BLUNDER = 50.0, 300.0
LOSS_EDGES = np.array([0, GOOD, BLUNDER, np.inf])


def probs(E, s, c):
    H = np.nan_to_num(E) ** c
    U = np.where(np.isnan(E), -np.inf, -H / s ** c)
    return np.exp(U - logsumexp(U, axis=1, keepdims=True)), H


def loss_classes(E, idx, P):
    """Fractions observées / prédites des 3 classes de perte (par position)."""
    rows = np.arange(len(idx))
    lp = E[rows, idx]
    obs = np.stack([(lp >= a) & (lp < b) for a, b in zip(LOSS_EDGES[:-1], LOSS_EDGES[1:])], 1).astype(float)
    Ef = np.nan_to_num(E, nan=-1)
    pred = np.stack([(P * ((Ef >= a) & (Ef < b))).sum(1) for a, b in zip(LOSS_EDGES[:-1], LOSS_EDGES[1:])], 1)
    return obs, pred


def fit_sharpness(y, pm):
    """logit P(y=1) = a + b logit(pm) ; renvoie (a, b)."""
    x = logit(np.clip(pm, 1e-4, 1 - 1e-4))

    def f(t):
        z = t[0] + t[1] * x
        p = expit(z)
        nll = -(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)).sum()
        g = p - y
        return nll, np.array([g.sum(), (g * x).sum()])
    return minimize(f, [0.0, 1.0], jac=True, method="L-BFGS-B").x


def heat_matrix(E, s_grid, c):
    """C de chaque position pour chaque s : (N, len(s_grid))."""
    out = np.empty((len(E), len(s_grid)))
    for j, s in enumerate(s_grid):
        P, H = probs(E, s, c)
        m1 = (P * H).sum(1)
        out[:, j] = s ** (-2 * c) * ((P * H ** 2).sum(1) - m1 ** 2)
    return out


def peak_stats(logs, C):
    """Pic (interpolation parabolique en log s), hauteur et largeur à mi-hauteur (décades)."""
    j = int(np.argmax(C))
    if 0 < j < len(C) - 1:
        y0, y1, y2 = C[j - 1:j + 2]
        den = y0 - 2 * y1 + y2
        dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        xs, cm = logs[j] + dx * (logs[1] - logs[0]), y1 - 0.25 * (y0 - y2) * dx
    else:
        xs, cm = logs[j], C[j]
    half = cm / 2
    lo = np.interp(half, C[:j + 1], logs[:j + 1]) if C[0] < half else np.nan
    hi = np.interp(-half, -C[j:], logs[j:]) if C[-1] < half else np.nan
    return np.exp(xs), cm, (hi - lo) / np.log(10) if np.isfinite(lo) and np.isfinite(hi) else np.nan


def subsample_null(E, n_target, rng):
    """Témoin de l'effet de taille : garde le meilleur coup et n_target - 1 autres tirés au hasard
    (les positions ont >= n_target coups). Renvoie une matrice E réduite."""
    out = np.full((len(E), n_target), np.nan)
    for k, row in enumerate(E):
        v = row[~np.isnan(row)]
        b = int(np.argmin(v))
        rest = np.delete(np.arange(len(v)), b)
        keep = np.concatenate([[b], rng.choice(rest, n_target - 1, replace=False)])
        out[k] = v[keep] - v[keep].min()
    return out
