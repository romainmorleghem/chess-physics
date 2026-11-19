"""Piste 1 : la loi de choix est-elle vraiment de Boltzmann ?

Modèle général (logit conditionnel) : p_i ∝ exp(-f(ΔE_i)), ΔE_i = e_best - e_i >= 0.
  - Boltzmann       : f(ΔE) = ΔE / T            (droite)
  - Exp. étirée     : f(ΔE) = (ΔE / s)^c        (c = 1 <=> Boltzmann ; Regan & Haworth 2011)
  - Non paramétrique: f linéaire par morceaux entre des nœuds fixes, f(0) = 0
Grâce à la propriété IIA du logit, la vraisemblance restreinte aux k candidats est
cohérente si le modèle est juste : on peut donc tester la forme de f sur les candidats.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

KNOTS = np.array([0, 10, 25, 50, 100, 200, 400, 1000.0])


def _nll_from_util(U, idx):
    U = np.where(np.isnan(U), -np.inf, U)
    return -(U[np.arange(len(idx)), idx] - logsumexp(U, axis=1)).sum()


def fit_stretched(E, idx):
    """MLE de (s, c) pour f = (ΔE/s)^c. Renvoie (s, c)."""
    En = np.nan_to_num(E, nan=0.0)
    mask = np.isnan(E)

    def nll(th):
        s, c = np.exp(th)
        U = -(En / s) ** c
        U[mask] = np.nan
        return _nll_from_util(U, idx)

    best = None
    for c0 in (0.5, 1.0, 1.5):
        r = minimize(nll, x0=[np.log(80.0), np.log(c0)], method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    return tuple(np.exp(best.x)), best.fun


def basis(E):
    """Fonctions chapeau (interpolation linéaire) sur KNOTS[1:], f(0)=0. E -> (..., K-1)."""
    Ec = np.clip(np.nan_to_num(E, nan=0.0), 0, KNOTS[-1])
    out = np.zeros(E.shape + (len(KNOTS) - 1,))
    for k in range(1, len(KNOTS)):
        left, mid = KNOTS[k - 1], KNOTS[k]
        up = (Ec - left) / (mid - left)
        if k < len(KNOTS) - 1:
            right = KNOTS[k + 1]
            down = (right - Ec) / (right - mid)
            out[..., k - 1] = np.clip(np.minimum(up, down), 0, 1)
        else:
            out[..., k - 1] = np.clip(up, 0, 1)
    return out


def fit_nonparam(E, idx):
    """f aux nœuds (f(0)=0), par MLE convexe. Renvoie le vecteur f(KNOTS)."""
    B = basis(E)
    mask = np.isnan(E)
    rows = np.arange(len(idx))

    def nll_grad(th):
        U = -(B @ th)
        U[mask] = -np.inf
        lse = logsumexp(U, axis=1)
        P = np.exp(U - lse[:, None])
        nll = -(U[rows, idx] - lse).sum()
        # dU/dth = -B ; grad = sum_n [ B[n,idx] - sum_j P_nj B[n,j] ]
        g = (B[rows, idx] - np.einsum("nj,njk->nk", P, B)).sum(0)
        return nll, g

    r = minimize(nll_grad, x0=np.full(len(KNOTS) - 1, 0.01), jac=True, method="L-BFGS-B")
    return np.concatenate([[0.0], r.x])
