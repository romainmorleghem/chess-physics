"""Piste 4 : fluctuation-dissipation. Une seule température suffit-elle ?

Logit p_i ∝ exp(-β H(ΔE_i)). L'équation du maximum de vraisemblance en β est
    Σ_n H_obs,n = Σ_n <H>_β,n          (égalité des énergies moyennes : « U »)
et, dans l'ensemble canonique, dU/dβ = -Var(H) : la même β doit aussi rendre compte des
fluctuations. On définit β_V par l'égalité des seconds moments
    Σ_n H_obs,n² = Σ_n <H²>_β,n
et le rapport de fluctuation R = Σ (H_obs - <H>)² / Σ Var_β(H) au β du maximum de vraisemblance.
Une seule température <=> β_V = β_U et R = 1. R > 1 : fluctuations excessives (plusieurs
« températures » mélangées, comme une température effective dans un verre).
H = ΔE (Boltzmann, β = 1/T) ou H = ΔE^c (loi étirée, β = s^-c). Pour le modèle à deux étages,
on compare seulement R (moments de ΔE prédits contre observés).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def moments(H, beta):
    """<H> et <H²> par position sous p ∝ exp(-β H) (NaN = coup absent)."""
    U = np.where(np.isnan(H), -np.inf, -beta * np.nan_to_num(H))
    P = np.exp(U - logsumexp(U, axis=1, keepdims=True))
    Hn = np.nan_to_num(H)
    return (P * Hn).sum(1), (P * Hn ** 2).sum(1)


def beta_match(H, idx, order):
    """β tel que Σ H_obs^order = Σ <H^order>_β (fonction décroissante de β)."""
    h = H[np.arange(len(idx)), idx]
    target = (h ** order).sum()

    def g(lb):
        return moments(H, np.exp(lb))[order - 1].sum() - target
    return np.exp(brentq(g, np.log(1e-8), np.log(1e3)))


def fluct_ratio(H, idx, beta):
    h = H[np.arange(len(idx)), idx]
    m1, m2 = moments(H, beta)
    return ((h - m1) ** 2).sum() / (m2 - m1 ** 2).sum()


def two_stage_P(theta, E):
    """Matrice des probabilités de choix du modèle à deux étages (cf. attention.py)."""
    from attention import KS
    s, c = np.exp(theta[0]), np.exp(theta[1])
    N = (~np.isnan(E)).sum(1)
    W = np.where(np.isnan(E), 0.0, np.exp(-(np.nan_to_num(E) / s) ** c))
    cum = np.cumsum(W, axis=1)
    lg = np.concatenate([theta[2:], [0.0]])
    pi = np.exp(lg - lg.max()); pi /= pi.sum()
    rank = np.arange(E.shape[1])[None, :]
    P = np.zeros_like(W)
    for pk, K in zip(pi, KS):
        Ke = np.minimum(K, N)
        P += pk * (rank < Ke[:, None]) * W / cum[np.arange(len(N)), Ke - 1][:, None]
    return P


def ratio_from_P(P, E, idx):
    En = np.nan_to_num(E)
    m1, m2 = (P * En).sum(1), (P * En ** 2).sum(1)
    h = En[np.arange(len(idx)), idx]
    return ((h - m1) ** 2).sum() / (m2 - m1 ** 2).sum(), h.mean(), m1.mean()
