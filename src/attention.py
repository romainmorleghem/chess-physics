"""Piste 7 : modèle à deux étages (ensemble de coups envisagés, puis choix).

Étage 1 : le joueur envisage les K coups les plus « saillants », K aléatoire de loi π sur
des points de support KS (le rang Stockfish sert d'approximation de la saillance ;
K = inf signifie « tous les coups »).
Étage 2 : choix parmi les K coups avec p ∝ exp(-(ΔE/s)^c).
    P(i) = w_i * Σ_K π_K 1[rang(i) < K] / Σ_{j<K} w_j
Le modèle à un étage (logit sur tous les coups) est le cas π(inf) = 1 ; il vérifie IIA,
le modèle à deux étages non. Ω_eff = E[min(K, N)] mesure le nombre de coups « accessibles ».
"""
import numpy as np
from scipy.optimize import minimize

KS = [1, 2, 3, 5, 8, 13, 10**6]


def _probs_played(theta, E, idx, Ncount, two_stage=True):
    s, c = np.exp(theta[0]), np.exp(theta[1])
    W = np.where(np.isnan(E), 0.0, np.exp(-(np.nan_to_num(E) / s) ** c))
    cum = np.cumsum(W, axis=1)
    rows = np.arange(len(idx))
    wi = W[rows, idx]
    if not two_stage:
        return wi / cum[:, -1]
    logits = np.concatenate([theta[2:], [0.0]])
    pi = np.exp(logits - logits.max())
    pi /= pi.sum()
    p = np.zeros(len(idx))
    for pk, K in zip(pi, KS):
        Ke = np.minimum(K, Ncount)
        p += pk * (idx < Ke) * wi / cum[rows, Ke - 1]
    return p


def nll(theta, E, idx, Ncount, two_stage=True):
    return -np.log(np.maximum(_probs_played(theta, E, idx, Ncount, two_stage), 1e-300)).sum()


def fit(E, idx, two_stage=True, x0=None):
    Ncount = (~np.isnan(E)).sum(1)
    if x0 is None:
        x0 = np.array([np.log(20.0), np.log(0.5)] + ([0.0] * (len(KS) - 1) if two_stage else []))
    r = minimize(nll, x0, args=(E, idx, Ncount, two_stage), method="L-BFGS-B",
                 bounds=[(np.log(0.1), np.log(5000)), (np.log(0.1), np.log(3))] +
                        [(-15, 15)] * (len(x0) - 2))
    return r.x, r.fun


def summarize(theta, E):
    """Ω_eff = E[min(K, N)] moyen, P(tous les coups envisagés), π, s, c."""
    Ncount = (~np.isnan(E)).sum(1)
    logits = np.concatenate([theta[2:], [0.0]])
    pi = np.exp(logits - logits.max())
    pi /= pi.sum()
    omega = np.mean([sum(pk * min(K, n) for pk, K in zip(pi, KS)) for n in Ncount])
    return dict(s=np.exp(theta[0]), c=np.exp(theta[1]), omega_eff=omega, p_all=pi[-1],
                **{f"pi_K{K if K < 10**6 else 'all'}": p for K, p in zip(KS, pi)})


def pred_out_top5(theta, E):
    """Taux prédit de coups joués hors des 5 premiers (somme des P(i) pour rang >= 5)."""
    Ncount = (~np.isnan(E)).sum(1)
    tot = 0.0
    for r in range(5, E.shape[1]):
        ok = Ncount > r
        if not ok.any():
            break
        tot += _probs_played(theta, E[ok], np.full(ok.sum(), r), Ncount[ok]).sum()
    return tot / len(E)
