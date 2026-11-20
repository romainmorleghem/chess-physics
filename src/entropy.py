"""Piste 8 (pilote) : énergie contre entropie des positions atteintes (tactique / stratégie).

Pour chaque position et chacun de ses K meilleurs coups i (spectre prof. 12) :
  - position fille (adversaire au trait) : tous ses coups évalués -> S_adv(i), L_adv(i)
  - position petite-fille, après la meilleure réponse adverse (joueur au trait) -> S_moi(i), L_moi(i)
avec, pour une position :
  S = log N_bons,  N_bons = nombre de coups à moins de `good_cp` du meilleur  (entropie « de Boltzmann »)
  L = <ΔE> sous la loi de choix de la tranche d'Elo (exp. étirée, s et c de results/spectrum_fit.csv) :
      perte moyenne attendue du joueur au trait (sa « faillibilité » dans cette position).
Modèles de choix (conditionnels aux K candidats, positions où le coup joué est parmi eux) :
  (A) entropique :  U_i = -(ΔE_i/s)^c + λ_moi S_moi(i) + λ_adv S_adv(i) [+ contrôles]
  (B) énergie libre (« évaluation pratique ») :
      E_i^prat = ΔE_i + κ_moi L_moi(i) - κ_adv L_adv(i),  U_i = -((E_i^prat - min_j E_j^prat)/s)^c
      Joueur rationnel conscient de sa faillibilité et de celle de l'adversaire : κ_moi = κ_adv = 1.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import fit as ft


def values(pvs, fcfg):
    return ft.cand_values(pvs, "cp", fcfg)


def summarize(pvs, fcfg, good_cp, s, c):
    v = values(pvs, fcfg)
    if len(v) == 0:
        return dict(S=0.0, L=0.0, N=0)
    dE = v.max() - v
    w = -(dE / s) ** c
    p = np.exp(w - logsumexp(w))
    return dict(S=float(np.log((dE <= good_cp).sum())), L=float((p * dE).sum()), N=len(v))


def nll_entropic(theta, dE, X, idx, band, nb, c):
    """theta = [log s par tranche (nb), λ (X.shape[2])]. dE, X : (n, K) et (n, K, p) ; NaN = absent."""
    ls = theta[:nb][band]
    lam = theta[nb:]
    U = -(np.nan_to_num(dE) / np.exp(ls)[:, None]) ** c + np.nan_to_num(X) @ lam
    U = np.where(np.isnan(dE), -np.inf, U)
    return -(U[np.arange(len(idx)), idx] - logsumexp(U, axis=1)).sum()


def nll_practical(theta, dE, Lm, La, idx, band, nb, c):
    ls = theta[:nb][band]
    km, ka = theta[nb], theta[nb + 1]
    Ep = dE + km * Lm - ka * La
    Ep = Ep - np.nanmin(Ep, axis=1, keepdims=True)
    U = -(np.nan_to_num(Ep) / np.exp(ls)[:, None]) ** c
    U = np.where(np.isnan(dE), -np.inf, U)
    return -(U[np.arange(len(idx)), idx] - logsumexp(U, axis=1)).sum()


def fit(fun, x0, *args):
    r = minimize(fun, x0, args=args, method="L-BFGS-B")
    return r.x, r.fun


def hessian_se(fun, x, *args, h=1e-3):
    """Erreurs-types par la hessienne numérique de la nll."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = fun(x, *args)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (fun(x + ei + ej, *args) - fun(x + ei, *args) - fun(x + ej, *args) + f0) / h ** 2
    return np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
