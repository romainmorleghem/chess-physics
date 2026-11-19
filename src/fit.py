"""Étape 4 : ajustement de la température par maximum de vraisemblance.

Modèle de Boltzmann sur les k coups candidats :
    p_i = exp(e_i / T) / sum_j exp(e_j / T)
soit, en langage physique, énergie E_i = e_best - e_i >= 0 (perte par rapport
au meilleur coup) et p_i ∝ exp(-E_i / T). La vraisemblance porte sur le coup
joué SACHANT qu'il est parmi les k candidats (les coups hors candidats sont
comptés séparément).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

LOGT_BOUNDS = (np.log(1e-3), np.log(1e6))


def cand_values(pvs, variant, cfg):
    clip = cfg["cp_clip"]
    v = []
    for p in pvs:
        cp = p["cp"] if p["mate"] is None else (clip if p["mate"] > 0 else -clip)
        v.append(float(np.clip(cp, -clip, clip)))
    v = np.array(v)
    if variant == "wp":
        v = 100.0 / (1.0 + np.exp(-cfg["wp_coef"] * v))
    return v


def build(records, variant, cfg):
    """records: résultats d'évaluation dont le coup joué est parmi les candidats.
    Renvoie une matrice E (énergies, NaN en bourrage) et l'indice du coup joué."""
    k = max(len(r["pvs"]) for r in records)
    E = np.full((len(records), k), np.nan)
    idx = np.empty(len(records), dtype=int)
    for n, r in enumerate(records):
        v = cand_values(r["pvs"], variant, cfg)
        E[n, :len(v)] = v.max() - v
        idx[n] = [p["move"] for p in r["pvs"]].index(r["played"])
    return E, idx


def nll(logT, E, idx):
    T = np.exp(logT)
    a = np.where(np.isnan(E), -np.inf, -E / T)
    return -(a[np.arange(len(idx)), idx] - logsumexp(a, axis=1)).sum()


def fit_T(E, idx):
    r = minimize_scalar(nll, bounds=LOGT_BOUNDS, args=(E, idx), method="bounded",
                        options={"xatol": 1e-6})
    T = float(np.exp(r.x))
    at_bound = min(abs(r.x - b) for b in LOGT_BOUNDS) < 1e-3
    return T, at_bound


def bootstrap(E, idx, n_boot, rng):
    n = len(idx)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)  # 1 position par partie => rééchantillonnage par parties
        out[b] = fit_T(E[s], idx[s])[0]
    return out
