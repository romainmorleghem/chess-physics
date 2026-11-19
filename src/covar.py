"""Pistes 3 et 6 : l'échelle s dépend-elle de covariables de la position (score, type) ?

Modèle additif sur l'échelle de la loi de choix p_i ∝ exp(-(ΔE_i/s)^c), c fixé :
    log s = Σ_f coef_f[niveau_f]        (premier facteur = tranche d'Elo, niveaux libres ;
                                         les autres ont leur niveau de référence à 0)
Test de chaque facteur par rapport de vraisemblance (modèle complet contre modèle sans lui),
effets par niveau avec IC bootstrap (rééchantillonnage des parties).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2

import clock as ck
import fit as ft
import phase as ph


def nll_rows(log_s, c, E, idx):
    U = np.where(np.isnan(E), -np.inf, -(np.nan_to_num(E) / np.exp(log_s)[:, None]) ** c)
    return -(U[np.arange(len(idx)), idx] - logsumexp(U, axis=1)).sum()


def nll_grad_rows(log_s, c, E, idx):
    """nll et sa dérivée par rapport à log s de chaque ligne (dU/dlog s = -c U)."""
    U = np.where(np.isnan(E), -np.inf, -(np.nan_to_num(E) / np.exp(log_s)[:, None]) ** c)
    lse = logsumexp(U, axis=1)
    P = np.exp(U - lse[:, None])
    Uf = np.where(np.isnan(E), 0.0, U)
    rows = np.arange(len(idx))
    nll = -(U[rows, idx] - lse).sum()
    g = c * (Uf[rows, idx] - (P * Uf).sum(1))      # d nll / d log s_n
    return nll, g


def fit_c(E, idx):
    """MLE commune (s, c) : renvoie c."""
    r = minimize(lambda x: nll_rows(np.full(len(idx), x[0]), np.exp(x[1]), E, idx),
                 [np.log(30.0), np.log(0.5)], method="Nelder-Mead")
    return float(np.exp(r.x[1]))


def fit_additive(E, idx, factors, c, x0=None):
    sizes = [int(f.max()) + 1 for f in factors]
    nfree = [sizes[0]] + [s - 1 for s in sizes[1:]]

    def unpack(x):
        out, k = [], 0
        for j, n in enumerate(nfree):
            v = x[k:k + n]
            out.append(v if j == 0 else np.concatenate([[0.0], v]))
            k += n
        return out

    def f(x):
        val, g = nll_grad_rows(sum(co[fa] for co, fa in zip(unpack(x), factors)), c, E, idx)
        grads = []
        for j, (fa, sz) in enumerate(zip(factors, sizes)):
            gb = np.bincount(fa, weights=g, minlength=sz)
            grads.append(gb if j == 0 else gb[1:])
        return val, np.concatenate(grads)

    if x0 is None:
        x0 = np.concatenate([np.full(nfree[0], np.log(20.0))] + [np.zeros(n) for n in nfree[1:]])
    r = minimize(f, x0, jac=True, method="L-BFGS-B")
    return unpack(r.x), r.fun, r.x


def dataset(cfg, cad, variant="cp"):
    """E, idx et un DataFrame de covariables pour une cadence (spectre complet, positions retenues)."""
    import chess
    from run import spectrum_records
    df = pd.read_csv(cfg["sample"]["output"])
    df["key"] = [f"{c}|{b}|{o}" for c, b, o in zip(df.cadence, df.band, df.order)]
    meta = df.set_index("key")
    recs, rows = [], []
    sp = spectrum_records(cfg)
    bands = sorted(b for c2, b in sp if c2 == cad)
    for (c2, band), rs in sp.items():
        if c2 != cad:
            continue
        for r in rs:
            m = meta.loc[r["key"]]
            v = ft.cand_values(r["pvs"], "cp", cfg["fit"])
            srt = np.sort(v)[::-1]
            board = chess.Board(m.fen)
            recs.append(r)
            rows.append(dict(key=r["key"], game_id=m.game_id, elo=bands.index(band), band=band,
                             ply=m.ply, frac=ck.frac_remaining(m.clock_before, m.time_control),
                             material=ph.material(m.fen), best_cp=srt[0],
                             gap12=srt[0] - srt[1] if len(srt) > 1 else np.nan,
                             n_good50=int((srt >= srt[0] - 50).sum()), n_legal=len(srt),
                             in_check=board.is_check(),
                             white=m.turn == "w"))
    E, idx = ft.build(recs, variant, cfg["fit"])
    return E, idx, pd.DataFrame(rows)


def binned(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def study(E, idx, factors, names, target, c, rng, n_boot, drop_tests=None, ref=0):
    """factors : liste de vecteurs d'entiers ; names : leurs noms ; target : nom du facteur dont on
    veut les effets avec IC. drop_tests : noms des facteurs à tester (défaut : tous sauf l'Elo).
    Renvoie (effets du facteur cible, tests)."""
    co, nll_full, xf = fit_additive(E, idx, factors, c)
    tests = []
    for nm in drop_tests or names[1:]:
        j = names.index(nm)
        sub = [f for k, f in enumerate(factors) if k != j]
        nll0 = fit_additive(E, idx, sub, c)[1]
        dof = int(factors[j].max())
        lr = 2 * (nll0 - nll_full)
        tests.append(dict(factor=nm, controls="+".join(n for n in names if n != nm), LR=lr, dof=dof,
                          p_value=chi2.sf(lr, dof)))
    t = names.index(target)
    bb = []
    for _ in range(n_boot):
        k = rng.integers(0, len(idx), len(idx))
        bb.append(fit_additive(E[k], idx[k], [f[k] for f in factors], c, x0=xf)[0][t])
    bb = np.array(bb)
    bb = bb - bb[:, [ref]]                       # effets relatifs au niveau de référence
    est = co[t] - co[t][ref]
    eff = pd.DataFrame(dict(level=np.arange(len(est)), n=np.bincount(factors[t], minlength=len(est)),
                            s_ratio=np.exp(est), lo=np.exp(np.quantile(bb, .025, axis=0)),
                            hi=np.exp(np.quantile(bb, .975, axis=0))))
    return eff, pd.DataFrame(tests), co, bb
