"""Piste 2 : la température en fonction du temps restant (recuit simulé).

c est fixé à C_FIXED (valeur quasi universelle trouvée en piste 1) ; on ajuste l'échelle s
par (groupe d'Elo g, tranche de temps restant b). Test de superposition (factorisation) :
    log s_gb = a_g + b_b   (modèle factorisé)   contre   s_gb libre,
par rapport de vraisemblance. Si la factorisation tient, toutes les courbes s(t)/s(Elo)
se superposent : le temps agit de la même façon à tous les niveaux.
"""
import numpy as np
from scipy.optimize import minimize

C_FIXED = 0.5


def frac_remaining(clock_before, time_control):
    try:
        base = float(str(time_control).split("+")[0])
    except ValueError:
        return np.nan
    return clock_before / base if base > 0 else np.nan


def _nll_cells(log_s_per_row, E, idx):
    s = np.exp(log_s_per_row)[:, None]
    U = np.where(np.isnan(E), -np.inf, -(np.nan_to_num(E) / s) ** C_FIXED)
    m = U.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(U - m).sum(1))
    return -(U[np.arange(len(idx)), idx] - lse).sum()


def fit_free(E, idx, g, b, G, B):
    """s libre par case (g, b). Renvoie log_s (G x B) et la nll totale."""
    out = np.full((G, B), np.nan)
    tot = 0.0
    for gi in range(G):
        for bi in range(B):
            m = (g == gi) & (b == bi)
            if m.sum() < 30:
                continue
            r = minimize(lambda x: _nll_cells(np.full(m.sum(), x[0]), E[m], idx[m]),
                         [np.log(20.0)], method="L-BFGS-B", bounds=[(np.log(0.1), np.log(5000))])
            out[gi, bi] = r.x[0]
            tot += r.fun
    return out, tot


def fit_factorized(E, idx, g, b, G, B):
    """log s = a_g + b_b (b_0 = 0). Renvoie (a, bb) et la nll totale."""
    def f(x):
        a, bb = x[:G], np.concatenate([[0.0], x[G:]])
        return _nll_cells(a[g] + bb[b], E, idx)
    r = minimize(f, np.concatenate([np.full(G, np.log(20.0)), np.zeros(B - 1)]), method="L-BFGS-B")
    return (r.x[:G], np.concatenate([[0.0], r.x[G:]])), r.fun
