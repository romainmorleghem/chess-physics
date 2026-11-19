"""Contrôle de la piste 2 : effet du temps restant à phase de jeu fixée.

c est fixé à 0,5 ; on ajuste un modèle additif sur l'échelle :
    log s = a[tranche d'Elo] + b[tranche de temps] + d[phase]     (b_0 = d_0 = 0)
La phase est mesurée soit par le numéro de demi-coup, soit par le matériel non-pion restant
(Q=4, T=2, F=C=1, les deux camps ; 24 au départ). Tests par rapport de vraisemblance :
  temps | phase  : (a + b + d) contre (a + d)   -> le temps agit-il encore à phase fixée ?
  phase | temps  : (a + b + d) contre (a + b)
"""
import numpy as np
from scipy.optimize import minimize

from clock import _nll_cells

MAT = {"q": 4, "r": 2, "b": 1, "n": 1}


def material(fen):
    return sum(MAT.get(ch.lower(), 0) for ch in fen.split()[0] if ch.isalpha())


def fit_additive(E, idx, factors, x0=None):
    """factors : liste de vecteurs d'indices entiers (0..L-1). Le premier facteur a tous ses
    niveaux libres (ordonnée), les suivants ont leur niveau 0 fixé à 0.
    Renvoie (liste des coefficients par facteur, nll)."""
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
        co = unpack(x)
        return _nll_cells(sum(c[fa] for c, fa in zip(co, factors)), E, idx)

    if x0 is None:
        x0 = np.concatenate([np.full(nfree[0], np.log(20.0))] + [np.zeros(n) for n in nfree[1:]])
    r = minimize(f, x0, method="L-BFGS-B")
    return unpack(r.x), r.fun, r.x
