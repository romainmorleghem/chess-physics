"""Piste 5 : désordre. Distribution des températures individuelles dans une tranche d'Elo.

Échantillon : joueurs Blitz actifs (>= min_games parties éligibles dans le préfixe du fichier), stratifiés
par tranche de leur Elo médian ; n_games parties tirées par joueur, une position par partie (le joueur au trait).
Les noms sont remplacés par un hachage (sha1 tronqué).

Modèle à effets aléatoires (c fixé) :
    log s_n = a[tranche] + d[demi-coup] + e·échec + u_joueur,   u ~ N(0, σ²)
vraisemblance marginale par quadrature de Gauss-Hermite. σ mesure le « désordre » : dispersion des températures
entre joueurs de même Elo, une fois la phase et l'échec contrôlés.
"""
import hashlib
import io
import random

import chess.pgn
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import sample as sm


def _hash(name):
    return hashlib.sha1(name.encode()).hexdigest()[:10]


def _eligible(h):
    if sm.cadence_of(h.get("Event", "")) != "Blitz" or "BOT" in (h.get("WhiteTitle", ""), h.get("BlackTitle", "")):
        return False
    return h.get("Termination") in ("Normal", "Time forfeit")


def select_players(cfg, log=print):
    p = cfg["players"]
    ratings = {}
    for h, _ in sm.iter_games_raw(p["raw_file"]):
        if not _eligible(h):
            continue
        for side in ("White", "Black"):
            try:
                ratings.setdefault(h[side], []).append(int(h[side + "Elo"]))
            except (KeyError, ValueError):
                pass
    rng = random.Random(cfg["seed"] + 11)
    chosen = {}
    for lo in range(p["elo_min"], p["elo_max"], p["elo_step"]):
        cands = sorted(n for n, r in ratings.items() if len(r) >= p["min_games"] and lo <= np.median(r) < lo + p["elo_step"])
        rng.shuffle(cands)
        for n in cands[:p["players_per_band"]]:
            chosen[n] = lo
        log(f"  tranche {lo}: {len(cands)} joueurs éligibles, {min(len(cands), p['players_per_band'])} retenus")
    return chosen


def sample_positions(cfg, chosen):
    p = cfg["players"]
    games = {n: [] for n in chosen}
    for h, mt in sm.iter_games_raw(p["raw_file"]):
        if not _eligible(h):
            continue
        for side in ("White", "Black"):
            n = h.get(side)
            if n in games:
                games[n].append((h.get("Site", "")[-8:], side[0].lower(), int(h[side + "Elo"]), h.get("TimeControl", ""), mt))
    rows = []
    for n, band in chosen.items():
        rng = random.Random(f"{cfg['seed']}-{n}")
        g = games[n]
        rng.shuffle(g)
        k = 0
        for gid, col, elo, tc, mt in g:
            if k >= p["n_games"]:
                break
            pos = _pick(mt, col, rng, cfg["sample"])
            if pos is None:
                continue
            rows.append(dict(player=_hash(n), band=band, k=k, game_id=gid, elo=elo, time_control=tc, **pos))
            k += 1
    return rows


def _pick(mt, col, rng, s):
    game = chess.pgn.read_game(io.StringIO(mt))
    if game is None or game.errors:
        return None
    nodes = list(game.mainline())
    want = 0 if col == "w" else 1                      # parité du demi-coup quand ce joueur est au trait
    el = [i for i in range(max(s["skip_first_plies"], 1), len(nodes) - 1) if i % 2 == want]
    rng.shuffle(el)
    for i in el:
        board = nodes[i].parent.board()
        if board.legal_moves.count() < s["min_legal_moves"]:
            continue
        prev = nodes[i - 2] if i >= 2 else None
        return dict(ply=i, n_plies=len(nodes), fen=board.fen(), move=nodes[i].move.uci(),
                    clock_before=prev.clock() if prev is not None else None)
    return None


# ---------- modèle à effets aléatoires ----------

def _loglik_rows(log_s, c, E, idx):
    U = np.where(np.isnan(E), -np.inf, -(np.nan_to_num(E) / np.exp(log_s)[:, None]) ** c)
    return U[np.arange(len(idx)), idx] - logsumexp(U, axis=1)


class RE:
    def __init__(self, E, idx, factors, player, c, n_nodes=21):
        self.E, self.idx, self.fac, self.c = E, idx, factors, c
        self.pl = player
        self.P = int(player.max()) + 1
        self.sizes = [int(f.max()) + 1 for f in factors]
        self.nfree = [self.sizes[0]] + [s - 1 for s in self.sizes[1:]]
        x, w = np.polynomial.hermite.hermgauss(n_nodes)
        self.gx, self.lw = np.sqrt(2) * x, np.log(w / np.sqrt(np.pi))

    def eta(self, x):
        out, k = 0.0, 0
        for j, (f, n) in enumerate(zip(self.fac, self.nfree)):
            v = x[k:k + n]
            out = out + (v if j == 0 else np.concatenate([[0.0], v]))[f]
            k += n
        return out

    def nll(self, x, sigma=None):
        sig = np.exp(x[-1]) if sigma is None else sigma
        eta = self.eta(x)
        L = np.empty((len(self.gx), self.P))
        for q, g in enumerate(self.gx):
            L[q] = np.bincount(self.pl, weights=_loglik_rows(eta + sig * g, self.c, self.E, self.idx), minlength=self.P)
        return -logsumexp(L + self.lw[:, None], axis=0).sum()

    def fit(self, x0=None, sigma0=False):
        nfe = sum(self.nfree)
        if x0 is None:
            x0 = np.concatenate([np.full(self.nfree[0], np.log(20.0)), np.zeros(nfe - self.nfree[0]), [np.log(0.2)]])
        if sigma0:   # modèle sans effet joueur
            r = minimize(lambda z: self.nll(np.append(z, 0.0), sigma=0.0), x0[:-1], method="L-BFGS-B")
            return np.append(r.x, -np.inf), r.fun
        r = minimize(self.nll, x0, method="L-BFGS-B", bounds=[(None, None)] * nfe + [(np.log(1e-3), np.log(3))])
        return r.x, r.fun

    def player_effects(self, x, mask=None):
        """Estimation a posteriori (moyenne) de u par joueur, sur les positions de mask."""
        sig = np.exp(x[-1])
        eta = self.eta(x)
        m = np.ones(len(self.idx), bool) if mask is None else mask
        L = np.empty((len(self.gx), self.P))
        for q, g in enumerate(self.gx):
            L[q] = np.bincount(self.pl[m], weights=_loglik_rows(eta[m] + sig * g, self.c, self.E[m], self.idx[m]),
                               minlength=self.P)
        W = L + self.lw[:, None]
        W = np.exp(W - W.max(0))
        return sig * (self.gx[:, None] * W).sum(0) / W.sum(0)
