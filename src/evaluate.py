"""Étape 3 : évaluations Stockfish MultiPV, profondeur fixe, en parallèle.

Reproductibilité : 1 thread par moteur, table de hachage vidée avant chaque
position (ucinewgame), profondeur fixe => résultat déterministe.
Les résultats sont ajoutés à un .jsonl (reprise possible après interruption).
"""
import json
import os
import time
from multiprocessing import Pool

import chess
import chess.engine

_engine = None
_cfg = None


def _init(cfg):
    global _engine, _cfg
    _cfg = cfg
    _engine = chess.engine.SimpleEngine.popen_uci(cfg["path"])
    _engine.configure({"Threads": 1, "Hash": cfg["hash_mb"]})


def _score(s, turn):
    p = s.pov(turn)
    return (None, p.mate()) if p.is_mate() else (p.score(), None)


def analyse_one(task):
    key, fen, played = task
    board = chess.Board(fen)
    infos = _engine.analyse(board, chess.engine.Limit(depth=_cfg["depth"]),
                            multipv=_cfg["multipv"], game=object())
    pvs = []
    for info in infos:
        if "pv" not in info:
            continue
        cp, mate = _score(info["score"], board.turn)
        pvs.append(dict(move=info["pv"][0].uci(), cp=cp, mate=mate))
    return dict(key=key, played=played, pvs=pvs,
                in_cands=played in [p["move"] for p in pvs])


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r
    return done


def evaluate(tasks, cfg, out_path, workers):
    """tasks: liste (key, fen, played). Ajoute au jsonl ; renvoie dict key->résultat."""
    done = load_done(out_path)
    todo = [t for t in tasks if t[0] not in done]
    t0 = time.time()
    if todo:
        # Pool.__exit__ appelle terminate() : évite le blocage dû au thread du moteur
        with Pool(workers, initializer=_init, initargs=(cfg,)) as pool, \
                open(out_path, "a") as f:
            for r in pool.imap(analyse_one, todo, chunksize=2):
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[r["key"]] = r
            dt = time.time() - t0
        return done, dt, len(todo)
    return done, 0.0, 0


def keep(r, max_abs_cp):
    """Filtre post-évaluation : pas de mat sur la meilleure ligne, |éval| <= max."""
    if not r["pvs"]:
        return False
    best = r["pvs"][0]
    return best["mate"] is None and abs(best["cp"]) <= max_abs_cp
