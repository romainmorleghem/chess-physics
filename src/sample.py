"""Étape 2 : échantillon pilote.

Lecture en flux du PGN .zst (éventuellement tronqué), sélection des parties
Blitz/Rapid dont les deux joueurs sont dans la même tranche d'Elo, puis tirage
d'UNE position par partie (positions indépendantes => bootstrap propre).
Échantillonnage par réservoir (algorithme R) avec graine fixe : chaque partie
éligible du fichier a la même probabilité d'être retenue.
"""
import csv
import io
import random

import chess
import chess.pgn
import zstandard


def iter_games_raw(path):
    """Renvoie (headers: dict, movetext: str) pour chaque partie, en tolérant un fichier tronqué."""
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        headers, moves = {}, []
        try:
            for line in text:
                if line.startswith("["):
                    if moves:  # nouvelle partie sans ligne vide (sécurité)
                        yield headers, " ".join(moves)
                        headers, moves = {}, []
                    sp = line.find(" ")
                    headers[line[1:sp]] = line[sp + 2: line.rfind('"')]
                elif line.strip():
                    moves.append(line.strip())
                elif moves:
                    yield headers, " ".join(moves)
                    headers, moves = {}, []
        except zstandard.ZstdError:
            pass  # fin d'un préfixe tronqué : la dernière partie partielle est ignorée


def cadence_of(event):
    if not event.startswith("Rated"):
        return None
    for c in ("Blitz", "Rapid"):
        if c in event:
            return c
    return None


def band_of(w, b, lo, hi, step):
    if not (lo <= w < hi and lo <= b < hi):
        return None
    bw, bb = (w - lo) // step, (b - lo) // step
    return lo + bw * step if bw == bb else None


def pick_position(movetext, rng, cfg):
    """Choisit au hasard une position éligible de la partie. None si aucune."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None or game.errors:
        return None
    nodes = list(game.mainline())
    n = len(nodes)
    # plies indexés 0..n-1 ; exclut les 10 premiers, le premier et le dernier coup
    eligible = list(range(max(cfg["skip_first_plies"], 1), n - 1))
    rng.shuffle(eligible)
    for i in eligible:
        node = nodes[i]
        board = node.parent.board()
        if board.legal_moves.count() < cfg["min_legal_moves"]:
            continue
        # horloge du joueur au trait AVANT le coup = son %clk au coup précédent
        prev_own = nodes[i - 2] if i >= 2 else None
        clk = prev_own.clock() if prev_own is not None else None
        return dict(ply=i, n_plies=n, fen=board.fen(), move=node.move.uci(),
                    turn="w" if board.turn else "b", clock_before=clk)
    return None


def run(cfg, log=print):
    s = cfg["sample"]
    rng = random.Random(cfg["seed"])
    lo, hi, step = s["elo_min"], s["elo_max"], s["elo_step"]
    R = s["reservoir_per_cell"]
    res, seen = {}, {}
    n_games = 0
    for h, mt in iter_games_raw(cfg["data"]["raw_file"]):
        n_games += 1
        if n_games % 1_000_000 == 0:
            log(f"  {n_games/1e6:.0f} M parties lues")
        cad = cadence_of(h.get("Event", ""))
        if cad is None or "BOT" in (h.get("WhiteTitle", ""), h.get("BlackTitle", "")):
            continue
        if h.get("Termination") not in ("Normal", "Time forfeit"):
            continue
        try:
            band = band_of(int(h["WhiteElo"]), int(h["BlackElo"]), lo, hi, step)
        except (KeyError, ValueError):
            continue
        if band is None:
            continue
        key = (cad, band)
        k = seen.get(key, 0)
        seen[key] = k + 1
        rec = (h.get("Site", "")[-8:], int(h["WhiteElo"]), int(h["BlackElo"]),
               h.get("TimeControl", ""), mt)
        cell = res.setdefault(key, [])
        if k < R:
            cell.append(rec)
        else:
            j = rng.randrange(k + 1)
            if j < R:
                cell[j] = rec

    rows = []
    for (cad, band), cell in sorted(res.items()):
        rng_cell = random.Random(f"{cfg['seed']}-{cad}-{band}")
        rng_cell.shuffle(cell)  # ordre aléatoire = ordre d'évaluation
        for order, (gid, we, be, tc, mt) in enumerate(cell):
            p = pick_position(mt, rng_cell, s)
            if p is None:
                continue
            rows.append(dict(cadence=cad, band=band, order=order, game_id=gid,
                             white_elo=we, black_elo=be, time_control=tc, **p))
    with open(s["output"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {k: (seen[k], sum(1 for r in rows if (r["cadence"], r["band"]) == k)) for k in seen}
    return n_games, summary
