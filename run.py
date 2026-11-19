"""Point d'entrée unique. Usage :
    python run.py all            # tout le pipeline
    python run.py download | sample | timing | evaluate | fit
Toutes les options sont dans config.yaml.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import evaluate as ev  # noqa: E402
import fit as ft  # noqa: E402
import sample as sm  # noqa: E402


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def step_download(cfg):
    d = cfg["data"]
    os.makedirs(os.path.dirname(d["raw_file"]), exist_ok=True)
    if os.path.exists(d["raw_file"]) and os.path.getsize(d["raw_file"]) >= d["max_bytes"]:
        print("Déjà téléchargé.")
        return
    subprocess.run(["curl", "-sS", "-C", "-", "-r", f"0-{d['max_bytes'] - 1}",
                    "-o", d["raw_file"], d["url"]], check=True)


def step_sample(cfg):
    n_games, summ = sm.run(cfg)
    print(f"Parties lues : {n_games:,}")
    t = pd.DataFrame([(c, b, s, k) for (c, b), (s, k) in summ.items()],
                     columns=["cadence", "band", "games_eligible", "positions_drawn"])
    print(t.sort_values(["cadence", "band"]).to_string(index=False))


def _tasks(df):
    return [(f"{r.cadence}|{r.band}|{r.order}", r.fen, r.move) for r in df.itertuples()]


def step_timing(cfg, n=200):
    df = pd.read_csv(cfg["sample"]["output"])
    per = math.ceil(n / df.groupby(["cadence", "band"]).ngroups)
    sub = df.sort_values("order").groupby(["cadence", "band"]).head(per).head(n)
    done, dt, m = ev.evaluate(_tasks(sub), cfg["engine"], cfg["engine"]["output"],
                              cfg["engine"]["workers"])
    keys = set(t[0] for t in _tasks(sub))
    kept = np.mean([ev.keep(done[k], cfg["filter"]["max_abs_cp"]) for k in keys])
    if m == 0:
        print("Déjà évalué (supprime data/evals.jsonl pour rechronométrer).")
        return
    per_pos = dt / m
    n_cells = df.groupby(["cadence", "band"]).ngroups
    total = n_cells * cfg["filter"]["target_per_cell"] / kept
    print(f"{m} positions en {dt:.0f} s -> {per_pos:.2f} s/position (avec {cfg['engine']['workers']} workers)")
    print(f"Taux de positions conservées après filtre : {kept:.0%}")
    print(f"Extrapolation : ~{total:,.0f} positions à évaluer -> ~{total * per_pos / 3600:.1f} h")


def step_evaluate(cfg):
    df = pd.read_csv(cfg["sample"]["output"]).sort_values(["cadence", "band", "order"])
    df["key"] = [f"{c}|{b}|{o}" for c, b, o in zip(df.cadence, df.band, df.order)]
    target, maxcp = cfg["filter"]["target_per_cell"], cfg["filter"]["max_abs_cp"]
    out = cfg["engine"]["output"]
    while True:
        done = ev.load_done(out)
        batch = []
        for _, g in df.groupby(["cadence", "band"]):
            kept = sum(ev.keep(done[k], maxcp) for k in g.key if k in done)
            evald = sum(k in done for k in g.key)
            rest = g[~g.key.isin(done)]
            if kept >= target or rest.empty:
                continue
            rate = max(kept / evald, 0.3) if evald else 0.8
            batch.append(rest.head(math.ceil((target - kept) / rate * 1.05) + 5))
        if not batch:
            break
        b = pd.concat(batch)
        print(f"  lot de {len(b)} positions...", flush=True)
        ev.evaluate(_tasks(b), cfg["engine"], out, cfg["engine"]["workers"])
    print("Évaluations terminées.")


def step_fit(cfg):
    import plot
    f, flt = cfg["fit"], cfg["filter"]
    os.makedirs(f["outdir"], exist_ok=True)
    df = pd.read_csv(cfg["sample"]["output"]).sort_values(["cadence", "band", "order"])
    done = ev.load_done(cfg["engine"]["output"])
    rng = np.random.default_rng(cfg["seed"])
    alpha = (1 - f["ci"]) / 2
    rows = []
    for (cad, band), g in df.groupby(["cadence", "band"]):
        recs = []
        for o in g.order:
            r = done.get(f"{cad}|{band}|{o}")
            if r is not None and ev.keep(r, flt["max_abs_cp"]):
                recs.append(r)
            if len(recs) >= flt["target_per_cell"]:
                break
        inc = [r for r in recs if r["in_cands"]]
        row = dict(cadence=cad, elo_band=f"{band}-{band + cfg['sample']['elo_step']}",
                   elo_mid=band + cfg["sample"]["elo_step"] / 2, n_positions=len(recs),
                   n_out_of_cands=len(recs) - len(inc),
                   out_rate=1 - len(inc) / len(recs) if recs else np.nan,
                   top1_rate=np.mean([r["pvs"][0]["move"] == r["played"] for r in recs]))
        if len(inc) < 10:
            print(f"  {cad} {band}: trop peu de positions ({len(inc)}), ignorée")
            continue
        for var in ("cp", "wp"):
            E, idx = ft.build(inc, var, f)
            T, at_b = ft.fit_T(E, idx)
            bs = ft.bootstrap(E, idx, f["n_boot"], rng)
            row.update({f"T_{var}": T, f"T_{var}_lo": np.quantile(bs, alpha),
                        f"T_{var}_hi": np.quantile(bs, 1 - alpha), f"T_{var}_at_bound": at_b})
        rows.append(row)
        print(f"  {cad} {band}: n={len(recs)}  T_cp={row['T_cp']:.1f}  T_wp={row['T_wp']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(f["outdir"], "summary.csv"), index=False, float_format="%.4g")
    plot.make(res, f["outdir"])
    cols = ["cadence", "elo_band", "n_positions", "out_rate", "T_cp", "T_cp_lo", "T_cp_hi",
            "T_wp", "T_wp_lo", "T_wp_hi"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


STEPS = dict(download=step_download, sample=step_sample, timing=step_timing,
             evaluate=step_evaluate, fit=step_fit)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    for s in (["download", "sample", "evaluate", "fit"] if a.step == "all" else [a.step]):
        print(f"== {s} ==")
        STEPS[s](cfg)
