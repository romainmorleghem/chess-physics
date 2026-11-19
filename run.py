"""Point d'entrée unique. Usage :
    python run.py all            # tout le pipeline
    python run.py download | sample | timing | evaluate | fit | shape
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


def select_records(cfg):
    """Pour chaque case : les target_per_cell premières positions (ordre aléatoire fixe)
    qui passent le filtre. Renvoie {(cadence, band): [résultats Stockfish]}."""
    flt = cfg["filter"]
    df = pd.read_csv(cfg["sample"]["output"]).sort_values(["cadence", "band", "order"])
    done = ev.load_done(cfg["engine"]["output"])
    out = {}
    for (cad, band), g in df.groupby(["cadence", "band"]):
        recs = []
        for o in g.order:
            r = done.get(f"{cad}|{band}|{o}")
            if r is not None and ev.keep(r, flt["max_abs_cp"]):
                recs.append(r)
            if len(recs) >= flt["target_per_cell"]:
                break
        out[(cad, band)] = recs
    return out


def step_fit(cfg):
    import plot
    f = cfg["fit"]
    os.makedirs(f["outdir"], exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    alpha = (1 - f["ci"]) / 2
    rows = []
    for (cad, band), recs in select_records(cfg).items():
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


def step_shape(cfg):
    """Piste 1 : forme de la loi de choix (exp. étirée par case, f non paramétrique par groupe)."""
    import plot
    import shape as sh
    f, n_boot = cfg["fit"], cfg["shape"]["n_boot"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    alpha = (1 - f["ci"]) / 2
    recs_all = select_records(cfg)
    rows = []
    for (cad, band), recs in recs_all.items():
        E, idx = ft.build([r for r in recs if r["in_cands"]], "cp", f)
        (s, c), nll_s = sh.fit_stretched(E, idx)
        T, _ = ft.fit_T(E, idx)
        lr = 2 * (ft.nll(np.log(T), E, idx) - nll_s)  # test du rapport de vraisemblance c = 1
        bs = np.array([sh.fit_stretched(E[i], idx[i])[0] for i in
                       (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))])
        rows.append(dict(cadence=cad, elo_mid=band + 100, n=len(idx), s=s, c=c,
                         c_lo=np.quantile(bs[:, 1], alpha), c_hi=np.quantile(bs[:, 1], 1 - alpha),
                         LR_vs_boltzmann=lr))
        print(f"  {cad} {band}: c={c:.2f} [{rows[-1]['c_lo']:.2f}-{rows[-1]['c_hi']:.2f}]  s={s:.0f}  LR={lr:.1f}", flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(f["outdir"], "shape_stretched.csv"), index=False, float_format="%.4g")
    # f(ΔE) non paramétrique par groupes d'Elo (tranches regroupées par 3)
    groups = cfg["shape"]["elo_groups"]
    curves = []
    for cad in sorted({k[0] for k in recs_all}):
        for lo, hi in groups:
            recs = [r for (c2, b), rs in recs_all.items() if c2 == cad and lo <= b < hi
                    for r in rs if r["in_cands"]]
            E, idx = ft.build(recs, "cp", f)
            fk = sh.fit_nonparam(E, idx)
            bs = np.array([sh.fit_nonparam(E[i], idx[i]) for i in
                           (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))])
            for k, dE in enumerate(sh.KNOTS):
                curves.append(dict(cadence=cad, elo_group=f"{lo}-{hi}", dE=dE, f=fk[k],
                                   f_lo=np.quantile(bs[:, k], alpha), f_hi=np.quantile(bs[:, k], 1 - alpha)))
    cv = pd.DataFrame(curves)
    cv.to_csv(os.path.join(f["outdir"], "shape_nonparam.csv"), index=False, float_format="%.4g")
    plot.make_shape(st, cv, f["outdir"])
    print(st[["cadence", "elo_mid", "c", "c_lo", "c_hi", "LR_vs_boltzmann"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))


STEPS = dict(download=step_download, sample=step_sample, timing=step_timing,
             evaluate=step_evaluate, fit=step_fit, shape=step_shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    for s in (["download", "sample", "evaluate", "fit", "shape"] if a.step == "all" else [a.step]):
        print(f"== {s} ==")
        STEPS[s](cfg)
