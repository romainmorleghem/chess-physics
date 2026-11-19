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


def _extra_eval(cfg, name, recs_by_cell):
    """Réévalue des positions déjà retenues avec d'autres réglages (cfg['extra'][name])."""
    x = cfg["extra"][name]
    ecfg = dict(cfg["engine"], depth=x["depth"], multipv=x["multipv"])
    df = pd.read_csv(cfg["sample"]["output"])
    fen = {f"{c}|{b}|{o}": (f, m) for c, b, o, f, m in zip(df.cadence, df.band, df.order, df.fen, df.move)}
    tasks = [(r["key"], *fen[r["key"]]) for recs in recs_by_cell.values() for r in recs]
    if x.get("n_positions"):
        rng = np.random.default_rng(cfg["seed"] + 2)
        tasks = [tasks[i] for i in sorted(rng.choice(len(tasks), x["n_positions"], replace=False))]
    print(f"  {len(tasks)} positions, profondeur {x['depth']}, MultiPV {x['multipv']}", flush=True)
    done, dt, m = ev.evaluate(tasks, ecfg, x["output"], ecfg["workers"])
    if m:
        print(f"  {m} évaluées en {dt / 60:.1f} min ({dt / m:.2f} s/position)")


def step_noise_eval(cfg):
    _extra_eval(cfg, "noise", select_records(cfg))


def step_spectrum_eval(cfg):
    _extra_eval(cfg, "spectrum", select_records(cfg))


def step_noise_check(cfg):
    """B : c à profondeur 16 vs profondeur 20 sur les mêmes positions (bootstrap apparié)."""
    import shape as sh
    f, x = cfg["fit"], cfg["extra"]["noise"]
    d16 = ev.load_done(cfg["engine"]["output"])
    d20 = ev.load_done(x["output"])
    keys = [k for k in d20 if d20[k]["in_cands"] and d16[k]["in_cands"] and d20[k]["pvs"]]
    E16, i16 = ft.build([d16[k] for k in keys], "cp", f)
    E20, i20 = ft.build([d20[k] for k in keys], "cp", f)
    same_best = np.mean([d16[k]["pvs"][0]["move"] == d20[k]["pvs"][0]["move"] for k in d20])
    rng = np.random.default_rng(cfg["seed"] + 3)
    rows = []
    boots = [rng.integers(0, len(keys), len(keys)) for _ in range(cfg["shape"]["n_boot"])]
    for lab, E, idx in (("prof16", E16, i16), ("prof20", E20, i20)):
        (s, c), _ = sh.fit_stretched(E, idx)
        T, _ = ft.fit_T(E, idx)
        bc = np.array([sh.fit_stretched(E[b], idx[b])[0][1] for b in boots])
        rows.append(dict(depth=lab, n=len(keys), c=c, c_lo=np.quantile(bc, .025),
                         c_hi=np.quantile(bc, .975), s=s, T=T, _bc=bc))
    dc = rows[1]["_bc"] - rows[0]["_bc"]
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_bc"} for r in rows])
    res.to_csv(os.path.join(f["outdir"], "noise_check.csv"), index=False, float_format="%.4g")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"Δc (20-16) = {rows[1]['c'] - rows[0]['c']:+.3f}  IC95 apparié [{np.quantile(dc, .025):+.3f}, {np.quantile(dc, .975):+.3f}]")
    print(f"Même meilleur coup à prof. 16 et 20 : {same_best:.0%}")


def step_spectrum_check(cfg):
    """A : forme de f(ΔE) sur tous les coups légaux + test IIA (taux hors top-5 prédit vs observé)."""
    import plot
    import shape as sh
    f, x, n_boot = cfg["fit"], cfg["extra"]["spectrum"], cfg["shape"]["n_boot"]
    dsp = ev.load_done(x["output"])
    rng = np.random.default_rng(cfg["seed"] + 4)
    rows, pooled = [], {}
    for (cad, band), recs in select_records(cfg).items():
        rs = [dsp[r["key"]] for r in recs if r["key"] in dsp and dsp[r["key"]]["in_cands"]]
        E, idx = ft.build(rs, "cp", f)
        obs_out = np.mean(idx >= 5)
        # ajustements sur le top-5 seul (vraisemblance conditionnelle), puis prédiction hors top-5
        top = idx < 5
        E5, i5 = E[top][:, :5], idx[top]
        T5, _ = ft.fit_T(E5, i5)
        (s5, c5), _ = sh.fit_stretched(E5, i5)
        # ajustements sur le spectre complet
        Tf, _ = ft.fit_T(E, idx)
        (sf, cf), nll_s = sh.fit_stretched(E, idx)
        lr = 2 * (ft.nll(np.log(Tf), E, idx) - nll_s)
        bc = np.array([sh.fit_stretched(E[b], idx[b])[0][1] for b in
                       (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))])
        rows.append(dict(cadence=cad, elo_mid=band + 100, n=len(idx), obs_out_top5=obs_out,
                         pred_out_boltz_top5fit=sh.predicted_out_rate(E, 5, lambda e: e / T5),
                         pred_out_stretch_top5fit=sh.predicted_out_rate(E, 5, lambda e: (e / s5) ** c5),
                         c_top5=c5, T_full=Tf, s=sf, c=cf, c_lo=np.quantile(bc, .025),
                         c_hi=np.quantile(bc, .975), LR_vs_boltzmann=lr))
        print(f"  {cad} {band}: c={cf:.2f} [{rows[-1]['c_lo']:.2f}-{rows[-1]['c_hi']:.2f}] hors top5 obs={obs_out:.2f} "
              f"préd. Boltz={rows[-1]['pred_out_boltz_top5fit']:.2f} étirée={rows[-1]['pred_out_stretch_top5fit']:.2f}", flush=True)
        pooled.setdefault(cad, []).append((band, E, idx))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(f["outdir"], "spectrum_fit.csv"), index=False, float_format="%.4g")
    curves = []
    for cad, lst in pooled.items():
        for lo, hi in cfg["shape"]["elo_groups"]:
            parts = [(E, idx) for b, E, idx in lst if lo <= b < hi]
            k = max(E.shape[1] for E, _ in parts)
            E = np.vstack([np.pad(E, ((0, 0), (0, k - E.shape[1])), constant_values=np.nan) for E, _ in parts])
            idx = np.concatenate([i for _, i in parts])
            fk = sh.fit_nonparam(E, idx)
            bs = np.array([sh.fit_nonparam(E[b], idx[b]) for b in
                           (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))])
            for j, dE in enumerate(sh.KNOTS):
                curves.append(dict(cadence=cad, elo_group=f"{lo}-{hi}", dE=dE, f=fk[j],
                                   f_lo=np.quantile(bs[:, j], .025), f_hi=np.quantile(bs[:, j], .975)))
    cv = pd.DataFrame(curves)
    cv.to_csv(os.path.join(f["outdir"], "spectrum_nonparam.csv"), index=False, float_format="%.4g")
    plot.make_shape(st, cv, f["outdir"], "spectrum.png",
                    note="Tous les coups légaux, Stockfish 18 profondeur 12. Barres : IC 95 % bootstrap sur les parties.")
    print(st[["cadence", "elo_mid", "c", "c_lo", "c_hi", "obs_out_top5", "pred_out_boltz_top5fit",
              "pred_out_stretch_top5fit"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def spectrum_records(cfg):
    """{(cadence, band): [records du spectre complet]} pour les positions retenues."""
    dsp = ev.load_done(cfg["extra"]["spectrum"]["output"])
    return {k: [dsp[r["key"]] for r in recs if r["key"] in dsp and dsp[r["key"]]["in_cands"]]
            for k, recs in select_records(cfg).items()}


def step_attention(cfg):
    """Piste 7 : modèle à deux étages, taille effective de l'ensemble de coups envisagés."""
    import attention as at
    import plot
    f, n_boot = cfg["fit"], cfg["attention"]["n_boot"]
    rng = np.random.default_rng(cfg["seed"] + 5)
    rows = []
    for (cad, band), rs in spectrum_records(cfg).items():
        E, idx = ft.build(rs, "cp", f)
        th1, nll1 = at.fit(E, idx, two_stage=False)
        th2, nll2 = at.fit(E, idx, two_stage=True)
        sm = at.summarize(th2, E)
        bo = []
        for _ in range(n_boot):
            b = rng.integers(0, len(idx), len(idx))
            bo.append(at.summarize(at.fit(E[b], idx[b], x0=th2)[0], E[b])["omega_eff"])
        rows.append(dict(cadence=cad, elo_mid=band + 100, n=len(idx), N_legal_mean=(~np.isnan(E)).sum(1).mean(),
                         obs_out_top5=np.mean(idx >= 5), pred_out_top5_two_stage=at.pred_out_top5(th2, E),
                         LR_two_vs_one=2 * (nll1 - nll2), omega_lo=np.quantile(bo, .025),
                         omega_hi=np.quantile(bo, .975), **sm))
        r = rows[-1]
        print(f"  {cad} {band}: Ω_eff={r['omega_eff']:.1f} [{r['omega_lo']:.1f}-{r['omega_hi']:.1f}] / {r['N_legal_mean']:.0f} coups, "
              f"P(tous)={r['p_all']:.2f}, c={r['c']:.2f}, LR={r['LR_two_vs_one']:.0f}", flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(f["outdir"], "attention.csv"), index=False, float_format="%.4g")
    plot.make_attention(st, f["outdir"])
    print(st[["cadence", "elo_mid", "omega_eff", "omega_lo", "omega_hi", "p_all", "c", "s",
              "obs_out_top5", "pred_out_top5_two_stage", "LR_two_vs_one"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))


def step_clock(cfg):
    """Piste 2 : échelle s (c fixé) en fonction de la fraction de temps restant ; test de superposition."""
    import clock as ck
    import plot
    f, n_boot = cfg["fit"], cfg["clock"]["n_boot"]
    edges = np.array(cfg["clock"]["frac_edges"], dtype=float)
    groups = cfg["shape"]["elo_groups"]
    df = pd.read_csv(cfg["sample"]["output"])
    meta = {f"{c}|{b}|{o}": ck.frac_remaining(cl, tc) for c, b, o, cl, tc in
            zip(df.cadence, df.band, df.order, df.clock_before, df.time_control)}
    rng = np.random.default_rng(cfg["seed"] + 6)
    rows, tests = [], []
    G, B = len(groups), len(edges) - 1
    for cad in ("Blitz", "Rapid"):
        recs, gl, bl = [], [], []
        for (c2, band), rs in spectrum_records(cfg).items():
            if c2 != cad:
                continue
            gi = next(i for i, (lo, hi) in enumerate(groups) if lo <= band < hi)
            for r in rs:
                x = meta[r["key"]]
                if np.isfinite(x):
                    recs.append(r); gl.append(gi); bl.append(np.searchsorted(edges, x, side="right") - 1)
        E, idx = ft.build(recs, "cp", f)
        g, b = np.array(gl), np.clip(np.array(bl), 0, B - 1)
        counts = np.zeros((G, B), int)
        np.add.at(counts, (g, b), 1)
        keep = counts[g, b] >= 30
        E, idx, g, b = E[keep], idx[keep], g[keep], b[keep]
        logs, nll_free = ck.fit_free(E, idx, g, b, G, B)
        (a, bb), nll_fac = ck.fit_factorized(E, idx, g, b, G, B)
        boots = []
        for _ in range(n_boot):
            k = rng.integers(0, len(idx), len(idx))
            boots.append(ck.fit_free(E[k], idx[k], g[k], b[k], G, B)[0])
        boots = np.array(boots)
        df_ = np.isfinite(logs).sum() - (G + B - 1)
        from scipy.stats import chi2
        lr = 2 * (nll_fac - nll_free)
        tests.append(dict(cadence=cad, LR_factorized_vs_free=lr, dof=df_, p_value=chi2.sf(lr, df_)))
        for gi in range(G):
            for bi in range(B):
                if np.isfinite(logs[gi, bi]):
                    rows.append(dict(cadence=cad, elo_group=f"{groups[gi][0]}-{groups[gi][1]}",
                                     frac_lo=edges[bi], frac_hi=edges[bi + 1], n=counts[gi, bi],
                                     s=np.exp(logs[gi, bi]),
                                     s_lo=np.exp(np.nanquantile(boots[:, gi, bi], .025)),
                                     s_hi=np.exp(np.nanquantile(boots[:, gi, bi], .975)),
                                     s_norm=np.exp(logs[gi, bi] - a[gi]), s_fact=np.exp(a[gi] + bb[bi])))
    st, te = pd.DataFrame(rows), pd.DataFrame(tests)
    st.to_csv(os.path.join(f["outdir"], "clock.csv"), index=False, float_format="%.4g")
    te.to_csv(os.path.join(f["outdir"], "clock_collapse_test.csv"), index=False, float_format="%.4g")
    plot.make_clock(st, f["outdir"])
    print(st[["cadence", "elo_group", "frac_lo", "n", "s", "s_lo", "s_hi", "s_norm"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    print(te.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


STEPS = dict(download=step_download, sample=step_sample, timing=step_timing,
             evaluate=step_evaluate, fit=step_fit, shape=step_shape,
             noise_eval=step_noise_eval, spectrum_eval=step_spectrum_eval,
             noise_check=step_noise_check, spectrum_check=step_spectrum_check,
             attention=step_attention, clock=step_clock)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    for s in (["download", "sample", "evaluate", "fit", "shape"] if a.step == "all" else [a.step]):
        print(f"== {s} ==")
        STEPS[s](cfg)
