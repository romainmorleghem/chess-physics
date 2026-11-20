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


def step_phase(cfg):
    """Contrôle piste 2 : effet du temps restant à phase de jeu fixée (numéro de coup ou matériel)."""
    import clock as ck
    import phase as ph
    import plot
    from scipy.stats import chi2
    f, pc = cfg["fit"], cfg["phase"]
    t_edges = np.array(cfg["clock"]["frac_edges"], dtype=float)
    df = pd.read_csv(cfg["sample"]["output"])
    meta = {f"{c}|{b}|{o}": (ck.frac_remaining(cl, tc), p, ph.material(fen)) for c, b, o, cl, tc, p, fen in
            zip(df.cadence, df.band, df.order, df.clock_before, df.time_control, df.ply, df.fen)}
    rng = np.random.default_rng(cfg["seed"] + 7)
    rows, tests = [], []
    sp = spectrum_records(cfg)
    for cad in ("Blitz", "Rapid"):
        recs, el, tb, plyv, matv = [], [], [], [], []
        bands = sorted(b for c2, b in sp if c2 == cad)
        for (c2, band), rs in sp.items():
            if c2 != cad:
                continue
            for r in rs:
                x, p, m = meta[r["key"]]
                if np.isfinite(x):
                    recs.append(r); el.append(bands.index(band)); plyv.append(p); matv.append(m)
                    tb.append(np.clip(np.searchsorted(t_edges, x, side="right") - 1, 0, len(t_edges) - 2))
        E, idx = ft.build(recs, "cp", f)
        el, tb = np.array(el), np.array(tb)
        phases = {"ply": np.searchsorted(pc["ply_edges"], np.array(plyv), side="right") - 1,
                  "material": len(pc["mat_edges"]) - 2 - (np.searchsorted(pc["mat_edges"], np.array(matv), side="right") - 1)}
        boots = [rng.integers(0, len(idx), len(idx)) for _ in range(pc["n_boot"])]
        # sans contrôle (référence)
        (a0, b0), nll_t, x_t = ph.fit_additive(E, idx, [el, tb])
        for lab, pv in [("aucun", None)] + list(phases.items()):
            if pv is None:
                co, nll_full, xf = (a0, b0), nll_t, x_t
                nll_np = ph.fit_additive(E, idx, [el])[1]
                fac = [el, tb]
            else:
                pv = np.clip(pv, 0, None)
                fac = [el, tb, pv]
                co, nll_full, xf = ph.fit_additive(E, idx, fac)
                nll_np = ph.fit_additive(E, idx, [el, pv])[1]
                lr_p = 2 * (nll_t - nll_full)
                tests.append(dict(cadence=cad, phase=lab, test="phase|temps", LR=lr_p, dof=int(pv.max()),
                                  p_value=chi2.sf(lr_p, int(pv.max()))))
            lr_t = 2 * (nll_np - nll_full)
            tests.append(dict(cadence=cad, phase=lab, test="temps|phase", LR=lr_t, dof=len(t_edges) - 2,
                              p_value=chi2.sf(lr_t, len(t_edges) - 2)))
            bb = np.array([ph.fit_additive(E[k], idx[k], [fa[k] for fa in fac], x0=xf)[0][1] for k in boots])
            for j in range(len(t_edges) - 1):
                rows.append(dict(cadence=cad, control=lab, frac_lo=t_edges[j], frac_hi=t_edges[j + 1],
                                 n=int((tb == j).sum()), s_ratio=np.exp(co[1][j]),
                                 lo=np.exp(np.quantile(bb[:, j], .025)), hi=np.exp(np.quantile(bb[:, j], .975))))
            if pv is not None:
                edges = pc["ply_edges"] if lab == "ply" else pc["mat_edges"]
                for j, d in enumerate(co[2]):
                    rows.append(dict(cadence=cad, control=f"effet_{lab}", frac_lo=j, frac_hi=np.nan,
                                     n=int((pv == j).sum()), s_ratio=np.exp(d), lo=np.nan, hi=np.nan))
            print(f"  {cad} contrôle={lab}: s(<10 %)/s(>75 %) = {np.exp(co[1][0] - co[1][-1]):.2f}, "
                  f"LR temps|phase = {lr_t:.1f}", flush=True)
    st, te = pd.DataFrame(rows), pd.DataFrame(tests)
    st.to_csv(os.path.join(f["outdir"], "phase.csv"), index=False, float_format="%.4g")
    te.to_csv(os.path.join(f["outdir"], "phase_tests.csv"), index=False, float_format="%.4g")
    plot.make_phase(st, f["outdir"])
    print(st[~st.control.str.startswith("effet")].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(te.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def step_lead(cfg):
    """Piste 3 : « champ extérieur » : s selon que le joueur au trait mène, est à égalité ou est mené."""
    import covar as cv
    import plot
    f, pc = cfg["fit"], cfg["lead"]
    rng = np.random.default_rng(cfg["seed"] + 8)
    effs, tests = [], []
    for cad in ("Blitz", "Rapid"):
        for var in ("cp", "wp"):
            E, idx, d = cv.dataset(cfg, cad, var)
            c = cv.fit_c(E, idx)
            lead = cv.binned(d.best_cp.values, np.array(pc["eval_edges"], float))
            ply = cv.binned(d.ply.values, np.array(cfg["phase"]["ply_edges"], float))
            ref = len(pc["eval_edges"]) // 2 - 1          # tranche centrale (égalité)
            eff, te, co, bb = cv.study(E, idx, [d.elo.values, lead, ply], ["elo", "eval", "ply"], "eval",
                                       c, rng, pc["n_boot"], ref=ref)
            nb = len(pc["eval_edges"]) - 1
            asym = bb[:, -1] - bb[:, 0]                   # mène largement / mené largement
            eff = eff.assign(cadence=cad, energy=var, c=c,
                             eval_lo=pc["eval_edges"][:-1], eval_hi=pc["eval_edges"][1:])
            te = te.assign(cadence=cad, energy=var, c=c)
            te = pd.concat([te, pd.DataFrame([dict(factor="asym(mène/mené)", controls="elo+ply", cadence=cad,
                            energy=var, c=c, LR=np.exp(co[1][nb - 1] - co[1][0]),
                            dof=np.nan, p_value=2 * min(np.mean(asym > 0), np.mean(asym < 0)))])])
            effs.append(eff); tests.append(te)
            print(f"  {cad} {var} (c={c:.2f}): " + "  ".join(f"{r.s_ratio:.2f}[{r.lo:.2f}-{r.hi:.2f}]"
                                                        for r in eff.itertuples()), flush=True)
    eff, te = pd.concat(effs), pd.concat(tests)
    eff.to_csv(os.path.join(f["outdir"], "lead.csv"), index=False, float_format="%.4g")
    te.to_csv(os.path.join(f["outdir"], "lead_tests.csv"), index=False, float_format="%.4g")
    plot.make_levels(eff, f["outdir"], "lead.png", "eval_lo", "eval_hi",
                     "évaluation du meilleur coup pour le joueur au trait (cp, prof. 12)",
                     "Piste 3 : s selon le score (rapporté à l'égalité)",
                     "log s = a(Elo) + e(score) + d(demi-coup), c ajusté par cadence et par énergie.")
    print(te.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def step_postype(cfg):
    """Piste 6 : type de position. s selon l'écart entre les deux meilleurs coups (position « tactique »
    à coup unique contre position calme), à Elo, demi-coup, score et nombre de coups légaux fixés.
    Question : est-ce le joueur (phase) ou la position qui « chauffe » ? Et c dépend-il du type ?"""
    import covar as cv
    import plot
    f, pc = cfg["fit"], cfg["postype"]
    rng = np.random.default_rng(cfg["seed"] + 9)
    effs, tests, plys, shapes = [], [], [], []
    for cad in ("Blitz", "Rapid"):
        for var in ("cp", "wp"):
            E, idx, d = cv.dataset(cfg, cad, var)
            c = cv.fit_c(E, idx)
            el = d.elo.values
            gap = cv.binned(d.gap12.values, np.array(pc["gap_edges"], float))
            nl = cv.binned(d.n_legal.values, np.array(pc["nlegal_edges"], float))
            lead = cv.binned(d.best_cp.values, np.array(cfg["lead"]["eval_edges"], float))
            ply = cv.binned(d.ply.values, np.array(cfg["phase"]["ply_edges"], float))
            chk = d.in_check.values.astype(int)
            fac, names = [el, gap, ply, nl, lead, chk], ["elo", "gap", "ply", "nlegal", "eval", "check"]
            eff, te, co, _ = cv.study(E, idx, fac, names, "gap", c, rng, pc["n_boot"])
            eff = eff.assign(cadence=cad, energy=var, c=c, gap_lo=pc["gap_edges"][:-1], gap_hi=pc["gap_edges"][1:])
            effs.append(eff); tests.append(te.assign(cadence=cad, energy=var, c=c))
            # effet du demi-coup (dernier niveau / premier) avec et sans contrôle du type de position
            co0 = cv.fit_additive(E, idx, [el, ply], c)[0]
            co1 = cv.fit_additive(E, idx, [el, ply, lead], c)[0]
            plys.append(dict(cadence=cad, energy=var, ply_effect_elo_only=np.exp(co0[1][-1]),
                             ply_effect_plus_eval=np.exp(co1[1][-1]), ply_effect_full=np.exp(co[2][-1]),
                         check_effect=np.exp(co[5][1]), nlegal_effects=" ".join(f"{v:.2f}" for v in np.exp(co[3]))))
            # forme : c profilé par tranche d'écart (a(Elo) libre)
            for g in range(len(pc["gap_edges"]) - 1):
                m = gap == g
                grid = np.arange(0.2, 1.21, 0.02)
                nl_ = np.array([cv.fit_additive(E[m], idx[m], [el[m], ply[m]], cc)[1] for cc in grid])
                ok = grid[nl_ <= nl_.min() + 1.92]                 # IC 95 % par vraisemblance profilée
                shapes.append(dict(cadence=cad, energy=var, gap_lo=pc["gap_edges"][g], n=int(m.sum()),
                                   c=grid[nl_.argmin()], c_lo=ok.min(), c_hi=ok.max()))
            print(f"  {cad} {var}: s(gap)/s(gap<10) = " + "  ".join(f"{r.s_ratio:.2f}[{r.lo:.2f}-{r.hi:.2f}]"
                  for r in eff.itertuples()) + f" ; demi-coup x{plys[-1]['ply_effect_full']:.2f}, échec x{plys[-1]['check_effect']:.2f}", flush=True)
    eff, te, pl, sh_ = pd.concat(effs), pd.concat(tests), pd.DataFrame(plys), pd.DataFrame(shapes)
    eff.to_csv(os.path.join(f["outdir"], "postype.csv"), index=False, float_format="%.4g")
    te.to_csv(os.path.join(f["outdir"], "postype_tests.csv"), index=False, float_format="%.4g")
    pl.to_csv(os.path.join(f["outdir"], "postype_ply.csv"), index=False, float_format="%.4g")
    sh_.to_csv(os.path.join(f["outdir"], "postype_shape.csv"), index=False, float_format="%.4g")
    plot.make_levels(eff, f["outdir"], "postype.png", "gap_lo", "gap_hi",
                     "écart entre les deux meilleurs coups (cp, prof. 12)",
                     "Piste 6 : s selon le type de position (réf. écart <10 cp)",
                     "log s = a(Elo) + g(écart) + d(demi-coup) + n(coups légaux) + e(score) + échec, c ajusté.")
    print(te.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(pl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(sh_.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def step_fdt(cfg):
    """Piste 4 : fluctuation-dissipation (β des moments d'ordre 1 et 2, rapport de fluctuation R)."""
    import attention as at
    import fdt
    import plot
    import covar as cv
    f, n_boot = cfg["fit"], cfg["fdt"]["n_boot"]
    rng = np.random.default_rng(cfg["seed"] + 10)
    rows = []
    for (cad, band), rs in spectrum_records(cfg).items():
        for var in ("cp", "wp"):
            E, idx = ft.build(rs, var, f)
            c = cv.fit_c(E, idx)
            row = dict(cadence=cad, elo_mid=band + 100, energy=var, n=len(idx), c=c)
            boots = [rng.integers(0, len(idx), len(idx)) for _ in range(n_boot)]
            for lab, H in (("boltz", E), ("stretch", E ** c)):
                bU, bV = fdt.beta_match(H, idx, 1), fdt.beta_match(H, idx, 2)
                R = fdt.fluct_ratio(H, idx, bU)
                bb = np.array([(fdt.beta_match(H[k], idx[k], 1) / fdt.beta_match(H[k], idx[k], 2),
                                fdt.fluct_ratio(H[k], idx[k], fdt.beta_match(H[k], idx[k], 1))) for k in boots])
                row.update({f"{lab}_TV_over_TU": bU / bV, f"{lab}_TV_over_TU_lo": np.quantile(bb[:, 0], .025),
                            f"{lab}_TV_over_TU_hi": np.quantile(bb[:, 0], .975), f"{lab}_R": R,
                            f"{lab}_R_lo": np.quantile(bb[:, 1], .025), f"{lab}_R_hi": np.quantile(bb[:, 1], .975)})
            if var == "cp":
                import shape as sh
                (s_, c_), _ = sh.fit_stretched(E, idx)
                U = np.where(np.isnan(E), -np.inf, -(np.nan_to_num(E) / s_) ** c_)
                Ps = np.exp(U - U.max(1, keepdims=True)); Ps /= Ps.sum(1, keepdims=True)
                R1, _, hm1 = fdt.ratio_from_P(Ps, E, idx)
                bb = np.array([fdt.ratio_from_P(Ps[k], E[k], idx[k])[0] for k in boots])
                row.update(stretchE_R=R1, stretchE_R_lo=np.quantile(bb, .025), stretchE_R_hi=np.quantile(bb, .975),
                           stretchE_mean_model=hm1)
                th, _ = at.fit(E, idx, two_stage=True)
                P = fdt.two_stage_P(th, E)
                R2, hobs, hmod = fdt.ratio_from_P(P, E, idx)
                bb = np.array([fdt.ratio_from_P(P[k], E[k], idx[k])[0] for k in boots])
                row.update(two_R=R2, two_R_lo=np.quantile(bb, .025), two_R_hi=np.quantile(bb, .975),
                           two_mean_obs=hobs, two_mean_model=hmod)
            rows.append(row)
            print(f"  {cad} {band} {var}: T_V/T_U Boltz={row['boltz_TV_over_TU']:.2f} R={row['boltz_R']:.2f} ; "
                  f"étirée T_V/T_U={row['stretch_TV_over_TU']:.2f} R={row['stretch_R']:.2f}"
                  + (f" ; en ΔE : étirée R={row['stretchE_R']:.2f}, 2 étages R={row['two_R']:.2f}" if var == "cp" else ""), flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(f["outdir"], "fdt.csv"), index=False, float_format="%.4g")
    plot.make_fdt(st, f["outdir"])


def step_players_sample(cfg):
    """Piste 5 : choix des joueurs et des positions (2 passages sur le préfixe du fichier)."""
    import players as pl
    chosen = pl.select_players(cfg)
    rows = pl.sample_positions(cfg, chosen)
    out = cfg["players"]["positions"]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {len(rows)} positions, {len(chosen)} joueurs -> {out}")


def step_players_eval(cfg):
    p = cfg["players"]
    df = pd.read_csv(p["positions"])
    ecfg = dict(cfg["engine"], depth=p["depth"], multipv="all")
    tasks = [(f"P|{r.player}|{r.k}", r.fen, r.move) for r in df.itertuples()]
    print(f"  {len(tasks)} positions, profondeur {p['depth']}, tous les coups", flush=True)
    done, dt, m = ev.evaluate(tasks, ecfg, p["evals"], ecfg["workers"])
    if m:
        print(f"  {m} évaluées en {dt / 60:.1f} min ({dt / m:.2f} s/position)")


def step_players(cfg):
    """Piste 5 : dispersion des températures individuelles (modèle à effets aléatoires) et contrôles."""
    import chess
    import covar as cv
    import players as pl
    import plot
    from scipy.optimize import minimize
    from scipy.stats import chi2, spearmanr
    f, p = cfg["fit"], cfg["players"]
    df = pd.read_csv(p["positions"])
    done = ev.load_done(p["evals"])
    recs, meta = [], []
    for r in df.itertuples():
        e = done.get(f"P|{r.player}|{r.k}")
        if e is None or not e["in_cands"] or not ev.keep(e, cfg["filter"]["max_abs_cp"]):
            continue
        recs.append(e); meta.append(dict(player=r.player, band=r.band, elo=r.elo, ply=r.ply,
                                         check=chess.Board(r.fen).is_check()))
    md = pd.DataFrame(meta)
    E, idx = ft.build(recs, "cp", f)
    pid = pd.factorize(md.player)[0]
    bands = sorted(md.band.unique())
    band = md.band.map({b: i for i, b in enumerate(bands)}).values
    ply = cv.binned(md.ply.values, np.array(cfg["phase"]["ply_edges"], float))
    chk = md.check.values.astype(int)
    c = cv.fit_c(E, idx)
    m = pl.RE(E, idx, [band, ply, chk], pid, c)
    x, nll1 = m.fit()
    x0, nll0 = m.fit(x0=x, sigma0=True)
    sig = np.exp(x[-1])
    lr = 2 * (nll0 - nll1)
    pval = 0.5 * chi2.sf(lr, 1)
    print(f"  {len(idx)} positions, {m.P} joueurs, c = {c:.2f} ; σ(log s) = {sig:.3f} ; LR = {lr:.1f}, p = {pval:.2g}", flush=True)
    # intervalle de σ par vraisemblance profilée
    grid = np.linspace(0.0, max(0.6, 2.5 * sig), 31)
    prof = []
    for g in grid:
        r = minimize(lambda z: m.nll(np.append(z, 0.0), sigma=g), x[:-1], method="L-BFGS-B")
        prof.append(r.fun)
    prof = np.array(prof)
    ok = grid[prof <= prof.min() + 1.92]
    # contrôle 1 : permutations des joueurs au sein de chaque tranche (σ doit être ~0)
    rng = np.random.default_rng(cfg["seed"] + 12)
    sig_null = []
    for _ in range(p["n_perm"]):
        perm = pid.copy()
        for b in range(len(bands)):
            w = np.where(band == b)[0]
            perm[w] = pid[rng.permutation(w)]
        mp = pl.RE(E, idx, [band, ply, chk], perm, c)
        sig_null.append(np.exp(mp.fit(x0=x)[0][-1]))
    sig_null = np.array(sig_null)
    # contrôle 2 : fiabilité moitié/moitié (positions paires / impaires de chaque joueur)
    k = md.groupby("player").cumcount().values
    u_a, u_b = m.player_effects(x, k % 2 == 0), m.player_effects(x, k % 2 == 1)
    rho = spearmanr(u_a, u_b).correlation
    # u contre l'écart d'Elo à la tranche
    u = m.player_effects(x)
    pe = md.assign(pid=pid).groupby("pid").agg(band=("band", "first"), elo=("elo", "mean"), n=("ply", "size"))
    rho_elo = spearmanr(u, pe.elo - pe.band).correlation
    a = np.exp(x[:len(bands)])
    res = pd.DataFrame([dict(n_positions=len(idx), n_players=m.P, c=c, sigma=sig, sigma_lo=ok.min(), sigma_hi=ok.max(),
                             LR=lr, p_value=pval, sigma_perm_mean=sig_null.mean(), sigma_perm_max=sig_null.max(),
                             split_half_spearman=rho, spearman_brown=2 * rho / (1 + rho),
                             spearman_u_vs_elo_in_band=rho_elo, s_band_min=a.min(), s_band_max=a.max())])
    res.to_csv(os.path.join(f["outdir"], "players.csv"), index=False, float_format="%.4g")
    pe.assign(u=u, u_even=u_a, u_odd=u_b, s_rel=np.exp(u)).to_csv(os.path.join(f["outdir"], "players_effects.csv"),
                                                                     index=False, float_format="%.4g")
    pd.DataFrame(dict(sigma=grid, nll=prof)).to_csv(os.path.join(f["outdir"], "players_profile.csv"), index=False,
                                                    float_format="%.6g")
    plot.make_players(pe.assign(u=u, u_even=u_a, u_odd=u_b), grid, prof, sig_null, bands, a, f["outdir"])
    print(res.T.to_string(float_format=lambda v: f"{v:.3g}"))
    print("  s par tranche :", " ".join(f"{b}:{v:.0f}" for b, v in zip(bands, a)))


def _entropy_positions(cfg):
    """Positions Blitz du pilote : les n_per_band premières (ordre aléatoire fixe) de chaque tranche."""
    e = cfg["entropy"]
    out = []
    for (cad, band), rs in spectrum_records(cfg).items():
        if cad == e["cadence"]:
            out += [(band, r) for r in rs[:e["n_per_band"]]]
    return out


def step_entropy_eval(cfg):
    """Pilote piste 8 : évalue les positions filles (K meilleurs coups) puis petites-filles (meilleure réponse)."""
    import chess
    e = cfg["entropy"]
    ecfg = dict(cfg["engine"], depth=e["depth"], multipv="all", workers=e["workers"])
    df = pd.read_csv(cfg["sample"]["output"])
    fen = {f"{c}|{b}|{o}": f for c, b, o, f in zip(df.cadence, df.band, df.order, df.fen)}
    pos = _entropy_positions(cfg)
    t1 = []
    for band, r in pos:
        b = chess.Board(fen[r["key"]])
        for k, pv in enumerate(r["pvs"][:e["K"]]):
            c = b.copy(); c.push_uci(pv["move"])
            if not c.is_game_over():
                t1.append((f"{r['key']}|{k}", c.fen(), "0000"))
    print(f"  {len(pos)} positions, {len(t1)} positions filles", flush=True)
    d1, dt, m = ev.evaluate(t1, ecfg, e["out_children"], ecfg["workers"])
    if m:
        print(f"  filles : {m} en {dt / 60:.1f} min", flush=True)
    t2 = []
    for key, f2, _ in t1:
        r = d1[key]
        if r["pvs"]:
            c = chess.Board(f2); c.push_uci(r["pvs"][0]["move"])
            if not c.is_game_over():
                t2.append((key, c.fen(), "0000"))
    d2, dt, m = ev.evaluate(t2, ecfg, e["out_grand"], ecfg["workers"])
    if m:
        print(f"  petites-filles : {m} en {dt / 60:.1f} min", flush=True)


def step_entropy(cfg):
    """Pilote piste 8 : le choix dépend-il de l'entropie des positions atteintes ? (modèles A et B)"""
    import chess
    import entropy as en
    from scipy.stats import chi2
    e, f = cfg["entropy"], cfg["fit"]
    sf = pd.read_csv(os.path.join(f["outdir"], "spectrum_fit.csv"))
    sc = {int(r.elo_mid) - 100: (r.s, r.c) for r in sf[sf.cadence == e["cadence"]].itertuples()}
    d1, d2 = ev.load_done(e["out_children"]), ev.load_done(e["out_grand"])
    df = pd.read_csv(cfg["sample"]["output"])
    fen = {f"{c}|{b}|{o}": f for c, b, o, f in zip(df.cadence, df.band, df.order, df.fen)}
    pos = _entropy_positions(cfg)
    K = e["K"]
    bands = sorted({b for b, _ in pos})
    n = len(pos)
    dE = np.full((n, K), np.nan); Sm, Sa, Lm, La, Na, chk, cap = (np.full((n, K), np.nan) for _ in range(7))
    idx = np.full(n, -1); band = np.zeros(n, int)
    for j, (b, r) in enumerate(pos):
        s_, c_ = sc[b]
        board = chess.Board(fen[r["key"]])
        v = en.values(r["pvs"], f)
        for k, pv in enumerate(r["pvs"][:K]):
            key = f"{r['key']}|{k}"
            mv = chess.Move.from_uci(pv["move"])
            dE[j, k] = v.max() - v[k]
            cap[j, k] = board.is_capture(mv); chk[j, k] = board.gives_check(mv)
            if key in d1 and d1[key]["pvs"]:
                a = en.summarize(d1[key]["pvs"], f, e["good_cp"], s_, c_)
                Sa[j, k], La[j, k], Na[j, k] = a["S"], a["L"], np.log(a["N"])
            else:                       # mat ou pat après le coup : l'adversaire n'a aucun coup
                Sa[j, k], La[j, k], Na[j, k] = 0.0, 0.0, 0.0
            if key in d2 and d2[key]["pvs"]:
                g = en.summarize(d2[key]["pvs"], f, e["good_cp"], s_, c_)
                Sm[j, k], Lm[j, k] = g["S"], g["L"]
            else:
                Sm[j, k], Lm[j, k] = 0.0, 0.0
        if r["played"] in [p["move"] for p in r["pvs"][:K]]:
            idx[j] = [p["move"] for p in r["pvs"][:K]].index(r["played"])
        band[j] = bands.index(b)
    keep = idx >= 0
    print(f"  {n} positions, coup joué parmi les {K} premiers : {keep.sum()}", flush=True)
    dE, Sm, Sa, Lm, La, Na, chk, cap, idx, band = (x[keep] for x in (dE, Sm, Sa, Lm, La, Na, chk, cap, idx, band))
    nb, c = len(bands), 0.5
    x0 = np.full(nb, np.log(20.0))
    rows = []
    specs = {"énergie seule": [],
             "A1 : S_moi, S_adv": [Sm, Sa],
             "A2 : + mobilité adv., capture, échec": [Sm, Sa, Na, cap, chk]}
    base_nll = None
    for lab, cols in specs.items():
        X = np.stack(cols, axis=2) if cols else np.zeros(dE.shape + (0,))
        th, nll = en.fit(en.nll_entropic, np.concatenate([x0, np.zeros(X.shape[2])]), dE, X, idx, band, nb, c)
        se = en.hessian_se(en.nll_entropic, th, dE, X, idx, band, nb, c)
        base_nll = nll if base_nll is None else base_nll
        names = ["lam_S_moi", "lam_S_adv", "lam_logN_adv", "lam_capture", "lam_check"][:X.shape[2]]
        row = dict(model=lab, n=len(idx), nll=nll, LR_vs_energy=2 * (base_nll - nll), dof=X.shape[2],
                   p_value=chi2.sf(2 * (base_nll - nll), max(X.shape[2], 1)) if X.shape[2] else np.nan)
        for nm, v_, s_ in zip(names, th[nb:], se[nb:]):
            row[nm], row[nm + "_se"] = v_, s_
        rows.append(row)
        print(f"  {lab}: nll={nll:.1f} " + " ".join(f"{nm}={v_:+.3f}±{s_:.3f}" for nm, v_, s_ in zip(names, th[nb:], se[nb:])), flush=True)
    # modèle B : évaluation pratique (énergie libre), profil sur grille de (κ_moi, κ_adv)
    grid = np.round(np.arange(e["kappa_min"], e["kappa_max"] + 1e-9, e["kappa_step"]), 3)
    i1 = int(np.argmin(np.abs(grid - 1.0)))
    i0 = int(np.argmin(np.abs(grid)))
    for lab, m in [("tous", np.ones(len(idx), bool))] + [
            (f"groupe {lo}-{hi}", np.isin(band, [i for i, b in enumerate(bands) if lo <= b < hi]))
            for lo, hi in cfg["shape"]["elo_groups"]]:
        bm = band[m] - band[m].min()
        nbm = bm.max() + 1
        P = en.profile_kappa(dE[m], Lm[m], La[m], idx[m], bm, nbm, c, grid)
        smy = en.profile_summary(P, grid)
        X = np.stack([Sm, Sa], axis=2)[m]
        tA, nA = en.fit(en.nll_entropic, np.concatenate([np.full(nbm, np.log(20.0)), [0, 0]]), dE[m], X, idx[m], bm, nbm, c)
        sA = en.hessian_se(en.nll_entropic, tA, dE[m], X, idx[m], bm, nbm, c)
        rows.append(dict(model=f"B profil, {lab}", n=int(m.sum()), **smy, nll_energy=P[i0, i0],
                         LR_vs_energy=2 * (P[i0, i0] - P.min()), LR_kappa1_vs_best=2 * (P[i1, i1] - P.min()),
                         p_kappa1=chi2.sf(2 * (P[i1, i1] - P.min()), 2),
                         lam_S_moi=tA[nbm], lam_S_moi_se=sA[nbm], lam_S_adv=tA[nbm + 1], lam_S_adv_se=sA[nbm + 1]))
        r = rows[-1]
        print(f"  {lab}: λ_moi={r['lam_S_moi']:+.2f}±{r['lam_S_moi_se']:.2f} λ_adv={r['lam_S_adv']:+.2f}±{r['lam_S_adv_se']:.2f} | "
              f"κ_moi={r['kappa_moi']:.2f} [{r['kappa_moi_lo']:.2f};{r['kappa_moi_hi']:.2f}] "
              f"κ_adv={r['kappa_adv']:.2f} [{r['kappa_adv_lo']:.2f};{r['kappa_adv_hi']:.2f}] "
              f"LR(κ=0)={r['LR_vs_energy']:.1f} LR(κ=1)={r['LR_kappa1_vs_best']:.1f}", flush=True)
        if lab == "tous":
            np.save(os.path.join(f["outdir"], "entropy_kappa_profile.npy"), P)
    # corrélations descriptives entre ΔE et les entropies (confusion)
    ok = ~np.isnan(dE)
    cor = pd.DataFrame(dict(dE=dE[ok], S_moi=Sm[ok], S_adv=Sa[ok], L_moi=Lm[ok], L_adv=La[ok],
                            check=chk[ok], capture=cap[ok])).corr("spearman").round(2)
    cor.to_csv(os.path.join(f["outdir"], "entropy_corr.csv"))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(f["outdir"], "entropy.csv"), index=False, float_format="%.4g")
    import plot
    plot.make_entropy(res, np.load(os.path.join(f["outdir"], "entropy_kappa_profile.npy")), grid, f["outdir"])
    print(cor.to_string())


def step_heat(cfg):
    """Chaleur spécifique C = β² Var(H), H = ΔE^c, β = s^-c (loi étirée ajustée par case).
    C mesure la sensibilité du choix à la température : grande quand plusieurs coups sont en compétition."""
    import covar as cv
    import plot
    from scipy.special import logsumexp
    f, h = cfg["fit"], cfg["heat"]
    sf = pd.read_csv(os.path.join(f["outdir"], "spectrum_fit.csv"))
    sc = {(r.cadence, int(r.elo_mid) - 100): (r.s, r.c) for r in sf.itertuples()}
    gap_edges = np.array(cfg["postype"]["gap_edges"], float)

    def per_pos(E, s, c):
        H = np.nan_to_num(E) ** c
        U = np.where(np.isnan(E), -np.inf, -H / s ** c)
        P = np.exp(U - logsumexp(U, axis=1, keepdims=True))
        m1 = (P * H).sum(1)
        return (P * H ** 2).sum(1) - m1 ** 2, m1, P

    rows, curves = [], []
    for cad in ("Blitz", "Rapid"):
        E, idx, d = cv.dataset(cfg, cad)
        gap = cv.binned(d.gap12.values, gap_edges)
        for k, b in enumerate(sorted(d.band.unique())):
            m = (d.band == b).values
            s, c = sc[(cad, b)]
            beta = s ** -c
            var, m1, _ = per_pos(E[m], s, c)
            hobs = np.nan_to_num(E[m][np.arange(m.sum()), idx[m]]) ** c
            for g in range(len(gap_edges) - 1):
                mg = gap[m] == g
                rows.append(dict(cadence=cad, elo_mid=b + 100, gap_lo=gap_edges[g], n=int(mg.sum()), s=s, c=c,
                                 C_model=beta ** 2 * var[mg].mean(),
                                 C_obs=beta ** 2 * ((hobs[mg] - m1[mg]) ** 2).mean()))
            rows.append(dict(cadence=cad, elo_mid=b + 100, gap_lo=-1, n=int(m.sum()), s=s, c=c,
                             C_model=beta ** 2 * var.mean(), C_obs=beta ** 2 * ((hobs - m1) ** 2).mean()))
        # C(s) sur l'ensemble de toutes les positions de la cadence (c = médiane), par tranche d'écart
        cm = float(np.median([sc[(cad, b)][1] for b in d.band.unique()]))
        for s in np.geomspace(1, 2000, h["n_grid"]):
            var, _, _ = per_pos(E, s, cm)
            C = s ** (-2 * cm) * var
            curves.append(dict(cadence=cad, s=s, c=cm, gap_lo=-1, C=C.mean()))
            for g in range(len(gap_edges) - 1):
                curves.append(dict(cadence=cad, s=s, c=cm, gap_lo=gap_edges[g], C=C[gap == g].mean()))
    st, cv_ = pd.DataFrame(rows), pd.DataFrame(curves)
    st.to_csv(os.path.join(f["outdir"], "heat.csv"), index=False, float_format="%.4g")
    cv_.to_csv(os.path.join(f["outdir"], "heat_curve.csv"), index=False, float_format="%.4g")
    plot.make_heat(st, cv_, f["outdir"])
    tot = st[st.gap_lo == -1]
    print(tot[["cadence", "elo_mid", "s", "C_model", "C_obs"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for cad in ("Blitz", "Rapid"):
        q = cv_[(cv_.cadence == cad)]
        for g, gg in q.groupby("gap_lo"):
            print(f"  {cad} écart>={g:g}: pic de C à s = {gg.s.values[gg.C.argmax()]:.1f} cp (C max = {gg.C.max():.3f})")
    piv = st[st.gap_lo >= 0].groupby(["cadence", "gap_lo"])[["C_model", "C_obs"]].mean()
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))


STEPS = dict(download=step_download, sample=step_sample, timing=step_timing,
             evaluate=step_evaluate, fit=step_fit, shape=step_shape,
             noise_eval=step_noise_eval, spectrum_eval=step_spectrum_eval,
             noise_check=step_noise_check, spectrum_check=step_spectrum_check,
             attention=step_attention, clock=step_clock,
             phase=step_phase, lead=step_lead,
             postype=step_postype, fdt=step_fdt, players_sample=step_players_sample,
             players_eval=step_players_eval, players=step_players,
             entropy_eval=step_entropy_eval, entropy=step_entropy, heat=step_heat)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    for s in (["download", "sample", "evaluate", "fit", "shape"] if a.step == "all" else [a.step]):
        print(f"== {s} ==")
        STEPS[s](cfg)
