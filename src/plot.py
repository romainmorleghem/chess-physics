"""Graphiques : T(Elo) avec barres d'erreur (2 variantes) et taux de coups hors candidats."""
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"Blitz": "#2a78d6", "Rapid": "#eb6834"}  # palette catégorielle validée, ordre fixe
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, ylabel):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Elo (milieu de tranche, les deux joueurs)", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def make(res, outdir):
    panels = [("T_cp", "T (centipions)", "Température — centipions écrêtés ±1000"),
              ("T_wp", "T (points de % de gain)", "Température — probabilité de gain"),
              ("out_rate", "fraction des coups", "Coups joués hors des 5 candidats")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (col, ylab, title) in zip(axes, panels):
        _style(ax, ylab)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        for cad, g in res.groupby("cadence"):
            g = g.sort_values("elo_mid")
            x = g.elo_mid + (-12 if cad == "Blitz" else 12)  # léger décalage pour lisibilité
            if col == "out_rate":
                ax.plot(x, g[col], "-o", color=COLORS[cad], lw=2, ms=6, label=cad,
                        mec="#fcfcfb", mew=1.5)
            else:
                yerr = [g[col] - g[col + "_lo"], g[col + "_hi"] - g[col]]
                ax.errorbar(x, g[col], yerr=yerr, fmt="-o", color=COLORS[cad], lw=2, ms=6,
                            capsize=3, elinewidth=1.2, label=cad, mec="#fcfcfb", mew=1.5)
            last = g.iloc[-1]
            ax.annotate(cad, (last.elo_mid, last[col]), xytext=(8, 0),
                        textcoords="offset points", color=INK, fontsize=9, va="center")
        ax.set_xlim(res.elo_mid.min() - 100, res.elo_mid.max() + 200)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, "Barres : IC 95 % par bootstrap (200 rééchantillonnages de parties, "
             "1 position par partie). Lichess oct. 2026 (préfixe 2 Go), Stockfish 18, prof. 16, MultiPV 5.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "T_vs_elo.png"))
    plt.close(fig)


SEQ = ["#86b6ef", "#3987e5", "#184f95"]  # rampe séquentielle (bleu clair -> foncé) : Elo croissant


def make_shape(st, cv, outdir, fname="shape.png", note=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    _style(ax, "exposant c")
    ax.set_title("Exposant c de exp(−(ΔE/s)^c)  (c = 1 : Boltzmann)", color=INK, fontsize=11, loc="left")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    for cad, g in st.groupby("cadence"):
        g = g.sort_values("elo_mid")
        x = g.elo_mid + (-12 if cad == "Blitz" else 12)
        ax.errorbar(x, g.c, yerr=[g.c - g.c_lo, g.c_hi - g.c], fmt="-o", color=COLORS[cad], lw=2,
                    ms=6, capsize=3, elinewidth=1.2, label=cad, mec="#fcfcfb", mew=1.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    for ax, cad in zip(axes[1:], ["Blitz", "Rapid"]):
        _style(ax, "f(ΔE) = −log(poids relatif)")
        ax.set_xlabel("ΔE = perte par rapport au meilleur coup (cp)", color=MUTED, fontsize=9)
        ax.set_title(f"{cad} : f(ΔE) non paramétrique (log-log)", color=INK, fontsize=11, loc="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        g = cv[(cv.cadence == cad) & (cv.dE > 0)]
        grps = sorted(g.elo_group.unique(), key=lambda x: int(x.split("-")[0]))
        for col, grp in zip(SEQ, grps):
            h = g[g.elo_group == grp]
            ax.errorbar(h.dE, h.f, yerr=[(h.f - h.f_lo).clip(lower=0), (h.f_hi - h.f).clip(lower=0)],
                        fmt="-o", color=col, lw=2, ms=5, capsize=2, elinewidth=1, label=f"Elo {grp}",
                        mec="#fcfcfb", mew=1)
        xs = [10, 1000]
        ax.plot(xs, [0.1, 10], color=MUTED, lw=1, ls="--", label="pente 1 (Boltzmann)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.text(0.01, 0.01, note or ("Vraisemblance conditionnelle aux 5 candidats (valide pour un logit, propriété IIA). "
             "Barres : IC 95 % bootstrap sur les parties."), color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


def make_attention(st, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    specs = [("omega_eff", "Ω_eff (coups)", "Taille effective de l'ensemble envisagé", True),
             ("p_all", "probabilité", "P(tous les coups envisagés)", False),
             ("obs_out_top5", "fraction des coups", "Coups hors top 5 : observé (●) et prédit 2 étages (○)", False)]
    for ax, (col, ylab, title, ci) in zip(axes, specs):
        _style(ax, ylab)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        for cad, g in st.groupby("cadence"):
            g = g.sort_values("elo_mid")
            x = g.elo_mid + (-12 if cad == "Blitz" else 12)
            if ci:
                ax.errorbar(x, g[col], yerr=[g[col] - g.omega_lo, g.omega_hi - g[col]], fmt="-o",
                            color=COLORS[cad], lw=2, ms=6, capsize=3, elinewidth=1.2, label=cad,
                            mec="#fcfcfb", mew=1.5)
            else:
                ax.plot(x, g[col], "-o", color=COLORS[cad], lw=2, ms=6, label=cad, mec="#fcfcfb", mew=1.5)
            if col == "obs_out_top5":
                ax.plot(x, g.pred_out_top5_two_stage, "--o", color=COLORS[cad], lw=1.2, ms=6, mfc="#fcfcfb")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, "Modèle : K coups les mieux classés envisagés (K aléatoire), puis choix ∝ exp(−(ΔE/s)^c). "
             "Spectre complet, profondeur 12. IC 95 % bootstrap (50).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "attention.png"))
    plt.close(fig)


def make_clock(st, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    grps = sorted(st.elo_group.unique(), key=lambda x: int(x.split("-")[0]))
    for ax, cad in zip(axes[:2], ["Blitz", "Rapid"]):
        _style(ax, "échelle s (cp, log), c = 0,5 fixé")
        ax.set_yscale("log")
        ax.set_xlabel("fraction du temps initial restant (milieu de tranche)", color=MUTED, fontsize=9)
        ax.set_title(f"{cad} : température en fonction du temps restant", color=INK, fontsize=11, loc="left")
        for col, grp in zip(SEQ, grps):
            h = st[(st.cadence == cad) & (st.elo_group == grp)]
            xm = (h.frac_lo + h.frac_hi.clip(upper=1.0)) / 2
            ax.errorbar(xm, h.s, yerr=[h.s - h.s_lo, h.s_hi - h.s], fmt="-o", color=col, lw=2, ms=6,
                        capsize=3, elinewidth=1.2, label=f"Elo {grp}", mec="#fcfcfb", mew=1.5)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[2]
    _style(ax, "s / s(Elo)  (modèle factorisé)")
    ax.set_xlabel("fraction du temps initial restant (milieu de tranche)", color=MUTED, fontsize=9)
    ax.set_title("Superposition : s normalisé par niveau", color=INK, fontsize=11, loc="left")
    for cad, ls in (("Blitz", "-"), ("Rapid", "--")):
        for col, grp in zip(SEQ, grps):
            h = st[(st.cadence == cad) & (st.elo_group == grp)]
            xm = (h.frac_lo + h.frac_hi.clip(upper=1.0)) / 2
            ax.plot(xm, h.s_norm, ls, marker="o", color=col, lw=1.6, ms=5, label=f"{cad} {grp}")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, ncol=2)
    fig.text(0.01, 0.01, "Spectre complet (profondeur 12), une position par partie. IC 95 % bootstrap (100). "
             "Bleu clair → foncé : Elo croissant ; trait plein Blitz, tirets Rapid.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "clock.png"))
    plt.close(fig)


def make_phase(st, outdir):
    """Effet du temps restant, s rapporté à la tranche <10 % : sans contrôle, à numéro de coup fixé,
    à matériel fixé."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    labs = {"aucun": "sans contrôle", "ply": "à numéro de coup fixé", "material": "à matériel fixé"}
    for ax, cad in zip(axes, ["Blitz", "Rapid"]):
        _style(ax, "s / s(tranche <10 %)  (log)")
        ax.set_yscale("log")
        ax.set_xlabel("fraction du temps initial restant (milieu de tranche)", color=MUTED, fontsize=9)
        ax.set_title(f"{cad} : effet du temps, phase contrôlée ou non", color=INK, fontsize=11, loc="left")
        for k, (ctl, lab) in enumerate(labs.items()):
            h = st[(st.cadence == cad) & (st.control == ctl)]
            xm = (h.frac_lo + h.frac_hi.clip(upper=1.0)) / 2 + (k - 1) * 0.012
            ax.errorbar(xm, h.s_ratio, yerr=[(h.s_ratio - h.lo).clip(lower=0), (h.hi - h.s_ratio).clip(lower=0)], fmt="-o", color=SEQ[k * 2 % len(SEQ)],
                        lw=2, ms=5, capsize=3, elinewidth=1.1, label=lab, mec="#fcfcfb", mew=1.2)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, "log s = a(Elo) + b(temps) [+ d(phase)], c = 0,5 fixé, spectre complet prof. 12. "
             "IC 95 % bootstrap (100).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "phase.png"))
    plt.close(fig)


def make_levels(eff, outdir, fname, xlo, xhi, xlabel, title, note):
    """Effet d'une covariable par niveau (s rapporté au niveau de référence), par cadence et énergie."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, var in zip(axes, ["cp", "wp"]):
        _style(ax, "s / s(référence)  (log)")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_title(f"{title} ({'centipions' if var == 'cp' else 'proba. de gain'})", color=INK, fontsize=10, loc="left")
        for k, (cad, col) in enumerate((("Blitz", SEQ[1]), ("Rapid", SEQ[2]))):
            h = eff[(eff.cadence == cad) & (eff.energy == var)]
            x = np.arange(len(h)) + (k - 0.5) * 0.12
            ax.errorbar(x, h.s_ratio, yerr=[(h.s_ratio - h.lo).clip(lower=0), (h.hi - h.s_ratio).clip(lower=0)],
                        fmt="-o", color=col, lw=2, ms=5, capsize=3, elinewidth=1.1, label=cad, mec="#fcfcfb", mew=1.2)
            ax.set_xticks(np.arange(len(h)))
            ax.set_xticklabels([f"{a:g}…{b:g}".replace("-10000", "−∞").replace("10000", "∞") for a, b in zip(h[xlo], h[xhi])],
                               fontsize=7, color=MUTED)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, note + " IC 95 % bootstrap.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


def make_fdt(st, outdir):
    """Rapport de fluctuation R (observé / prédit) par Elo : Boltzmann, loi étirée, deux étages."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, cad in zip(axes, ["Blitz", "Rapid"]):
        _style(ax, "R = fluctuations observées / prédites  (log)")
        ax.set_yscale("log")
        ax.set_xlabel("Elo (milieu de tranche)", color=MUTED, fontsize=9)
        ax.set_title(f"{cad} : test de fluctuation-dissipation (ΔE en cp)", color=INK, fontsize=10, loc="left")
        h = st[(st.cadence == cad) & (st.energy == "cp")]
        for k, (lab, col, name) in enumerate((("boltz", SEQ[0], "Boltzmann"), ("stretchE", SEQ[1], "exp. étirée"),
                                              ("two", SEQ[2], "deux étages"))):
            y, lo, hi = h[f"{lab}_R"], h[f"{lab}_R_lo"], h[f"{lab}_R_hi"]
            ax.errorbar(h.elo_mid + (k - 1) * 15, y, yerr=[(y - lo).clip(lower=0), (hi - y).clip(lower=0)], fmt="-o",
                        color=col, lw=2, ms=5, capsize=3, elinewidth=1.1, label=name, mec="#fcfcfb", mew=1.2)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, "R = 1 : une seule température rend compte des fluctuations de ΔE (spectre complet, prof. 12). "
             "IC 95 % bootstrap.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "fdt.png"))
    plt.close(fig)


def make_players(pe, grid, prof, sig_null, bands, a, outdir):
    """Piste 5 : profil de σ, fiabilité moitié/moitié, s par joueur."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    _style(ax, "−Δ log-vraisemblance profilée")
    ax.plot(grid, prof - prof.min(), "-o", color=SEQ[2], lw=2, ms=4)
    ax.axhline(1.92, color=MUTED, lw=0.8, ls=":")
    for v in sig_null:
        ax.axvline(v, color=SEQ[0], lw=0.8, alpha=0.7)
    ax.set_xlabel("σ = dispersion de log s entre joueurs (traits clairs : joueurs permutés)", color=MUTED, fontsize=8)
    ax.set_title("Désordre : profil de σ", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, max(10, 1.2 * (prof[0] - prof.min())))
    ax = axes[1]
    _style(ax, "u (positions impaires)")
    ax.scatter(pe.u_even, pe.u_odd, s=14, color=SEQ[1], alpha=0.8, edgecolors="none")
    ax.set_xlabel("u (positions paires)", color=MUTED, fontsize=9)
    ax.set_title("Fiabilité moitié/moitié de l'effet joueur", color=INK, fontsize=11, loc="left")
    ax = axes[2]
    _style(ax, "s du joueur (cp, log)")
    ax.set_yscale("log")
    bi = {b: i for i, b in enumerate(bands)}
    ax.scatter(pe.band + 100 + np.random.default_rng(0).uniform(-40, 40, len(pe)),
               a[pe.band.map(bi).values] * np.exp(pe.u), s=14, color=SEQ[1], alpha=0.8, edgecolors="none")
    ax.plot(np.array(bands) + 100, a, "-", color=INK, lw=1.5, label="s de la tranche")
    ax.set_xlabel("Elo (tranche du joueur)", color=MUTED, fontsize=9)
    ax.set_title("Températures individuelles (a posteriori)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, "Blitz, 25 positions par joueur, tous les coups légaux (prof. 10). log s = a(Elo) + d(demi-coup) + échec + u, "
             "u ~ N(0, σ²), c fixé.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "players.png"))
    plt.close(fig)


def make_heat(st, cv_, outdir):
    """Chaleur spécifique : C(s) de l'ensemble des positions (par écart) et C par tranche d'Elo."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    _style(ax, "chaleur spécifique C = β² Var(ΔE^c)")
    ax.set_xscale("log")
    ax.set_xlabel("échelle s (cp) — « température »", color=MUTED, fontsize=9)
    ax.set_title("Blitz : C(s) sur l'ensemble des positions", color=INK, fontsize=10, loc="left")
    q = cv_[cv_.cadence == "Blitz"]
    gl = sorted(g for g in q.gap_lo.unique() if g >= 0)
    cols = plt.cm.Blues(np.linspace(0.35, 0.95, len(gl)))
    for k, (g, col) in enumerate(zip(gl, cols)):
        h = q[q.gap_lo == g]
        lab = f"écart {g:g}–{gl[k + 1]:g} cp" if k + 1 < len(gl) else f"écart > {g:g} cp"
        ax.plot(h.s, h.C, color=col, lw=1.6, label=lab)
    h = q[q.gap_lo == -1]
    ax.plot(h.s, h.C, color=INK, lw=2.2, label="toutes")
    sb = st[(st.cadence == "Blitz") & (st.gap_lo == -1)].s
    ax.axvspan(sb.min(), sb.max(), color="#eb6834", alpha=0.15, label="s humains (800-2600)")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    ax = axes[1]
    _style(ax, "C moyen par position")
    ax.set_xlabel("Elo (milieu de tranche)", color=MUTED, fontsize=9)
    ax.set_title("C prédit (modèle) et « observé » (fluctuations)", color=INK, fontsize=10, loc="left")
    for cad, col in (("Blitz", "#2a78d6"), ("Rapid", "#eb6834")):
        h = st[(st.cadence == cad) & (st.gap_lo == -1)]
        ax.plot(h.elo_mid, h.C_model, "-o", color=col, lw=2, ms=5, label=f"{cad} modèle")
        ax.plot(h.elo_mid, h.C_obs, "--s", color=col, lw=1.4, ms=4, label=f"{cad} observé")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, "Loi étirée ajustée par case (spectre complet, prof. 12). C observé = β²(H_obs − ⟨H⟩)² ; "
             "rapport observé/modèle = R de la piste 4.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "heat.png"))
    plt.close(fig)


def make_entropy(res, P, grid, outdir):
    """Pilote énergie/entropie : λ par groupe d'Elo et profil de vraisemblance de (κ_moi, κ_adv)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    _style(ax, "λ (log-cote par nat d'entropie)")
    g = res[res.model.str.startswith("B profil, groupe")]
    x = np.arange(len(g))
    for k, (col, nm, lab) in enumerate((("#2a78d6", "lam_S_moi", "λ_moi : garder ses options"),
                                        ("#eb6834", "lam_S_adv", "λ_adv : options de l'adversaire"))):
        ax.errorbar(x + (k - 0.5) * 0.1, g[nm], yerr=1.96 * g[nm + "_se"], fmt="-o", color=col, lw=2, ms=6,
                    capsize=3, label=lab, mec="#fcfcfb", mew=1.2)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("groupe ")[1] for m in g.model], color=MUTED, fontsize=9)
    ax.set_xlabel("groupe d'Elo", color=MUTED, fontsize=9)
    ax.set_title("Préférence pour l'entropie des positions atteintes", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1]
    D = P - P.min()
    im = ax.imshow(np.minimum(D, 60).T, origin="lower", extent=[grid[0], grid[-1], grid[0], grid[-1]],
                   cmap="Blues_r", aspect="auto")
    ax.contour(grid, grid, D.T, levels=[3.0], colors=INK, linewidths=1.2)
    ax.plot([1], [1], "o", color="#eb6834", ms=8, label="joueur « rationnel » (κ = 1)")
    ax.plot([0], [0], "s", color=MUTED, ms=6, label="énergie seule (κ = 0)")
    ax.set_xlabel("κ_moi (poids de mes erreurs futures)", color=MUTED, fontsize=9)
    ax.set_ylabel("κ_adv (poids des erreurs adverses)", color=MUTED, fontsize=9)
    ax.set_title("Énergie libre : profil de −log-vraisemblance", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.01, 0.01, "Pilote Blitz, 837 positions, 8 coups candidats ; entropies et pertes attendues à prof. 10. "
             "Barres : IC 95 % ; contour : région de confiance 95 % (Δnll = 3).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "entropy.png"))
    plt.close(fig)


def make_transition(sh, cu, hi, sz, fs, outdir):
    """Recherche d'une transition : histogramme de la perte (coup unique), bascule P(trouve), effet de taille."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    gmax = hi.gap_lo.max()
    ax = axes[0]
    _style(ax, "fraction des positions")
    ax.set_xlabel("perte du coup joué ΔE (cp, échelle log ; 0 placé à 0,5)", color=MUTED, fontsize=9)
    ax.set_title(f"Blitz, coup unique (écart > {gmax:g} cp)", color=INK, fontsize=10, loc="left")
    h = hi[(hi.cadence == "Blitz") & (hi.gap_lo == gmax)]
    xc = np.where(h.lo == 0, 0.5, np.sqrt(h.lo.clip(lower=1) * h.hi))
    ax.set_xscale("log")
    obs_, pred_ = h.obs.where(h.obs > 0), h.pred.where(h.pred > 1e-4)
    ax.plot(xc, obs_, "o", color=COLORS["Blitz"], ms=5, label="observé")
    ax.plot(xc, pred_, "s", mfc="none", color=MUTED, ms=6, label="loi étirée (prédit)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1]
    _style(ax, "P(coup à < 50 cp) observée")
    ax.set_xlabel("P(coup à < 50 cp) prédite par la loi étirée", color=MUTED, fontsize=9)
    ax.set_title("Bascule trouve / gaffe (Blitz)", color=INK, fontsize=10, loc="left")
    gl = sorted(cu.gap_lo.unique())
    cols = plt.cm.Blues(np.linspace(0.35, 0.95, len(gl)))
    for g, col in zip(gl, cols):
        q = cu[(cu.cadence == "Blitz") & (cu.gap_lo == g)]
        b = sh[(sh.cadence == "Blitz") & (sh.gap_lo == g)].b.iloc[0]
        ax.plot(q.p_model, q.p_obs, "-o", color=col, lw=1.5, ms=3, label=f"écart ≥ {g:g} (b = {b:.2f})")
    ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=1)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    ax = axes[2]
    _style(ax, "C (repondéré à écart fixé)")
    ax.set_xscale("log")
    ax.set_xlabel("échelle s (cp)", color=MUTED, fontsize=9)
    ax.set_title("Effet de taille : C(s) par nb de coups légaux (Blitz, hors échec)", color=INK, fontsize=9, loc="left")
    q = fs[(fs.cadence == "Blitz") & (fs["mode"] == "repondéré écart")]
    nl = sorted(q.n_lo.unique())
    cols = plt.cm.Oranges(np.linspace(0.35, 0.95, len(nl)))
    lab = sz[(sz.cadence == "Blitz") & (sz["mode"] == "repondéré écart")].set_index("n_lo")
    for n_, col in zip(nl, cols):
        r = q[q.n_lo == n_]
        ax.plot(r.s, r.C, color=col, lw=1.8, label=f"{n_:g}–{lab.loc[n_, 'n_hi']:g} coups")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.text(0.01, 0.01, "Spectre complet, prof. 12 ; loi étirée ajustée par case. b : pente de logit P(trouve) obs. "
             "sur logit P prédite (1 = modèle continu, > 1 = bascule plus brutale).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, "transition.png"))
    plt.close(fig)
