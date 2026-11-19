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
