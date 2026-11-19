"""Graphiques : T(Elo) avec barres d'erreur (2 variantes) et taux de coups hors candidats."""
import os

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
