#!/usr/bin/env python3
"""
build_enterprise_explorer.py
----------------------------
Color vs. everything-else: does the census explain the colors?

The blueness of the permissive-accretion variants decomposes into two
halves (dust reddening collapse + intrinsically bluer nodust colors),
so the panels are decomposed the same way -- each candidate explanation
is tested against the color component it could actually cause:

  Panel A  dust reddening (dust - nodust, face-on)  vs  log10 dust proxy
           The mechanism panel: SKIRT's dust input vs what dust did.
  Panel B  NODUST face-on B-K  vs  log10 sSFR
           Young-star fraction candidate for the intrinsic half.
  Panel C  NODUST face-on B-K  vs  Z*_mw
           Stellar metallicity candidate for the intrinsic half.
  Panel D  NODUST face-on B-K  vs  log10 M*
           Context: is intrinsic color just tracking integrated history?

Shared visual language with build_enterprise_figures.py (imported):
family = viridis ramp ordered by noBH M*, variant = marker shape,
noBH = open circle. Spearman rho printed per panel in grey with an
explicit small-n caveat in the figure note -- descriptive, not a test.

Points whose SFR rests on < 20 young star particles are drawn with a
grey edge ring in Panel B and listed on stdout: their sSFR is particle
noise and should be read as an upper limit.

Usage:
    python build_enterprise_explorer.py
    python build_enterprise_explorer.py --outdir /some/where
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Shared style/helpers -- single source of truth for the talk's visuals.
from build_enterprise_figures import (BASE, FAMILIES, VARIANTS,
                                      VARIANT_MARKERS, GREY,
                                      family_colors, load_colors, face_on,
                                      save)

EXT_CENSUS_CSV = BASE / "enterprise_census_extended.csv"
N_YOUNG_MIN = 20     # below this, SFR is particle noise -> flagged


# ---------------------------------------------------------------------------
# Data assembly: one row per run with colors + census quantities
# ---------------------------------------------------------------------------

def assemble():
    df = load_colors()
    dust = face_on(df, "dust")
    nod = face_on(df, "nodust")

    cen = pd.read_csv(EXT_CENSUS_CSV)

    rows = []
    for f in FAMILIES:
        for v in VARIANTS:
            run = f if v == "noBH" else f"{f}_{v}"
            c = cen[cen.run == run]
            if len(c) != 1:
                print(f"WARNING: {run} not uniquely in extended census; "
                      f"skipped")
                continue
            c = c.iloc[0]
            rows.append(dict(
                run=run, family=f, variant=v,
                bk_dust=dust.loc[f, v], bk_nodust=nod.loc[f, v],
                reddening=dust.loc[f, v] - nod.loc[f, v],
                mstar=c.mstar_30kpc, ssfr=c.ssfr, zstar=c.zstar_mw,
                n_young=c.n_young, mdust=c.mdust_proxy))
    d = pd.DataFrame(rows)

    low = d[d.n_young < N_YOUNG_MIN]
    if len(low):
        print(f"SFR resting on < {N_YOUNG_MIN} young particles "
              f"(treat sSFR as upper limit):")
        print(low[["run", "n_young"]].to_string(index=False))
    return d


# ---------------------------------------------------------------------------
# Panel machinery
# ---------------------------------------------------------------------------

def scatter_panel(ax, d, xcol, ycol, shades, logx=False,
                  flag_low_young=False):
    x_all, y_all = [], []
    for _, r in d.iterrows():
        x = np.log10(r[xcol]) if logx else r[xcol]
        if not np.isfinite(x) or not np.isfinite(r[ycol]):
            continue
        marker, fill = VARIANT_MARKERS[r.variant]
        kw = dict(marker=marker, ms=8, color=shades[r.family],
                  linestyle="none", zorder=4)
        if fill == "none":
            kw.update(markerfacecolor="none", markeredgewidth=1.6)
        if flag_low_young and r.n_young < N_YOUNG_MIN:
            kw.update(markeredgecolor=GREY, markeredgewidth=2.2)
        ax.plot(x, r[ycol], **kw)
        x_all.append(x)
        y_all.append(r[ycol])

    if len(x_all) >= 3:
        rho, _ = spearmanr(x_all, y_all)
        ax.annotate(f"Spearman \u03c1 = {rho:+.2f}",
                    xy=(0.03, 0.03), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=10, color=GREY)


def variant_legend(ax):
    from matplotlib.lines import Line2D
    handles = []
    for v in VARIANTS:
        marker, fill = VARIANT_MARKERS[v]
        kw = dict(marker=marker, color=GREY, linestyle="none", ms=7)
        if fill == "none":
            kw.update(markerfacecolor="none", markeredgewidth=1.4)
        handles.append(Line2D([], [], label=v, **kw))
    ax.legend(handles=handles, frameon=False, fontsize=9,
              loc="upper right", handletextpad=0.2, borderaxespad=0.2)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "products" / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    d = assemble()
    mstar_fam = {f: float(d[(d.family == f) & (d.variant == "noBH")]
                          .mstar.iloc[0]) for f in FAMILIES}
    shades, _ = family_colors(mstar_fam)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    (axA, axB), (axC, axD) = axes

    # A: mechanism -- what dust did vs what dust SKIRT was given
    scatter_panel(axA, d, "mdust", "reddening", shades, logx=True)
    axA.set_xlabel("log\u2081\u2080  dust reservoir 0.4\u00b7M\u1d22(cold) "
                   "[M\u2299]")
    axA.set_ylabel("dust reddening of B\u2212K  [mag]")
    axA.set_title("A \u00b7 attenuation follows the dust reservoir")
    variant_legend(axA)

    # B: intrinsic color vs sSFR
    scatter_panel(axB, d, "ssfr", "bk_nodust", shades, logx=True,
                  flag_low_young=True)
    axB.set_xlabel("log\u2081\u2080  sSFR  [yr\u207b\u00b9]")
    axB.set_ylabel("nodust B\u2212K (Vega)  [mag]")
    axB.set_title("B \u00b7 intrinsic color vs. specific SFR")

    # C: intrinsic color vs stellar metallicity
    scatter_panel(axC, d, "zstar", "bk_nodust", shades)
    axC.set_xlabel("mass-weighted stellar metallicity Z\u2217")
    axC.set_ylabel("nodust B\u2212K (Vega)  [mag]")
    axC.set_title("C \u00b7 intrinsic color vs. metallicity")

    # D: intrinsic color vs stellar mass
    scatter_panel(axD, d, "mstar", "bk_nodust", shades, logx=True)
    axD.set_xlabel("log\u2081\u2080  M\u2217 / M\u2299")
    axD.set_ylabel("nodust B\u2212K (Vega)  [mag]")
    axD.set_title("D \u00b7 intrinsic color vs. stellar mass")

    fig.suptitle("What sets the colors? "
                 "(attenuation half: panel A; intrinsic half: B\u2013D)",
                 fontsize=15)
    fig.text(0.01, 0.01,
             "n = 20 with family structure; Spearman \u03c1 descriptive, "
             "not a significance test \u00b7 grey-ringed points: "
             f"SFR on < {N_YOUNG_MIN} young particles (upper limit)",
             fontsize=9, color=GREY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, args.outdir, "fig_enterprise_explorer")

    # Companion table on stdout: per-variant means of the candidates,
    # for talking-point numbers without opening the figure.
    print("\n--- per-variant means (across 5 families) ---")
    tab = (d.groupby("variant")
             .agg(bk_dust=("bk_dust", "mean"),
                  bk_nodust=("bk_nodust", "mean"),
                  reddening=("reddening", "mean"),
                  log_ssfr=("ssfr", lambda s: np.log10(s).mean()),
                  zstar=("zstar", "mean"),
                  log_mdust=("mdust", lambda s: np.log10(s).mean()))
             .reindex(VARIANTS))
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()