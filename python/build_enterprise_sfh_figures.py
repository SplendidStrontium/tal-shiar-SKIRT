#!/usr/bin/env python3
"""
build_enterprise_sfh_figures.py
-------------------------------
Two new figure families from the SFH extraction
(products/enterprise_sfh.csv + enterprise_sfh_summary.csv):

  fig_enterprise_sfh_grid      5 columns (families, mass-ordered
                               left->right, matching the viridis ramp)
                               x 2 rows (clean pairs). Wide layout for
                               16:9 slides. Baseline variant (noBH/BH6)
                               in grey; changed variant (BH/BH8) in the
                               family color -- the eye reads "what did
                               the knob change."

  fig_enterprise_ssfr_color    the INTRINSIC leg of the decomposition:
                               log sSFR (recent window) vs. nodust
                               face-on B-K, 20 points, connectors only
                               within clean pairs. Companion to the
                               dust-reddening figures: this explains
                               the solid bars in fig_enterprise_
                               decomposition, those explain the
                               hatched ones.

Standing decisions honored:
  * No panel and no connector ever crosses the confounded BH<->BH6
    boundary.
  * Family mass ordering lives in the viridis ramp (imported), stated
    once per caption.
  * Columns of the SFH grid share a y-axis (same family across pairs:
    comparable levels are real information, same logic as the
    slopegraph's shared axis). Rows/columns of DIFFERENT families
    do not share.
  * Quenched runs (zero SF in the recent window) appear as upper
    limits with a leftward arrow, never silently dropped.

Kept as a separate file so build_enterprise_figures.py (v4, in review)
stays frozen; shared conventions are imported from it.

Usage:
    python build_enterprise_sfh_figures.py
    python build_enterprise_sfh_figures.py --logy --outdir /some/where
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared conventions from the frozen figure script (same directory).
# Importing also applies the shared rcParams.
from build_enterprise_figures import (BASE, FAMILIES, VARIANTS, PAIRS,
                                      GREY, VARIANT_MARKERS,
                                      load_colors, face_on, load_masses,
                                      family_colors, save)

SFH_CSV = BASE / "products" / "enterprise_sfh.csv"
SUM_CSV = BASE / "products" / "enterprise_sfh_summary.csv"


def run_name(family, variant):
    return family if variant == "noBH" else f"{family}_{variant}"


def load_sfh():
    df = pd.read_csv(SFH_CSV)
    df["t_mid"] = 0.5 * (df.t_left + df.t_right)
    return df


# ---------------------------------------------------------------------------
# Figure A: SFH grid, families x clean pairs
# ---------------------------------------------------------------------------

def fig_sfh_grid(mstar_fam, outdir, logy=False):
    sfh = load_sfh()
    shades, order = family_colors(mstar_fam)   # order: most massive first

    pair_rows = [("noBH", "BH",  "adding a black hole\n(noBH \u2192 BH)"),
                 ("BH6",  "BH8", "feedback \u00d710 weaker\n"
                                 "(BH6 \u2192 BH8)")]

    fig, axes = plt.subplots(2, len(order), figsize=(13.5, 5.8),
                             sharex=True, sharey="col")

    for col, fam in enumerate(order):
        for row, (v0, v1, _) in enumerate(pair_rows):
            ax = axes[row, col]
            for v, color, lw, z in ((v0, GREY, 1.6, 3),
                                    (v1, shades[fam], 2.0, 4)):
                sub = sfh[sfh.run == run_name(fam, v)]
                if sub.empty:
                    print(f"  WARNING: {run_name(fam, v)} missing "
                          f"from SFH CSV; skipped")
                    continue
                ax.step(sub.t_mid, sub.sfr, where="mid",
                        color=color, lw=lw, zorder=z, label=v)
            if logy:
                ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                      handlelength=1.2, handletextpad=0.4,
                      borderaxespad=0.15, labelspacing=0.2)

        # Family name atop each column, in the family's shade
        axes[0, col].set_title(fam, fontsize=13, color=shades[fam])

    # Pair labels on the left edge, one per row
    for row, (_, _, lab) in enumerate(pair_rows):
        axes[row, 0].set_ylabel(f"{lab}\n\nSFR  [M\u2299/yr]", fontsize=10)
    for col in range(len(order)):
        axes[1, col].set_xlabel("t  [Gyr]", fontsize=11)

    fig.suptitle("Star formation histories: same initial conditions, "
                 "different BH physics (main halo)", fontsize=15, y=0.99)
    fig.text(0.5, 0.012,
             "grey: baseline variant \u00b7 color: changed variant \u00b7 "
             "families mass-ordered, most massive left \u00b7 "
             "columns share a y-axis; rows are independent clean pairs",
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.09, right=0.985,
                        hspace=0.12, wspace=0.30)
    stem = "fig_enterprise_sfh_grid" + ("_logy" if logy else "")
    save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# Figure B: sSFR vs intrinsic color (the intrinsic leg)
# ---------------------------------------------------------------------------

def fig_ssfr_color(mstar_fam, outdir):
    """log sSFR (recent window, from the SFH summary) vs. nodust
    face-on B-K. Connectors within clean pairs only.

    Speaker-notes honesty: this shows sSFR and intrinsic color move
    TOGETHER along one locus -- it does not by itself establish
    causation, and the net observed color still depends on the dust
    leg (see decomposition figure)."""
    pivot = face_on(load_colors(), "nodust")
    summ = pd.read_csv(SUM_CSV).set_index("run")
    mstar_by_run, _ = load_masses()
    shades, order = family_colors(mstar_fam)
    window_gyr = float(summ.sfr_window_gyr.iloc[0])

    # Assemble points; log floor for quenched runs (upper limits)
    ssfr = {}
    for f in FAMILIES:
        for v in VARIANTS:
            run = run_name(f, v)
            if run not in summ.index or run not in mstar_by_run:
                print(f"  WARNING: {run} missing; skipped")
                continue
            ssfr[(f, v)] = summ.loc[run, "sfr_recent"] / mstar_by_run[run]

    pos = [s for s in ssfr.values() if s > 0]
    floor = np.log10(min(pos)) - 0.4         # limits sit below the data

    def xval(key):
        s = ssfr[key]
        return np.log10(s) if s > 0 else floor

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # Connectors first (underneath): clean pairs only, never BH<->BH6
    for f in FAMILIES:
        for (v0, v1), _ in PAIRS:
            if (f, v0) in ssfr and (f, v1) in ssfr:
                ax.plot([xval((f, v0)), xval((f, v1))],
                        [pivot.loc[f, v0], pivot.loc[f, v1]],
                        "-", color=shades[f], lw=1.3, alpha=0.7, zorder=2)

    for (f, v), s in ssfr.items():
        marker, fill = VARIANT_MARKERS[v]
        x, y = xval((f, v)), pivot.loc[f, v]
        ax.plot(x, y, marker=marker, fillstyle=fill, ms=9,
                color=shades[f], linestyle="none", zorder=4)
        if s <= 0:                           # quenched: leftward limit
            ax.annotate("", xy=(x - 0.35, y), xytext=(x - 0.06, y),
                        arrowprops=dict(arrowstyle="->", lw=1.3,
                                        color=shades[f]), zorder=4)

    # Spearman on detections only (limits excluded, honestly noted)
    det = [(np.log10(s), pivot.loc[f, v])
           for (f, v), s in ssfr.items() if s > 0]
    xs, ys = map(np.asarray, zip(*det))
    rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    ax.annotate(f"Spearman \u03c1 = {rho:+.2f} "
                f"({len(det)} detections; limits excluded)",
                xy=(0.03, 0.03), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=10, color=GREY)

    # Variant marker key (grey, shape-only)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, fillstyle=fs, linestyle="none",
                      ms=8, color="0.35", label=v)
               for v, (m, fs) in VARIANT_MARKERS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9,
              loc="upper right", handletextpad=0.3)

    win_myr = window_gyr * 1e3
    ax.set_xlabel(f"log\u2081\u2080  sSFR (last {win_myr:.0f} Myr)  "
                  f"[yr\u207b\u00b9]")
    ax.set_ylabel("\u2190 bluer    intrinsic B\u2212K (Vega, nodust)  "
                  "[mag]    redder \u2192")
    ax.set_title("Recent star formation tracks the intrinsic color")
    fig.text(0.5, 0.012,
             "connectors: clean twin pairs only (noBH\u2192BH, "
             "BH6\u2192BH8) \u00b7 point shade: darker = more massive "
             "family \u00b7 arrows: sSFR upper limits",
             ha="center", va="bottom", fontsize=9, color=GREY)
    fig.subplots_adjust(top=0.92, bottom=0.17, left=0.13, right=0.96)
    save(fig, outdir, "fig_enterprise_ssfr_color")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "products" / "figures")
    ap.add_argument("--logy", action="store_true",
                    help="log y-axis on the SFH grid (bursty dwarfs "
                         "span decades; try both)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    _, mstar_fam = load_masses()

    print("Building SFH figures ->", args.outdir)
    fig_sfh_grid(mstar_fam, args.outdir, logy=args.logy)
    fig_ssfr_color(mstar_fam, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()