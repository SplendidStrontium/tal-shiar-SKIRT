#!/usr/bin/env python3
"""
build_enterprise_sfh_figures.py  (v3)
-------------------------------------
Figures from the SFH extraction (products/enterprise_sfh.csv +
enterprise_sfh_summary.csv).

v3 changes (figure review round 1):
  * SPLIT by clean pair. The two pairs don't relate across the
    confounded BH<->BH6 axis, so there is no reason to co-plot them;
    each figure is now one pair, 1x5 family panels:
        fig_enterprise_sfh_addBH        noBH vs BH
        fig_enterprise_sfh_feedback     BH6  vs BH8
        fig_enterprise_ssfr_color_addBH
        fig_enterprise_ssfr_color_feedback
  * Each SFH panel gets its own y-axis (single row: nothing to share).
    Per-panel legends dropped; one caption line replaces ten legends.
  * Marker scheme in the sSFR figures simplified to legend-free:
    open = baseline variant, filled = changed variant, family color
    throughout. (Deck-wide VARIANT_MARKERS still rule any figure that
    mixes all four variants.)
  * Trailing PARTIAL time bin dropped per run (t_right > t_now): the
    last arange bin held ~0.0008 Gyr of history divided by the full
    0.2 Gyr width -- the 'cliff to zero' at the right edge was an
    artifact, not quenching.
  * ssfr figure layout fixed: concise ylabel (no more overflow),
    rho annotation moved clear of the data, redder/bluer cues as
    small in-axes hints.

Standing decisions honored:
  * No panel and no connector ever crosses the confounded BH<->BH6
    boundary -- now enforced by construction, one pair per figure.
  * Family mass ordering: panels left->right most massive first,
    matching the viridis ramp; stated once per caption.
  * Quenched runs appear as upper limits with a leftward arrow,
    never silently dropped.

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

from build_enterprise_figures import (BASE, FAMILIES, VARIANTS, PAIRS,
                                      GREY,
                                      load_colors, face_on, load_masses,
                                      family_colors, save)

SFH_CSV = BASE / "products" / "enterprise_sfh.csv"
SUM_CSV = BASE / "products" / "enterprise_sfh_summary.csv"

# (pair, short title, filename tag) -- clean axes only, by construction
PAIR_FIGS = [(("noBH", "BH"),
              "adding a black hole (noBH \u2192 BH)", "addBH"),
             (("BH6", "BH8"),
              "feedback \u00d710 weaker (BH6 \u2192 BH8)", "feedback")]


def run_name(family, variant):
    return family if variant == "noBH" else f"{family}_{variant}"


def load_sfh():
    """SFH rows with the trailing partial bin removed per run.
    The extractor's arange grid ends past t_now, so the last bin
    contains a sliver of history divided by the full bin width --
    a spurious cliff. Drop any bin whose right edge exceeds t_now."""
    df = pd.read_csv(SFH_CSV)
    t_now = pd.read_csv(SUM_CSV).set_index("run").t_now
    df = df[df.t_right <= df.run.map(t_now) + 1e-9].copy()
    df["t_mid"] = 0.5 * (df.t_left + df.t_right)
    return df


# ---------------------------------------------------------------------------
# SFH figures: one clean pair, 1x5 family panels
# ---------------------------------------------------------------------------

def fig_sfh_pair(sfh, mstar_fam, outdir, pair, title, tag, logy=False):
    v0, v1 = pair
    shades, order = family_colors(mstar_fam)   # most massive first

    fig, axes = plt.subplots(1, len(order), figsize=(13.5, 3.6),
                             sharex=True)
    for ax, fam in zip(axes, order):
        for v, color, lw, z in ((v0, GREY, 1.6, 3),
                                (v1, shades[fam], 2.0, 4)):
            sub = sfh[sfh.run == run_name(fam, v)]
            if sub.empty:
                print(f"  WARNING: {run_name(fam, v)} missing from "
                      f"SFH CSV; skipped")
                continue
            ax.step(sub.t_mid, sub.sfr, where="mid",
                    color=color, lw=lw, zorder=z)
        if logy:
            ax.set_yscale("log")
        ax.set_title(fam, fontsize=13, color=shades[fam])
        ax.set_xlabel("t  [Gyr]", fontsize=11)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("SFR  [M\u2299/yr]", fontsize=12)
    fig.suptitle(f"Star formation histories: {title}   "
                 f"(main halo, matched initial conditions)",
                 fontsize=15, y=0.99)
    fig.text(0.5, 0.015,
             f"grey: {v0} \u00b7 color: {v1} \u00b7 families "
             f"mass-ordered, most massive left \u00b7 "
             f"y-axes independent per family",
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.06, right=0.99,
                        wspace=0.30)
    stem = f"fig_enterprise_sfh_{tag}" + ("_logy" if logy else "")
    save(fig, outdir, stem)

def fig_sfh_feedback_highlight(sfh, mstar_fam, outdir,
                               families=("r488", "r613"), logy=False):
    """One standalone figure per family for the BH6->BH8 feedback
    story, sized identically so they can be placed and animated
    independently on one slide (typical case r488, extreme case
    r613). Bottom captions omitted deliberately: spoken aloud.
    Full five-family version remains the backup slide."""
    v0, v1 = "BH6", "BH8"
    shades, _ = family_colors(mstar_fam)

    for fam in families:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for v, color, lw, z in ((v0, GREY, 1.8, 3),
                                (v1, shades[fam], 2.2, 4)):
            sub = sfh[sfh.run == run_name(fam, v)]
            if sub.empty:
                print(f"  WARNING: {run_name(fam, v)} missing from "
                      f"SFH CSV; skipped")
                continue
            ax.step(sub.t_mid, sub.sfr, where="mid",
                    color=color, lw=lw, zorder=z, label=v)
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=11, loc="upper left",
                  handlelength=1.4, handletextpad=0.4)
        ax.set_title(fam, fontsize=15, color=shades[fam])
        ax.set_xlabel("t  [Gyr]", fontsize=12)
        ax.set_ylabel("SFR  [M\u2299/yr]", fontsize=13)
        ax.tick_params(labelsize=11)
        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.96)
        save(fig, outdir, f"fig_enterprise_sfh_feedback_{fam}")


# ---------------------------------------------------------------------------
# sSFR vs intrinsic color: one clean pair per figure, legend-free markers
# ---------------------------------------------------------------------------

def fig_ssfr_color_pair(mstar_fam, outdir, pair, title, tag):
    """log sSFR (recent window) vs. nodust face-on B-K for ONE clean
    pair: 10 points, open = baseline, filled = changed, connector per
    family. The intrinsic leg of the decomposition.

    Speaker-notes honesty: co-movement along a locus, not causation
    by itself; the net observed color still needs the dust leg."""
    v0, v1 = pair
    pivot = face_on(load_colors(), "nodust")
    summ = pd.read_csv(SUM_CSV).set_index("run")
    mstar_by_run, _ = load_masses()
    shades, order = family_colors(mstar_fam)
    window_gyr = float(summ.sfr_window_gyr.iloc[0])

    ssfr = {}
    for f in FAMILIES:
        for v in (v0, v1):
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

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    # Connectors underneath
    for f in FAMILIES:
        if (f, v0) in ssfr and (f, v1) in ssfr:
            ax.plot([xval((f, v0)), xval((f, v1))],
                    [pivot.loc[f, v0], pivot.loc[f, v1]],
                    "-", color=shades[f], lw=1.4, alpha=0.7, zorder=2)

    # Points: open = baseline, filled = changed -- no legend lookup
    for (f, v), s in ssfr.items():
        fill = "none" if v == v0 else "full"
        x, y = xval((f, v)), pivot.loc[f, v]
        ax.plot(x, y, marker="o", fillstyle=fill, ms=10, mew=1.8,
                color=shades[f], linestyle="none", zorder=4)
        if s <= 0:                           # quenched: leftward limit
            ax.annotate("", xy=(x - 0.35, y), xytext=(x - 0.06, y),
                        arrowprops=dict(arrowstyle="->", lw=1.3,
                                        color=shades[f]), zorder=4)

    # Spearman on this pair's detections, above the frame (never on data)
    det = [(np.log10(s), pivot.loc[f, v])
           for (f, v), s in ssfr.items() if s > 0]
    if len(det) >= 3:
        xs, ys = map(np.asarray, zip(*det))
        rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
        note = f"Spearman \u03c1 = {rho:+.2f}"
        if len(det) < len(ssfr):
            note += f" ({len(det)}/{len(ssfr)} detected; limits excluded)"
        ax.annotate(note, xy=(0.02, 1.02), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=10, color=GREY)

    # redder/bluer cues inside the axes, small and grey
    ax.annotate("redder \u2191", xy=(0.99, 0.99),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=9, color=GREY)
    ax.annotate("bluer \u2193", xy=(0.99, 0.01),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9, color=GREY)

    win_myr = window_gyr * 1e3
    ax.set_xlabel(f"log\u2081\u2080  sSFR (last {win_myr:.0f} Myr)  "
                  f"[yr\u207b\u00b9]")
    ax.set_ylabel("intrinsic B\u2212K (Vega, nodust)  [mag]")
    ax.set_title(title, fontsize=13, pad=28)
    fig.text(0.5, 0.013,
             f"open: {v0} \u00b7 filled: {v1} \u00b7 one connector per "
             f"family \u00b7 darker = more massive family \u00b7 "
             f"arrows: sSFR upper limits",
             ha="center", va="bottom", fontsize=9, color=GREY)
    fig.subplots_adjust(top=0.85, bottom=0.17, left=0.13, right=0.96)
    save(fig, outdir, f"fig_enterprise_ssfr_color_{tag}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "products" / "figures")
    ap.add_argument("--logy", action="store_true",
                    help="log y-axis on the SFH panels (bursty dwarfs "
                         "span decades; try both)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    _, mstar_fam = load_masses()
    sfh = load_sfh()

    print("Building pair-split SFH figures ->", args.outdir)
    for pair, title, tag in PAIR_FIGS:
        fig_sfh_pair(sfh, mstar_fam, args.outdir, pair, title, tag,
                     logy=args.logy)
        fig_ssfr_color_pair(mstar_fam, args.outdir, pair, title, tag)
    fig_sfh_feedback_highlight(sfh, mstar_fam, args.outdir,
                               logy=args.logy)
    print("Done.")


if __name__ == "__main__":
    main()