#!/usr/bin/env python3
"""
build_enterprise_figures.py
---------------------------
Money-shot figures for the Enterprise twin-study, from
enterprise_colors.csv (+ enterprise_census.csv for masses).

Figures (each written as PDF vector + 200 dpi PNG):

  fig_enterprise_slopegraph      face-on dust B-K, TWO side-by-side
                                 panels sharing a y-axis: noBH<->BH and
                                 BH6<->BH8. No connector between panels.
  fig_enterprise_dust_reddening  same paired layout, y = (dust - nodust).
  fig_enterprise_color_mass      only with --color-mass; superseded by
                                 the explorer figure's panel D and
                                 recommended dropped from the talk.

Design decisions (deliberate, flag if you disagree):
  * v2: the BH<->BH6 dotted connector is GONE. Even a dotted line
    asserts a trajectory across a comparison that changes accretion
    threshold AND feedback together. Separate panels on a shared
    y-axis keep levels comparable while making trajectories exist
    only within the clean single-variable pairs.
  * Each panel carries its own mean +/- std of the pair delta across
    the five families (the numbers already quoted from --all).
  * Families are colored on a sequential (viridis) ramp ordered by
    noBH main-halo M*; end labels sit on the OUTER edges (left panel
    labels left, right panel labels right) with a vertical dodge so
    they can never overlap, fixing the r613/r618 and r488/r741
    collisions in v1.
  * Arrow glyphs in rotated y-labels: after 90 deg rotation '\u2192'
    renders visually UP and '\u2190' DOWN (confirmed empirically).
  * Because viridis contains blues, color-direction cues on B-K axes
    use neutral grey (same palette rule as the poster figures).

Usage:
    python build_enterprise_figures.py
    python build_enterprise_figures.py --color-mass   # also build fig 3
    python build_enterprise_figures.py --outdir /some/where
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inputs / sample
# ---------------------------------------------------------------------------

BASE = Path("/mnt/data0/pkrsnak/romulus/enterprise")
COLORS_CSV = BASE / "products" / "enterprise_colors.csv"
CENSUS_CSV = BASE / "enterprise_census.csv"

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]

# The two clean single-variable pairs, with the knob each one turns.
PAIRS = [(("noBH", "BH"), "adding a black hole\n(restricted accretion)"),
         (("BH6", "BH8"), "feedback efficiency 0.05 \u2192 0.005\n"
                          "(permissive accretion)")]

DPI_PNG = 200

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GREY = "0.45"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_colors():
    df = pd.read_csv(COLORS_CSV)
    df = df[df.instrument.str.startswith("sed_")]
    return df


def face_on(df, medium):
    sub = df[(df.medium == medium) & np.isclose(df.inc_deg, 0.0)]
    return (sub.pivot_table(index="family", columns="variant", values="BK")
               .reindex(index=FAMILIES, columns=VARIANTS))


def load_masses():
    cen = pd.read_csv(CENSUS_CSV)
    main = cen[cen.halo_id == 1].set_index("run")
    mstar_by_run = main.mstar.to_dict()
    mstar_fam = {f: mstar_by_run[f] for f in FAMILIES}
    return mstar_by_run, mstar_fam


def family_colors(mstar_fam):
    order = sorted(FAMILIES, key=lambda f: mstar_fam[f], reverse=True)
    cmap = plt.get_cmap("viridis")
    shades = {f: cmap(0.10 + 0.75 * i / (len(order) - 1))
              for i, f in enumerate(order)}
    return shades, order


def fam_label(f, mstar_fam):
    return f"{f}  ({mstar_fam[f] / 1e8:.1f}\u00d710\u2078)"


def save(fig, outdir, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI_PNG})):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        print(f"  wrote {p}")
    plt.close(fig)


def dodge(ys, min_sep):
    """Push label y-positions apart (ascending) until separated by
    >= min_sep, preserving order. Returns array aligned with input."""
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    out = ys.copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_sep:
            out[b] = out[a] + min_sep
    return out


# ---------------------------------------------------------------------------
# Paired slopegraph (figures 1 and 2)
# ---------------------------------------------------------------------------

def paired_slopegraph(pivot, mstar_fam, ylabel, suptitle, zero_line=False):
    shades, order = family_colors(mstar_fam)

    vals = pivot.loc[FAMILIES, VARIANTS].to_numpy(dtype=float)
    pad = 0.06 * (vals.max() - vals.min())
    ylo, yhi = vals.min() - pad, vals.max() + pad
    if zero_line:
        ylo = min(ylo, -0.02)
    min_sep = 0.045 * (yhi - ylo)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4), sharey=True,
                             gridspec_kw=dict(wspace=0.10))

    for ax, ((v0, v1), knob), side in zip(axes, PAIRS, ("left", "right")):
        y0 = pivot[v0].loc[FAMILIES].to_numpy(dtype=float)
        y1 = pivot[v1].loc[FAMILIES].to_numpy(dtype=float)

        for f, a, b in zip(FAMILIES, y0, y1):
            c = shades[f]
            ax.plot([0, 1], [a, b], "-", color=c, lw=2.4, zorder=3)
            ax.plot([0, 1], [a, b], "o", color=c, ms=7, zorder=4,
                    linestyle="none")

        # Outer-edge family labels with vertical dodge
        if side == "left":
            ylab = dodge(y0, min_sep)
            for f, yl in zip(FAMILIES, ylab):
                ax.annotate(fam_label(f, mstar_fam), xy=(0, yl),
                            xytext=(-10, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=11,
                            color=shades[f])
            ax.set_xlim(-1.15, 1.25)
        else:
            ylab = dodge(y1, min_sep)
            for f, yl in zip(FAMILIES, ylab):
                ax.annotate(fam_label(f, mstar_fam), xy=(1, yl),
                            xytext=(10, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=11,
                            color=shades[f])
            ax.set_xlim(-0.25, 2.15)

        delta = y1 - y0
        ax.set_title(knob, fontsize=12)
        ax.annotate(f"mean \u0394 = {delta.mean():+.2f} \u00b1 "
                    f"{delta.std(ddof=1):.2f}",
                    xy=(0.5, 0.02), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=11, color=GREY)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([v0, v1])
        if zero_line:
            ax.axhline(0.0, color=GREY, lw=0.8, zorder=1)

    axes[0].set_ylim(ylo, yhi)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, fontsize=15)
    fig.text(0.5, 0.925,
             "panels share the y-axis; accretion physics differs between "
             "panels, so compare levels, not trajectories",
             ha="center", va="top", fontsize=10, color=GREY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def fig_slopegraph(df, mstar_fam, outdir):
    pivot = face_on(df, "dust")
    ylabel = ("\u2190 bluer      B\u2212K (Vega)  [mag]      redder \u2192")
    fig = paired_slopegraph(
        pivot, mstar_fam, ylabel,
        suptitle="Same initial conditions, different BH physics "
                 "(face-on, with dust)")
    save(fig, outdir, "fig_enterprise_slopegraph")


def fig_dust_reddening(df, mstar_fam, outdir):
    pivot = face_on(df, "dust") - face_on(df, "nodust")
    fig = paired_slopegraph(
        pivot, mstar_fam,
        ylabel="dust reddening of B\u2212K  [mag]\n"
               "(dust \u2212 nodust, face-on)",
        suptitle="How much dust reddens each galaxy",
        zero_line=True)
    save(fig, outdir, "fig_enterprise_dust_reddening")


# ---------------------------------------------------------------------------
# Figure 3 (opt-in only): color vs mass
# ---------------------------------------------------------------------------

VARIANT_MARKERS = {"noBH": ("o", "none"),
                   "BH":   ("o", "full"),
                   "BH6":  ("s", "full"),
                   "BH8":  ("D", "full")}


def fig_color_mass(df, mstar_by_run, mstar_fam, outdir):
    """Superseded by the explorer figure's panel D (same content, no
    connecting lines). Kept behind --color-mass for completeness."""
    shades, order = family_colors(mstar_fam)
    pivot = face_on(df, "dust")

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for f in order:
        for v in VARIANTS:
            run = f if v == "noBH" else f"{f}_{v}"
            if run not in mstar_by_run:
                print(f"  WARNING: {run} missing from census; skipped")
                continue
            marker, fill = VARIANT_MARKERS[v]
            kw = dict(marker=marker, ms=9, color=shades[f], zorder=4,
                      linestyle="none")
            if fill == "none":
                kw.update(markerfacecolor="none", markeredgewidth=1.8)
            ax.plot(np.log10(mstar_by_run[run]), pivot.loc[f, v], **kw)

    from matplotlib.lines import Line2D
    handles = []
    for v in VARIANTS:
        marker, fill = VARIANT_MARKERS[v]
        kw = dict(marker=marker, color=GREY, linestyle="none", ms=8)
        if fill == "none":
            kw.update(markerfacecolor="none", markeredgewidth=1.6)
        handles.append(Line2D([], [], label=v, **kw))
    ax.legend(handles=handles, title="variant", frameon=False, loc="best")

    ax.set_xlabel("log\u2081\u2080  M\u2217 / M\u2299   (main halo, 30 kpc)")
    ax.set_ylabel("\u2190 bluer      B\u2212K (Vega)  [mag]      redder \u2192")
    ax.set_title("Color vs. stellar mass across BH variants")
    fig.tight_layout()
    save(fig, outdir, "fig_enterprise_color_mass")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "products" / "figures")
    ap.add_argument("--color-mass", action="store_true",
                    help="also build the (superseded) color-mass figure")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_colors()
    mstar_by_run, mstar_fam = load_masses()

    print("Building figures ->", args.outdir)
    fig_slopegraph(df, mstar_fam, args.outdir)
    fig_dust_reddening(df, mstar_fam, args.outdir)
    if args.color_mass:
        fig_color_mass(df, mstar_by_run, mstar_fam, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()