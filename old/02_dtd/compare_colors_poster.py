#!/usr/bin/env python3
"""
compare_colors.py
-----------------
First comparison figure: observed vs simulated B-K as populations.

Observed : clean dereddened sample from catalog_dered.csv
           (rows with ~deredden_suspect AND a finite ebv_sfd), color = BK_dered.
Simulated: all halos from sim_colors.csv, color = BK.

Outputs two standalone figures -- bk_histogram.pdf/.png (distribution) and
bk_cdf.pdf/.png (cumulative) -- and prints
KS / Anderson-Darling stats. This is the "are simulated dwarf colors drawn from
the observed population?" test. The BH split and the color-mass diagram come
once the census (M*, tform<0 labels) is joined in.

Usage:
    python compare_colors.py --observed catalog_dered.csv --sim sim_colors.csv
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---- poster styling -------------------------------------------------------
# Larger relative font sizes so text stays legible after the figure is scaled
# down into a poster column. PDF output is vector, so this is about the
# text-to-plot ratio, not resolution.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "lines.linewidth": 2.2,
})
PNG_DPI = 200          # crisper raster if a PNG (not the PDF) ends up on the poster
BLUE = "#2166ac"
RED = "#b2182b"
# translucent backing so the stat callout stays legible against the plot
NOTE_BBOX = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75)


def annotate_color_axis(ax, axis="x", palette="color"):
    """Mark the bluer/redder direction on a B-K axis (larger B-K = redder).

    palette="color"   -> blue/red words (use where nothing else is blue/red)
    palette="neutral" -> grey words (use where blue/red already means something,
                         e.g. the BH-split panels)
    """
    bc, rc = (BLUE, RED) if palette == "color" else ("0.3", "0.3")
    kw = dict(fontsize=13, fontweight="bold", annotation_clip=False)
    if axis == "x":
        ax.annotate("\u2190 bluer", xy=(0.0, -0.16), xycoords="axes fraction",
                    ha="left", va="top", color=bc, **kw)
        ax.annotate("redder \u2192", xy=(1.0, -0.16), xycoords="axes fraction",
                    ha="right", va="top", color=rc, **kw)
    else:
        # NOTE: text is rotated 90deg, which also rotates the arrow glyph.
        # A rotated "\u2192" renders pointing UP, "\u2190" renders pointing DOWN.
        ax.annotate("redder \u2192", xy=(-0.135, 1.0), xycoords="axes fraction",
                    ha="right", va="top", color=rc, rotation=90, **kw)
        ax.annotate("\u2190 bluer", xy=(-0.135, 0.0), xycoords="axes fraction",
                    ha="right", va="bottom", color=bc, rotation=90, **kw)


def load_observed(path):
    df = pd.read_csv(path)
    clean = df[~df["deredden_suspect"].astype(bool) & df["ebv_sfd"].notna()]
    return clean["BK_dered"].dropna().values


def load_sim(path):
    df = pd.read_csv(path)
    dust = df["BK"].dropna().values
    nodust = (df["BK_nodust"].dropna().values
              if "BK_nodust" in df.columns else np.array([]))
    return dust, nodust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed", default="catalog_dered.csv")
    ap.add_argument("--sim", default="sim_colors.csv")
    ap.add_argument("--out-hist", default="bk_histogram.pdf")
    ap.add_argument("--out-cdf", default="bk_cdf.pdf")
    ap.add_argument("--show-ks", action="store_true",
                    help="annotate figures with KS D/p (for talks); "
                         "default shows the median color offset")
    args = ap.parse_args()

    obs = load_observed(args.observed)
    sim, sim_nd = load_sim(args.sim)
    have_nd = sim_nd.size > 0
    print(f"observed (clean):  n={len(obs):3d}  median={np.median(obs):.2f}  "
          f"range {obs.min():.2f}-{obs.max():.2f}")
    print(f"simulated (dust):  n={len(sim):3d}  median={np.median(sim):.2f}  "
          f"range {sim.min():.2f}-{sim.max():.2f}")
    if have_nd:
        print(f"simulated (nodust):n={len(sim_nd):3d}  median={np.median(sim_nd):.2f}  "
              f"range {sim_nd.min():.2f}-{sim_nd.max():.2f}")

    ks = stats.ks_2samp(obs, sim)
    print(f"\nKS two-sample:    D={ks.statistic:.3f}  p={ks.pvalue:.3g}")
    try:
        ad = stats.anderson_ksamp([obs, sim])
        ad_p = getattr(ad, "pvalue", None)
        if ad_p is None:  # older scipy exposes it as significance_level
            ad_p = getattr(ad, "significance_level", float("nan"))
        print(f"Anderson-Darling: A2={ad.statistic:.3f}  p={ad_p:.3g}")
    except Exception as e:
        print("Anderson-Darling skipped:", e)
    print(f"median offset (dust sim - obs): {np.median(sim) - np.median(obs):+.2f}")
    if have_nd:
        ks_nd = stats.ks_2samp(obs, sim_nd)
        print(f"KS (nodust vs obs): D={ks_nd.statistic:.3f}  p={ks_nd.pvalue:.3g}")
        print(f"median offset (nodust sim - obs): "
              f"{np.median(sim_nd) - np.median(obs):+.2f}")
        print(f"--> dust-treatment bracket on sim median offset: "
              f"{np.median(sim_nd) - np.median(obs):+.2f} (nodust)  to  "
              f"{np.median(sim) - np.median(obs):+.2f} (30,000 K dust)")
    print(f"NOTE: n_sim={len(sim)} -> modest power; read p-values with that in mind.")

    # ---- Figure 1: distribution (histogram) ----
    fig, axh = plt.subplots(figsize=(6.8, 5.1))
    bins = np.linspace(1.0, 5.0, 21)
    axh.hist(obs, bins=bins, density=True, histtype="stepfilled", alpha=0.40,
             color="0.55", label=f"observed (n={len(obs)})")
    if have_nd:
        # shade the sim dust-treatment bracket (between the two sim medians)
        axh.axvspan(np.median(sim_nd), np.median(sim), color="C3", alpha=0.07,
                    zorder=0)
        axh.hist(sim_nd, bins=bins, density=True, histtype="step", lw=2.4,
                 ls="--", color="C0", label=f"sim, nodust (n={len(sim_nd)})")
        axh.axvline(np.median(sim_nd), color="C0", ls="--", lw=1.2)
    axh.hist(sim, bins=bins, density=True, histtype="step", lw=2.6,
             color="C3", label=f"sim, dust (n={len(sim)})")
    axh.axvline(np.median(obs), color="0.45", ls="--", lw=1.2)
    axh.axvline(np.median(sim), color="C3", ls="--", lw=1.2)
    axh.set_xlabel("B - K (Vega)")
    axh.set_ylabel("normalized density")
    axh.set_title("B$-$K color distribution: ROMULUS vs. observed nearby galaxies")
    annotate_color_axis(axh, "x")
    # narrow callout in the empty upper-left so it clears the central peaks
    if args.show_ks:
        note = f"KS D = {ks.statistic:.2f}\np = {ks.pvalue:.2g}"
    else:
        note = (f"median offset\n(sim - obs)\n"
                f"{np.median(sim)-np.median(obs):+.2f} mag")
    axh.annotate(note, xy=(0.03, 0.97), xycoords="axes fraction", fontsize=12.5,
                 va="top", bbox=NOTE_BBOX)
    axh.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_hist, bbox_inches="tight")
    fig.savefig(args.out_hist.replace(".pdf", ".png"), dpi=PNG_DPI,
                bbox_inches="tight")
    print(f"\nwrote {args.out_hist} (+ .png)")

    # ---- Figure 2: cumulative distribution (CDF) ----
    fig, axc = plt.subplots(figsize=(6.8, 5.1))
    series = [(obs, "0.40", "-", "observed"), (sim, "C3", "-", "simulated, dust")]
    if have_nd:
        series.append((sim_nd, "C0", "--", "simulated, nodust"))
    for data, c, ls, lab in series:
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        axc.step(x, y, where="post", color=c, lw=2.4, ls=ls, label=lab)
    axc.set_xlabel("B - K (Vega)")
    axc.set_ylabel("cumulative fraction")
    axc.set_title("B$-$K cumulative distribution: ROMULUS vs. observed nearby galaxies")
    annotate_color_axis(axc, "x")
    # upper-left of an increasing CDF is empty space
    if args.show_ks:
        note = f"KS D = {ks.statistic:.2f}\np = {ks.pvalue:.2g}"
    else:
        note = (f"median offset\n(sim - obs)\n"
                f"{np.median(sim)-np.median(obs):+.2f} mag")
    axc.annotate(note, xy=(0.03, 0.97), xycoords="axes fraction", fontsize=12.5,
                 va="top", bbox=NOTE_BBOX)
    axc.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_cdf, bbox_inches="tight")
    fig.savefig(args.out_cdf.replace(".pdf", ".png"), dpi=PNG_DPI,
                bbox_inches="tight")
    print(f"wrote {args.out_cdf} (+ .png)")


if __name__ == "__main__":
    main()