#!/usr/bin/env python3
"""
build_figures.py
----------------
Color-mass diagram + black-hole split, joining the simulated colors to the
halo census and the clean observed sample.

Inputs (defaults, all in cwd):
  sim_colors.csv       halo, BK (dust), BK_nodust, ...
  proposal_census.csv  name, Mstar_Msol, N_bh, M_bh_Msol, MHI_Msol, ...
  catalog_dered.csv    observed; clean = ~deredden_suspect & ebv_sfd finite;
                       color = BK_dered, log stellar mass = log_mass_26

Outputs:
  color_mass_diagram.pdf/.png   (the headline: B-K vs log M*, controls for mass)
  bh_split.pdf/.png             (sim with-BH vs no-BH B-K, jittered strip)
  bh_split_hist.pdf/.png        (same split as histograms -- see note below)
and prints the with-BH / no-BH statistics with an explicit small-n caveat.

The CMD uses the DUST (observable) colors against the observed sample, and
overplots the nodust (intrinsic) colors with a thin connector so the dust
reddening per halo is visible.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
# translucent backing so in-plot stat labels stay legible over bars/lines/points
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
        ax.annotate("redder \u2192", xy=(-0.115, 1.0), xycoords="axes fraction",
                    ha="right", va="top", color=rc, rotation=90, **kw)
        ax.annotate("\u2190 bluer", xy=(-0.115, 0.0), xycoords="axes fraction",
                    ha="right", va="bottom", color=bc, rotation=90, **kw)


def load_sim(sim_path, census_path):
    sim = pd.read_csv(sim_path)
    cen = pd.read_csv(census_path)
    df = sim.merge(cen[["name", "Mstar_Msol", "N_bh", "M_bh_Msol"]],
                   left_on="halo", right_on="name", how="left")
    missing = df.loc[df["Mstar_Msol"].isna(), "halo"].tolist()
    if missing:
        print(f"WARNING: no census row for {missing} -> dropped from CMD")
    df = df.dropna(subset=["Mstar_Msol"])
    df["logM"] = np.log10(df["Mstar_Msol"])
    df["has_bh"] = df["N_bh"].fillna(0) > 0
    return df


def load_observed(path):
    o = pd.read_csv(path)
    clean = o[~o["deredden_suspect"].astype(bool) & o["ebv_sfd"].notna()].copy()
    return clean.dropna(subset=["log_mass_26", "BK_dered"])


def color_mass_diagram(sim, obs, out):
    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    ax.scatter(obs["log_mass_26"], obs["BK_dered"], s=14, c="0.72",
               alpha=0.55, edgecolors="none", zorder=1,
               label=f"observed (n={len(obs)})")
    # observed running median in mass bins
    bins = np.linspace(obs["log_mass_26"].min(), obs["log_mass_26"].max(), 7)
    bc, med = [], []
    for i in range(len(bins) - 1):
        sel = (obs["log_mass_26"] >= bins[i]) & (obs["log_mass_26"] < bins[i + 1])
        if sel.sum() >= 3:
            bc.append(0.5 * (bins[i] + bins[i + 1]))
            med.append(np.median(obs.loc[sel, "BK_dered"]))
    ax.plot(bc, med, "-", color="0.35", lw=2.4, zorder=2, label="observed median")

    # dust -> nodust connector per halo (shows the attenuation each halo gets)
    if "BK_nodust" in sim.columns:
        for _, r in sim.iterrows():
            if np.isfinite(r["BK_nodust"]):
                ax.plot([r["logM"], r["logM"]], [r["BK_nodust"], r["BK"]],
                        color="0.6", lw=0.9, zorder=2)
        ax.scatter(sim["logM"], sim["BK_nodust"], s=30, marker="o",
                   facecolors="none", edgecolors="0.5", zorder=3,
                   label="sim, intrinsic (nodust)")

    nob, wbh = sim[~sim["has_bh"]], sim[sim["has_bh"]]
    ax.scatter(nob["logM"], nob["BK"], s=95, marker="o", facecolor="C0",
               edgecolor="k", lw=0.7, zorder=4, label=f"sim, no BH (n={len(nob)})")
    ax.scatter(wbh["logM"], wbh["BK"], s=150, marker="*", facecolor="C3",
               edgecolor="k", lw=0.7, zorder=4, label=f"sim, BH (n={len(wbh)})")

    ax.set_xlabel(r"$\log_{10}\,M_\star\ [M_\odot]$")
    ax.set_ylabel("B - K (Vega)")
    ax.set_title("Color$-$mass diagram: ROMULUS halos vs. observed nearby galaxies")
    # blue/red is already spent on no-BH/BH in the legend, so use the grey cue
    annotate_color_axis(ax, "y", palette="neutral")
    # the connectors span each halo's dust-treatment systematic
    if "BK_nodust" in sim.columns:
        ax.annotate("vertical bars = dust-threshold systematic\n"
                    "(no dust  \u2192  30,000 K dust)",
                    xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=11, color="0.4",
                    bbox=NOTE_BBOX)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=PNG_DPI, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def _bh_stats(a, b, label):
    """Print with-BH vs no-BH stats for one color column; return the KS result."""
    print(f"\nBH split [{label}]:")
    print(f"  with-BH n={len(a)}  median {np.median(a):.2f}")
    print(f"  no-BH   n={len(b)}  median {np.median(b):.2f}")
    print(f"  median difference (BH - noBH): {np.median(a) - np.median(b):+.2f}")
    ks = stats.ks_2samp(a, b)
    print(f"  KS: D={ks.statistic:.3f}  p={ks.pvalue:.3g}")
    try:
        perm = stats.permutation_test(
            (a, b), lambda x, y: np.median(x) - np.median(y),
            permutation_type="independent", n_resamples=20000,
            alternative="two-sided", random_state=0)
        print(f"  permutation test on median diff: p={perm.pvalue:.3g}")
    except Exception as e:
        print(f"  permutation test skipped: {type(e).__name__}: {e}")
    return ks


def _bh_panel(ax, a, b, ylabel, title):
    """Draw one jittered strip panel (no-BH then BH) with median bars."""
    rng = np.random.default_rng(0)
    for i, (grp, c) in enumerate([(b, "C0"), (a, "C3")]):
        x = i + rng.uniform(-0.07, 0.07, len(grp))
        ax.scatter(x, grp, s=90, c=c, edgecolor="k", lw=0.7, zorder=3)
        ax.hlines(np.median(grp), i - 0.2, i + 0.2, color="k", lw=2.8, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"no BH\n(n={len(b)})", f"BH\n(n={len(a)})"])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, 1.5)


def bh_split(sim, out, show_ks=False):
    # one panel per available color: dust (observable) and, if present, nodust
    cols = [("BK", "dust (observable)")]
    if "BK_nodust" in sim.columns and sim["BK_nodust"].notna().any():
        cols.append(("BK_nodust", "intrinsic (no dust)"))

    a0 = sim.loc[sim["has_bh"], "BK"].dropna().values
    b0 = sim.loc[~sim["has_bh"], "BK"].dropna().values
    if a0.size == 0 or b0.size == 0:
        print(f"\nBH split: one group is empty (with-BH={a0.size}, no-BH={b0.size}); "
              f"no split test possible.")
        return

    panels = []
    for col, title in cols:
        a = sim.loc[sim["has_bh"], col].dropna().values
        b = sim.loc[~sim["has_bh"], col].dropna().values
        ks = _bh_stats(a, b, title)
        panels.append((a, b, title, ks))
    print(f"  NOTE: n={a0.size}+{b0.size} is small; this test has low power and a "
          f"non-detection is not evidence of 'no difference'.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6),
                             sharey=True, squeeze=False)
    for j, (ax, (a, b, title, ks)) in enumerate(zip(axes[0], panels)):
        _bh_panel(ax, a, b, "B - K (Vega)" if j == 0 else "", title)
        # color-sense cue only on the leftmost panel (shared y); neutral palette
        # because blue/red here already encode no-BH/BH, not galaxy color.
        if j == 0:
            annotate_color_axis(ax, "y", palette="neutral")
        if show_ks:
            note = f"KS p = {ks.pvalue:.2f}"
        else:
            note = f"BH - no BH:\n{np.median(a) - np.median(b):+.2f} mag"
        ax.annotate(note, xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=13, va="top", bbox=NOTE_BBOX)
    fig.suptitle("Black-hole color split (ROMULUS): with-BH vs. no-BH",
                 fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=PNG_DPI, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def bh_split_hist(sim, out, show_ks=False):
    """Histogram version of the BH split (mentor request, for comparison).

    Caveat: with n=4 (no BH) and ~11 (BH), a histogram is dominated by binning
    and the two groups have very different counts, so the bars should not be
    over-read. Median lines are drawn for reference. The jittered strip
    (bh_split) is usually the clearer display at this n; this is provided side
    by side for comparison.
    """
    cols = [("BK", "dust (observable)")]
    if "BK_nodust" in sim.columns and sim["BK_nodust"].notna().any():
        cols.append(("BK_nodust", "intrinsic (no dust)"))

    a0 = sim.loc[sim["has_bh"], "BK"].dropna().values
    b0 = sim.loc[~sim["has_bh"], "BK"].dropna().values
    if a0.size == 0 or b0.size == 0:
        print(f"\nBH split (hist): one group empty; skipped.")
        return

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.6),
                             sharey=True, squeeze=False)
    for j, (ax, (col, title)) in enumerate(zip(axes[0], cols)):
        a = sim.loc[sim["has_bh"], col].dropna().values
        b = sim.loc[~sim["has_bh"], col].dropna().values
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        bins = np.linspace(lo - 0.05, hi + 0.05, 8)   # shared bins, ~0.3 mag wide
        ax.hist(b, bins=bins, histtype="step", lw=2.6, color="C0",
                label=f"no BH (n={len(b)})")
        ax.hist(a, bins=bins, histtype="step", lw=2.6, color="C3",
                label=f"BH (n={len(a)})")
        ax.axvline(np.median(b), color="C0", ls="--", lw=1.6)
        ax.axvline(np.median(a), color="C3", ls="--", lw=1.6)
        ax.set_xlabel("B - K (Vega)")
        if j == 0:
            ax.set_ylabel("count")
        ax.set_title(title)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are whole
        annotate_color_axis(ax, "x", palette="neutral")
        ks = stats.ks_2samp(a, b)
        note = (f"KS p = {ks.pvalue:.2f}" if show_ks
                else f"BH - no BH: {np.median(a) - np.median(b):+.2f} mag")
        ax.legend(frameon=False, loc="upper right", title=note, title_fontsize=12)
    fig.suptitle("Black-hole color split (ROMULUS)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=PNG_DPI, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="sim_colors.csv")
    ap.add_argument("--census", default="proposal_census.csv")
    ap.add_argument("--observed", default="catalog_dered.csv")
    ap.add_argument("--show-ks", action="store_true",
                    help="annotate the BH split with KS p (for talks); "
                         "default shows the median color difference")
    args = ap.parse_args()

    sim = load_sim(args.sim, args.census)
    obs = load_observed(args.observed)
    print(f"sim halos with mass: {len(sim)}  "
          f"(with-BH={int(sim['has_bh'].sum())}, no-BH={int((~sim['has_bh']).sum())})")
    print(f"observed clean: {len(obs)}")

    color_mass_diagram(sim, obs, "color_mass_diagram.pdf")
    bh_split(sim, "bh_split.pdf", show_ks=args.show_ks)
    bh_split_hist(sim, "bh_split_hist.pdf", show_ks=args.show_ks)


if __name__ == "__main__":
    main()