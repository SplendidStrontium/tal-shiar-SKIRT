#!/usr/bin/env python3
"""
build_enterprise_figures.py
---------------------------
Money-shot figures for the Enterprise twin-study, from
enterprise_colors.csv (+ census CSVs for masses and dust reservoirs).

Figures (each written as PDF vector + 200 dpi PNG):

  fig_enterprise_slopegraph            face-on dust B-K, two panels
                                       SHARING a y-axis (the observable:
                                       levels are comparable). Result
                                       figure; carries the pair deltas.
  fig_enterprise_dust_reddening_addBH  noBH<->BH reddening, standalone,
                                       own y-axis.
  fig_enterprise_dust_reddening_feedback
                                       BH6<->BH8 reddening, standalone,
                                       own y-axis -- lets the 0-0.2
                                       range fill the frame so the
                                       dust-ratio annotations breathe.
  fig_enterprise_mass_color_context    20 anonymous points; context
                                       slide; Spearman rho in corner.

v4 changes (figure review round 3):
  * Dust-reddening split into TWO standalone files with independent
    y-ranges. Rationale (reviewer's, and correct): the content is the
    within-pair change; the panels' accretion physics differs, so
    cross-panel level comparison isn't a slide's job, and the shared
    axis was crushing the feedback panel's readability.
  * Slopegraph intentionally unchanged: for the OBSERVABLE, levels are
    comparable and meaningful across panels (r488 sitting bluer on the
    right is real information), so it keeps the shared axis -- and its
    place in the deck is still under review (result vs mechanism
    division of labor; see conversation notes).

Standing decisions:
  * No connector between clean pairs ever crosses the confounded
    BH<->BH6 boundary.
  * Family labels are names only; mass ordering lives in the viridis
    ramp, stated once in the caption.
  * Pair deltas live in titles (reliably empty space), spelled out.
  * Dust-ratio annotations ("dust x0.4") from the extended census tie
    reddening changes to reservoir changes; skipped gracefully if the
    extended census is absent.
  * '\u2192' renders UP after 90 deg label rotation, '\u2190' DOWN.

Usage:
    python build_enterprise_figures.py
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
EXT_CENSUS_CSV = BASE / "enterprise_census_extended.csv"

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]

PAIRS = [(("noBH", "BH"), "adding a black hole (restricted accretion)"),
         (("BH6", "BH8"), "feedback efficiency 0.05 \u2192 0.005 "
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


def load_dust_proxy():
    if not EXT_CENSUS_CSV.exists():
        print(f"  NOTE: {EXT_CENSUS_CSV} not found; "
              f"dust-ratio annotations skipped")
        return None
    d = pd.read_csv(EXT_CENSUS_CSV)

    def split(run):
        return run.split("_", 1) if "_" in run else (run, "noBH")
    d["family"] = d.run.map(lambda r: split(r)[0])
    d["variant"] = d.run.map(lambda r: split(r)[1])
    return (d.pivot_table(index="family", columns="variant",
                          values="mdust_proxy")
              .reindex(index=FAMILIES, columns=VARIANTS))


def family_colors(mstar_fam):
    order = sorted(FAMILIES, key=lambda f: mstar_fam[f], reverse=True)
    cmap = plt.get_cmap("viridis")
    shades = {f: cmap(0.10 + 0.75 * i / (len(order) - 1))
              for i, f in enumerate(order)}
    return shades, order


def save(fig, outdir, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI_PNG})):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        print(f"  wrote {p}")
    plt.close(fig)


def dodge(ys, min_sep):
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    out = ys.copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_sep:
            out[b] = out[a] + min_sep
    return out


def ratio_label(r):
    if not np.isfinite(r) or r <= 0:
        return ""
    if r >= 10:
        return f"\u00d7{r:.0f}"
    return f"\u00d7{r:.2g}"


def mass_caption(mstar_fam):
    mmax = max(mstar_fam.values()) / 1e8
    mmin = min(mstar_fam.values()) / 1e8
    return ("line shade: darker = more massive family "
            f"(M\u2217 = {mmax:.0f}\u00d710\u2078 \u2192 "
            f"{mmin:.1f}\u00d710\u2078 M\u2299)")


def draw_pair(ax, pivot, v0, v1, shades, mstar_fam, min_sep,
              seg_annot=None, annot_fontsize=10, annot_prefix="dust"):
    """Draw one matched pair on ax: lines, points, right-edge dodged
    family labels, optional midpoint dust-ratio annotations."""
    y0 = pivot[v0].loc[FAMILIES].to_numpy(dtype=float)
    y1 = pivot[v1].loc[FAMILIES].to_numpy(dtype=float)

    for f, a, b in zip(FAMILIES, y0, y1):
        c = shades[f]
        ax.plot([0, 1], [a, b], "-", color=c, lw=2.4, zorder=3)
        ax.plot([0, 1], [a, b], "o", color=c, ms=7, zorder=4,
                linestyle="none")

    def edge_text(f):
        if seg_annot is None:
            return f
        den = seg_annot.loc[f, v0]
        lab = ratio_label(seg_annot.loc[f, v1] / den) if den else ""
        return f"{f}  \u00b7  {annot_prefix} {lab}" if lab else f

    ylab = dodge(y1, min_sep)
    for f, yl in zip(FAMILIES, ylab):
        ax.annotate(edge_text(f), xy=(1, yl), xytext=(10, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=11, color=shades[f])

    ax.set_xticks([0, 1])
    ax.set_xticklabels([v0, v1])
    return y1 - y0

def fig_delta_result(df, mstar_fam, outdir):
    """The result figure: twin-pair color deltas hugging zero."""
    pivot = face_on(df, "dust")
    shades, order = family_colors(mstar_fam)
    row_labels = ["adding a black hole\n(noBH \u2192 BH)",
                  "feedback \u00d710 weaker\n(BH6 \u2192 BH8)"]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for row, ((v0, v1), _) in enumerate(PAIRS):
        y = 1 - row
        d = (pivot[v1] - pivot[v0]).loc[FAMILIES].to_numpy(dtype=float)
        for f, x in zip(FAMILIES, d):
            ax.plot(x, y + 0.13, "o", ms=9, color=shades[f], zorder=4,
                    linestyle="none")
        ax.errorbar(d.mean(), y - 0.13, xerr=d.std(ddof=1), fmt="D",
                    color="black", ms=7, capsize=4, lw=1.6, zorder=5)
        ax.annotate(f"{d.mean():+.2f} \u00b1 {d.std(ddof=1):.2f}",
                    xy=(d.mean(), y - 0.30), ha="center", va="top",
                    fontsize=10, color=GREY)
    ax.axvline(0, color="0.25", lw=1.2, zorder=1)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_ylim(-0.55, 1.5)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xlabel("\u0394(B\u2212K) between twins  [mag]\n"
                  "bluer \u2190 0 \u2192 redder", fontsize=11)
    ax.set_title("Changing BH physics barely moves the color")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", ms=7,
                      color=shades[f], label=f) for f in order]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left",
              handletextpad=0.2, borderaxespad=0.2)
    ax.annotate("colored dots: one twin pair per family \u00b7 "
                "black: mean \u00b1 std across families",
                xy=(0.5, -0.32), xycoords="axes fraction",
                ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, outdir, "fig_enterprise_delta_result")


# ---------------------------------------------------------------------------
# Figure 1: slopegraph of the observable (shared y-axis, two panels)
# ---------------------------------------------------------------------------

def fig_slopegraph(df, mstar_fam, outdir):
    pivot = face_on(df, "dust")
    shades, order = family_colors(mstar_fam)

    vals = pivot.loc[FAMILIES, VARIANTS].to_numpy(dtype=float)
    pad = 0.07 * (vals.max() - vals.min())
    ylo, yhi = vals.min() - pad, vals.max() + pad
    min_sep = 0.05 * (yhi - ylo)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.6), sharey=True)
    for ax, ((v0, v1), knob) in zip(axes, PAIRS):
        delta = draw_pair(ax, pivot, v0, v1, shades, mstar_fam, min_sep)
        ax.set_title(f"{knob}\n\u27e8\u0394(B\u2212K)\u27e9 = "
                     f"{delta.mean():+.2f} \u00b1 "
                     f"{delta.std(ddof=1):.2f} mag", fontsize=11.5)
        ax.set_xlim(-0.20, 1.65)

    axes[0].set_ylim(ylo, yhi)
    axes[0].set_ylabel("\u2190 bluer      B\u2212K (Vega)  [mag]      "
                       "redder \u2192")
    fig.suptitle("Same initial conditions, different BH physics "
                 "(face-on, with dust)", fontsize=15, y=0.99)
    fig.text(0.5, 0.015,
             mass_caption(mstar_fam) + " \u00b7 panels share the y-axis; "
             "accretion physics differs between panels \u2014 compare "
             "levels, not trajectories",
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.13, right=0.93,
                        wspace=0.12)
    save(fig, outdir, "fig_enterprise_slopegraph")


# ---------------------------------------------------------------------------
# Figures 2a/2b: dust reddening, one standalone figure per pair
# ---------------------------------------------------------------------------

def fig_dust_reddening_single(df, mstar_fam, outdir, pair_idx, stem):
    pivot = face_on(df, "dust") - face_on(df, "nodust")
    shades, order = family_colors(mstar_fam)
    (v0, v1), knob = PAIRS[pair_idx]

    # Independent y-range from THIS pair's data only
    vals = pivot[[v0, v1]].loc[FAMILIES].to_numpy(dtype=float)
    pad = 0.10 * (vals.max() - vals.min())
    ylo = min(vals.min() - pad, -0.005)
    yhi = vals.max() + pad
    min_sep = 0.06 * (yhi - ylo)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    delta = draw_pair(ax, pivot, v0, v1, shades, mstar_fam, min_sep,
                      seg_annot=load_dust_proxy(), annot_fontsize=10)
    ax.set_ylim(ylo, yhi)
    ax.axhline(0.0, color=GREY, lw=0.8, zorder=1)
    ax.set_xlim(-0.20, 2.55)
    ax.set_title(f"{knob}\n\u27e8\u0394 reddening\u27e9 = "
                 f"{delta.mean():+.2f} \u00b1 "
                 f"{delta.std(ddof=1):.2f} mag", fontsize=12)
    ax.set_ylabel("dust reddening of B\u2212K  [mag]\n"
                  "(dust \u2212 nodust, face-on)")

    fig.suptitle("How much dust reddens each galaxy", fontsize=15, y=0.99)
    fig.text(0.5, 0.050, mass_caption(mstar_fam),
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.text(0.5, 0.015,
             "\u201cdust \u00d7N\u201d = change in the cold-gas dust reservoir",
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.17, right=0.90)

    save(fig, outdir, stem)


def fig_dust_reddening(df, mstar_fam, outdir):
    fig_dust_reddening_single(df, mstar_fam, outdir, 0,
                              "fig_enterprise_dust_reddening_addBH")
    fig_dust_reddening_single(df, mstar_fam, outdir, 1,
                              "fig_enterprise_dust_reddening_feedback")


# ---------------------------------------------------------------------------
# Figure 3: anonymous mass-color context
# ---------------------------------------------------------------------------

VARIANT_MARKERS = {"noBH": ("o", "none"),   # kept: explorer imports this
                   "BH":   ("o", "full"),
                   "BH6":  ("s", "full"),
                   "BH8":  ("D", "full")}


def fig_mass_color_context(df, mstar_by_run, outdir):
    """20 anonymous simulated dwarfs: the known mass-color relation.
    Speaker-notes honesty: part of this correlation is created by the
    feedback physics itself, so present as 'our sample follows the
    known relation', not independent confirmation of it."""
    pivot = face_on(df, "dust")

    xs, ys = [], []
    for f in FAMILIES:
        for v in VARIANTS:
            run = f if v == "noBH" else f"{f}_{v}"
            if run not in mstar_by_run:
                print(f"  WARNING: {run} missing from census; skipped")
                continue
            xs.append(np.log10(mstar_by_run[run]))
            ys.append(pivot.loc[f, v])
    xs, ys = np.asarray(xs), np.asarray(ys)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(xs, ys, "o", ms=9, color="0.25", alpha=0.85, linestyle="none")

    rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    ax.annotate(f"20 simulated dwarfs",
                xy=(0.03, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=11, color=GREY)
    ## \u00b7 Spearman \u03c1 = {rho:+.2f}

    ax.set_xlabel("log\u2081\u2080  M\u2217 / M\u2299")
    ax.set_ylabel("\u2190 bluer      B\u2212K (Vega)  [mag]      redder \u2192")
    ax.set_title("More massive dwarfs are redder")
    fig.tight_layout()
    save(fig, outdir, "fig_enterprise_mass_color_context")

# ---------------------------------------------------------------------------
# Figure 4: analysis of feedback in bh6 to bh8
# ---------------------------------------------------------------------------

def fig_bh_selfregulation(mstar_fam, outdir):
    """First link of the feedback-pair causal chain: weaker per-event
    feedback -> BHs self-regulate to higher mass. Every line rises."""
    cen = pd.read_csv(CENSUS_CSV)
    main = cen[cen.halo_id == 1].copy()

    def split(run):
        return run.split("_", 1) if "_" in run else (run, "noBH")
    main["family"] = main.run.map(lambda r: split(r)[0])
    main["variant"] = main.run.map(lambda r: split(r)[1])
    piv = (main.pivot_table(index="family", columns="variant",
                            values="mbh_max")
               .reindex(index=FAMILIES, columns=VARIANTS))
    logp = np.log10(piv[["BH6", "BH8"]])

    shades, order = family_colors(mstar_fam)
    vals = logp[["BH6", "BH8"]].loc[FAMILIES].to_numpy(dtype=float)
    pad = 0.10 * (vals.max() - vals.min())
    ylo, yhi = vals.min() - pad, vals.max() + pad
    min_sep = 0.06 * (yhi - ylo)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    # temporarily point PAIRS[1] machinery at the BH-mass pivot,
    # with growth-factor edge labels

    delta = draw_pair(ax, logp, "BH6", "BH8", shades, mstar_fam, min_sep,
                      seg_annot=piv, annot_fontsize=10,
                      annot_prefix="grew")
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.20, 1.65)
    ax.set_title("feedback efficiency 0.05 \u2192 0.005\n", fontsize=12)
    """
    f"\u27e8\u0394 log\u2081\u2080 M\u2099\u2095\u27e9 = ",
    f"{delta.mean():+.2f} \u00b1 {delta.std(ddof=1):.2f} dex "
    f"(\u2248\u00d7{10**delta.mean():.0f})"
    """
    ax.set_ylabel(r"$\log_{10}\; M_{\mathrm{BH}} / M_\odot$   (most massive BH)")
    ## main halo
    fig.suptitle("Weaker feedback \u2192 bigger black holes",
                 fontsize=16, y=0.95)
    fig.text(0.5, 0.015,
         r"“grew ×N” = $M_{\mathrm{BH}}$ ratio BH8/BH6",
         ha="center", va="bottom", fontsize=9.5, color=GREY)
    ## self-regulation: BHs grow until their feedback compensates
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.15, right=0.97)
    save(fig, outdir, "fig_enterprise_bh_selfregulation")

# ---------------------------------------------------------------------------
# Figure 5: why color doesn't budge, show effect on stars vs. dust
# ---------------------------------------------------------------------------

def fig_decomposition_single(df, mstar_fam, outdir, pair_idx, stem):
    """One clean pair: each twin color change split into intrinsic
    (stellar) and dust contributions.
      solid bar   = intrinsic = \u0394(nodust B\u2212K)
      hatched bar = dust      = \u0394(reddening)
      black tick  = total     = \u0394(dust B\u2212K) = solid + hatched
    Independent y-range from THIS pair's data only, zero always
    included (same rationale as the reddening split: the content is
    the within-pair decomposition, not cross-pair levels)."""
    p_dust = face_on(df, "dust")
    p_nod = face_on(df, "nodust")
    shades, order = family_colors(mstar_fam)
    (v0, v1), knob = PAIRS[pair_idx]

    d_tot = (p_dust[v1] - p_dust[v0]).loc[FAMILIES].to_numpy(dtype=float)
    d_int = (p_nod[v1] - p_nod[v0]).loc[FAMILIES].to_numpy(dtype=float)
    d_dust = d_tot - d_int

    allv = np.concatenate([d_tot, d_int, d_dust, [0.0]])
    pad = 0.12 * (allv.max() - allv.min())
    ylo, yhi = allv.min() - pad, allv.max() + pad

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    w = 0.32
    xs = np.arange(len(FAMILIES))
    for x, f, di, dd, dt in zip(xs, FAMILIES, d_int, d_dust, d_tot):
        ax.bar(x - w / 2, di, width=w, color=shades[f], zorder=3)
        ax.bar(x + w / 2, dd, width=w, color=shades[f], alpha=0.45,
               hatch="//", edgecolor=shades[f], zorder=3)
        ax.plot(x, dt, "_", color="black", ms=16, mew=2.2, zorder=5)
    ax.axhline(0, color="0.25", lw=1.0, zorder=1)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(xs)
    ax.set_xticklabels(FAMILIES, fontsize=11)
    ax.set_title(f"{knob}\n\u27e8\u0394(B\u2212K)\u27e9 = "
                 f"{d_tot.mean():+.2f} \u00b1 "
                 f"{d_tot.std(ddof=1):.2f} mag", fontsize=12)
    ax.set_ylabel("\u0394(B\u2212K) between twins  [mag]")
    ax.annotate("solid: intrinsic (stars) \u00b7 hatched: dust \u00b7 "
                "black tick: total",
                xy=(0.02, 0.97), xycoords="axes fraction",
                va="top", fontsize=9.5, color=GREY)

    fig.suptitle("Why the color barely moves: stars vs. dust "
                 "contributions", fontsize=15, y=0.99)
    fig.text(0.5, 0.015,
             "bar shade: darker = more massive family \u00b7 "
             "intrinsic + dust = total by construction",
             ha="center", va="bottom", fontsize=9.5, color=GREY)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.14, right=0.97)
    save(fig, outdir, stem)


def fig_decomposition(df, mstar_fam, outdir):
    fig_decomposition_single(df, mstar_fam, outdir, 0,
                             "fig_enterprise_decomposition_addBH")
    fig_decomposition_single(df, mstar_fam, outdir, 1,
                             "fig_enterprise_decomposition_feedback")

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=BASE / "products" / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_colors()
    mstar_by_run, mstar_fam = load_masses()

    print("Building figures ->", args.outdir)
    fig_slopegraph(df, mstar_fam, args.outdir)
    fig_dust_reddening(df, mstar_fam, args.outdir)
    fig_delta_result(df, mstar_fam, args.outdir)
    fig_mass_color_context(df, mstar_by_run, args.outdir)
    fig_bh_selfregulation(mstar_fam, args.outdir)
    fig_decomposition(df, mstar_fam, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()