"""
build_color_hist.py

Overlapping histograms comparing ROMULUS Tangos values against SKIRT nodust
output, for three quantities: B-K color, absolute B, absolute Ks.

Both sides are dust-free and on an absolute scale, so they are like-for-like.
NOTE: the SKIRT side is Vega (pyphot zero-points, Johnson B - 2MASS Ks).
The ROMULUS/Tangos photometric system is unconfirmed -- see the figure note.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# --- config ---------------------------------------------------------------

# r239: SKIRT and ROMULUS disagree by ~2.8 mag in B and ~4.0 mag in Ks
# (a factor of ~14 in luminosity), while its B-K colour offset lands dead on
# the population mean. That is a "which stars are in the galaxy" problem, not
# a photometric one, so it is set aside here rather than left to dominate the
# per-band figures. Excluded loudly, not silently.
EXCLUDE = [239]

NBINS = 10          # ~14 halos per histogram; 8-12 is the sane range
FIG_NOTE = "SKIRT: Vega (Johnson B / 2MASS Ks)"

# --- read -----------------------------------------------------------------

comp = pd.read_csv("rom_vs_skirt_colors.csv")

# only rows where both a ROMULUS and a SKIRT value exist
matched = comp[comp["source"] == "both"]

dropped = matched[matched["HaloID"].isin(EXCLUDE)]
matched = matched[~matched["HaloID"].isin(EXCLUDE)]

print(f"matched halos: {len(matched)}")
for _, r in dropped.iterrows():
    print(f"  excluded halo {int(r['HaloID'])}  "
          f"(delta_B {r['delta_B']:+.2f}, delta_K {r['delta_K']:+.2f})")
print()

# --- one plotting function, called three times -----------------------------

def hist_pair(rom_vals, skirt_vals, xlabel, arrows, title, outfile, nbins=NBINS):
    """Overlapping ROMULUS/SKIRT histograms on ONE shared set of bin edges.

    Passing bins=<int> to each hist() call would let matplotlib choose edges
    separately per dataset, and the bars would no longer be comparable.
    """
    lo = min(rom_vals.min(), skirt_vals.min())
    hi = max(rom_vals.max(), skirt_vals.max())
    bins = np.linspace(lo - 0.1, hi + 0.1, nbins + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rom_vals, bins=bins, alpha=0.6,
            label="ROMULUS (Tangos)", color="tab:blue")
    ax.hist(skirt_vals, bins=bins, alpha=0.6,
            label="SKIRT (no dust)", color="tab:orange")

    ax.set_xlabel(f"{xlabel}\n{arrows}")
    ax.set_ylabel("Number of halos")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.legend()

    # state our own photometric system; make no claim about the other side
    ax.text(0.99, 0.97, FIG_NOTE, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    fig.savefig(outfile.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  wrote {outfile}  (bin width {bins[1] - bins[0]:.3f} mag)")


n = len(matched)

# Color: bigger B-K = redder.
hist_pair(matched["rom_color"], matched["skirt_color"],
          "B - K color [mag]",
          "\u2190 bluer                    redder \u2192",
          f"B - K color, {n} halos",
          "hist_bk_color.png")

# Magnitude: bigger = FAINTER. Opposite sense to the colour axis.
hist_pair(matched["rom_B"], matched["skirt_B"],
          "absolute B [mag]",
          "\u2190 brighter                    fainter \u2192",
          f"Absolute B, {n} halos",
          "hist_absmag_B.png")

hist_pair(matched["rom_K"], matched["skirt_K"],
          "absolute Ks [mag]",
          "\u2190 brighter                    fainter \u2192",
          f"Absolute Ks, {n} halos",
          "hist_absmag_Ks.png")

# --- numbers to go with the figures ---------------------------------------

print()
for lab, col in [("B - K ", "delta"), ("B band", "delta_B"), ("Ks band", "delta_K")]:
    v = matched[col]
    se = v.std() / np.sqrt(len(v))
    print(f"  delta {lab}: mean {v.mean():+6.3f} +/- {se:.3f}   median {v.median():+6.3f}")