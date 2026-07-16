"""
build_color_hist.py

this script is used to build a histogram of colors to compare the value given by ROMULUS Tangos and the color produced by SKIRT.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------- read ----------

comp = pd.read_csv("rom_vs_skirt_colors.csv")
# only take the rows where ROM and SKIRT colors exist
matched = comp[comp["source"] == "both"]

# ------- bins ------------

# ONE set of bin edges, shared by both histograms.
lo = min(matched["rom_color"].min(), matched["skirt_color"].min())
hi = max(matched["rom_color"].max(), matched["skirt_color"].max())
bins = np.linspace(lo - 0.1, hi + 0.1, 13)   # 13 edges = 12 bins

# ------ draw -------------

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.hist(matched["rom_color"], bins=bins, alpha=0.6,
        label="ROMULUS intrinsic", color="tab:blue")
ax.hist(matched["skirt_color"], bins=bins, alpha=0.6,
        label="SKIRT (no dust)", color="tab:orange")

# ------ label and save ---------

ax.set_xlabel("B - K color [mag]\n← bluer                    redder →")
ax.set_ylabel("Number of halos")
ax.set_title(f"ROMULUS vs SKIRT color, {len(matched)} matched halos")
ax.legend()

fig.tight_layout()
fig.savefig("bk_rom_vs_skirt_hist.png", dpi=200)
