#!/usr/bin/env python3
"""
dust_threshold_diagnostic.py

Answers one question before we touch the enterprise ski files:
how much of the (metal-traced) dust mass do we keep or lose as a
function of the gas temperature threshold?

For every enterprise run (halo 1 only), computes

    M_dust(<T) = DUST_FRACTION * sum( m_gas * Z_gas )  over gas with
                 T < threshold and r < 30 kpc,

for a ladder of thresholds spanning the NIHAO convention (8e3 K)
through the Tal Shiar choice (3e4 K) and beyond. Reports each as a
fraction of the no-temperature-cut metal mass in the same aperture.

Interpretation guide (from the plan):
  * curve flat between 1e4 and 3e4  -> threshold doesn't matter;
    document the choice and move on.
  * substantial mass enters between 8e3 and 3e4 -> threshold matters;
    bracket it in SKIRT (one halo at two thresholds, measure dB-K)
    and report as a bracketed systematic, Tal Shiar style.

Same discovery conventions as census_enterprise.py: glob the output
number, never trust achOutName; BHs are star-family tform < 0 (not
relevant here -- we only touch gas -- but stated for the record).

Usage (Hamilton, usual conda env):
    python dust_threshold_diagnostic.py                # all 20 runs
    python dust_threshold_diagnostic.py --only r488    # one family
    python dust_threshold_diagnostic.py --no-plot      # CSV only
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless Hamilton
import matplotlib.pyplot as plt

import pynbody

# ----------------------------------------------------------------- config
BASE = "/mnt/data0/pkrsnak/romulus/enterprise"
SNAP_GLOB = "*.004096"
OUT_CSV = os.path.join(BASE, "dust_threshold_scan.csv")
OUT_FIG = os.path.join(BASE, "dust_threshold_scan")   # .png / .pdf appended

HALO_ID = 1                      # major halo per snapshot
APERTURE_KPC = 30.0              # matches make_particles / Tal Shiar chain
DUST_FRACTION = 0.4              # dust-to-metals, NIHAO mainstream

# Threshold ladder: NIHAO convention -> Tal Shiar choice -> beyond.
THRESHOLDS_K = [8e3, 1e4, 1.5e4, 2e4, 3e4, 5e4, 1e5]

# Highlight these two in the console summary (the actual decision).
T_LO, T_HI = 8e3, 3e4

VARIANT_ORDER = ["noBH", "BH", "BH6", "BH8"]
VARIANT_COLOR = {"noBH": "0.35", "BH": "tab:blue",
                 "BH6": "tab:orange", "BH8": "tab:red"}


# ------------------------------------------------------------- discovery
# (mirrors census_enterprise.py so the two scripts agree on what a "run" is)

def discover_runs(base, only=None):
    runs = []
    for d in sorted(glob.glob(os.path.join(base, "r*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        parts = name.split("_", 1)
        family = parts[0]
        variant = parts[1] if len(parts) > 1 else "noBH"
        if only and family != only:
            continue
        runs.append(dict(path=d, run=name, family=family, variant=variant))
    return runs


def find_snapshot(run_dir):
    hits = [p for p in glob.glob(os.path.join(run_dir, SNAP_GLOB))
            if os.path.isfile(p)]
    if len(hits) == 1:
        return hits[0]
    print(f"  !! {os.path.basename(run_dir)}: "
          f"{len(hits)} files match {SNAP_GLOB} -- skipping.")
    return None


# ------------------------------------------------------------------ scan
def scan_run(run):
    """Return list of row dicts (one per threshold) for one run."""
    snap_path = find_snapshot(run["path"])
    if snap_path is None:
        return []

    print(f"  loading {os.path.basename(snap_path)} ...")
    s = pynbody.load(snap_path)
    s.physical_units()

    try:
        halo = s.halos()[HALO_ID]
    except Exception as e:
        print(f"  !! {run['run']}: no halo catalog / halo {HALO_ID} ({e})")
        return []

    if len(halo.g) == 0:
        print(f"  !! {run['run']}: halo {HALO_ID} has no gas -- skipping.")
        return []

    # Center on the FULL halo (DM included -- it defines the potential
    # well), then measure gas radii relative to that center without
    # mutating the snapshot.
    try:
        cen = np.asarray(pynbody.analysis.halo.center(
            halo, mode="ssc", retcen=True))
    except Exception:
        m = halo["mass"]
        cen = np.asarray((halo["pos"] * m[:, None]).sum(axis=0) / m.sum())

    gpos = np.asarray(halo.g["pos"].in_units("kpc"))
    r = np.linalg.norm(gpos - np.asarray(cen), axis=1)
    in_ap = r < APERTURE_KPC

    gmass = np.asarray(halo.g["mass"].in_units("Msol"))[in_ap]
    gtemp = np.asarray(halo.g["temp"])[in_ap]          # K
    gZ = np.asarray(halo.g["metals"])[in_ap]           # metal mass fraction

    metal_mass = gmass * gZ
    m_metal_total = float(metal_mass.sum())            # no temp cut
    if m_metal_total <= 0:
        print(f"  !! {run['run']}: zero metal mass in aperture -- skipping.")
        return []

    rows = []
    for T in THRESHOLDS_K:
        cold = gtemp < T
        m_dust = DUST_FRACTION * float(metal_mass[cold].sum())
        rows.append(dict(
            family=run["family"], variant=run["variant"], run=run["run"],
            threshold_K=T,
            n_gas_below=int(cold.sum()),
            m_gas_below=float(gmass[cold].sum()),
            m_dust=m_dust,
            dust_frac=m_dust / (DUST_FRACTION * m_metal_total),
            m_metal_total_aperture=m_metal_total,
        ))

    f_lo = next(r_["dust_frac"] for r_ in rows if r_["threshold_K"] == T_LO)
    f_hi = next(r_["dust_frac"] for r_ in rows if r_["threshold_K"] == T_HI)
    print(f"  {run['run']}: dust kept at {T_LO:.0f} K = {f_lo:5.1%}, "
          f"at {T_HI:.0f} K = {f_hi:5.1%}  "
          f"(enters between the two: {f_hi - f_lo:5.1%})")
    return rows


# ------------------------------------------------------------------ plot
def make_plot(df, path_stem):
    fig, ax = plt.subplots(figsize=(7, 5))
    for run_name, sub in df.groupby("run"):
        sub = sub.sort_values("threshold_K")
        v = sub["variant"].iloc[0]
        ax.plot(sub["threshold_K"], sub["dust_frac"],
                color=VARIANT_COLOR.get(v, "k"), alpha=0.8, lw=1.5)
    # legend: one entry per variant, not per run
    for v in VARIANT_ORDER:
        ax.plot([], [], color=VARIANT_COLOR[v], lw=1.5, label=v)

    for T, label in [(T_LO, "NIHAO 8e3 K"), (T_HI, "Tal Shiar 3e4 K")]:
        ax.axvline(T, color="k", ls=":", lw=1, alpha=0.6)
        ax.text(T, 1.02, label, ha="center", va="bottom", fontsize=8,
                transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_xlabel("gas temperature threshold [K]")
    ax.set_ylabel("fraction of aperture dust mass retained")
    ax.set_ylim(0, 1.05)
    ax.legend(title="variant", fontsize=9)
    ax.set_title(f"Enterprise halo {HALO_ID}: dust mass vs. temperature cut "
                 f"(r < {APERTURE_KPC:.0f} kpc)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path_stem + ".png", dpi=200)
    fig.savefig(path_stem + ".pdf")
    print(f"Wrote {path_stem}.png / .pdf")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", metavar="FAMILY")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    runs = discover_runs(BASE, only=args.only)
    if not runs:
        sys.exit(f"No run directories found under {BASE}")
    print(f"Found {len(runs)} run directories.\n")

    all_rows = []
    for run in runs:
        print(f"[{run['run']}]")
        all_rows.extend(scan_run(run))

    if not all_rows:
        sys.exit("Nothing collected -- nothing written.")

    df = pd.DataFrame(all_rows)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER,
                                   ordered=True)
    df = df.sort_values(["family", "variant", "threshold_K"])
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")

    # Decision table: dust fraction retained at each threshold, wide format.
    wide = df.pivot_table(index="run", columns="threshold_K",
                          values="dust_frac", observed=True)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("\nFraction of aperture dust mass retained vs. threshold [K]:")
        print(wide.to_string())

    gap = wide[T_HI] - wide[T_LO]
    print(f"\nDust mass entering between {T_LO:.0f} K and {T_HI:.0f} K "
          f"(the decision gap):")
    print(f"  median {gap.median():.1%}   min {gap.min():.1%}   "
          f"max {gap.max():.1%}")

    if not args.no_plot:
        make_plot(df, OUT_FIG)


if __name__ == "__main__":
    main()