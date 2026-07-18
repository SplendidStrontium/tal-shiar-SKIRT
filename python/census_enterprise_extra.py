#!/usr/bin/env python3
"""
census_enterprise_extra.py
--------------------------
Extend the Enterprise census with the quantities needed to test the three
candidate explanations for "suppressed star formation yet bluer":

    sfr_25myr    [Msun/yr]  stars formed in the last 25 Myr / 25 Myr
                            (matches the proposal's SFR convention)
    ssfr         [1/yr]     sfr_25myr / mstar_30kpc
    zstar_mw     [mass fr.] mass-weighted mean stellar metallicity
    mgas_cold    [Msun]     gas mass with T < 8,000 K inside the aperture
    mz_cold      [Msun]     metal mass in that cold gas
    mdust_proxy  [Msun]     0.4 * mz_cold -- up to the dust-to-metals
                            factor, this IS the dust SKIRT saw. If this
                            collapsed in BH6/BH8, the dust-reddening
                            table is mechanistically explained.
    mstar_30kpc  [Msun]     recomputed here for internal consistency of
                            ssfr (should track census mstar closely;
                            printed side by side as a cross-check)

Conventions carried over from census_enterprise.py -- do not drift:
  * Snapshot located by globbing {run}/*.004096. NEVER via achOutName.
  * Main halo = amiga halo 1 in every run (verified in the census).
  * BH particles live in the STAR family with tform < 0 (ChaNGa
    sentinel). They are excluded from ALL stellar quantities here by
    masking on sign of tform directly.
  * Aperture: 30 kpc sphere about the main halo center, matching the
    census stellar masses and the SKIRT particle cut.
  * Cold-gas threshold 8,000 K = the Camps & Trayford convention the
    Enterprise ski files use (NOT Tal Shiar's 30,000 K).

Output: enterprise_census_extended.csv in BASE -- the original census
main-halo rows merged (key-based, validated one-to-one) with the new
columns. The original census file is not touched.

Usage:
    python census_enterprise_extra.py            # all 20 runs
    python census_enterprise_extra.py --run r488_BH6   # one run, verbose
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pynbody

BASE = Path("/mnt/data0/pkrsnak/romulus/enterprise")
CENSUS_CSV = BASE / "enterprise_census.csv"
OUT_CSV = BASE / "enterprise_census_extended.csv"

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]

APERTURE_KPC = 30.0
SFR_WINDOW_MYR = 25.0
COLD_T_K = 8000.0
DUST_TO_METALS = 0.4


def run_names():
    names = []
    for fam in FAMILIES:
        for var in VARIANTS:
            names.append(fam if var == "noBH" else f"{fam}_{var}")
    return names


def find_snapshot(run_dir):
    """Glob for the final output. Exactly one *.004096 expected (verified
    in the original census); anything else is an error worth stopping for."""
    hits = [p for p in glob.glob(str(run_dir / "*.004096"))
            if not p.endswith((".amiga.grp", ".amiga.gtp", ".amiga.stat"))]
    # keep only the bare snapshot (no double extension after 004096)
    hits = [h for h in hits if h.endswith(".004096")]
    if len(hits) != 1:
        raise RuntimeError(f"{run_dir}: expected 1 snapshot, found {hits}")
    return hits[0]


_CENSUS_CACHE = None

def census_center(run):
    """Main-halo center from the existing census, in kpc.
    Reusing the census center (rather than re-deriving one) also makes the
    aperture *identical* to the census by construction."""
    global _CENSUS_CACHE
    if _CENSUS_CACHE is None:
        _CENSUS_CACHE = pd.read_csv(CENSUS_CSV)
    row = _CENSUS_CACHE[(_CENSUS_CACHE.run == run)
                        & (_CENSUS_CACHE.halo_id == 1)]
    if len(row) != 1:
        raise RuntimeError(f"{run}: expected 1 main-halo census row, "
                           f"got {len(row)}")
    for cols in (("cen_x", "cen_y", "cen_z"),
                 ("cx", "cy", "cz"), ("xc", "yc", "zc"),
                 ("x_c", "y_c", "z_c"), ("center_x", "center_y", "center_z")):
        if all(c in row.columns for c in cols):
            return row[list(cols)].to_numpy(dtype=float)[0]
    raise RuntimeError("No center columns recognized in census CSV. "
                       f"Columns present: {list(row.columns)}")


def measure_one(run, verbose=False):
    run_dir = BASE / run
    snap_path = find_snapshot(run_dir)
    if verbose:
        print(f"{run}: {Path(snap_path).name}")

    cen_kpc = census_center(run)

    s = pynbody.load(snap_path)
    s.physical_units()

    def in_aperture(fam):
        pos = np.asarray(fam["pos"].in_units("kpc"))
        d2 = ((pos - cen_kpc) ** 2).sum(axis=1)
        return fam[d2 < APERTURE_KPC ** 2]

    stars_all = in_aperture(s.star)
    tform = stars_all["tform"]
    real = stars_all[tform > 0]          # BH sentinel: tform < 0 -> excluded

    mstar = float(real["mass"].sum().in_units("Msol"))

    # --- SFR over the last SFR_WINDOW_MYR ---
    t_now = float(s.properties["time"].in_units("Gyr"))
    t_cut = t_now - SFR_WINDOW_MYR / 1000.0
    young = real[real["tform"].in_units("Gyr") > t_cut]
    m_young = float(young["mass"].sum().in_units("Msol"))
    sfr = m_young / (SFR_WINDOW_MYR * 1e6)          # Msun/yr
    ssfr = sfr / mstar if mstar > 0 else np.nan

    # --- mass-weighted stellar metallicity ---
    zstar = float((real["mass"] * real["metals"]).sum()
                  / real["mass"].sum()) if len(real) else np.nan

    # --- cold gas + its metals (the dust reservoir SKIRT saw) ---
    gas = in_aperture(s.gas)
    cold = gas[gas["temp"].in_units("K") < COLD_T_K]
    mgas_cold = float(cold["mass"].sum().in_units("Msol"))
    mz_cold = float((cold["mass"].in_units("Msol") * cold["metals"]).sum())

    row = dict(run=run,
               mstar_30kpc=mstar,
               sfr_25myr=sfr,
               ssfr=ssfr,
               zstar_mw=zstar,
               n_young=len(young),
               mgas_cold=mgas_cold,
               mz_cold=mz_cold,
               mdust_proxy=DUST_TO_METALS * mz_cold)

    if verbose:
        print(f"  t_now = {t_now:.3f} Gyr, aperture {APERTURE_KPC} kpc")
        print(f"  stars: {len(stars_all)} in family, {len(real)} real, "
              f"{len(stars_all) - len(real)} BH-sentinel excluded")
        for k, v in row.items():
            if k != "run":
                print(f"  {k:12s} = {v:.4g}" if isinstance(v, float)
                      else f"  {k:12s} = {v}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="single run, verbose")
    args = ap.parse_args()

    targets = [args.run] if args.run else run_names()
    rows = []
    for run in targets:
        rows.append(measure_one(run, verbose=bool(args.run)))
    extra = pd.DataFrame(rows)

    if args.run:
        return 0

    census = pd.read_csv(CENSUS_CSV)
    main_halo = census[census.halo_id == 1]

    merged = pd.merge(main_halo, extra, on="run",
                      how="outer", indicator=True, validate="one_to_one")
    bad = merged[merged._merge != "both"]
    if len(bad):
        print("MERGE MISMATCH -- rows without a partner:")
        print(bad[["run", "_merge"]].to_string(index=False))
    merged = merged.drop(columns="_merge")

    # Cross-check: aperture M* here vs census M*
    ratio = merged.mstar_30kpc / merged.mstar
    print("\nmstar cross-check (this script / census):")
    print(merged.assign(ratio=ratio)[["run", "mstar", "mstar_30kpc",
                                      "ratio"]].round(3).to_string(index=False))
    if (np.abs(ratio - 1) > 0.05).any():
        print("WARNING: >5% disagreement somewhere above -- investigate "
              "before trusting ssfr.")

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(merged)} rows -> {OUT_CSV}")

    # Quick look: the dust-reservoir table, since it's the sharpest test
    fam_var = merged.run.map(
        lambda r: (r.split("_", 1) if "_" in r else (r, "noBH")))
    merged["family"] = fam_var.map(lambda t: t[0])
    merged["variant"] = fam_var.map(lambda t: t[1])
    piv = (merged.pivot_table(index="family", columns="variant",
                              values="mdust_proxy")
                 .reindex(index=FAMILIES, columns=VARIANTS))
    print("\n--- dust reservoir proxy 0.4*Mz(cold) [Msun] ---")
    print(piv.applymap(lambda v: f"{v:.2e}").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())