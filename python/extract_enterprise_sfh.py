#!/usr/bin/env python3
"""
extract_enterprise_sfh.py
-------------------------
Star-formation histories for all 20 Enterprise runs, from the raw
z=0 snapshots (output 004096), main halo only.

Does the SAME computation as pynbody.plot.stars.sfh -- a histogram of
star-particle formation times weighted by formation mass, divided by
bin width -- but decoupled from plotting so the (expensive) snapshot
loads happen once and the figure script can iterate cheaply.

To cross-check any single run against the real thing:

    import pynbody, pynbody.plot.stars
    s = pynbody.load(snap); s.physical_units()
    st = s.halos()[1].star
    st = st[st['tform'].in_units('Gyr') > 0]     # <-- BH sinks out FIRST
    pynbody.plot.stars.sfh(st, massform=True, bins=NBINS)

Critical gotchas (standing decisions):
  * BH sink particles live in the star family with tform < 0.
    pynbody.plot.stars.sfh does NOT filter these, and by default sets
    its time range from the data min/max, so an unfiltered call
    silently stretches the bins to negative time. We filter
    tform > 0 before anything else.
  * Snapshots are found by globbing *.004096 in each run directory.
    Never parse achOutName (stale fossils from pre-rename names).
  * noBH runs are named bare (e.g. 'r488'); BH variants are
    '{family}_{variant}' -- matches enterprise_census.csv.

massform fallback:
  'massform' (mass at formation) is preferred; if the aux array is
  missing for a run we fall back to current 'mass' (underestimates
  early SF because of stellar mass loss) and RECORD which was used
  in the summary CSV, so no silent inconsistency between panels.

Outputs (products/):
  enterprise_sfh.csv          tidy: run, t_left, t_right, sfr  [Msol/yr]
  enterprise_sfh_summary.csv  run, t_now, n_stars, mform_total,
                              sfr_recent, sfr_window_gyr, mass_source

Resumable: runs already present in enterprise_sfh.csv are skipped;
delete the CSVs (or a run's rows) to force recompute.

Usage (on Hamilton, in the pynbody environment):
    python extract_enterprise_sfh.py
    python extract_enterprise_sfh.py --dt 0.1 --sfr-window 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pynbody
import pynbody.analysis.cosmology as cosmo

BASE = Path("/mnt/data0/pkrsnak/romulus/enterprise")
OUT_SFH = BASE / "products" / "enterprise_sfh.csv"
OUT_SUM = BASE / "products" / "enterprise_sfh_summary.csv"

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]


def run_name(family, variant):
    return family if variant == "noBH" else f"{family}_{variant}"


def find_snapshot(run):
    """Glob the base snapshot by output number. The pattern '*.004096'
    anchors at the end of the name, so aux arrays like
    foo.004096.amiga.grp do not match."""
    d = BASE / run
    hits = sorted(d.glob("*.004096"))
    if len(hits) != 1:
        raise RuntimeError(f"{run}: expected exactly one *.004096 in {d}, "
                           f"found {[h.name for h in hits]}")
    return hits[0]


def load_main_halo_stars(snap_path):
    """Load snapshot, return (star SubSnap of main halo with BH sinks
    removed, t_now in Gyr).

    NOTE: main-halo selection here is h[1] from s.halos(). If
    census_enterprise.py selects the main halo differently, mirror
    that here so masses/SFRs refer to the same object.
    """
    s = pynbody.load(str(snap_path))
    s.physical_units()
    t_now = float(cosmo.age(s))          # Gyr at this snapshot

    h = s.halos()
    st = h[1].star

    # BH sinks: star-family particles with tform < 0 (in-band sentinel).
    # Filter on tform directly, never on derived age.
    tform = np.asarray(st["tform"].in_units("Gyr"), dtype=float)
    keep = tform > 0
    return st[keep], t_now


def formation_masses(stars, force_mass=False):
    """massform if loadable, else current mass. Returns (array Msol,
    source string)."""
    if force_mass:
        mf = np.asarray(stars["mass"].in_units("Msol"), dtype=float)
        return mf, "mass"
    try:
        mf = np.asarray(stars["massform"].in_units("Msol"), dtype=float)
        return mf, "massform"
    except Exception as e:                       # missing aux array etc.
        print(f"    massform unavailable ({type(e).__name__}); "
              f"falling back to current mass")
        mf = np.asarray(stars["mass"].in_units("Msol"), dtype=float)
        return mf, "mass"


def sfh_one_run(run, dt, sfr_window, args_force_mass):
    snap = find_snapshot(run)
    print(f"  {run}: {snap.name}")
    stars, t_now = load_main_halo_stars(snap)

    tform = np.asarray(stars["tform"].in_units("Gyr"), dtype=float)
    mform, source = formation_masses(stars, force_mass=args_force_mass)

    # Fixed bins 0 -> t_now: identical grid for every run so twins
    # overlay bin-for-bin.  (Same math as pynbody.plot.stars.sfh:
    # sum of formation mass per bin / bin width.)
    edges = np.arange(0.0, t_now + dt, dt)
    hist, edges = np.histogram(tform, bins=edges, weights=mform)
    sfr = hist / (dt * 1e9)                      # Msol / yr

    sfh = pd.DataFrame({"run": run,
                        "t_left": edges[:-1],
                        "t_right": edges[1:],
                        "sfr": sfr})

    recent = mform[tform > (t_now - sfr_window)].sum() / (sfr_window * 1e9)
    summary = {"run": run,
               "t_now": t_now,
               "n_stars": len(tform),
               "mform_total": mform.sum(),
               "sfr_recent": recent,             # Msol/yr over the window
               "sfr_window_gyr": sfr_window,
               "mass_source": source}
    return sfh, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.2,
                    help="SFH bin width [Gyr] (default 0.2)")
    ap.add_argument("--sfr-window", type=float, default=0.1,
                    help="recent-SFR averaging window [Gyr] "
                         "(default 0.1 = 100 Myr)")
    ap.add_argument("--force-mass", action="store_true",
                    help="Use current mass instead of formation mass")
    args = ap.parse_args()

    OUT_SFH.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT_SFH.exists():
        done = set(pd.read_csv(OUT_SFH).run.unique())
        print(f"Resuming: {len(done)} run(s) already extracted")

    runs = [run_name(f, v) for f in FAMILIES for v in VARIANTS]
    for run in runs:
        if run in done:
            print(f"  {run}: done, skipping")
            continue
        sfh, summary = sfh_one_run(run, args.dt, args.sfr_window, args.force_mass)

        # Append-mode writes, header only on first touch (sweep-log style)
        sfh.to_csv(OUT_SFH, mode="a", header=not OUT_SFH.exists(),
                   index=False)
        pd.DataFrame([summary]).to_csv(
            OUT_SUM, mode="a", header=not OUT_SUM.exists(), index=False)

    print(f"Wrote {OUT_SFH}")
    print(f"Wrote {OUT_SUM}")


if __name__ == "__main__":
    main()