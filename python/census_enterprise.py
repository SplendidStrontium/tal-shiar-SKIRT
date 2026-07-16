#!/usr/bin/env python
"""
census_enterprise.py

Snapshot census for the Enterprise BH-variant comparison project.

Walks the enterprise directory, finds every run dir (r<family> with
optional _BH/_BH6/_BH8 suffix), loads the final snapshot (output
004096) with pynbody, reads the amiga halo catalog, and tabulates
per-halo properties into one long-format CSV (one row per halo per
run) for later key-based merging and cross-variant matching.

Conventions baked in (learned the hard way):
  * Snapshots are found by globbing the output number (*.004096).
    NEVER derive filenames from achOutName in the param files --
    those are stale fossils from pre-rename working names.
  * Black holes live in the STAR family with tform < 0 (ChaNGa
    sentinel). Filter on the sign of tform directly.
  * BH6/BH8 wrote binary aux arrays (iBinaryOutput=1); noBH/BH wrote
    ASCII. pynbody handles both, but if an aux array loads garbage in
    one variant only, suspect this first.

Usage (on Hamilton, in your usual conda env -- NOT the system python2):
    python census_enterprise.py                 # full census
    python census_enterprise.py --inventory     # dry run: what's on disk
    python census_enterprise.py --only r488     # one family only
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pynbody

# ----------------------------------------------------------------- config
BASE = "/mnt/data0/pkrsnak/romulus/enterprise"
SNAP_GLOB = "*.004096"     # final output number; aux arrays (.004096.HI
                           # etc.) are NOT matched by this pattern
OUT_CSV = os.path.join(BASE, "enterprise_census.csv")
MIN_NSTAR = 50             # skip halos with fewer real star particles
MAX_HALO_ID = 100          # amiga orders halos by size; don't walk past this
VARIANT_ORDER = ["noBH", "BH", "BH6", "BH8"]


# ------------------------------------------------------------- discovery
def discover_runs(base, only=None):
    """Return list of dicts: {path, run, family, variant}."""
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
    """Return the unique final snapshot path, or None (with a complaint)."""
    hits = [p for p in glob.glob(os.path.join(run_dir, SNAP_GLOB))
            if os.path.isfile(p)]
    if len(hits) == 1:
        return hits[0]
    tag = "no" if not hits else f"{len(hits)}"
    print(f"  !! {os.path.basename(run_dir)}: {tag} files match {SNAP_GLOB}"
          f" -- skipping. Check for restarts ending at a different step.")
    for p in hits:
        print(f"       candidate: {os.path.basename(p)}")
    return None


def inventory(runs):
    """Dry run: report what exists without loading anything."""
    print(f"{'run':<12} {'snapshot':<44} {'grp':<4} {'stat':<5} {'param'}")
    for r in runs:
        snap = find_snapshot(r["path"])
        snap_name = os.path.basename(snap) if snap else "MISSING"
        has = lambda ext: bool(glob.glob(os.path.join(r["path"], "*" + ext)))
        print(f"{r['run']:<12} {snap_name:<44} "
              f"{'y' if has('.amiga.grp') else 'NO':<4} "
              f"{'y' if has('.amiga.stat') else 'NO':<5} "
              f"{'y' if has('.param') else 'NO'}")


# ---------------------------------------------------------------- census
def halo_center(halo):
    """Shrinking-sphere center; falls back to stellar center of mass."""
    try:
        return np.asarray(
            pynbody.analysis.halo.center(halo, mode="ssc", retcen=True))
    except Exception:
        sub = halo.s if len(halo.s) > 0 else halo.dm
        m = sub["mass"]
        return np.asarray((sub["pos"] * m[:, None]).sum(axis=0) / m.sum())


def census_run(run):
    """Return list of per-halo row dicts for one run."""
    snap_path = find_snapshot(run["path"])
    if snap_path is None:
        return []

    print(f"  loading {os.path.basename(snap_path)} ...")
    s = pynbody.load(snap_path)
    s.physical_units()

    try:
        h = s.halos()  # reads .amiga.grp
    except Exception as e:
        print(f"  !! {run['run']}: could not load halo catalog ({e})")
        return []

    n_halos = len(h)
    hires_dm = float(s.dm["mass"].min())  # high-res DM particle mass
    z = float(s.properties.get("z", np.nan))

    rows = []
    for hid in range(1, min(n_halos, MAX_HALO_ID) + 1):
        try:
            halo = h[hid]
        except Exception:
            continue

        tf = halo.s["tform"] if len(halo.s) > 0 else np.array([])
        real = tf > 0          # actual stars
        bh = tf < 0            # ChaNGa BH sentinel

        n_star = int(real.sum())
        n_bh = int(bh.sum())

        # Keep BH-hosting halos even below the star threshold -- a BH in a
        # tiny halo is exactly the kind of thing we want to see in a census.
        if n_star < MIN_NSTAR and n_bh == 0:
            continue

        mstar = float(halo.s["mass"][real].sum()) if n_star else 0.0
        mbh_max = float(halo.s["mass"][bh].max()) if n_bh else 0.0
        mbh_tot = float(halo.s["mass"][bh].sum()) if n_bh else 0.0
        mgas = float(halo.g["mass"].sum()) if len(halo.g) else 0.0
        mtot = float(halo["mass"].sum())

        dm_min = float(halo.dm["mass"].min()) if len(halo.dm) else np.nan
        # 1.0 = clean zoom halo; >1 means low-res DM contamination
        contam = dm_min / hires_dm if np.isfinite(dm_min) else np.nan

        cen = halo_center(halo)

        rows.append(dict(
            family=run["family"], variant=run["variant"], run=run["run"],
            halo_id=hid, z=z,
            n_star=n_star, n_gas=len(halo.g), n_dm=len(halo.dm),
            mstar=mstar, mgas=mgas, mtot=mtot,
            n_bh=n_bh, mbh_max=mbh_max, mbh_tot=mbh_tot,
            cen_x=cen[0], cen_y=cen[1], cen_z=cen[2],
            dm_min_ratio=contam,
            snapshot=os.path.basename(snap_path),
        ))

    # ---- per-run console summary + sanity checks
    kept = len(rows)
    total_bh = sum(r["n_bh"] for r in rows)
    print(f"  {run['run']}: {n_halos} halos in catalog, {kept} kept, "
          f"{total_bh} BH particles across kept halos")
    if run["variant"] == "noBH" and total_bh > 0:
        print(f"  ** WARNING: noBH run {run['run']} contains tform<0 "
              f"particles -- investigate before trusting anything. **")
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true",
                    help="list snapshots/catalogs on disk, load nothing")
    ap.add_argument("--only", metavar="FAMILY",
                    help="restrict to one family, e.g. r488")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    runs = discover_runs(BASE, only=args.only)
    if not runs:
        sys.exit(f"No run directories found under {BASE}")
    print(f"Found {len(runs)} run directories.\n")

    if args.inventory:
        inventory(runs)
        return

    all_rows = []
    for run in runs:
        print(f"[{run['run']}]")
        all_rows.extend(census_run(run))

    if not all_rows:
        sys.exit("No halos collected -- nothing written.")

    df = pd.DataFrame(all_rows)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER,
                                   ordered=True)
    df = df.sort_values(["family", "variant", "mstar"],
                        ascending=[True, True, False])
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")

    # compact overview: biggest halo per run
    top = df.sort_values("mstar", ascending=False).groupby(
        "run", observed=True).head(1)
    cols = ["run", "halo_id", "mstar", "mgas", "n_bh", "mbh_max",
            "dm_min_ratio"]
    with pd.option_context("display.float_format", "{:.3e}".format):
        print("\nMost massive (by M*) halo per run:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()