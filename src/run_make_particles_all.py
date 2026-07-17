#!/usr/bin/env python3
"""
run_make_particles_all.py — Enterprise particle extraction driver

Loops all 20 enterprise runs, invoking make_particles.py once per run as a
SUBPROCESS (not an import). This is deliberate: pynbody holds the full zoom
volume in memory, and a fresh process per run guarantees everything is
released before the next 10M+ particle snapshot loads. Same reason the
census felt heavy — but here each run's memory dies with its process.

Reads:   /mnt/data0/pkrsnak/romulus/enterprise/{run}/   (Jillian's symlinks,
         read-only by convention — we never write there)
Writes:  /mnt/data0/pkrsnak/romulus/enterprise/products/{run}/particles/
         plus an aggregated products/diagnostics_all.csv at the end.

Conventions (same as census_enterprise.py):
  * snapshots found by globbing *.004096 — never derived from achOutName
  * run dirs discovered as enterprise/r*; 'products' explicitly excluded
    (belt and suspenders: the r* glob already skips it, but we don't want
    the layout to depend on that accident)

Usage (Hamilton, usual conda env, from tal-shiar-SKIRT/src):
    python run_make_particles_all.py                # all 20 runs
    python run_make_particles_all.py --only r488    # one family
    python run_make_particles_all.py --dry-run      # show commands only
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

# ----------------------------------------------------------------- config
BASE = Path("/mnt/data0/pkrsnak/romulus/enterprise")
PRODUCTS = BASE / "products"
SNAP_GLOB = "*.004096"
RADIUS_PC = 30000
HALO_ID = 1

# make_particles.py lives next to this driver
MAKE_PARTICLES = Path(__file__).resolve().parent / "make_particles.py"

VARIANT_ORDER = ["noBH", "BH", "BH6", "BH8"]


# ------------------------------------------------------------- discovery
def discover_runs(base, only=None):
    runs = []
    for d in sorted(glob.glob(str(base / "r*"))):
        if not os.path.isdir(d):          # follows symlinks, as intended
            continue
        name = os.path.basename(d)
        if name == "products":            # explicit, not glob-luck
            continue
        parts = name.split("_", 1)
        family = parts[0]
        variant = parts[1] if len(parts) > 1 else "noBH"
        if only and family != only:
            continue
        runs.append(dict(path=Path(d), run=name, family=family,
                         variant=variant))
    return runs


def find_snapshot(run_dir):
    hits = [p for p in glob.glob(str(run_dir / SNAP_GLOB))
            if os.path.isfile(p)]
    if len(hits) == 1:
        return Path(hits[0])
    print(f"  !! {run_dir.name}: {len(hits)} files match {SNAP_GLOB} "
          f"-- skipping.")
    return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", metavar="FAMILY")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if diagnostics.csv already exists")
    args = ap.parse_args()

    if not MAKE_PARTICLES.exists():
        sys.exit(f"make_particles.py not found at {MAKE_PARTICLES}")

    runs = discover_runs(BASE, only=args.only)
    if not runs:
        sys.exit(f"No run directories found under {BASE}")
    print(f"Found {len(runs)} run directories.\n")

    results = []   # (run, status, seconds)
    for run in runs:
        snap = find_snapshot(run["path"])
        if snap is None:
            results.append((run["run"], "NO_SNAPSHOT", 0.0))
            continue

        outdir = PRODUCTS / run["run"] / "particles"
        done_marker = outdir / "diagnostics.csv"
        if done_marker.exists() and not args.force:
            print(f"[{run['run']}] already extracted "
                  f"({done_marker} exists) -- skipping. Use --force to redo.")
            results.append((run["run"], "SKIPPED", 0.0))
            continue

        cmd = [sys.executable, str(MAKE_PARTICLES),
               "--snapshot", str(snap),
               "--output", str(outdir),
               "--radius", str(RADIUS_PC),
               "--halo", str(HALO_ID)]

        if args.dry_run:
            print(f"[{run['run']}] would run:\n  {' '.join(cmd)}")
            results.append((run["run"], "DRY", 0.0))
            continue

        print(f"\n{'='*70}\n[{run['run']}] extracting...\n{'='*70}")
        t0 = timer()
        # stream child output live; per-run log kept alongside its products
        outdir.mkdir(parents=True, exist_ok=True)
        log_path = outdir / "extraction.log"
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            proc.wait()
        dt = timer() - t0

        if proc.returncode == 0:
            results.append((run["run"], "OK", dt))
        else:
            # canary behavior: record and continue, don't abort the sweep
            print(f"  ** {run['run']} FAILED (exit {proc.returncode}) -- "
                  f"see {log_path} **")
            results.append((run["run"], f"FAILED({proc.returncode})", dt))

    # ------------------------------------------------------------ summary
    print(f"\n{'='*70}\nSweep summary\n{'='*70}")
    for name, status, dt in results:
        print(f"  {name:<12} {status:<14} {dt:6.1f} s")

    if args.dry_run:
        return

    # ---------------------------------------------- aggregate diagnostics
    diag_files = sorted(glob.glob(str(PRODUCTS / "*" / "particles"
                                      / "diagnostics.csv")))
    if not diag_files:
        print("\nNo diagnostics files found; nothing to aggregate.")
        return

    frames = []
    for f in diag_files:
        run_name = Path(f).parent.parent.name
        parts = run_name.split("_", 1)
        df = pd.read_csv(f)
        df.insert(0, "run", run_name)
        df.insert(1, "family", parts[0])
        df.insert(2, "variant", parts[1] if len(parts) > 1 else "noBH")
        frames.append(df)

    alldiag = pd.concat(frames, ignore_index=True)
    alldiag["variant"] = pd.Categorical(alldiag["variant"],
                                        categories=VARIANT_ORDER,
                                        ordered=True)
    alldiag = alldiag.sort_values(["family", "variant"])
    out_csv = PRODUCTS / "diagnostics_all.csv"
    alldiag.to_csv(out_csv, index=False)
    print(f"\nAggregated {len(alldiag)} runs -> {out_csv}")

    cols = ["run", "mstar", "mgas", "n_bh", "mbh_tot",
            "mean_age_gyr", "t50_gyr", "r_half_kpc",
            "sfr_25myr", "f_gas_cold_8e3"]
    with pd.option_context("display.float_format", "{:.3g}".format):
        print("\nCross-variant comparison table:")
        print(alldiag[cols].to_string(index=False))


if __name__ == "__main__":
    main()