#!/usr/bin/env python3
"""
run_skirt_production.py — Enterprise SKIRT production driver

Runs the production {run}_dust.ski / {run}_nodust.ski files as-is (no
rewriting) for every run in the enterprise sample: 5 families x 4 BH
variants = 20 runs = 40 SKIRT simulations. Expects skis configured for
production: 2 inclinations (0, 90), 5e7 photons, 500 pixels, maxLevel 11.

Adapted from the Tal Shiar production driver. Enterprise-specific behavior:

  * Layout: reads particles from products/{run}/particles/, writes SKIRT
    output to products/{run}/production/. Never touches the raw snapshot
    symlinks (Jillian's tree).
  * Resumable: after each successful SKIRT run, a .done_{label} marker is
    written in the run's production dir. On restart, completed runs are
    skipped automatically. Use --force to redo. A crash, Ctrl-C, or node
    reboot never costs finished runs.
  * Convergence archiving: each dust run's spatial convergence file is
    copied to a timestamped name immediately after the run finishes, so
    re-runs can never silently overwrite the evidence.
  * Sweep order: families run smallest to largest (r741 first, r488 last)
    so a morning check of the log shows several completed runs early.
  * Sweep log: --detach writes to products/production_sweep.log by default
    (it gets long; it belongs with the data), APPEND mode with a session
    header, so relaunches never clobber earlier history.

Usage:
    # Quick single-run sanity check before committing the night (~2 min):
    python run_skirt_production.py --galaxy r741 --dry-run

    # The real thing: full 40-run sweep, detached, survives SSH logout:
    python run_skirt_production.py --detach

    # Monitor:
    tail -f /mnt/data0/pkrsnak/romulus/enterprise/products/production_sweep.log

    # Single run, foreground:
    python run_skirt_production.py --galaxy r488_BH8

    # Redo a run that has done-markers:
    python run_skirt_production.py --galaxy r488_BH8 --force
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
# Families ordered smallest to largest (by particle load) so early sweep
# progress is visible quickly. Variant naming matches on-disk convention:
# noBH runs are the bare family name. Keep in sync with
# generate_ski_enterprise.py.
FAMILIES = ["r741", "r618", "r613", "r568", "r488"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]


def run_names():
    names = []
    for fam in FAMILIES:
        for var in VARIANTS:
            names.append(fam if var == "noBH" else f"{fam}_{var}")
    return names


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
SKI_DIRNAME = "ski_enterprise"

SCRIPT_DIR = Path(__file__).resolve().parent          # .../tal-shiar-SKIRT/src
DEFAULT_SKI_DIR = SCRIPT_DIR / SKI_DIRNAME
DEFAULT_PRODUCTS_DIR = "/mnt/data0/pkrsnak/romulus/enterprise/products"

DEFAULT_SKIRT = "/mnt/data0/jillian/SKIRT/release/SKIRT/main/skirt"

# 2 inclinations x (FullInstrument + SEDInstrument) = 4 sed.dat per SKIRT run
EXPECTED_SEDS_PER_LABEL = 4


def log(msg):
    """Print with timestamp so the sweep log is useful."""
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def preflight(run, particle_dir, ski_dir, skirt_bin, required_particles,
              ski_files):
    """Check everything is in place before committing to real runs."""
    log(f"[{run}] Preflight checks...")

    missing = [f for f in required_particles if not (particle_dir / f).exists()]
    if missing:
        log(f"  ERROR: missing particle files in {particle_dir}: {missing}")
        return False
    for f in required_particles:
        size_mb = (particle_dir / f).stat().st_size / 1e6
        log(f"  {particle_dir / f}: {size_mb:.2f} MB")

    for f in ski_files:
        path = ski_dir / f
        if not path.exists():
            log(f"  ERROR: missing ski file: {path} "
                f"(check SKI_DIRNAME='{SKI_DIRNAME}' or pass --ski-dir)")
            return False
        size_kb = path.stat().st_size / 1e3
        log(f"  {path}: {size_kb:.1f} KB")

    if not Path(skirt_bin).exists():
        log(f"  ERROR: SKIRT binary not found at {skirt_bin}")
        return False
    log(f"  {skirt_bin}: OK")

    return True


def stage_particle_files(output_dir, particle_dir, required_particles):
    """Symlink particle files into output dir so SKIRT's relative paths resolve."""
    for f in required_particles:
        link = output_dir / f
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(particle_dir / f)
    log(f"  Symlinked {len(required_particles)} particle files into {output_dir}")


def run_emulation(skirt_bin, ski_path, workdir):
    """Schema + particle-file canary before committing to a long run."""
    log(f"  emulation: {ski_path.name} ...")
    t0 = perf_counter()
    result = subprocess.run(
        [skirt_bin, "-e", str(ski_path.resolve())],
        cwd=workdir,
        capture_output=True, text=True,
    )
    dt = perf_counter() - t0
    if result.returncode == 0:
        log(f"    OK ({dt:.1f}s)")
        return True
    log(f"    FAILED ({dt:.1f}s)")
    print("--- stdout ---")
    print(result.stdout)
    print("--- stderr ---")
    print(result.stderr)
    return False


def run_skirt(skirt_bin, ski_path, workdir, label):
    """Real SKIRT run. Streams stdout live so the sweep log captures progress."""
    log(f"  === Running {label}: {ski_path.name} ===")
    t0 = perf_counter()
    result = subprocess.run(
        [skirt_bin, str(ski_path.resolve())],
        cwd=workdir,
    )
    dt = perf_counter() - t0
    if result.returncode == 0:
        log(f"  {label} finished in {dt:.1f}s ({dt/60:.1f} min)")
    else:
        log(f"  {label} FAILED after {dt:.1f}s, returncode={result.returncode}")
    return dt, result.returncode


def archive_convergence(run, output_dir):
    """
    Copy the dust run's convergence file to a timestamped name so later
    re-runs can never silently overwrite it. (Learned the hard way during
    the maxLevel resolution ladder.)
    """
    src = output_dir / f"{run}_dust_spatial_convergence_convergence.dat"
    if not src.exists():
        log(f"  NOTE: no convergence file found at {src.name}")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = output_dir / f"{run}_dust_convergence_{stamp}.dat"
    shutil.copy2(src, dst)
    log(f"  Archived convergence file -> {dst.name}")


def run_one(run, args, ski_dir, skirt_bin):
    """
    Full production cycle for a single run (dust + nodust). Returns a dict
    label->(dt, rc), possibly containing 'SKIPPED' sentinels, or None if
    preflight/emulation gated it off.
    """
    products_dir = Path(args.products_dir).resolve()
    particle_dir = products_dir / run / "particles"
    output_dir = products_dir / run / "production"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_particles = ["stars.txt", "youngStars.txt", "gas.txt"]
    ski_files = [f"{run}_dust.ski", f"{run}_nodust.ski"]

    log("-" * 70)
    log(f"[{run}] particle_dir: {particle_dir}")
    log(f"[{run}] output_dir:   {output_dir}")

    # Assemble the worklist, honoring done-markers and skip flags
    labels = []
    if not args.skip_dust:
        labels.append("dust")
    if not args.skip_nodust:
        labels.append("nodust")

    todo = []
    results = {}
    for label in labels:
        marker = output_dir / f".done_{label}"
        if marker.exists() and not args.force:
            log(f"[{run}] {label}: already done ({marker.name} exists) "
                f"-- skipping. Use --force to redo.")
            results[label] = ("SKIPPED", 0)
        else:
            todo.append(label)

    if not todo:
        log(f"[{run}] nothing to do.")
        return results

    if not preflight(run, particle_dir, ski_dir, skirt_bin,
                     required_particles, ski_files):
        log(f"[{run}] preflight failed — skipping this run.")
        return None

    stage_particle_files(output_dir, particle_dir, required_particles)

    # Emulation canary — only for the labels we're actually about to run
    log(f"[{run}] Emulation pass (schema canary)...")
    for label in todo:
        ski = ski_dir / f"{run}_{label}.ski"
        if not run_emulation(skirt_bin, ski, output_dir):
            log(f"[{run}] emulation failed — skipping this run.")
            return None

    if args.dry_run:
        log(f"[{run}] dry run: emulation passed, no real run.")
        return results

    # Real runs
    for label in todo:
        ski = ski_dir / f"{run}_{label}.ski"
        dt, rc = run_skirt(skirt_bin, ski, output_dir, label)
        results[label] = (dt, rc)
        if rc == 0:
            (output_dir / f".done_{label}").write_text(
                f"{datetime.now().isoformat()}  wall={dt:.1f}s\n")
            if label == "dust":
                archive_convergence(run, output_dir)
        else:
            log(f"[{run}] stopping this run due to {label} failure "
                f"(no done-marker written; will retry on next sweep).")
            break

    # Output spot-check: 4 sed.dat per completed label
    for label in labels:
        seds = sorted(output_dir.glob(f"{run}_{label}_*_sed.dat"))
        log(f"[{run}] {label}: {len(seds)} sed.dat files "
            f"(expected {EXPECTED_SEDS_PER_LABEL} = 2 inclinations x 2 instruments)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Production SKIRT driver for the Enterprise sweep")
    parser.add_argument("--galaxy", default=None, metavar="RUN",
                        help="Run a single run (e.g. r741, r488_BH8) instead "
                             "of the full 20-run sweep")
    parser.add_argument("--products-dir", default=DEFAULT_PRODUCTS_DIR,
                        help=f"Products base dir (default: {DEFAULT_PRODUCTS_DIR})")
    parser.add_argument("--ski-dir", default=None,
                        help=f"Directory containing the ski files "
                             f"(default: {DEFAULT_SKI_DIR})")
    parser.add_argument("--skirt", default=DEFAULT_SKIRT,
                        help="Path to skirt binary")
    parser.add_argument("--skip-dust", action="store_true",
                        help="Skip dust runs (all runs)")
    parser.add_argument("--skip-nodust", action="store_true",
                        help="Skip nodust runs (all runs)")
    parser.add_argument("--force", action="store_true",
                        help="Redo runs even if .done markers exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run emulation only; don't start real SKIRT runs")
    parser.add_argument("--detach", action="store_true",
                        help="Re-exec via nohup and detach. Log appends to "
                             "{products-dir}/production_sweep.log")
    args = parser.parse_args()

    ski_dir = Path(args.ski_dir).resolve() if args.ski_dir else DEFAULT_SKI_DIR
    runs = [args.galaxy] if args.galaxy else run_names()

    # --- nohup detach mode: re-exec ourselves once for the whole sweep ---
    if args.detach:
        log_path = Path(args.products_dir).resolve() / "production_sweep.log"
        cmd = [sys.executable, os.path.abspath(__file__)]
        for k, v in vars(args).items():
            if k == "detach" or v is None or v is False:
                continue
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])

        log(f"Detaching. Log: {log_path}")
        log(f"  Re-executing: {' '.join(cmd)}")
        log(f"  Monitor with: tail -f {log_path}")
        # APPEND mode + session banner: relaunches never clobber history.
        with open(log_path, 'a') as logf:
            logf.write(f"\n{'#' * 70}\n"
                       f"# New sweep session: {datetime.now().isoformat()}\n"
                       f"# Command: {' '.join(cmd)}\n"
                       f"{'#' * 70}\n")
            logf.flush()
            subprocess.Popen(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log(f"Detached. Exiting parent. Sleep well.")
        return

    # --- Normal (attached) execution ---
    log("=" * 70)
    log(f"Enterprise SKIRT production sweep")
    log(f"  runs:    {len(runs)} -> {runs}")
    log(f"  ski_dir: {ski_dir}")
    log(f"  skirt:   {args.skirt}")
    log("=" * 70)

    total_start = perf_counter()
    sweep = {}
    for run in runs:
        sweep[run] = run_one(run, args, ski_dir, args.skirt)
    total_dt = perf_counter() - total_start

    # --- Grand summary ---
    log("")
    log("=" * 70)
    log("Sweep summary")
    log("=" * 70)
    n_ok = n_fail = 0
    for run, results in sweep.items():
        if results is None:
            log(f"  {run:10s}: GATED (preflight/emulation)")
            n_fail += 1
            continue
        if not results:
            log(f"  {run:10s}: no runs (dry-run or all skipped)")
            continue
        parts = []
        for label, (dt, rc) in results.items():
            if dt == "SKIPPED":
                parts.append(f"{label} skipped(done)")
            elif rc == 0:
                parts.append(f"{label} {dt/60:.1f}min OK")
                n_ok += 1
            else:
                parts.append(f"{label} {dt/60:.1f}min FAIL(rc={rc})")
                n_fail += 1
        log(f"  {run:10s}: " + " | ".join(parts))
    log(f"  {'TOTAL':10s}: {total_dt:.0f}s ({total_dt/60:.1f} min, "
        f"{total_dt/3600:.2f} hr) | {n_ok} OK, {n_fail} failed/gated")


if __name__ == "__main__":
    main()