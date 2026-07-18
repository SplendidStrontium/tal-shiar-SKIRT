#!/usr/bin/env python3
"""
run_skirt_production.py — Tal Shiar SKIRT Pipeline, production driver

Runs the production {galaxy}_dust.ski / {galaxy}_nodust.ski files as-is (no
rewriting) for every halo in HALOS. Expects the ski files to be configured
for the intended production: face-on (1 inclination), 5e7 photons, 500 pixels.

What this does, per halo:
    1. Preflight: confirms ski files, particle files, and SKIRT binary exist.
    2. Stages particle files (symlinks) into the output directory so SKIRT
       can find them via relative paths.
    3. Runs `skirt -e` emulation on both ski files as a schema canary —
       if this fails, skip to the next halo rather than burning a long run.
    4. Runs SKIRT for real: dust first, then nodust. Times each.
    5. Reports wall time and lists output files.
A grand summary across all halos is printed at the end.

Usage:
    # All 15 halos, foreground
    python run_skirt_production.py

    # A single halo
    python run_skirt_production.py --galaxy r320

    # Detached via nohup (survives terminal close; logs to the sweep log)
    python run_skirt_production.py --detach

    # Dry run (emulation only, no real SKIRT run)
    python run_skirt_production.py --dry-run

    # Skip dust or nodust if already done (applies to every halo)
    python run_skirt_production.py --skip-dust
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
# Full face-on sample: three dust-study halos + twelve new proposal halos.
# Keep in sync with generate_ski.py.
HALOS = ["r107", "r142", "r320",                          # original dust study
         "r154", "r168", "r204", "r219", "r223", "r239",  # new proposal sample
         "r284", "r306", "r316", "r330", "r372", "r429"]

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# The .ski files live in src/ski/. This script is in src/, so the ski dir is
# a subdirectory of the script's own directory. SET THIS if you rename it.
SKI_DIRNAME = "ski"

SCRIPT_DIR = Path(__file__).resolve().parent          # .../tal-shiar-SKIRT/src
DEFAULT_SKI_DIR = SCRIPT_DIR / SKI_DIRNAME             # .../src/ski
DEFAULT_ROMULUS_DIR = "/mnt/data0/pkrsnak/romulus"

DEFAULT_SKIRT = "/mnt/data0/jillian/SKIRT/release/SKIRT/main/skirt"


def log(msg):
    """Print with timestamp so the nohup log is useful."""
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def preflight(galaxy, particle_dir, ski_dir, skirt_bin, required_particles, ski_files):
    """Check everything is in place before we commit to a real run for this halo."""
    log(f"[{galaxy}] Preflight checks...")

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
    """Schema validation — fast fail for ski errors before committing to a long run."""
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
    """Real SKIRT run. Streams stdout live so the log captures progress."""
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


def run_one_halo(galaxy, args, ski_dir, skirt_bin):
    """
    Full production cycle for a single halo. Returns a dict label->(dt, rc),
    or None if preflight/emulation gated it off before any real run.
    """
    particle_dir = Path(args.romulus_dir).resolve() / galaxy
    output_dir = particle_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    required_particles = ["stars.txt", "youngStars.txt", "gas.txt"]
    ski_files = [f"{galaxy}_dust.ski", f"{galaxy}_nodust.ski"]

    log("-" * 70)
    log(f"[{galaxy}] particle_dir: {particle_dir}")
    log(f"[{galaxy}] output_dir:   {output_dir}")

    if not preflight(galaxy, particle_dir, ski_dir, skirt_bin,
                     required_particles, ski_files):
        log(f"[{galaxy}] preflight failed — skipping this halo.")
        return None

    stage_particle_files(output_dir, particle_dir, required_particles)

    runs = []
    if not args.skip_dust:
        runs.append(("dust", ski_dir / f"{galaxy}_dust.ski"))
    if not args.skip_nodust:
        runs.append(("nodust", ski_dir / f"{galaxy}_nodust.ski"))
    if not runs:
        log(f"[{galaxy}] nothing to do (both --skip-dust and --skip-nodust set).")
        return {}

    # Emulation canary
    log(f"[{galaxy}] Emulation pass (schema canary)...")
    for label, ski in runs:
        if not run_emulation(skirt_bin, ski, output_dir):
            log(f"[{galaxy}] emulation failed — skipping this halo.")
            return None

    if args.dry_run:
        log(f"[{galaxy}] dry run: emulation passed, no real run.")
        return {}

    # Real runs
    results = {}
    for label, ski in runs:
        dt, rc = run_skirt(skirt_bin, ski, output_dir, label)
        results[label] = (dt, rc)
        if rc != 0:
            log(f"[{galaxy}] stopping this halo due to {label} failure.")
            break

    # Per-halo output inventory + SED spot-check (face-on => 2 sed.dat expected)
    seds = sorted(output_dir.glob("*_sed.dat"))
    log(f"[{galaxy}] {len(seds)} sed.dat files (expected 2 = 1 inclination x 2 runs)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Production SKIRT driver for Tal Shiar pipeline")
    parser.add_argument("--galaxy", default=None,
                        help="Run a single halo instead of the full HALOS sweep")
    parser.add_argument("--romulus-dir", default=DEFAULT_ROMULUS_DIR,
                        help=f"Base dir holding per-halo subdirs (default: {DEFAULT_ROMULUS_DIR})")
    parser.add_argument("--ski-dir", default=None,
                        help=f"Directory containing the ski files (default: {DEFAULT_SKI_DIR})")
    parser.add_argument("--output-subdir", default="production",
                        help="Per-halo subdir for production outputs (default: production)")
    parser.add_argument("--skirt", default=DEFAULT_SKIRT, help="Path to skirt binary")
    parser.add_argument("--skip-dust", action="store_true", help="Skip dust run (all halos)")
    parser.add_argument("--skip-nodust", action="store_true", help="Skip nodust run (all halos)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run emulation only; don't start real SKIRT runs")
    parser.add_argument("--detach", action="store_true",
                        help="Re-exec via nohup and detach. Log at ./production_sweep.log (cwd)")
    args = parser.parse_args()

    ski_dir = Path(args.ski_dir).resolve() if args.ski_dir else DEFAULT_SKI_DIR
    halos = [args.galaxy] if args.galaxy else HALOS

    # --- nohup detach mode: re-exec ourselves under nohup, once, for the whole sweep ---
    if args.detach:
        # Log goes in the launch directory (cwd), not the ski dir — keeps it
        # out of the folder full of .ski files.
        log_path = Path.cwd() / "production_sweep.log"
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
        with open(log_path, 'w') as logf:
            subprocess.Popen(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log(f"Detached. Exiting parent.")
        return

    # --- Normal (attached) execution ---
    log("=" * 70)
    log(f"Tal Shiar SKIRT production sweep")
    log(f"  halos:   {len(halos)} -> {halos}")
    log(f"  ski_dir: {ski_dir}")
    log(f"  skirt:   {args.skirt}")
    log("=" * 70)

    total_start = perf_counter()
    sweep = {}   # galaxy -> results dict (or None)
    for galaxy in halos:
        sweep[galaxy] = run_one_halo(galaxy, args, ski_dir, args.skirt)
    total_dt = perf_counter() - total_start

    # --- Grand summary ---
    log("")
    log("=" * 70)
    log("Sweep summary")
    log("=" * 70)
    for galaxy, results in sweep.items():
        if results is None:
            log(f"  {galaxy:6s}: SKIPPED (preflight/emulation gate)")
            continue
        if not results:
            log(f"  {galaxy:6s}: no runs (dry-run or both skipped)")
            continue
        parts = []
        for label, (dt, rc) in results.items():
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            parts.append(f"{label} {dt/60:.1f}min {status}")
        log(f"  {galaxy:6s}: " + " | ".join(parts))
    log(f"  {'TOTAL':6s}: {total_dt:.0f}s ({total_dt/60:.1f} min, {total_dt/3600:.2f} hr)")


if __name__ == "__main__":
    main()