#!/usr/bin/env python3
"""
run_skirt_production.py — Tal Shiar SKIRT Pipeline, production driver

Runs the production r142_dust.ski and r142_nodust.ski files as-is (no
rewriting). Expects the ski files to be configured for the full intended
production: 12 inclinations, 1e7 photons, 500 pixels.

What this does:
    1. Preflight: confirms ski files, particle files, and SKIRT binary exist.
    2. Stages particle files (symlinks) into the output directory so SKIRT
       can find them via relative paths.
    3. Runs `skirt -e` emulation on both ski files as a schema canary —
       if this fails, there is no point starting a 1+ hour real run.
    4. Runs SKIRT for real: dust first, then nodust. Times each.
    5. Reports total wall time and lists output files.

Usage:
    # Foreground (will tie up terminal; good for first production run)
    python run_skirt_production.py

    # Detached via nohup (survives terminal close, logs to production.log)
    python run_skirt_production.py --detach

    # Dry run (emulation only, no real SKIRT run)
    python run_skirt_production.py --dry-run

    # Skip dust or nodust if already done
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
# Galaxy selection — change this for r107 / r320
# ---------------------------------------------------------------------------
GALAXY_ID = "r320"

DEFAULT_SKIRT = "/mnt/data0/jillian/SKIRT/release/SKIRT/main/skirt"


def log(msg):
    """Print with timestamp so the nohup log is useful."""
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def preflight(particle_dir, ski_dir, skirt_bin, required_particles, ski_files):
    """Check everything is in place before we commit to hours of runtime."""
    log(f"Preflight checks...")

    # Particle files
    missing = [f for f in required_particles if not (particle_dir / f).exists()]
    if missing:
        log(f"  ERROR: missing particle files in {particle_dir}: {missing}")
        return False
    for f in required_particles:
        size_mb = (particle_dir / f).stat().st_size / 1e6
        log(f"  {particle_dir / f}: {size_mb:.2f} MB")

    # Ski files
    for f in ski_files:
        path = ski_dir / f
        if not path.exists():
            log(f"  ERROR: missing ski file: {path}")
            return False
        size_kb = path.stat().st_size / 1e3
        log(f"  {path}: {size_kb:.1f} KB")

    # Skirt binary
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
    # Live stream rather than capture, so nohup log shows progress in real time.
    # SKIRT writes its own per-run .log file in workdir anyway — this is just
    # for the driver's own progress visibility.
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


def main():
    parser = argparse.ArgumentParser(description="Production SKIRT driver for Tal Shiar pipeline")
    parser.add_argument("--particle-dir", default=f"/mnt/data0/pkrsnak/romulus/{GALAXY_ID}",
                        help="Directory containing stars.txt, youngStars.txt, gas.txt")
    parser.add_argument("--ski-dir", default=None,
                        help="Directory containing production ski files "
                             "(default: auto-detect between current dir and ~/tal-shiar-SKIRT/python)")
    parser.add_argument("--output-subdir", default="production",
                        help="Subdir of --particle-dir for production outputs (default: production)")
    parser.add_argument("--skirt", default=DEFAULT_SKIRT, help="Path to skirt binary")
    parser.add_argument("--skip-dust", action="store_true", help="Skip dust run")
    parser.add_argument("--skip-nodust", action="store_true", help="Skip nodust run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run emulation only; don't start real SKIRT runs")
    parser.add_argument("--detach", action="store_true",
                        help="Re-exec via nohup and detach. Driver log at <output>/production.log")
    args = parser.parse_args()

    particle_dir = Path(args.particle_dir).resolve()

    # Auto-detect ski-dir if not given
    if args.ski_dir is None:
        candidates = [
            Path.cwd(),
            Path.home() / "tal-shiar-SKIRT" / "python",
            Path.home() / "tal-shiar-SKIRT" / "src",
        ]
        ski_dir = None
        for cand in candidates:
            if (cand / f"{GALAXY_ID}_dust.ski").exists():
                ski_dir = cand
                break
        if ski_dir is None:
            log("ERROR: could not find r142_dust.ski in any of:")
            for c in candidates:
                log(f"  {c}")
            log("  Pass --ski-dir explicitly.")
            sys.exit(1)
    else:
        ski_dir = Path(args.ski_dir).resolve()

    output_dir = particle_dir / args.output_subdir
    output_dir.mkdir(exist_ok=True)

    required_particles = ["stars.txt", "youngStars.txt", "gas.txt"]
    ski_files = [f"{GALAXY_ID}_dust.ski", f"{GALAXY_ID}_nodust.ski"]

    # --- nohup detach mode: re-exec ourselves under nohup ---
    if args.detach:
        log_path = output_dir / "production.log"
        # Build command with all args except --detach
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
                start_new_session=True,  # detach from terminal
            )
        log(f"Detached. PID logged in {log_path}. Exiting parent.")
        return

    # --- Normal (attached) execution ---
    log("=" * 70)
    log(f"Tal Shiar SKIRT production run")
    log(f"  particle_dir: {particle_dir}")
    log(f"  ski_dir:      {ski_dir}")
    log(f"  output_dir:   {output_dir}")
    log(f"  skirt:        {args.skirt}")
    log("=" * 70)

    if not preflight(particle_dir, ski_dir, args.skirt, required_particles, ski_files):
        sys.exit(1)

    stage_particle_files(output_dir, particle_dir, required_particles)

    # Which runs to execute
    runs = []
    if not args.skip_dust:
        runs.append(("dust", ski_dir / f"{GALAXY_ID}_dust.ski"))
    if not args.skip_nodust:
        runs.append(("nodust", ski_dir / f"{GALAXY_ID}_nodust.ski"))

    if not runs:
        log("Nothing to do (both --skip-dust and --skip-nodust set).")
        return

    # Emulation sweep
    log("")
    log("Emulation pass (schema canary)...")
    for label, ski in runs:
        if not run_emulation(args.skirt, ski, output_dir):
            log("Emulation failed. Aborting before committing to real run.")
            sys.exit(1)

    if args.dry_run:
        log("Dry run complete. Emulation passed; exiting without real SKIRT run.")
        return

    # Real runs
    log("")
    log("Starting production SKIRT runs...")
    total_start = perf_counter()
    results = {}
    for label, ski in runs:
        dt, rc = run_skirt(args.skirt, ski, output_dir, label)
        results[label] = (dt, rc)
        if rc != 0:
            log(f"STOPPING due to {label} failure.")
            break

    total_dt = perf_counter() - total_start

    # Summary
    log("")
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    for label, (dt, rc) in results.items():
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        log(f"  {label:8s}: {dt:7.1f}s ({dt/60:5.1f} min)  {status}")
    log(f"  {'TOTAL':8s}: {total_dt:7.1f}s ({total_dt/60:5.1f} min, {total_dt/3600:.2f} hr)")

    # Output inventory
    outputs = sorted(output_dir.glob(f"{GALAXY_ID}_*"))
    log(f"")
    log(f"Output files in {output_dir}: {len(outputs)} total")
    type_counts = {}
    for p in outputs:
        ext = p.suffix
        type_counts[ext] = type_counts.get(ext, 0) + 1
    for ext, count in sorted(type_counts.items()):
        log(f"  *{ext}: {count}")

    # Sanity: expected file count
    # Per inclination, dust run produces: 1 sed + 1 total.fits + 3 broadband.fits = 5
    # Plus per-run: 1 log + 1 parameters.xml + 1 convergence + 6 media_density_cuts = 9
    # nodust is same but no convergence/density cuts: (1+1+3)*12 + 2 = 62
    # Total expected: ~(5*12 + 9) + (5*12 + 2) = 131
    log("")
    log("  Spot-check SEDs:")
    seds = sorted(output_dir.glob("*_sed.dat"))
    log(f"    {len(seds)} sed.dat files (expected 24 = 12 inclinations x 2 runs)")


if __name__ == "__main__":
    main()