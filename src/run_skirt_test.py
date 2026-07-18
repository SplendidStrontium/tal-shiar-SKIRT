#!/usr/bin/env python3
"""
run_skirt_test.py — Tal Shiar SKIRT Pipeline, test driver

Runs a low-photon validation of the SKIRT setup for one halo at a single
inclination (face-on by default) with both the dust and no-dust ski files,
then projects the cost of the full face-on production sweep.

What this does:
    1. Reads the production {galaxy}_dust.ski / {galaxy}_nodust.ski files
       from the ski directory (src/ski/).
    2. Rewrites them with (a) test photon count, (b) test pixel count,
       (c) only the requested inclinations, into a test_runs/ subdir
    3. Runs `skirt -e` (emulation mode) on each to catch schema errors
       in seconds rather than minutes
    4. Runs SKIRT for real with timing telemetry
    5. Prints per-run wall time and projects production cost

Why a separate driver instead of re-running generate_ski.py:
    - Production .ski files stay pristine (no test-parameter contamination)
    - Test output goes into test_runs/ and doesn't pollute production output
    - Lets you iterate on test configs (photon count, inclinations, pixels)
      without touching generate_ski.py

Usage:
    # Default: galaxy..., face-on, 1e6 photons, 256 pixels
    python run_skirt_test.py

    # A different halo
    python run_skirt_test.py --galaxy r320

    # More photons
    python run_skirt_test.py --photons 5e6

Assumes the halo's particle files live at
/mnt/data0/pkrsnak/romulus/{galaxy}/ (override with --particle-dir).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# The .ski files live in src/ski/. This script is in src/, so the ski dir is
# a subdirectory of the script's own directory. SET THIS if you rename it.
SKI_DIRNAME = "ski_enterprise"

# Resolve from __file__ so it works regardless of the cwd you launch from.
SCRIPT_DIR = Path(__file__).resolve().parent          # .../tal-shiar-SKIRT/src
DEFAULT_SKI_DIR = SCRIPT_DIR / SKI_DIRNAME             # .../src/ski_enterprise

# Production configuration this test projects toward (kept in sync with
# generate_ski.py). Used only for the cost projection at the end.
PROD_PHOTONS = 5e7
N_RUNS = 20             # full dust+nodust sweep is 2 x N_RUNS SKIRT runs

# Default SKIRT binary — override with --skirt if needed
DEFAULT_SKIRT = "/mnt/data0/jillian/SKIRT/release/SKIRT/main/skirt"


def rewrite_ski_for_test(src_path, dst_path, num_photons, num_pixels, keep_incs_deg, max_level=None):
    """
    Rewrite a production .ski file for test running:
      - numPackets attribute on MonteCarloSimulation -> num_photons
      - numPixelsX/Y attributes on FullInstrument    -> num_pixels
      - Keep only FullInstrument blocks whose inclination is in keep_incs_deg

    keep_incs_deg: list of floats, matches against the production ski's
    embedded inclinations (rounded to 2 decimals).
    """
    text = Path(src_path).read_text()

    # 1. Swap numPackets
    text = re.sub(r'numPackets="[^"]+"', f'numPackets="{num_photons}"', text)

    # 2. Swap numPixelsX / numPixelsY everywhere
    text = re.sub(r'numPixelsX="[^"]+"', f'numPixelsX="{num_pixels}"', text)
    text = re.sub(r'numPixelsY="[^"]+"', f'numPixelsY="{num_pixels}"', text)

    # after the numPixels substitutions:
    if max_level is not None:
        text = re.sub(r'maxLevel="[^"]+"', f'maxLevel="{max_level}"', text)

    # 3. Filter FullInstrument blocks by inclination
    keep_set = {round(x, 2) for x in keep_incs_deg}

    def instrument_filter(match):
        block = match.group(0)
        inc_match = re.search(r'inclination="([^"]+) deg"', block)
        if inc_match:
            inc_val = round(float(inc_match.group(1)), 2)
            if inc_val in keep_set:
                return block
            else:
                return ""  # drop this instrument block
        return block

    # Match a FullInstrument from opening tag through closing </FullInstrument>,
    # including the nested wavelengthGrid element. DOTALL so . matches newlines.
    pattern = re.compile(
        r'<FullInstrument\s[^>]*>.*?</FullInstrument>', re.DOTALL
    )
    text = pattern.sub(instrument_filter, text)
    sed_pattern = re.compile(r'<SEDInstrument\s[^>]*/>')
    text = sed_pattern.sub(instrument_filter, text)

    # Clean up any empty lines left by removed blocks
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    Path(dst_path).write_text(text)

    # Sanity-confirm the rewrite produced the right number of instruments
    n_kept = text.count("<FullInstrument ")
    if n_kept != len(keep_set):
        print(f"  WARNING: requested {len(keep_set)} inclinations but "
              f"ski file has {n_kept} FullInstrument blocks. "
              f"Inclinations may not be rounding to values in the source ski.")
    return n_kept


def run_emulation(skirt_bin, ski_path, workdir):
    """
    Run SKIRT in emulation mode to validate the ski without photon transport.

    This is our schema-error canary: the SpatialGridConvergenceProbe bug took
    a full SKIRT startup + minutes before failing; emulation fails in seconds.

    workdir must be the dir containing stars.txt / youngStars.txt / gas.txt,
    because emulation resolves particle file references just like a real run.

    Returns True on success, False on failure (stdout printed either way).
    """
    print(f"  emulation: {ski_path.name} ... ", end="", flush=True)
    t0 = perf_counter()
    result = subprocess.run(
        [skirt_bin, "-e", str(ski_path.resolve())],
        cwd=workdir,
        capture_output=True, text=True,
    )
    dt = perf_counter() - t0
    if result.returncode == 0:
        print(f"OK ({dt:.1f}s)")
        return True
    else:
        print(f"FAILED ({dt:.1f}s)")
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)
        return False


def run_skirt(skirt_bin, ski_path, workdir):
    """
    Run SKIRT for real. Working directory is set so relative paths in the
    ski (stars.txt, gas.txt, youngStars.txt) resolve to the particle files.

    Returns wall time in seconds.
    """
    print(f"  running:   {ski_path.name} ... ", end="", flush=True)
    t0 = perf_counter()
    result = subprocess.run(
        [skirt_bin, str(ski_path.resolve())],
        cwd=workdir,
        capture_output=True, text=True,
    )
    dt = perf_counter() - t0
    if result.returncode == 0:
        print(f"OK ({dt:.1f}s wall)")
        # Extract "Finished simulation" line or similar telemetry
        for line in result.stdout.splitlines()[-20:]:
            if "Finished" in line or "Total CPU" in line or "photon packets" in line.lower():
                print(f"    {line.strip()}")
    else:
        print(f"FAILED ({dt:.1f}s)")
        print("--- last 30 lines of stdout ---")
        print("\n".join(result.stdout.splitlines()[-30:]))
        print("--- stderr ---")
        print(result.stderr)
    return dt, result.returncode


def main():
    parser = argparse.ArgumentParser(description="SKIRT test-run driver for Tal Shiar pipeline")
    parser.add_argument("--galaxy", default="r741",
                        help="Run to test (e.g. r741, r488_BH8)")
    parser.add_argument("--particle-dir", default=None,
                        help="Dir with stars.txt, youngStars.txt, gas.txt "
                             "(default: /mnt/data0/pkrsnak/romulus/enterprise/{galaxy})")
    parser.add_argument("--ski-dir", default=None,
                        help=f"Directory containing {{galaxy}}_dust.ski / "
                             f"{{galaxy}}_nodust.ski (default: {DEFAULT_SKI_DIR})")
    parser.add_argument("--test-subdir", default="test_runs",
                        help="Subdirectory of --particle-dir for test outputs")
    parser.add_argument("--photons", default="1e6",
                        help="Photon packet count for test (default 1e6)")
    parser.add_argument("--pixels", type=int, default=256,
                        help="Pixels per side for test images (default 256)")
    parser.add_argument("--inclinations", type=float, nargs="+", default=[0.0, 90.0],
                        help="Inclinations in degrees to keep (default: 0, 90)")
    parser.add_argument("--skirt", default=DEFAULT_SKIRT,
                        help=f"Path to skirt binary (default: {DEFAULT_SKIRT})")
    parser.add_argument("--skip-dust", action="store_true",
                        help="Only run the no-dust ski file")
    parser.add_argument("--skip-nodust", action="store_true",
                        help="Only run the dust ski file")
    parser.add_argument("--max-level", type=int, default=None,
                        help="Maximum level for test (default: None)")
    args = parser.parse_args()

    galaxy = args.galaxy

    # Resolve galaxy-dependent defaults after we know --galaxy
    particle_dir = Path(args.particle_dir).resolve() if args.particle_dir \
        else Path(f"/mnt/data0/pkrsnak/romulus/enterprise/products/{galaxy}/particles").resolve()
    ski_dir = Path(args.ski_dir).resolve() if args.ski_dir \
        else DEFAULT_SKI_DIR
    test_dir = particle_dir.parent / args.test_subdir
    test_dir.mkdir(exist_ok=True)

    # --- Preflight: confirm particle files exist ---
    print(f"Galaxy:       {galaxy}")
    print(f"Particle dir: {particle_dir}")
    print(f"Ski dir:      {ski_dir}")
    required = ["stars.txt", "youngStars.txt", "gas.txt"]
    missing = [f for f in required if not (particle_dir / f).exists()]
    if missing:
        print(f"ERROR: missing particle files: {missing}")
        print(f"  Run make_particles.py first, or check --particle-dir")
        sys.exit(1)
    for f in required:
        size_mb = (particle_dir / f).stat().st_size / 1e6
        nlines = sum(1 for _ in open(particle_dir / f))
        print(f"  {f}: {size_mb:.2f} MB, {nlines} lines")

    # --- Preflight: confirm ski files exist ---
    dust_src = ski_dir / f"{galaxy}_dust.ski"
    nodust_src = ski_dir / f"{galaxy}_nodust.ski"
    if not dust_src.exists():
        print(f"ERROR: {dust_src} not found "
              f"(check SKI_DIRNAME='{SKI_DIRNAME}' or pass --ski-dir)")
        sys.exit(1)
    if not nodust_src.exists():
        print(f"ERROR: {nodust_src} not found "
              f"(check SKI_DIRNAME='{SKI_DIRNAME}' or pass --ski-dir)")
        sys.exit(1)

    # --- Preflight: confirm SKIRT binary exists and is executable ---
    if not Path(args.skirt).exists():
        print(f"ERROR: SKIRT binary not found at {args.skirt}")
        sys.exit(1)

    print(f"\nTest config:")
    print(f"  photons:      {args.photons}")
    print(f"  pixels:       {args.pixels}")
    print(f"  inclinations: {args.inclinations}")
    print(f"  output dir:   {test_dir}")

    # --- Stage test ski files ---
    # Symlink particle files into test_dir so SKIRT can find them via the
    # relative paths in the ski.
    for f in required:
        link = test_dir / f
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(particle_dir / f)

    dust_test = test_dir / f"{galaxy}_dust_test.ski"
    nodust_test = test_dir / f"{galaxy}_nodust_test.ski"

    print("\nRewriting ski files for test...")
    n_dust = rewrite_ski_for_test(dust_src, dust_test,
                                   args.photons, args.pixels, args.inclinations, args.max_level)
    n_nodust = rewrite_ski_for_test(nodust_src, nodust_test,
                                     args.photons, args.pixels, args.inclinations, args.max_level)
    print(f"  {dust_test.name}:   {n_dust} instruments")
    print(f"  {nodust_test.name}: {n_nodust} instruments")

    # --- Emulation pass ---
    print("\nEmulation pass (schema validation)...")
    all_ok = True
    runs_to_do = []
    if not args.skip_dust:
        runs_to_do.append(("dust", dust_test))
    if not args.skip_nodust:
        runs_to_do.append(("nodust", nodust_test))

    for label, ski in runs_to_do:
        if not run_emulation(args.skirt, ski, test_dir):
            all_ok = False

    if not all_ok:
        print("\nEmulation failed. Not running SKIRT. Fix ski errors above.")
        sys.exit(1)

    # --- Real runs ---
    print("\nSKIRT runs...")
    timings = {}
    for label, ski in runs_to_do:
        dt, rc = run_skirt(args.skirt, ski, test_dir)
        timings[label] = (dt, rc)

    # --- Summary + production cost projection ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = 0
    for label, (dt, rc) in timings.items():
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {label:8s}: {dt:7.1f}s  {status}")
        total += dt
    print(f"  {'total':8s}: {total:7.1f}s")

    # Projection to production (face-on, PROD_PHOTONS, dust+nodust, N_HALOS).
    # Dust scales ~linearly with photon count; nodust is NoMedium (no transport)
    # and barely depends on photons, so we project it flat. Pixels are nearly
    # free, so the 256 -> 500 change is ignored. Assumes the test was run
    # face-on (matching production); a multi-inclination test will overestimate.
    test_photons = float(args.photons)
    photon_scale = PROD_PHOTONS / test_photons
    dust_dt = timings.get("dust", (0.0, 0))[0]
    nodust_dt = timings.get("nodust", (0.0, 0))[0]
    per_halo = dust_dt * photon_scale + nodust_dt
    full_sweep = per_halo * N_RUNS
    print()
    print(f"Production projection (face-on, {PROD_PHOTONS:.0e} photons, dust+nodust):")
    print(f"  per halo:   ~{per_halo:7.0f} s  = ~{per_halo/60:.1f} min")
    print(f"  {N_RUNS} runs:   ~{full_sweep:7.0f} s  = ~{full_sweep/60:.1f} min "
          f"= ~{full_sweep/3600:.2f} hr")
    print(f"  (dust photon scale: {photon_scale:.0f}x; nodust projected flat)")

    # --- List output files ---
    print(f"\nOutput files in {test_dir}:")
    outputs = sorted(test_dir.glob(f"{galaxy}_*"))
    for p in outputs[:20]:
        size = p.stat().st_size
        size_str = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.1f} KB"
        print(f"  {p.name}: {size_str}")
    if len(outputs) > 20:
        print(f"  ... and {len(outputs) - 20} more")


if __name__ == "__main__":
    main()