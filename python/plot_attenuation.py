#!/usr/bin/env python3
"""
plot_attenuation.py — Tal Shiar SKIRT Pipeline, analysis step 1

Reads production SED files from SKIRT and produces two key diagnostic plots:

    1. A(lambda, i) curve family: 12 curves (one per inclination) of
       attenuation in magnitudes vs wavelength. Shows how attenuation
       strengthens with inclination and how the shape of the extinction
       curve changes.

    2. A(V) vs cos(i): single curve showing the canonical attenuation
       scalar vs orientation. Face-on (cos i = 1) on the right,
       edge-on (cos i = 0) on the left. Pure cos(i)^-1 = slab geometry;
       deviations reveal disk thickness effects.

Run after production completes. Looks for files matching:
    r142_dust_i{NN}_{inc}deg_sed.dat
    r142_nodust_i{NN}_{inc}deg_sed.dat

and matches them by inclination index.

Usage:
    # Default: looks in ./production (or --input-dir)
    python plot_attenuation.py --input-dir /mnt/data0/pkrsnak/romulus/r142/production

    # Custom output location
    python plot_attenuation.py --input-dir ... --output-dir ./plots

    # Skip saving, just show (requires X forwarding — not useful on Hamilton)
    python plot_attenuation.py --show
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no X needed on Hamilton
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

# ---------------------------------------------------------------------------
# Galaxy selection — change this for r107 / r320
# ---------------------------------------------------------------------------
GALAXY_ID = "r142"

# --- File pattern matching ---

# Inclination labels look like i00_00p00deg, i01_08p18deg, etc.
# Capture group 1: index, Capture group 2: degree value with 'p' as decimal.
INC_PATTERN = re.compile(r"_i(\d{2})_(\d{2}p\d{2})deg_sed\.dat$")


def parse_inc_from_filename(path):
    """Return (idx, inc_deg) from a SKIRT SED filename, or None if no match."""
    m = INC_PATTERN.search(str(path))
    if not m:
        return None
    idx = int(m.group(1))
    deg_str = m.group(2).replace("p", ".")
    return idx, float(deg_str)


def load_sed(path):
    """
    Read a SKIRT _sed.dat file.

    Returns (wavelength_micron, flux_jy) as numpy arrays.
    The file has comment lines starting with '#' followed by two-column data.
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected shape in {path}: {data.shape}")
    return data[:, 0], data[:, 1]


def collect_seds(input_dir, variant):
    """
    Collect all SED files for a given variant ('dust' or 'nodust').

    Returns a dict: {inc_idx: (inc_deg, wavelength, flux)}
    Sorted by inclination index.
    """
    pattern = f"{GALAXY_ID}_{variant}_i*_sed.dat"
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        # Fall back to pattern without r142 prefix, in case naming changes
        paths = sorted(input_dir.glob(f"*_{variant}_i*_sed.dat"))

    result = {}
    for p in paths:
        parsed = parse_inc_from_filename(p)
        if parsed is None:
            print(f"  WARNING: could not parse inclination from {p.name}")
            continue
        idx, inc_deg = parsed
        wl, flux = load_sed(p)
        result[idx] = (inc_deg, wl, flux)
    return result


def compute_attenuation(flux_dust, flux_nodust):
    """
    A(lambda) = -2.5 log10(F_dust / F_nodust).

    Handles zero/negative fluxes gracefully: returns NaN where the ratio
    is undefined (shouldn't happen in practice, but MC noise at the UV
    tail of edge-on runs could produce zero flux bins).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = flux_dust / flux_nodust
        A = np.where(ratio > 0, -2.5 * np.log10(ratio), np.nan)
    return A


# --- Plots ---

def plot_attenuation_curves(inclinations_deg, wavelengths, A_matrix, out_path):
    """
    Plot 1: A(lambda) for each inclination as colored curves.

    A_matrix shape: (n_inc, n_wavelength).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    n = len(inclinations_deg)
    colors = viridis(np.linspace(0, 1, n))

    for i, inc in enumerate(inclinations_deg):
        ax.plot(wavelengths, A_matrix[i], color=colors[i],
                label=f"{inc:5.1f}°", lw=1.4)

    ax.set_xscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"Attenuation $A(\lambda)$ (mag)")
    ax.set_title(f"{GALAXY_ID} — Attenuation curves by inclination")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Inclination", loc="upper right", fontsize=8,
              ncol=2, frameon=True)

    # Annotate common photometric bands at the top
    bands = {"FUV": 0.154, "NUV": 0.230, "u": 0.356, "g": 0.477,
             "r": 0.623, "i": 0.762, "z": 0.913,
             "J": 1.25, "H": 1.63, "K": 2.19}
    ymax = A_matrix[np.isfinite(A_matrix)].max()
    for label, lam in bands.items():
        if wavelengths.min() <= lam <= wavelengths.max():
            ax.axvline(lam, color="gray", lw=0.3, alpha=0.5)
            ax.text(lam, ymax * 1.02, label, ha="center", va="bottom",
                    fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_av_vs_cos_i(inclinations_deg, wavelengths, A_matrix, out_path,
                     ref_wavelength=0.551):
    """
    Plot 2: A(V) vs cos(i).

    ref_wavelength in microns. 0.551 = Johnson V-band center.
    Also plots A(FUV) and A(NUV) in a second panel for comparison.
    """
    cos_i = np.cos(np.deg2rad(inclinations_deg))

    # Pick the grid wavelength nearest to our reference bands
    def nearest_idx(lam):
        return int(np.argmin(np.abs(wavelengths - lam)))

    iV = nearest_idx(0.551)
    iFUV = nearest_idx(0.154)
    iNUV = nearest_idx(0.230)
    iK = nearest_idx(2.19)

    AV = A_matrix[:, iV]
    AFUV = A_matrix[:, iFUV]
    ANUV = A_matrix[:, iNUV]
    AK = A_matrix[:, iK]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(cos_i, AV, "o-", color="C0", label=f"V ({wavelengths[iV]:.3f} μm)", lw=2)
    ax.plot(cos_i, AFUV, "s-", color="C3", label=f"FUV ({wavelengths[iFUV]:.3f} μm)", lw=1.5, alpha=0.8)
    ax.plot(cos_i, ANUV, "^-", color="C1", label=f"NUV ({wavelengths[iNUV]:.3f} μm)", lw=1.5, alpha=0.8)
    ax.plot(cos_i, AK, "v-", color="C4", label=f"K ({wavelengths[iK]:.3f} μm)", lw=1.5, alpha=0.8)

    # Slab-geometry reference: A(i) = A_face / cos(i) — plotted for the V-band
    # anchored at the face-on measurement. Clipped to match data range so it
    # doesn't blow up the y-axis.
    if np.isfinite(AV[-1]) and AV[-1] > 0:
        # face-on is cos_i = 1 (last entry, inc=0)
        cos_grid = np.linspace(0.05, 1.0, 50)
        slab_V = AV[-1] / cos_grid
        ax.plot(cos_grid, slab_V, ":", color="C0", alpha=0.5, lw=1,
                label="V, slab A/cos(i)")

    ax.set_xlabel(r"$\cos\,i$ (1 = face-on, 0 = edge-on)")
    ax.set_ylabel(r"Attenuation $A(\lambda, i)$ (mag)")
    ax.set_title(f"{GALAXY_ID} — Attenuation vs orientation")

    # NIHAO / Trcka / Trayford convention: face-on on left, edge-on on right
    # so attenuation curves rise left-to-right. Achieve by putting cos_i=1
    # on the left and cos_i=0 on the right.
    ax.set_xlim(1.02, -0.02)

    # Clip y so slab divergence near cos_i=0 doesn't crush the data curves.
    # Use data max + 20% as ceiling, min of 0.
    y_max = np.nanmax(A_matrix) * 1.2 if np.any(np.isfinite(A_matrix)) else 10
    ax.set_ylim(min(-0.1, np.nanmin(A_matrix)), y_max)

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    # Annotate limits — face-on is at cos_i=1, which is on the LEFT of the plot
    ax.text(0.02, 0.02, "face-on", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="gray")
    ax.text(0.98, 0.02, "edge-on", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(inclinations_deg, wavelengths, A_matrix):
    """Print a quick text table of A at key bands for each inclination."""
    def nearest_idx(lam):
        return int(np.argmin(np.abs(wavelengths - lam)))

    iFUV = nearest_idx(0.154)
    iNUV = nearest_idx(0.230)
    iV = nearest_idx(0.551)
    iK = nearest_idx(2.19)

    print()
    print("  Attenuation summary (magnitudes):")
    print(f"  {'inc(deg)':>9s}  {'cos(i)':>7s}  {'A(FUV)':>7s}  {'A(NUV)':>7s}  {'A(V)':>7s}  {'A(K)':>7s}")
    for i, inc in enumerate(inclinations_deg):
        cos_i = np.cos(np.deg2rad(inc))
        print(f"  {inc:>9.2f}  {cos_i:>7.3f}  "
              f"{A_matrix[i, iFUV]:>7.3f}  {A_matrix[i, iNUV]:>7.3f}  "
              f"{A_matrix[i, iV]:>7.3f}  {A_matrix[i, iK]:>7.3f}")


def save_attenuation_table(inclinations_deg, wavelengths, A_matrix, out_path):
    """
    Save A(lambda, i) as a text table. Rows = inclinations, columns = wavelengths.
    Useful as input to downstream analysis or plotting in other tools.
    """
    header = "# A(lambda, i) in magnitudes\n"
    header += "# rows: inclinations (deg), cols: wavelengths (micron)\n"
    header += "# first row: wavelength values with leading 0 placeholder\n"
    header += "# subsequent rows: inclination(deg) followed by A values\n"

    # Build the output matrix with a header row of wavelengths
    n_inc = len(inclinations_deg)
    n_wl = len(wavelengths)
    out = np.zeros((n_inc + 1, n_wl + 1))
    out[0, 0] = np.nan  # placeholder
    out[0, 1:] = wavelengths
    out[1:, 0] = inclinations_deg
    out[1:, 1:] = A_matrix

    np.savetxt(out_path, out, header=header.rstrip("\n"), fmt="%.6e")
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot A(lambda, i) from SKIRT output")
    parser.add_argument("--input-dir",
                        default=f"/mnt/data0/pkrsnak/romulus/{GALAXY_ID}/production",
                        help="Directory containing SKIRT _sed.dat files")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: --input-dir)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(exist_ok=True)

    print(f"Loading SEDs from {input_dir}...")
    dust_seds = collect_seds(input_dir, "dust")
    nodust_seds = collect_seds(input_dir, "nodust")

    if not dust_seds:
        print(f"  ERROR: no dust SEDs found in {input_dir}")
        sys.exit(1)
    if not nodust_seds:
        print(f"  ERROR: no nodust SEDs found in {input_dir}")
        sys.exit(1)

    print(f"  Found {len(dust_seds)} dust SEDs, {len(nodust_seds)} nodust SEDs")

    # Match by inclination index
    shared_idx = sorted(set(dust_seds.keys()) & set(nodust_seds.keys()))
    only_dust = set(dust_seds.keys()) - set(nodust_seds.keys())
    only_nodust = set(nodust_seds.keys()) - set(dust_seds.keys())

    if only_dust:
        print(f"  WARNING: dust-only inclinations (no nodust match): {sorted(only_dust)}")
    if only_nodust:
        print(f"  WARNING: nodust-only inclinations (no dust match): {sorted(only_nodust)}")

    if not shared_idx:
        print("  ERROR: no inclinations have both dust and nodust SEDs")
        sys.exit(1)

    # Sanity: wavelength grids must match across all files
    _, wl_ref, _ = dust_seds[shared_idx[0]]
    for idx in shared_idx:
        for label, seds in [("dust", dust_seds), ("nodust", nodust_seds)]:
            _, wl, _ = seds[idx]
            if not np.allclose(wl, wl_ref):
                print(f"  ERROR: wavelength grid mismatch in {label} i{idx:02d}")
                sys.exit(1)

    # Build A(lambda, i) matrix
    inclinations_deg = np.array([dust_seds[i][0] for i in shared_idx])
    A_matrix = np.zeros((len(shared_idx), len(wl_ref)))
    for row, idx in enumerate(shared_idx):
        _, _, f_dust = dust_seds[idx]
        _, _, f_nodust = nodust_seds[idx]
        A_matrix[row] = compute_attenuation(f_dust, f_nodust)

    print(f"  A matrix: {A_matrix.shape[0]} inclinations × {A_matrix.shape[1]} wavelengths")
    n_nan = np.isnan(A_matrix).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values in A (zero/negative flux ratios)")

    # Print diagnostic table
    print_summary_table(inclinations_deg, wl_ref, A_matrix)

    # Plots
    print()
    print("Plotting...")
    plot_attenuation_curves(inclinations_deg, wl_ref, A_matrix,
                            output_dir / f"{GALAXY_ID}_attenuation_curves.png")
    plot_av_vs_cos_i(inclinations_deg, wl_ref, A_matrix,
                     output_dir / f"{GALAXY_ID}_av_vs_cos_i.png")

    # Save numerical table for downstream use
    save_attenuation_table(inclinations_deg, wl_ref, A_matrix,
                           output_dir / f"{GALAXY_ID}_attenuation_table.dat")

    print()
    print("Done.")


if __name__ == "__main__":
    main()