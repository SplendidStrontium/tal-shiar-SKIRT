#!/usr/bin/env python3
"""
compare_galaxies.py — Tal Shiar SKIRT Pipeline, cross-galaxy comparison

Combines attenuation tables from r142, r107, r320 into two SEPARATE
figures (one per slide):

    1. comparison_av_vs_orientation.png
       A(V) vs cos(i) for all three galaxies.

    2. comparison_attenuation_curve_edge_on.png
       A(lambda) at edge-on for all three galaxies, with FUV/NUV/V/K
       photometric bands marked. X-axis in nanometers.

Run after plot_attenuation.py has produced *_attenuation_table.dat for
each galaxy. Reads from each galaxy's production/ directory.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GALAXIES = ["r107", "r142", "r320"]
ROMULUS_DIR = Path("/mnt/data0/pkrsnak/romulus")

# Distinct, accessible colors for three galaxies
COLORS = {"r107": "#CC6677", "r142": "#117733", "r320": "#332288"}
MARKERS = {"r107": "s", "r142": "o", "r320": "^"}

# Reference wavelengths in microns (matches plot_attenuation.py)
WL_FUV = 0.154
WL_NUV = 0.230
WL_V = 0.551
WL_K = 2.19


def load_table(galaxy):
    """
    Read galaxy's attenuation table written by plot_attenuation.py.
    Format: row 0 has wavelengths (col 0 is NaN), rows 1+ are
    [inclination_deg, A(lambda_0), A(lambda_1), ...].
    Returns (inclinations_deg, wavelengths_micron, A_matrix).
    """
    path = ROMULUS_DIR / galaxy / "production" / f"{galaxy}_attenuation_table.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    raw = np.loadtxt(path)
    wavelengths = raw[0, 1:]
    inclinations = raw[1:, 0]
    A = raw[1:, 1:]
    return inclinations, wavelengths, A


def nearest_idx(wavelengths, target):
    return int(np.argmin(np.abs(wavelengths - target)))


# ---------------------------------------------------------------------------
# Plot 1: A(V) vs cos(i)
# ---------------------------------------------------------------------------

def plot_av_vs_orientation(data, out_path):
    """A(V) vs cos(i) for all galaxies, on its own figure."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for gal, (inc, wl, A) in data.items():
        cos_i = np.cos(np.deg2rad(inc))
        iV = nearest_idx(wl, WL_V)
        ax.plot(cos_i, A[:, iV],
                marker=MARKERS[gal], color=COLORS[gal],
                lw=2, ms=7, label=gal)

    ax.set_xlabel(r"$\cos\,i$ (1 = face-on, 0 = edge-on)")
    ax.set_ylabel(r"$A(V)$ (mag)")
    ax.set_title("Attenuation vs. inclination")
    ax.set_xlim(1.02, -0.02)  # face-on left, edge-on right
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    # Move face-on / edge-on annotations below the data so they don't
    # collide with the r107 line near A(V) ~ 0. Sit them just under the
    # axis using axes-fraction coordinates with negative y.
    ax.annotate("face-on", xy=(1.0, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -28), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="gray")
    ax.annotate("edge-on", xy=(0.0, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -28), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: A(lambda) at edge-on, x-axis in nm, with band labels
# ---------------------------------------------------------------------------

def plot_attenuation_curve_edge_on(data, out_path, target_inc_deg=81.82):
    """A(lambda) near edge-on for all galaxies, x-axis in nm, bands labeled."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Track min A so band labels can sit below the data
    y_min_data = 0.0
    y_max_data = 0.0

    for gal, (inc, wl, A) in data.items():
        i_row = int(np.argmin(np.abs(inc - target_inc_deg)))
        actual_inc = inc[i_row]
        wl_nm = wl * 1000.0  # micron -> nm
        ax.plot(wl_nm, A[i_row],
                marker=MARKERS[gal], color=COLORS[gal],
                lw=2, ms=5, label=f"{gal} (i = {actual_inc:.1f}°)")
        y_min_data = min(y_min_data, np.nanmin(A[i_row]))
        y_max_data = max(y_max_data, np.nanmax(A[i_row]))

    ax.set_xscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel(r"$A(\lambda)$ (mag)")
    ax.set_title("Attenuation vs. wavelength")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=11, frameon=True)

    # Override matplotlib's default log-scale "10^N" tick formatting with
    # explicit values in hundreds-of-nm so the audience sees actual numbers.
    # Pick ticks that span the data range without being too dense.
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    major_ticks = [200, 300, 500, 1000, 2000]
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in major_ticks]))
    # Suppress the default minor ticks/labels (which would otherwise add
    # 10^2, 10^3 in scientific notation around our explicit ticks).
    ax.xaxis.set_minor_locator(NullLocator())

    # Give the band labels some headroom below the data (negative side).
    y_range = y_max_data - y_min_data
    pad = max(0.10 * y_range, 0.15)
    ax.set_ylim(bottom=y_min_data - pad, top=y_max_data * 1.05)

    # Annotate the four key photometric bands. Vertical line + label
    # at the bottom of the axes, all in nm.
    bands = {"FUV": WL_FUV, "NUV": WL_NUV, "V": WL_V, "K": WL_K}
    label_y = y_min_data - pad * 0.55
    for label, lam_micron in bands.items():
        lam_nm = lam_micron * 1000.0
        ax.axvline(lam_nm, color="gray", lw=0.7, ls="--", alpha=0.55)
        ax.text(lam_nm, label_y, label,
                ha="center", va="center", fontsize=10, color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.5, alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Text summary (unchanged from previous version)
# ---------------------------------------------------------------------------

def print_summary(data):
    print()
    print("=" * 76)
    print("CROSS-GALAXY COMPARISON")
    print("=" * 76)

    rows = []
    for gal in GALAXIES:
        inc, wl, A = data[gal]
        cos_i = np.cos(np.deg2rad(inc))

        iV = nearest_idx(wl, WL_V)
        iFUV = nearest_idx(wl, WL_FUV)

        i_face = int(np.argmax(cos_i))
        i_peak = int(np.nanargmax(A[:, iV]))

        rows.append({
            "gal":       gal,
            "AV_face":   A[i_face, iV],
            "AV_peak":   A[i_peak, iV],
            "AV_dyn":    A[i_peak, iV] - A[i_face, iV],
            "AFUV_face": A[i_face, iFUV],
            "AFUV_peak": A[i_peak, iFUV],
            "i_peak":    inc[i_peak],
        })

    hdr = f"{'gal':<6}{'A(V) face':>10}{'A(V) peak':>10}{'ΔA(V)':>9}" \
          f"{'A(FUV) face':>13}{'A(FUV) peak':>13}{'i_peak':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['gal']:<6}{r['AV_face']:>10.3f}{r['AV_peak']:>10.3f}"
              f"{r['AV_dyn']:>9.3f}"
              f"{r['AFUV_face']:>13.3f}{r['AFUV_peak']:>13.3f}"
              f"{r['i_peak']:>8.1f}°")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Cross-galaxy attenuation comparison")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for comparison plots (default: cwd)")
    parser.add_argument("--target-inc", type=float, default=81.82,
                        help="Inclination (deg) for the A(lambda) plot (default: 81.82°)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    print("Loading attenuation tables...")
    data = {}
    for gal in GALAXIES:
        try:
            data[gal] = load_table(gal)
            print(f"  {gal}: {data[gal][0].shape[0]} inclinations × {data[gal][1].shape[0]} wavelengths")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return

    # Two separate figures
    plot_av_vs_orientation(
        data,
        output_dir / "comparison_av_vs_orientation.png",
    )
    plot_attenuation_curve_edge_on(
        data,
        output_dir / "comparison_attenuation_curve_edge_on.png",
        target_inc_deg=args.target_inc,
    )

    print_summary(data)


if __name__ == "__main__":
    main()