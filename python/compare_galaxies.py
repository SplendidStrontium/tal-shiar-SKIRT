#!/usr/bin/env python3
"""
compare_galaxies.py — Tal Shiar SKIRT Pipeline, cross-galaxy comparison

Combines attenuation tables from r142, r107, r320 into a side-by-side
comparison plot:
    Left panel:  A(V) vs cos(i) for all three galaxies
    Right panel: A(lambda) at edge-on for all three galaxies

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

# Match plot_attenuation.py's reference wavelengths
WL_FUV = 0.154
WL_NUV = 0.230
WL_V = 0.551
WL_K = 2.19


def load_table(galaxy):
    """
    Read galaxy's attenuation table written by plot_attenuation.py.
    Format: row 0 has wavelengths (col 0 is NaN), rows 1+ are
    [inclination_deg, A(lambda_0), A(lambda_1), ...].
    Returns (inclinations_deg, wavelengths, A_matrix).
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


def plot_av_vs_cos_i_panel(ax, data):
    """Left panel: A(V) vs cos(i) for all galaxies."""
    for gal, (inc, wl, A) in data.items():
        cos_i = np.cos(np.deg2rad(inc))
        iV = nearest_idx(wl, WL_V)
        ax.plot(cos_i, A[:, iV],
                marker=MARKERS[gal], color=COLORS[gal],
                lw=2, ms=7, label=gal)

    ax.set_xlabel(r"$\cos\,i$ (1 = face-on, 0 = edge-on)")
    ax.set_ylabel(r"$A(V)$ (mag)")
    ax.set_title(r"V-band attenuation vs orientation")
    ax.set_xlim(1.02, -0.02)  # face-on left, edge-on right
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.text(0.02, 0.04, "face-on", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="gray")
    ax.text(0.98, 0.04, "edge-on", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")


def plot_attenuation_curves_panel(ax, data, target_inc_deg=81.82):
    """Right panel: A(lambda) at peak inclination for all galaxies."""
    for gal, (inc, wl, A) in data.items():
        # Find row nearest to target inclination
        i_row = int(np.argmin(np.abs(inc - target_inc_deg)))
        actual_inc = inc[i_row]
        ax.plot(wl, A[i_row],
                marker=MARKERS[gal], color=COLORS[gal],
                lw=2, ms=5, label=f"{gal} (i={actual_inc:.1f}°)")

    ax.set_xscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$A(\lambda)$ (mag)")
    ax.set_title(rf"Attenuation curve near edge-on (i$\sim${target_inc_deg:.0f}°)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    # Annotate photometric bands
    bands = {"FUV": WL_FUV, "NUV": WL_NUV, "V": WL_V, "K": WL_K}
    for label, lam in bands.items():
        ax.axvline(lam, color="gray", lw=0.3, alpha=0.4)


def print_summary(data):
    """Text summary for presentation."""
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

        # face-on (cos_i closest to 1)
        i_face = int(np.argmax(cos_i))
        # peak (max attenuation in V)
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

    # Print
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
                        help="Output directory for comparison plot (default: cwd)")
    parser.add_argument("--target-inc", type=float, default=81.82,
                        help="Inclination (deg) for the A(lambda) panel (default: 81.82°)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    # Load all three
    print("Loading attenuation tables...")
    data = {}
    for gal in GALAXIES:
        try:
            data[gal] = load_table(gal)
            print(f"  {gal}: {data[gal][0].shape[0]} inclinations × {data[gal][1].shape[0]} wavelengths")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return

    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_av_vs_cos_i_panel(axes[0], data)
    plot_attenuation_curves_panel(axes[1], data, target_inc_deg=args.target_inc)

    fig.suptitle("Tal Shiar SKIRT — Three-galaxy comparison", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "comparison_three_galaxies.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    print_summary(data)

if __name__ == "__main__":
    main()