#!/usr/bin/env python3
"""
make_money_shots.py — Tal Shiar SKIRT Pipeline, RGB image gallery

Builds telescope-like RGB composites from the SKIRT _total.fits datacubes.
For each galaxy, generates a face-on / 45° / edge-on triplet, then a 3x3
grid showing all three galaxies at all three inclinations.

Mapping: SDSS r → R, SDSS g → G, GALEX NUV → B (boosted)
        NUV-as-blue makes dust effects visible: NUV is heavily
        attenuated by dust while r-band is much less affected.

Stretch: per-galaxy auto-anchor on r-band 99.5th percentile.

Usage:
    python make_money_shots.py --output-dir .
"""

import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GALAXIES = ["r142", "r107", "r320"]   # row order in the 3x3 grid
ROMULUS_DIR = Path("/mnt/data0/pkrsnak/romulus")

# Inclinations to display: face-on / 45° (closest is i05 at 40.91°) / edge-on
INCLINATIONS = [
    ("face-on",  0,  0.0),
    ("45°",      5, 40.91),
    ("edge-on", 11, 90.0),
]

# Broadband cube slice ordering. SKIRT's PredefinedBandWavelengthGrid
# (GALEX + SDSS + 2MASS enabled) writes 10 slices in this fixed order.
BAND_INDEX = {
    "FUV": 0, "NUV": 1,
    "u":   2, "g":   3, "r":  4, "i": 5, "z": 6,
    "J":   7, "H":   8, "Ks": 9,
}

# Lupton parameters
LUPTON_Q = 10
LUPTON_BLUE_BOOST = 3.0   # multiply NUV input to compensate for low UV flux
STRETCH_DIVISOR = 5.0     # smaller = brighter overall (try 10 or 20 for more punch)

# FOV: cube is 500x500 over ±35 kpc; we crop to ±15 kpc center for display
CUBE_HALF_KPC = 35.0
FOV_HALF_KPC = 15.0

# Per-galaxy stretch cache
_stretch_cache = {}

NUV_SMOOTH_SIGMA = 4.0   # Gaussian sigma in pixels; smooths sparse young-star particles


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fits_path(galaxy, inc_idx, inc_deg):
    name = f"i{inc_idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return ROMULUS_DIR / galaxy / "production" / f"{galaxy}_dust_{name}_total.fits"


def load_band_image(galaxy, inc_idx, inc_deg, band_name):
    path = fits_path(galaxy, inc_idx, inc_deg)
    if not path.exists():
        raise FileNotFoundError(f"Missing FITS: {path}")
    if band_name not in BAND_INDEX:
        raise KeyError(f"Unknown band '{band_name}'")
    with fits.open(path) as hdul:
        cube = hdul[0].data  # (n_bands, ny, nx)
    return cube[BAND_INDEX[band_name]]


def get_stretch_for_galaxy(galaxy):
    """Per-galaxy stretch anchored on face-on r-band 99.5th percentile."""
    if galaxy not in _stretch_cache:
        face_r = load_band_image(galaxy, 0, 0.0, "r")
        positive = face_r[face_r > 0]
        anchor = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        _stretch_cache[galaxy] = anchor / STRETCH_DIVISOR
    return _stretch_cache[galaxy]


def build_rgb(galaxy, inc_idx, inc_deg):
    """
    RGB from SDSS i/r/g for a redder, Hubble-style palette.
    Mapping: i (slice 5) → R, r (slice 4) → G, g (slice 3) → B.
    Per-galaxy auto-stretch anchored on face-on r-band.
    Blue channel is Gaussian-smoothed to diffuse sparse young-star particles.
    """
    r_img = load_band_image(galaxy, inc_idx, inc_deg, "i")
    g_img = load_band_image(galaxy, inc_idx, inc_deg, "r")
    b_img = load_band_image(galaxy, inc_idx, inc_deg, "g")
    b_img = gaussian_filter(b_img, sigma=NUV_SMOOTH_SIGMA)
    stretch = get_stretch_for_galaxy(galaxy)
    return make_lupton_rgb(r_img, g_img, b_img, Q=LUPTON_Q, stretch=stretch)


def crop_center(rgb):
    """Crop ±35 kpc cube to ±FOV_HALF_KPC center."""
    n = rgb.shape[0]
    pix_per_kpc = n / (2 * CUBE_HALF_KPC)
    half_pix = int(FOV_HALF_KPC * pix_per_kpc)
    cy, cx = n // 2, n // 2
    return rgb[cy - half_pix:cy + half_pix, cx - half_pix:cx + half_pix]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _imshow_rgb(ax, rgb, title=None):
    cropped = crop_center(rgb)
    ax.imshow(cropped, origin="lower",
              extent=[-FOV_HALF_KPC, FOV_HALF_KPC,
                      -FOV_HALF_KPC, FOV_HALF_KPC])
    if title:
        ax.set_title(title)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])


def plot_per_galaxy_triplet(galaxy, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (label, inc_idx, inc_deg) in zip(axes, INCLINATIONS):
        rgb = build_rgb(galaxy, inc_idx, inc_deg)
        _imshow_rgb(ax, rgb, title=f"{label} (i = {inc_deg:.1f}°)")
    fig.suptitle(f"{galaxy}", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = output_dir / f"{galaxy}_money_shot_triplet.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_three_by_three(output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))

    for row_idx, galaxy in enumerate(GALAXIES):
        for col_idx, (label, inc_idx, inc_deg) in enumerate(INCLINATIONS):
            ax = axes[row_idx, col_idx]
            rgb = build_rgb(galaxy, inc_idx, inc_deg)
            cropped = crop_center(rgb)
            ax.imshow(cropped, origin="lower",
                      extent=[-FOV_HALF_KPC, FOV_HALF_KPC,
                              -FOV_HALF_KPC, FOV_HALF_KPC])
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])

            if row_idx == 0:
                ax.set_title(f"{label} (i = {inc_deg:.1f}°)", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{galaxy}\n\ny (kpc)", fontsize=11)
            else:
                ax.set_ylabel("")
            if row_idx == 2:
                ax.set_xlabel("x (kpc)")
            else:
                ax.set_xlabel("")

    fig.suptitle("Tal Shiar SKIRT — Galaxy gallery (SDSS r/g + GALEX NUV)",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    out_path = output_dir / "money_shot_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LUPTON_Q, LUPTON_BLUE_BOOST, STRETCH_DIVISOR

    parser = argparse.ArgumentParser(description="Build RGB galaxy gallery from SKIRT datacubes")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for PNGs (default: cwd)")
    parser.add_argument("--lupton-Q", type=float, default=LUPTON_Q,
                        help=f"Lupton Q (asinh transition; default {LUPTON_Q})")
    parser.add_argument("--blue-boost", type=float, default=LUPTON_BLUE_BOOST,
                        help=f"Multiplier for blue (NUV) channel (default {LUPTON_BLUE_BOOST})")
    parser.add_argument("--stretch-divisor", type=float, default=STRETCH_DIVISOR,
                        help=f"Stretch = anchor/divisor; smaller = brighter (default {STRETCH_DIVISOR})")
    args = parser.parse_args()

    LUPTON_Q = args.lupton_Q
    LUPTON_BLUE_BOOST = args.blue_boost
    STRETCH_DIVISOR = args.stretch_divisor

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)
    print(f"Output dir: {output_dir}")
    print(f"Lupton Q = {LUPTON_Q}, blue_boost = {LUPTON_BLUE_BOOST}, "
          f"stretch_divisor = {STRETCH_DIVISOR}")
    print()

    # Flux range check
    print("Flux range check (SDSS r-band, face-on):")
    for galaxy in GALAXIES:
        try:
            img = load_band_image(galaxy, 0, 0.0, "r")
            vmax = float(np.nanmax(img))
            positive = img[img > 0]
            vmean = float(np.nanmean(positive)) if positive.size else 0.0
            v995 = float(np.percentile(positive, 99.5)) if positive.size else 0.0
            print(f"  {galaxy}: max = {vmax:.3e}, mean(>0) = {vmean:.3e}, "
                  f"99.5%ile = {v995:.3e}")
        except FileNotFoundError as e:
            print(f"  {galaxy}: MISSING ({e})")
            return
    print()

    # Stretch values that will be used
    print("Per-galaxy stretch values:")
    for galaxy in GALAXIES:
        s = get_stretch_for_galaxy(galaxy)
        print(f"  {galaxy}: stretch = {s:.4e}")
    print()

    print("Building per-galaxy triplets...")
    for galaxy in GALAXIES:
        try:
            plot_per_galaxy_triplet(galaxy, output_dir)
        except Exception as e:
            print(f"  ERROR for {galaxy}: {type(e).__name__}: {e}")

    print("\nBuilding 3x3 grid...")
    try:
        plot_three_by_three(output_dir)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()