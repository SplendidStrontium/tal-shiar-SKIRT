#!/usr/bin/env python3
"""
make_dust_comparison.py — Tal Shiar SKIRT Pipeline, dust vs no-dust comparison

Builds side-by-side grayscale images of a single galaxy at a single orientation,
comparing the dust-included radiative transfer run to the dust-free run. Both
panels share the same intensity scale and stretch so the difference reflects
real attenuation, not display normalization.

Defaults are tuned for visually striking dust comparisons:
  - NUV band (~10x attenuation in dusty galaxies, vs ~2.6x in r-band)
  - No-dust-anchored stretch (dusty panel reads as "dim because of dust")
  - Gaussian smoothing on UV bands (suppresses young-star particle noise)

By default produces figures at i09 (~73.6°) and i10 (~81.8°) for galaxy r320,
in all three stretch modes. Generates 6 PNGs per run by default.

Usage:
    python make_dust_comparison.py                              # NUV, all 3 stretches, i09+i10
    python make_dust_comparison.py --band r                     # r-band instead
    python make_dust_comparison.py --stretch-mode nodust-anchor # one stretch only
    python make_dust_comparison.py --inc 10 --band NUV          # one inclination
"""

import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
try:
    from matplotlib.colors import AsinhNorm
    _HAS_ASINH_NORM = True
except ImportError:
    _HAS_ASINH_NORM = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROMULUS_DIR = Path("/mnt/data0/pkrsnak/romulus")

# Inclination index → degrees, read off the production/ filenames.
# 12 orientations evenly spaced in cos(i) from face-on (i00) to edge-on (i11).
INCLINATION_TABLE = {
    0:  0.00,   1:  8.18,   2: 16.36,   3: 24.55,
    4: 32.73,   5: 40.91,   6: 49.09,   7: 57.27,
    8: 65.45,   9: 73.64,  10: 81.82,  11: 90.00,
}

# SKIRT PredefinedBandWavelengthGrid slice ordering
# (GALEX + SDSS + 2MASS = 10 slices)
BAND_INDEX = {
    "FUV": 0, "NUV": 1,
    "u":   2, "g":   3, "r":  4, "i": 5, "z": 6,
    "J":   7, "H":   8, "Ks": 9,
}

# Default inclinations to render
DEFAULT_INC_INDICES = [9, 10]

# Default band — NUV gives the most dramatic dust contrast (~10x attenuation)
DEFAULT_BAND = "NUV"

# Display FOV — cube is 500x500 over ±35 kpc; crop to ±15 kpc for slide
CUBE_HALF_KPC = 35.0
FOV_HALF_KPC = 15.0

# Bands to smooth (UV bands trace sparse young-star particles, look granular).
# Sigma in pixels.
SMOOTH_BANDS = {"FUV", "NUV"}
SMOOTH_SIGMA = 4.0

# Stretch parameters
STRETCH_PERCENTILE = 99.5     # vmax = this percentile of the anchor image
ASINH_LINEAR_FRAC = 0.1       # asinh "linear width" as fraction of vmax;
                              # smaller = more aggressive nonlinear compression

STRETCH_MODES = ["dusty-anchor", "nodust-anchor", "log"]
DEFAULT_STRETCH_MODES = STRETCH_MODES  # render all three by default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fits_path(galaxy, dust_label, inc_idx):
    """Build the FITS path for a given galaxy / dust mode / inclination index.

    dust_label: 'dust' or 'nodust'
    """
    inc_deg = INCLINATION_TABLE[inc_idx]
    name = f"i{inc_idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return ROMULUS_DIR / galaxy / "production" / f"{galaxy}_{dust_label}_{name}_total.fits"


def load_band(galaxy, dust_label, inc_idx, band):
    """Load a single-band image from the SKIRT total cube."""
    if band not in BAND_INDEX:
        raise KeyError(f"Unknown band '{band}'. Valid: {list(BAND_INDEX)}")
    path = fits_path(galaxy, dust_label, inc_idx)
    if not path.exists():
        raise FileNotFoundError(f"Missing FITS: {path}")
    with fits.open(path) as hdul:
        cube = hdul[0].data  # (n_bands, ny, nx)
    img = cube[BAND_INDEX[band]]
    if band in SMOOTH_BANDS:
        img = gaussian_filter(img, sigma=SMOOTH_SIGMA)
    return img


def crop_center(img):
    """Crop ±CUBE_HALF_KPC cube to ±FOV_HALF_KPC center."""
    n = img.shape[0]
    pix_per_kpc = n / (2 * CUBE_HALF_KPC)
    half_pix = int(FOV_HALF_KPC * pix_per_kpc)
    cy, cx = n // 2, n // 2
    return img[cy - half_pix:cy + half_pix, cx - half_pix:cx + half_pix]


# ---------------------------------------------------------------------------
# Stretch computation
# ---------------------------------------------------------------------------

def compute_stretch(dusty_crop, nodust_crop, mode):
    """Return (vmin, vmax, stretch_kind, stretch_param) for the chosen mode.

    Modes:
      'dusty-anchor':  vmax = P99.5 of dusty image (no-dust may saturate)
      'nodust-anchor': vmax = P99.5 of no-dust image (dusty looks dimmer — most
                       intuitive for non-specialists; reads as "dust takes light away")
      'log':           log scale anchored on no-dust image; compresses bulge,
                       expands faint disk so dust lanes / extended emission show

    stretch_kind is 'asinh' or 'log'. stretch_param is linear_width (asinh only).
    """
    if mode == "dusty-anchor":
        anchor = dusty_crop
        kind = "asinh"
    elif mode == "nodust-anchor":
        anchor = nodust_crop
        kind = "asinh"
    elif mode == "log":
        anchor = nodust_crop
        kind = "log"
    else:
        raise ValueError(f"Unknown stretch mode: {mode}")

    positive = anchor[anchor > 0]
    if positive.size == 0:
        raise ValueError(f"Anchor image has no positive pixels (mode={mode})")

    vmax = float(np.percentile(positive, STRETCH_PERCENTILE))

    if kind == "asinh":
        vmin = 0.0
        param = ASINH_LINEAR_FRAC * vmax
    else:  # log
        # vmin set just above zero — pick a low percentile of positive pixels
        # so noise floor doesn't dominate, but disk faintly visible
        vmin = float(np.percentile(positive, 50.0))
        param = None

    return vmin, vmax, kind, param


def apply_stretch(img, vmin, vmax, kind, param):
    """Return a (display_array, Normalize) tuple ready for imshow.

    For asinh + new matplotlib, returns the raw image with an AsinhNorm.
    For asinh + old matplotlib, manually applies asinh and uses linear Normalize.
    For log, manually applies log10 (handling zeros) and uses linear Normalize.
    """
    if kind == "asinh":
        if _HAS_ASINH_NORM:
            return img, AsinhNorm(linear_width=param, vmin=vmin, vmax=vmax)
        # Manual asinh fallback: y = asinh(x / lw) / asinh(vmax / lw)
        clipped = np.clip(img, vmin, vmax)
        scaled = np.arcsinh(clipped / param) / np.arcsinh(vmax / param)
        return scaled, Normalize(vmin=0.0, vmax=1.0)

    if kind == "log":
        # Floor at vmin to avoid log(0); clip top at vmax
        floored = np.clip(img, vmin, vmax)
        scaled = (np.log10(floored) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
        return scaled, Normalize(vmin=0.0, vmax=1.0)

    raise ValueError(f"Unknown stretch kind: {kind}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(galaxy, inc_idx, band, stretch_mode, output_dir):
    inc_deg = INCLINATION_TABLE[inc_idx]
    print(f"\n{galaxy} i{inc_idx:02d} ({inc_deg:.1f}°) | band={band} | stretch={stretch_mode}")

    dusty = load_band(galaxy, "dust", inc_idx, band)
    nodust = load_band(galaxy, "nodust", inc_idx, band)

    dusty_crop = crop_center(dusty)
    nodust_crop = crop_center(nodust)

    vmin, vmax, kind, param = compute_stretch(dusty_crop, nodust_crop, stretch_mode)

    flux_ratio = float(nodust_crop.max()) / float(dusty_crop.max()) if dusty_crop.max() > 0 else float("nan")
    print(f"  vmin = {vmin:.3e}, vmax = {vmax:.3e}, kind = {kind}")
    if param is not None:
        print(f"  asinh linear_width = {param:.3e}")
    print(f"  dusty max = {float(dusty_crop.max()):.3e}, nodust max = {float(nodust_crop.max()):.3e}")
    print(f"  flux ratio (nodust/dusty max) = {flux_ratio:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    extent = [-FOV_HALF_KPC, FOV_HALF_KPC, -FOV_HALF_KPC, FOV_HALF_KPC]

    for ax, img, title in [
        (axes[0], dusty_crop,  "With dust"),
        (axes[1], nodust_crop, "No dust"),
    ]:
        display_img, display_norm = apply_stretch(img, vmin, vmax, kind, param)
        ax.imshow(display_img, origin="lower", extent=extent,
                  cmap="gray", norm=display_norm)
        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlabel("x (kpc)", color="white")
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    axes[0].set_ylabel("y (kpc)", color="white")
    axes[1].set_ylabel("")

    # Inclination annotation in upper-right of right panel
    axes[1].text(0.97, 0.97, f"i = {inc_deg:.1f}°",
                 transform=axes[1].transAxes,
                 ha="right", va="top",
                 color="white", fontsize=12,
                 bbox=dict(facecolor="black", edgecolor="white",
                           alpha=0.6, pad=4))

    # Subtitle includes stretch mode for tracking which version is which
    fig.suptitle(f"{galaxy} — {band}, {stretch_mode} stretch",
                 fontsize=14, color="white", y=0.98)
    fig.tight_layout()

    out_path = output_dir / f"{galaxy}_dust_comparison_i{inc_idx:02d}_{band}_{stretch_mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global STRETCH_PERCENTILE, ASINH_LINEAR_FRAC

    parser = argparse.ArgumentParser(
        description="Dust vs no-dust comparison figures from SKIRT datacubes")
    parser.add_argument("--galaxy", default="r320",
                        help="Galaxy name (default: r320)")
    parser.add_argument("--inc", type=int, action="append",
                        help=f"Inclination index (0-11). Repeatable. Default: {DEFAULT_INC_INDICES}")
    parser.add_argument("--band", default=DEFAULT_BAND, choices=list(BAND_INDEX),
                        help=f"Band name (default: {DEFAULT_BAND})")
    parser.add_argument("--stretch-mode", action="append", choices=STRETCH_MODES,
                        help=f"Stretch mode. Repeatable. Default: all of {STRETCH_MODES}")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory (default: cwd)")
    parser.add_argument("--stretch-percentile", type=float, default=STRETCH_PERCENTILE,
                        help=f"Percentile of anchor image used as vmax (default {STRETCH_PERCENTILE})")
    parser.add_argument("--asinh-frac", type=float, default=ASINH_LINEAR_FRAC,
                        help=f"asinh linear_width as fraction of vmax (default {ASINH_LINEAR_FRAC})")
    args = parser.parse_args()

    inc_indices = args.inc if args.inc else DEFAULT_INC_INDICES
    for idx in inc_indices:
        if idx not in INCLINATION_TABLE:
            parser.error(f"--inc {idx} not in valid range 0-11")

    stretch_modes = args.stretch_mode if args.stretch_mode else DEFAULT_STRETCH_MODES

    STRETCH_PERCENTILE = args.stretch_percentile
    ASINH_LINEAR_FRAC = args.asinh_frac

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    print(f"Galaxy:        {args.galaxy}")
    print(f"Band:          {args.band}" + (f" (smoothed, sigma={SMOOTH_SIGMA}px)"
                                            if args.band in SMOOTH_BANDS else ""))
    print(f"Inclinations:  {inc_indices} "
          f"({[f'{INCLINATION_TABLE[i]:.1f}°' for i in inc_indices]})")
    print(f"Stretch modes: {stretch_modes}")
    print(f"Output dir:    {output_dir}")

    saved = []
    for inc_idx in inc_indices:
        for stretch_mode in stretch_modes:
            try:
                saved.append(plot_comparison(args.galaxy, inc_idx, args.band,
                                             stretch_mode, output_dir))
            except FileNotFoundError as e:
                print(f"  SKIPPING i{inc_idx:02d} {stretch_mode}: {e}")
            except Exception as e:
                print(f"  ERROR for i{inc_idx:02d} {stretch_mode}: {type(e).__name__}: {e}")

    print(f"\nDone. {len(saved)} figure(s) saved.")


if __name__ == "__main__":
    main()