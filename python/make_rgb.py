#!/usr/bin/env python3
"""
make_rgb.py -- Lupton asinh RGB composites from SKIRT v9 *_total.fits datacubes.

Usage (on Hamilton):
    python make_rgb.py /mnt/data0/pkrsnak/romulus/r488_BH6/*_total.fits
    python make_rgb.py cube.fits --outdir figures/rgb --Q 8 --stretch 0.5

One PNG is written per input cube, named <cube_stem>_rgb.png.

Requires: numpy, astropy, matplotlib (pillow via matplotlib for PNG output).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Wavelength windows (microns) used to synthesize each display channel.
# Roughly "what an optical telescope camera sees":
#   B ~ 0.40-0.50 um, G ~ 0.50-0.60 um, R ~ 0.60-0.75 um
# Widen the R window toward 0.9-1.0 um if you want dust lanes / old
# populations to pop more (more of a gri-like composite).
# ----------------------------------------------------------------------
DEFAULT_WINDOWS = {
    "R": (0.70, 1.00),   # catches 0.749 + 0.895 um slices (R/I)
    "G": (0.55, 0.70),   # catches 0.618 um slice (V/R)
    "B": (0.40, 0.55),   # catches 0.470 um slice (B)
}


def load_cube(path):
    """Return (cube[nlam, ny, nx], wavelengths_um) from a SKIRT v9 FITS file."""
    with fits.open(path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=float)
        if cube.ndim != 3:
            raise ValueError(f"{path}: expected 3D datacube, got shape {cube.shape}")

        # SKIRT v9 stores the wavelength grid in a table extension (HDU 1).
        wav = None
        if len(hdul) > 1 and hdul[1].data is not None:
            try:
                col = hdul[1].data.columns[0].name
                wav = np.asarray(hdul[1].data[col], dtype=float).ravel()
            except Exception:
                wav = None

        # Fallback: linear WCS along axis 3 (rare for SKIRT, but cheap insurance).
        if wav is None or wav.size != cube.shape[0]:
            hdr = hdul[0].header
            n = cube.shape[0]
            crval = hdr.get("CRVAL3", None)
            cdelt = hdr.get("CDELT3", hdr.get("CD3_3", None))
            crpix = hdr.get("CRPIX3", 1.0)
            if crval is None or cdelt is None:
                raise ValueError(
                    f"{path}: no wavelength table in HDU 1 and no usable "
                    "CRVAL3/CDELT3 in the header."
                )
            wav = crval + (np.arange(n) + 1 - crpix) * cdelt

    # SKIRT writes wavelengths in micron by default; if these look like
    # meters (tiny numbers), convert.
    if np.nanmedian(wav) < 1e-3:
        wav = wav * 1e6

    return cube, wav


def channel_image(cube, wav, lo, hi):
    """Mean of cube slices with lo <= lambda(um) <= hi."""
    sel = (wav >= lo) & (wav <= hi)
    if not sel.any():
        raise ValueError(
            f"No wavelength slices in [{lo}, {hi}] um; "
            f"cube covers {wav.min():.3f}-{wav.max():.3f} um."
        )
    return np.nanmean(cube[sel], axis=0)


def channel_scale(img, pct):
    """The pct-th percentile of positive pixels -- the number that maps to 1."""
    pos = img[np.isfinite(img) & (img > 0)]
    return float(np.percentile(pos, pct)) if pos.size else 1.0


def normalize(img, scale):
    """Apply a fixed scale (from this image or a reference image)."""
    return np.clip(np.nan_to_num(img), 0, None) / max(scale, 1e-30)

def center_crop(img, frac):
    """Keep the central frac of the frame (SKIRT instruments are galaxy-centered)."""
    if frac >= 1.0:
        return img
    ny, nx = img.shape[:2]
    hy, hx = max(1, int(ny * frac / 2)), max(1, int(nx * frac / 2))
    return img[ny // 2 - hy: ny // 2 + hy, nx // 2 - hx: nx // 2 + hx]


def make_one(path, outdir, windows, Q, stretch, pct, minimum, crop, upscale, scales=None):
    cube, wav = load_cube(path)

    chans = {k: center_crop(channel_image(cube, wav, *windows[k]), crop)
             for k in ("R", "G", "B")}
    if scales is None:
        scales = {k: channel_scale(chans[k], pct) for k in chans}   # old per-image behavior
    r = normalize(chans["R"], scales["R"])
    g = normalize(chans["G"], scales["G"])
    b = normalize(chans["B"], scales["B"])
    rgb = make_lupton_rgb(r, g, b, Q=Q, stretch=stretch, minimum=minimum)

    out = outdir / (Path(path).stem + "_rgb.png")
    ny, nx = rgb.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(nx * upscale / dpi, ny * upscale / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="lower", interpolation="lanczos")
    ax.set_axis_off()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  {Path(path).name} -> {out}  "
          f"(cube {cube.shape}, lambda {wav.min():.2f}-{wav.max():.2f} um)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cubes", nargs="+", help="SKIRT *_total.fits datacube(s)")
    p.add_argument("--outdir",
               default="/mnt/data0/pkrsnak/romulus/enterprise/products/figures/rgb",
               help="output directory for PNGs")
    p.add_argument("--Q", type=float, default=10.0,
                   help="Lupton asinh softening; higher = flatter, shows faint "
                        "outskirts (try 5-15)")
    p.add_argument("--stretch", type=float, default=0.5,
                   help="linear stretch before asinh; lower = brighter center "
                        "saturates sooner (try 0.1-1)")
    p.add_argument("--pct", type=float, default=99.5,
                   help="percentile of positive pixels mapped to 1.0 per "
                        "channel before stretching")
    p.add_argument("--minimum", type=float, default=0.0,
                   help="black-point subtracted from each channel")
    p.add_argument("--crop", type=float, default=0.35,
               help="central fraction of the frame to keep (1.0 = no crop)")
    p.add_argument("--upscale", type=int, default=3,
               help="integer upscale factor for the output PNG")
    p.add_argument("--ref", default=None,
                   help="cube whose channel scales are applied to ALL images "
                        "(shared normalization; omit for per-image behavior)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scales = None
    if args.ref:
        rcube, rwav = load_cube(args.ref)
        scales = {k: channel_scale(
                      center_crop(channel_image(rcube, rwav, *DEFAULT_WINDOWS[k]),
                                  args.crop), args.pct)
                  for k in ("R", "G", "B")}
        print(f"Reference {Path(args.ref).name}: scales "
              f"R={scales['R']:.4g} G={scales['G']:.4g} B={scales['B']:.4g}")

    failures = 0
    for path in args.cubes:
        try:
            make_one(path, outdir, DEFAULT_WINDOWS,
                     args.Q, args.stretch, args.pct, args.minimum, args.crop, args.upscale, scales)
        except Exception as e:
            failures += 1
            print(f"  FAILED {path}: {e}", file=sys.stderr)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()