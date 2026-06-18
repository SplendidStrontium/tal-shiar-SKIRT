#!/usr/bin/env python3
"""
extract_sim_colors.py
----------------------
Integrated B-K color for each SKIRT halo, in the same Vega system as the
observed catalog (Johnson B - 2MASS Ks).

We read the DUST (internal-dust-attenuated), face-on (i00) integrated SED:
    {halo}/production/{halo}_dust_i00_00p00deg_sed.dat
because that matches the foreground-dereddened observed colors -- both sides
carry the galaxy's own internal dust, neither carries a Milky Way screen.

Synthetic photometry is done with pyphot (bundled Vega zero-points). B-K is a
flux RATIO, so the absolute SKIRT flux normalization is irrelevant; only the
spectral SHAPE, the wavelength scale, and the filter/Vega zero-points matter.
That is why the per-band magnitudes below are not distance-calibrated but the
color is exact.

Usage:
    python extract_sim_colors.py                 # inspect one halo (verbose)
    python extract_sim_colors.py --halo r142     # inspect a different halo
    python extract_sim_colors.py --all           # process all 15 -> sim_colors.csv

Run the verbose single-halo mode FIRST and check the printed header parse
(flux style, wavelength span) before trusting --all.
"""

import argparse
import os
import re
import sys

import numpy as np
import astropy.units as u
import pyphot

ROMULUS_DIR = "/mnt/data0/pkrsnak/romulus"
HALOS = ["r107", "r142", "r154", "r168", "r204", "r219", "r223", "r239",
         "r284", "r306", "r316", "r320", "r330", "r372", "r429"]
SED_TEMPLATE = "{halo}/production/{halo}_dust_sed_i00_00p00deg_sed.dat"

# >>> mentor-review-pending: the catalog B is Johnson B_T; Johnson vs Bessell B
#     differs by <~0.05 mag. GROUND_JOHNSON_B is the matching choice.
FILTER_B = "GROUND_JOHNSON_B"
FILTER_KS = "2MASS_Ks"

_WAVE_TO_AA = {"angstrom": 1.0, "a": 1.0, "aa": 1.0, "nm": 10.0,
               "micron": 1e4, "um": 1e4, "micrometer": 1e4, "m": 1e10}


def parse_header(path):
    """Return [(col_index, description), ...] from SKIRT '# column N:' lines."""
    cols = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*column\s+(\d+)\s*:\s*(.*)", line, re.I)
            if m:
                cols.append((int(m.group(1)) - 1, m.group(2).strip()))
    return cols


def identify_columns(cols):
    """Find the wavelength column and the TOTAL-flux column + its flux style."""
    wave_idx = wave_unit = None
    flux_idx = flux_style = flux_unit = None
    for idx, desc in cols:
        low = desc.lower()
        mu = re.search(r"\(([^)]*)\)", desc)
        unit = mu.group(1).strip() if mu else None
        if wave_idx is None and ("wavelength" in low or re.search(r"\blambda\b", low)
                                 and "f_lambda" not in low and "*" not in low):
            wave_idx, wave_unit = idx, unit
        if flux_idx is None and "total flux" in low:
            flux_idx, flux_unit = idx, unit
            un = (unit or "").replace(" ", "").lower()
            if "f_nu" in low or "/hz" in un or "jy" in un:
                flux_style = "fnu"
            elif "lambda*f_lambda" in low or un in ("w/m2", "w/m^2"):
                flux_style = "neutral"   # lambda*F_lambda
            else:
                flux_style = "flambda"
    if wave_idx is None:           # SKIRT default: col 1 = wavelength (micron)
        wave_idx, wave_unit = 0, "micron"
    if flux_idx is None:           # SKIRT default: col 2 = total flux
        flux_idx, flux_style, flux_unit = 1, "flambda", "?"
    return wave_idx, wave_unit, flux_idx, flux_style, flux_unit


def load_sed_as_flam(path):
    """Load a SKIRT SED and return (wave_AA, flam_proportional, meta).

    Only the shape matters for color, so we convert any flux style to an
    F_lambda-proportional array (the overall constant cancels in B-K)."""
    cols = parse_header(path)
    wi, wu, fi, fs, fu = identify_columns(cols)
    data = np.loadtxt(path, comments="#")
    wave = data[:, wi].astype(float)
    flux = data[:, fi].astype(float)

    wave_AA = wave * _WAVE_TO_AA.get((wu or "micron").strip().lower(), 1e4)
    if fs == "neutral":            # lambda*F_lambda -> divide by lambda
        flam = flux / wave_AA
    elif fs == "fnu":              # F_nu -> F_lambda ~ F_nu / lambda^2
        flam = flux / wave_AA ** 2
    else:                          # already F_lambda shape
        flam = flux.copy()

    good = np.isfinite(wave_AA) & np.isfinite(flam) & (flam > 0)
    wave_AA, flam = wave_AA[good], flam[good]
    order = np.argsort(wave_AA)
    meta = dict(wave_unit=wu, flux_style=fs, flux_unit=fu, ncols=len(cols))
    return wave_AA[order], flam[order], meta


def synth_BK(wave_AA, flam):
    lib = pyphot.get_library()
    b, k = lib[FILTER_B], lib[FILTER_KS]
    sl = wave_AA * u.AA
    sf = flam * (u.erg / u.s / u.cm ** 2 / u.AA)
    mB = -2.5 * np.log10(b.get_flux(sl, sf).value) - b.Vega_zero_mag
    mK = -2.5 * np.log10(k.get_flux(sl, sf).value) - k.Vega_zero_mag
    return float(mB), float(mK)


def process(halo, verbose=False):
    path = os.path.join(ROMULUS_DIR, SED_TEMPLATE.format(halo=halo))
    if not os.path.exists(path):
        print(f"  [missing SED] {path}", file=sys.stderr)
        return None
    wave_AA, flam, meta = load_sed_as_flam(path)
    mB, mK = synth_BK(wave_AA, flam)

    if verbose:
        print(f"--- {halo} ---\n  raw header:")
        with open(path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                print("   ", line.rstrip())
        print(f"  parsed -> wave_unit={meta['wave_unit']}  "
              f"flux_style={meta['flux_style']}  flux_unit={meta['flux_unit']}  "
              f"ncols={meta['ncols']}")
        print(f"  wavelength span: {wave_AA.min():.0f}-{wave_AA.max():.0f} AA "
              f"({len(wave_AA)} points)")
        b_ok = wave_AA.min() < 3600 and wave_AA.max() > 5500   # B coverage
        k_ok = wave_AA.max() > 23000                            # Ks coverage
        if not (b_ok and k_ok):
            print("  !! WARNING: SED may not fully cover B (~3600-5500 AA) and/or "
                  "Ks (~19000-23500 AA). Check the instrument wavelength grid.")
        print(f"  B={mB:.3f}  Ks={mK:.3f}  ->  B-K = {mB - mK:.3f}")
        print("  (only B-K is meaningful; per-band zero-points are not distance-calibrated)")

    return dict(halo=halo, mag_B=mB, mag_Ks=mK, BK=mB - mK,
                flux_style=meta["flux_style"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="process all 15 halos")
    ap.add_argument("--halo", default="r284", help="single halo to inspect")
    ap.add_argument("--out", default="sim_colors.csv")
    args = ap.parse_args()

    if not args.all:
        process(args.halo, verbose=True)
        return

    import pandas as pd
    rows = [r for h in HALOS if (r := process(h)) is not None]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}  ({len(df)}/{len(HALOS)} halos)")
    print(f"B-K range: {df['BK'].min():.2f} to {df['BK'].max():.2f}, "
          f"median {df['BK'].median():.2f}")


if __name__ == "__main__":
    main()