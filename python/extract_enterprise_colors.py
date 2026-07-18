#!/usr/bin/env python3
"""
extract_enterprise_colors.py
----------------------------
Integrated B-K (Vega) for every Enterprise SKIRT run: 5 families x 4 BH
variants x {dust, nodust} x {face-on, edge-on}.

Mirrors the validated Tal Shiar extractor (pyphot GROUND_JOHNSON_B and
2MASS_Ks, Vega zero-points, astropy spectral_density for the F_nu -> F_lambda
conversion -- the fix that removed the +11.308 mag offset). Differences from
the Tal Shiar version:

  * Filenames are PARSED, not assumed: run name, medium (dust/nodust),
    instrument label, and inclination are pulled from each *_sed.dat path,
    and every SED becomes one row in a long-format CSV. Downstream code
    filters (e.g. face-on + dust) rather than this script deciding.
  * Run names follow the on-disk convention: noBH runs are the bare family
    name (r488), others are family_variant (r488_BH6). NEVER derived from
    achOutName -- that lesson is already paid for.
  * B-K is a flux ratio, so absolute normalization / distance cancels;
    per-band magnitudes in the CSV are not distance-calibrated but the
    color is exact.

Usage:
    python extract_enterprise_colors.py --inventory
        # No photometry. Walks the products tree, lists every *_sed.dat,
        # groups by run, and reports the census against the expected
        # 4 SEDs per (run, medium) label. RUN THIS FIRST and eyeball it.

    python extract_enterprise_colors.py --one r741
        # Verbose single-run extraction: prints the parsed header (flux
        # style, wavelength span, N points) and the B/K/B-K numbers for
        # every SED belonging to r741. Sanity-check the header parse and
        # that face-on dust B-K lands in a plausible dwarf range (~1-4).

    python extract_enterprise_colors.py --all
        # Full extraction -> enterprise_colors.csv (long format) in
        # PRODUCTS_DIR, plus printed pivot tables:
        #   face-on dust B-K   [family x variant]
        #   dust - nodust      [family x variant]  (dust reddening)
        #   edge-on - face-on  [family x variant]  (inclination reddening)
        # and the two clean twin-pair deltas (BH - noBH, BH8 - BH6) with
        # mean +/- std across families.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sample + layout (keep in sync with generate_ski_enterprise.py)
# ---------------------------------------------------------------------------

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]

PRODUCTS_DIR = Path("/mnt/data0/pkrsnak/romulus/enterprise/products")

EXPECTED_SEDS_PER_LABEL = 4   # 2 inclinations x (FullInstrument + SEDInstrument)
MEDIA = ["dust", "nodust"]
IGNORE_INSTRUMENT_PREFIXES = ("test",)


def run_names():
    """20 run names, on-disk convention: noBH = bare family name."""
    names = []
    for fam in FAMILIES:
        for var in VARIANTS:
            names.append(fam if var == "noBH" else f"{fam}_{var}")
    return names


def split_run(run):
    """'r488_BH6' -> ('r488', 'BH6');  'r488' -> ('r488', 'noBH')."""
    if "_" in run:
        fam, var = run.split("_", 1)
    else:
        fam, var = run, "noBH"
    return fam, var


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
# SKIRT names outputs {ski_prefix}_{instrument}_sed.dat, and the ski prefix
# is {run}_{medium}. Instrument labels embed inclination as e.g. 00p00deg /
# 90p00deg. We anchor on the LONGEST matching run name so 'r488_BH6_dust_...'
# is not claimed by run 'r488'.

_RUNS_BY_LENGTH = sorted(run_names(), key=len, reverse=True)
_INC_RE = re.compile(r"(\d{2})p(\d{2})deg")


def parse_sed_path(path):
    """Return dict(run, family, variant, medium, instrument, inc_deg)
    or None if the filename doesn't belong to the Enterprise sample."""
    stem = path.name
    if not stem.endswith("_sed.dat"):
        return None
    core = stem[: -len("_sed.dat")]

    run = next((r for r in _RUNS_BY_LENGTH if core.startswith(r + "_")), None)
    if run is None:
        return None
    rest = core[len(run) + 1:]

    medium = next((m for m in MEDIA if rest == m or rest.startswith(m + "_")), None)
    if medium is None:
        return None
    instrument = rest[len(medium):].lstrip("_") or "(unnamed)"

    if instrument.startswith(IGNORE_INSTRUMENT_PREFIXES):
        return None

    m = _INC_RE.search(instrument)
    inc_deg = float(f"{m.group(1)}.{m.group(2)}") if m else np.nan

    fam, var = split_run(run)
    return dict(run=run, family=fam, variant=var, medium=medium,
                instrument=instrument, inc_deg=inc_deg, path=str(path))


def find_seds(products_dir):
    """Recursive glob + parse. Returns (parsed_rows, unclaimed_paths)."""
    rows, unclaimed = [], []
    for p in sorted(products_dir.rglob("*_sed.dat")):
        info = parse_sed_path(p)
        (rows if info else unclaimed).append(info or str(p))
    return rows, unclaimed


# ---------------------------------------------------------------------------
# SED reading + synthetic photometry
# ---------------------------------------------------------------------------

def read_skirt_sed(path):
    """Read a SKIRT *_sed.dat. Returns (wave_micron, F_lambda in
    erg/s/cm2/AA, header_info) handling F_nu (Jy) or F_lambda styles.
    """
    import astropy.units as u

    header = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header.append(line.rstrip())
            else:
                break
    text = "\n".join(header)

    data = np.loadtxt(path)
    wave_micron, flux = data[:, 0], data[:, 1]

    if re.search(r"\bJy\b", text):
        style = "F_nu (Jy)"
        f_nu = flux * u.Jy
        f_lam = f_nu.to(u.erg / u.s / u.cm**2 / u.AA,
                        equivalencies=u.spectral_density(wave_micron * u.micron))
        f_lam = f_lam.value
    elif re.search(r"W/m2/micron|erg/s/cm2/A", text):
        style = "F_lambda"
        # Normalize to erg/s/cm2/AA whichever spelling appears
        unit = (u.W / u.m**2 / u.micron) if "W/m2/micron" in text \
            else (u.erg / u.s / u.cm**2 / u.AA)
        f_lam = (flux * unit).to(u.erg / u.s / u.cm**2 / u.AA).value
    else:
        raise ValueError(f"Unrecognized flux style in header of {path}:\n{text}")

    info = dict(style=style, n=len(wave_micron),
                wmin=wave_micron.min(), wmax=wave_micron.max())
    return wave_micron, f_lam, info


_FILTERS = None


def get_filters():
    global _FILTERS
    if _FILTERS is None:
        import pyphot
        lib = pyphot.get_library()
        _FILTERS = (lib["GROUND_JOHNSON_B"], lib["2MASS_Ks"])
    return _FILTERS


def bk_color(wave_micron, f_lam):
    """(mB_vega, mK_vega, B-K). Magnitudes are NOT distance-calibrated."""
    import astropy.units as u
    b, k = get_filters()
    wave_aa = (wave_micron * u.micron).to(u.AA)
    f = f_lam * u.erg / u.s / u.cm**2 / u.AA
    fb = b.get_flux(wave_aa, f)
    fk = k.get_flux(wave_aa, f)
    mB = -2.5 * np.log10(fb.value) - b.Vega_zero_mag
    mK = -2.5 * np.log10(fk.value) - k.Vega_zero_mag
    return mB, mK, mB - mK


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def do_inventory(products_dir):
    rows, unclaimed = find_seds(products_dir)
    print(f"Scanned: {products_dir}")
    print(f"Parsed SEDs: {len(rows)}   Unclaimed *_sed.dat: {len(unclaimed)}\n")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No SEDs parsed -- check PRODUCTS_DIR and filename convention.")
        for p in unclaimed:
            print("  ?", p)
        return 1

    counts = (df.groupby(["run", "medium"]).size()
                .unstack(fill_value=0)
                .reindex(run_names()))
    counts["TOTAL"] = counts.sum(axis=1)
    print(counts.to_string(), "\n")

    bad = []
    for run in run_names():
        for med in MEDIA:
            n = len(df[(df.run == run) & (df.medium == med)])
            if n != EXPECTED_SEDS_PER_LABEL:
                bad.append((run, med, n))
    if bad:
        print(f"MISMATCH vs expected {EXPECTED_SEDS_PER_LABEL} per (run, medium):")
        for run, med, n in bad:
            print(f"  {run:10s} {med:6s} -> {n}")
    else:
        print(f"All 20 runs x 2 media have exactly "
              f"{EXPECTED_SEDS_PER_LABEL} SEDs. Census clean.")
    if unclaimed:
        print("\nUnclaimed files (not matching any run name):")
        for p in unclaimed:
            print("  ?", p)

    print("\nInstrument labels seen:")
    for inst, n in df.instrument.value_counts().sort_index().items():
        print(f"  {inst:30s} x{n}")
    return 0


def extract(df_files, verbose=False):
    out = []
    for _, r in df_files.iterrows():
        wave, f_lam, info = read_skirt_sed(Path(r.path))
        mB, mK, bk = bk_color(wave, f_lam)
        if verbose:
            print(f"  {Path(r.path).name}")
            print(f"    header: {info['style']}, {info['n']} pts, "
                  f"{info['wmin']:.3f}-{info['wmax']:.3f} um")
            print(f"    mB={mB:+.3f}  mK={mK:+.3f}  B-K={bk:+.3f}")
        out.append({**{k: r[k] for k in
                       ("run", "family", "variant", "medium",
                        "instrument", "inc_deg")},
                    "mB": mB, "mK": mK, "BK": bk})
    return pd.DataFrame(out)


def pick_one_per_cell(df):
    """One row per (run, medium, inc_deg): if both a FullInstrument and a
    SEDInstrument SED exist for the same view, prefer the SED instrument
    (pure integrated SED; the Full one is identical in principle but keep
    the choice explicit and printed)."""
    def rank(inst):
        return 0 if "sed" in inst.lower() else 1
    d = df.copy()
    d["_rank"] = d.instrument.map(rank)
    d = (d.sort_values("_rank")
           .groupby(["run", "medium", "inc_deg"], as_index=False)
           .first()
           .drop(columns="_rank"))
    return d


def do_all(products_dir):
    rows, unclaimed = find_seds(products_dir)
    df_files = pd.DataFrame(rows)
    n_before = len(df_files)
    df_files = df_files[df_files.instrument.str.startswith("sed_")]
    print(f"Photometry restricted to SEDInstrument outputs: "
          f"{len(df_files)} of {n_before} files "
          f"(FullInstrument 10-pt broadband grids excluded -- "
          f"not valid input for filter convolution)")
    if unclaimed:
        print(f"NOTE: {len(unclaimed)} unclaimed *_sed.dat ignored "
              f"(run --inventory to see them)")
    print(f"Extracting photometry for {len(df_files)} SEDs ...")
    df = extract(df_files)

    csv_path = products_dir / "enterprise_colors.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows -> {csv_path}\n")

    d = pick_one_per_cell(df)
    face = d[np.isclose(d.inc_deg, 0.0)]
    edge = d[np.isclose(d.inc_deg, 90.0)]

    def pivot(sub, label):
        p = (sub.pivot_table(index="family", columns="variant", values="BK")
                .reindex(index=FAMILIES, columns=VARIANTS))
        print(f"--- {label} ---")
        print(p.round(3).to_string(), "\n")
        return p

    p_dust = pivot(face[face.medium == "dust"], "face-on DUST B-K (Vega)")
    p_nod = pivot(face[face.medium == "nodust"], "face-on NODUST B-K (Vega)")

    print("--- dust reddening: dust - nodust (face-on) ---")
    print((p_dust - p_nod).round(3).to_string(), "\n")

    pe_dust = (edge[edge.medium == "dust"]
               .pivot_table(index="family", columns="variant", values="BK")
               .reindex(index=FAMILIES, columns=VARIANTS))
    print("--- inclination reddening: edge-on - face-on (dust) ---")
    print((pe_dust - p_dust).round(3).to_string(), "\n")

    print("--- twin pairs, face-on dust B-K ---")
    for lo, hi, label in [("noBH", "BH", "BH - noBH"),
                          ("BH6", "BH8", "BH8 - BH6")]:
        delta = p_dust[hi] - p_dust[lo]
        print(f"{label}: per family "
              + "  ".join(f"{f}={delta[f]:+.3f}" for f in FAMILIES))
        print(f"{label}: mean {delta.mean():+.3f} +/- {delta.std(ddof=1):.3f} "
              f"(std across 5 families)\n")
    print("Caveat: matched runs diverge stochastically; the cross-family "
          "trend is the evidence, not any single pair.")
    return 0


def do_one(products_dir, run):
    rows, _ = find_seds(products_dir)
    df_files = pd.DataFrame(rows)
    sub = df_files[df_files.run == run]
    if sub.empty:
        print(f"No SEDs found for run '{run}'. Known runs: {run_names()}")
        return 1
    print(f"{run}: {len(sub)} SEDs\n")
    extract(sub, verbose=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-dir", type=Path, default=PRODUCTS_DIR)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inventory", action="store_true")
    g.add_argument("--one", metavar="RUN")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.inventory:
        return do_inventory(args.products_dir)
    if args.one:
        return do_one(args.products_dir, args.one)
    return do_all(args.products_dir)


if __name__ == "__main__":
    sys.exit(main())