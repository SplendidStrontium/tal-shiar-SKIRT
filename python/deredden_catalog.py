#!/usr/bin/env python3
"""
deredden_catalog.py
--------------------
Galactic FOREGROUND extinction correction for the Nearby Galaxy Catalog.

Removes the Milky Way dust screen ONLY (SFD / Schlafly & Finkbeiner), and
leaves each galaxy's own internal dust untouched. That is deliberate: SKIRT
produces dust-attenuated colors that contain internal dust but NO Milky Way
foreground, so to compare apples-to-apples the observed side must have the
foreground removed and the internal attenuation kept.

Pipeline
  1. Resolve each galaxy name -> (RA, Dec)        [SIMBAD, NED fallback]
  2. Look up E(B-V) at those coords               [dustmaps SFD]
  3. A_B, A_Ks = R_band * E(B-V); deredden mags; recompute B-K
  4. Write augmented CSV + print a before/after diagnostic on the red tail

Two steps need network access to hosts outside a locked-down sandbox
(SIMBAD and the dustmaps data server). Run on Hamilton, or first do the
one-time `python -c "import dustmaps.sfd; dustmaps.sfd.fetch()"`.

Coordinate resolution is cached to disk (--coord-cache), so it is a one-time
cost; re-runs are instant and do not re-hit SIMBAD.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Extinction coefficients: Schlafly & Finkbeiner (2011), Table 6, R_V = 3.1.
# Values are A_band / E(B-V)_SFD and are applied DIRECTLY to the raw SFD map
# value -- they already fold in the ~14% SFD recalibration.
#   B  = Landolt B  = 3.626
#   Ks = 2MASS Ks   = 0.306   (Yuan+2013 reference point; SF11-interpolated ~0.31)
# The color correction is therefore  (B-K)_dered = (B-K)_obs - 3.320 * E(B-V).
#
# >>> mentor-review-pending: confirms the catalog's B is treated as Landolt/
#     Johnson B and its Ks as 2MASS Ks. Both are safe defaults for this catalog,
#     but worth a one-line confirmation.
# ---------------------------------------------------------------------------
R_BAND = {"B": 3.626, "Ks": 0.306}

# SFD is unreliable in the Galactic plane (it overestimates E(B-V) in dense,
# structured disk dust) and toward the Magellanic Clouds (the Clouds' own dust
# emission contaminates the map). Colors dereddened on those lines of sight are
# not trustworthy, so we EXCLUDE them rather than trust the number.
# >>> mentor-review-pending: |b| < 10 deg is the conservative ZoA cut; 5 deg is
#     the looser option. Adjust LOW_GAL_LAT_DEG to taste.
LOW_GAL_LAT_DEG = 10.0
MAGELLANIC = {"SMC", "LMC"}  # SFD contaminated by the Clouds' own emission

# Informational only (not used for exclusion): a large foreground means any SFD
# error is amplified, and a dereddened B-K below the physical floor is
# unusually blue -- both are worth eyeballing, but neither alone is a reason to
# cut a high-latitude galaxy (e.g. genuinely blue star-formers like NGC5408).
EBV_INFO = 0.3
BK_DERED_FLOOR = 1.5

# [CHANGE 1] SIMBAD is strict about identifier format. These fixes are pure
# formatting / unambiguous aliases -- they do NOT change which object we mean.
_KK_PREFIX = re.compile(r"^(KKSG|KKR|KKH|KK)(\d)")
_NAME_ALIASES = {
    "SexA": "Sextans A",
    "HolmIX": "Holmberg IX",
    "HolmIV": "Holmberg IV",
    "HolmII": "Holmberg II",
    "CamA": "Camelopardalis A",
}


def normalize_name(name):
    """Map a catalog name to a SIMBAD-friendly identifier (formatting only)."""
    if name in _NAME_ALIASES:
        return _NAME_ALIASES[name]
    return _KK_PREFIX.sub(r"\1 \2", name)  # 'KK55' -> 'KK 55', 'KKH12' -> 'KKH 12'


def resolve_coordinates(names, cache_path):
    """Resolve a list of names to ICRS (ra_deg, dec_deg).

    SIMBAD first, NED as a fallback for names SIMBAD misses. Results are
    cached to `cache_path` keyed by name so this only ever runs once.
    """
    # load existing cache; [CHANGE 1] keep only successful (finite) entries so a
    # re-run automatically retries anything that failed last time.
    if cache_path and os.path.exists(cache_path):
        c = pd.read_csv(cache_path).set_index("name")
        cache = {n: (r.ra_deg, r.dec_deg) for n, r in c.iterrows()
                 if np.isfinite(r.ra_deg) and np.isfinite(r.dec_deg)}
    else:
        cache = {}

    todo = [n for n in names if n not in cache]
    if todo:
        from astroquery.simbad import Simbad
        from astroquery.ipac.ned import Ned
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        Ned.TIMEOUT = 60  # [CHANGE 1] the default was short enough to ReadTimeout
        simbad = Simbad()
        for i, name in enumerate(todo, 1):
            ra = dec = np.nan
            query = normalize_name(name)
            try:
                res = simbad.query_object(query)
                if res is not None and len(res):
                    row = res[0]
                    # newer astroquery returns 'ra'/'dec' in degrees;
                    # older returns 'RA'/'DEC' sexagesimal -- handle both.
                    if "ra" in res.colnames:
                        ra, dec = float(row["ra"]), float(row["dec"])
                    else:
                        c = SkyCoord(row["RA"], row["DEC"],
                                     unit=(u.hourangle, u.deg))
                        ra, dec = c.ra.deg, c.dec.deg
            except Exception as e:
                print(f"  [SIMBAD miss] {name}: {type(e).__name__}", file=sys.stderr)

            if np.isnan(ra):  # NED fallback, retried -- NED tolerates name variants
                for attempt in range(3):
                    try:
                        res = Ned.query_object(name)
                        if res is not None and len(res):
                            ra = float(res[0]["RA"])
                            dec = float(res[0]["DEC"])
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  [NED miss]    {name}: {type(e).__name__}",
                                  file=sys.stderr)
                        else:
                            time.sleep(2 * (attempt + 1))  # back off and retry

            if np.isfinite(ra) and np.isfinite(dec):  # [CHANGE 1] cache successes only
                cache[name] = (ra, dec)
            if i % 25 == 0:
                print(f"  resolved {i}/{len(todo)}")
            time.sleep(0.1)  # be polite to the name servers

        if cache_path:
            (pd.DataFrame([(n, r[0], r[1]) for n, r in cache.items()],
                          columns=["name", "ra_deg", "dec_deg"])
               .to_csv(cache_path, index=False))

    ra = np.array([cache.get(n, (np.nan, np.nan))[0] for n in names])
    dec = np.array([cache.get(n, (np.nan, np.nan))[1] for n in names])
    return ra, dec


def get_ebv(ra_deg, dec_deg):
    """Raw SFD98 E(B-V) at each (ra, dec). NaN coords -> NaN E(B-V)."""
    from dustmaps.sfd import SFDQuery
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ebv = np.full(len(ra_deg), np.nan)
    good = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    if good.any():
        sfd = SFDQuery()
        coords = SkyCoord(ra_deg[good] * u.deg, dec_deg[good] * u.deg,
                          frame="icrs")
        ebv[good] = sfd(coords)
    return ebv


def add_galactic_latitude(df):
    """[CHANGE 2] Add Galactic latitude and flag the SFD-unreliable low-|b| zone."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    gal_b = np.full(len(df), np.nan)
    good = np.isfinite(df["ra_deg"].values) & np.isfinite(df["dec_deg"].values)
    if good.any():
        c = SkyCoord(df.loc[good, "ra_deg"].values * u.deg,
                     df.loc[good, "dec_deg"].values * u.deg, frame="icrs").galactic
        gal_b[good] = c.b.deg
    df["gal_b"] = gal_b
    df["low_gal_lat"] = np.abs(df["gal_b"]) < LOW_GAL_LAT_DEG
    return df


def deredden(df):
    """Apply foreground correction to B and Ks; recompute observed/dered B-K."""
    df["A_B"] = R_BAND["B"] * df["ebv_sfd"]
    df["A_Ks"] = R_BAND["Ks"] * df["ebv_sfd"]
    df["bmag_dered"] = df["bmag"] - df["A_B"]
    df["ks_mag_dered"] = df["ks_mag"] - df["A_Ks"]
    df["BK_obs"] = df["bmag"] - df["ks_mag"]
    df["BK_dered"] = df["bmag_dered"] - df["ks_mag_dered"]
    # [CHANGE 3] EXCLUDE on line-of-sight reliability only: Galactic plane or
    # Magellanic contamination. Do NOT exclude on the resulting color.
    df["deredden_suspect"] = df["low_gal_lat"] | df["name"].isin(MAGELLANIC)
    # informational flags (surface for review, do not drive exclusion)
    df["high_foreground"] = df["ebv_sfd"] > EBV_INFO
    df["very_blue"] = df["BK_dered"] < BK_DERED_FLOOR
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default="nearby_dwarf_catalog.csv")
    ap.add_argument("--out", dest="outfile", default="nearby_dwarf_catalog_dered.csv")
    ap.add_argument("--coord-cache", default="catalog_coords.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    print(f"loaded {len(df)} rows from {args.infile}")

    # 1. coordinates (skip if the CSV already carries ra_deg/dec_deg)
    if {"ra_deg", "dec_deg"}.issubset(df.columns):
        print("using ra_deg/dec_deg already present in the CSV")
        ra, dec = df["ra_deg"].values, df["dec_deg"].values
    else:
        print("resolving coordinates (SIMBAD -> NED, cached)...")
        ra, dec = resolve_coordinates(df["name"].tolist(), args.coord_cache)
        df["ra_deg"], df["dec_deg"] = ra, dec

    n_unresolved = int(np.isnan(ra).sum())
    if n_unresolved:
        print(f"WARNING: {n_unresolved} names did not resolve -> "
              f"E(B-V) and dereddened values will be NaN for those. "
              f"Names: {', '.join(df.loc[np.isnan(ra), 'name'].tolist())}")

    # 2. E(B-V), Galactic latitude, then 3. deredden + flag
    print("querying SFD dust map...")
    df["ebv_sfd"] = get_ebv(ra, dec)
    df = add_galactic_latitude(df)
    df = deredden(df)

    # 4. write + diagnostic
    df.to_csv(args.outfile, index=False)
    print(f"\nwrote {args.outfile}")

    print("\n=== before/after on the reddest observed objects (the foreground tail) ===")
    red = df.sort_values("BK_obs", ascending=False).head(12)
    for _, r in red.iterrows():
        flag = " [SUSPECT]" if r["deredden_suspect"] else ""
        print(f"  {r['name']:16s} b={r['gal_b']:6.1f}  E(B-V)={r['ebv_sfd']:5.2f}  "
              f"B-K: {r['BK_obs']:5.2f} -> {r['BK_dered']:5.2f}{flag}")

    ceil = 4.5
    before = int((df["BK_obs"] > ceil).sum())
    after = int((df["BK_dered"] > ceil).sum())
    print(f"\ngalaxies with B-K > {ceil}: {before} (observed) -> {after} (dereddened)")
    print(f"median B-K: {df['BK_obs'].median():.2f} (obs) -> "
          f"{df['BK_dered'].median():.2f} (dered)")

    n_lowlat = int(df["low_gal_lat"].sum())
    n_mag = int(df["name"].isin(MAGELLANIC).sum())
    n_suspect = int(df["deredden_suspect"].sum())
    n_highfg = int(df["high_foreground"].sum())
    n_blue = int(df["very_blue"].sum())
    n_unres = int(df["ebv_sfd"].isna().sum())
    print(f"\nEXCLUDE (SFD unreliable): {n_suspect}  "
          f"[low |b|<{LOW_GAL_LAT_DEG:g}: {n_lowlat}, Magellanic: {n_mag}]")
    if n_suspect:
        print("  " + ", ".join(df.loc[df["deredden_suspect"], "name"].tolist()))
    print(f"\nINFO only (review, not excluded):")
    print(f"  large foreground E(B-V)>{EBV_INFO}: {n_highfg}")
    print(f"  unusually blue B-K<{BK_DERED_FLOOR}: {n_blue}  "
          + ("(" + ", ".join(df.loc[df["very_blue"], "name"].tolist()) + ")" if n_blue else ""))
    print(f"\nstill unresolved (E(B-V) NaN): {n_unres}")
    if n_unres:
        print("  " + ", ".join(df.loc[df["ebv_sfd"].isna(), "name"].tolist()))

    n_clean = int((~df["deredden_suspect"] & df["ebv_sfd"].notna()).sum())
    print(f"\nclean comparison sample (not suspect, resolved): {n_clean} / {len(df)}")


if __name__ == "__main__":
    main()