#!/usr/bin/env python3
"""
proposal_census.py — Tal Shiar SKIRT Pipeline, Reines BH-proposal census

Extracts the apples-to-apples comparison quantities the proposal needs on the
simulation side, for one or more Romulus halos:

    M_star      stellar mass             (Msun)
    M_HI        neutral atomic hydrogen  (Msun)    <- 21cm-comparable, NOT all cold gas
    SFR_25Myr   star formation rate      (Msun/yr) averaged over 25 Myr (matches Fig 2)

plus a dwarf flag (M_star < 3e9 Msun, the proposal's definition).

This deliberately does NOT touch black holes — BH presence and active-vs-lurking
classification are deferred. Colors (B-K) are handled separately via SKIRT.

----------------------------------------------------------------------------
Aperture choice (FIXED, not R_vir):
    All quantities are summed inside a fixed spherical aperture centered on the
    stellar density peak. We use a fixed aperture rather than R_vir because the
    SKIRT-prep snapshots have their dark matter stripped (r107 loads with 0 DM
    particles), which makes a particle-based R_vir estimate unreliable — it
    would miss most of the enclosing mass. A 30 kpc radius captures essentially
    all of a dwarf's stars and neutral gas while matching the 30 kpc cut used in
    make_particles.py. Override with --aperture-kpc if needed.

HI prescription (mentor-review-pending):
    M_HI = sum_gas  m_i * X_H,i * f_HI,i
      X_H,i  = 1 - Y - Z_i            hydrogen mass fraction (Y = helium mass frac)
      f_HI,i = neutral atomic H fraction  n_HI / n_H
    f_HI is taken from pynbody's post-hoc ionization-equilibrium calculation
    (Haardt-Madau UVB tables) when available, with a temperature-based fallback.
    This is ATOMIC HI only (no H2), which is the right quantity to compare with
    21cm HI masses in the Nearby Galaxy Catalog. H2 is not split out; in dwarfs
    the molecular fraction is small, so this is a conservative HI estimate.

    >>> CONFIRM this matches the HI definition used in Sharma+2022 before
        trusting the numbers, and VERIFY the pynbody ionfrac normalization for
        your version (is the returned quantity n_HI/n_H, with X_H applied
        separately as done here, or does it already fold in the H mass frac?).

Usage:
    python proposal_census.py \
        --snapshot /mnt/data0/pkrsnak/romulus/r142.007779.tipsy --name r142 \
        --snapshot /mnt/data0/pkrsnak/romulus/r154.007779.tipsy --name r154 \
        --snapshot /mnt/data0/pkrsnak/romulus/r168.007779.tipsy --name r168 \
        --output proposal_census.csv

    # repeat --snapshot/--name per galaxy, order must match (same pattern as
    # galaxy_diagnostic.py)
"""

import argparse
import csv
import warnings

import numpy as np
import pynbody
import pynbody.analysis


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DWARF_MSTAR_MAX = 3e9          # proposal's dwarf definition (Msun)
SFR_WINDOW_YR_DEFAULT = 25e6   # matches Fig 2 axis "Star Formation Rate (25 Myr)"
APERTURE_KPC_DEFAULT = 30.0    # fixed aperture; matches make_particles radius cut
HELIUM_MASS_FRACTION_DEFAULT = 0.24   # Y, for X_H = 1 - Y - Z
HI_TEMP_FLOOR_K = 1.5e4        # fallback only: gas below this treated as ~neutral

CSV_COLUMNS = [
    "name", "snapshot", "aperture_kpc",
    "N_star_in_ap", "Mstar_Msol", "is_dwarf",
    "N_gas_in_ap", "MHI_Msol", "SFR_window_Myr", "SFR_Msol_per_yr",
    "hi_method", "N_bh", "M_bh_Msol"
]


# ---------------------------------------------------------------------------
# Loading / centering (mirrors make_particles.py + galaxy_diagnostic.py)
# ---------------------------------------------------------------------------
def load_and_center(path):
    """Load a tipsy snapshot, convert to physical units, center on stellar peak."""
    data = pynbody.load(path)
    data.physical_units()
    # Shrinking-sphere centering on stars; move_all so positions are wrt center.
    pynbody.analysis.halo.center(data.s, mode='ssc', move_all=True)
    return data


# ---------------------------------------------------------------------------
# HI neutral fraction
# ---------------------------------------------------------------------------
def neutral_fraction(gas):
    """
    Return (f_HI, method) where f_HI is the neutral atomic hydrogen fraction
    n_HI/n_H per gas particle.

    Primary: pynbody ionization-equilibrium calculation (UVB tables).
    Fallback: temperature step — gas below HI_TEMP_FLOOR_K treated as neutral.
    """
    try:
        import pynbody.analysis.ionfrac as ionfrac
        # n_HI / n_H from post-hoc ionization equilibrium.
        # NB: confirm the return convention for your pynbody version. Some
        # versions also expose this as the derived array gas['HI'].
        f = np.asarray(ionfrac.calculate(gas, ion='hi'), dtype=float)
        f = np.clip(f, 0.0, 1.0)
        return f, "pynbody_ionfrac_hi"
    except Exception as e:
        warnings.warn(
            f"ionfrac HI unavailable ({type(e).__name__}: {e}); "
            f"falling back to a T<{HI_TEMP_FLOOR_K:.0e} K neutral step. "
            f"This crude fallback lumps in molecular gas — replace before "
            f"trusting M_HI.",
            RuntimeWarning,
        )
        T = np.asarray(gas['temp'].view(np.ndarray), dtype=float)
        f = (T < HI_TEMP_FLOOR_K).astype(float)
        return f, "temperature_fallback"


# ---------------------------------------------------------------------------
# Per-galaxy quantities
# ---------------------------------------------------------------------------
def compute_quantities(data, aperture_kpc, sfr_window_yr, he_fraction):
    """Compute M*, M_HI, SFR within a fixed spherical aperture about the origin."""
    # Radii from the (already-centered) origin, in kpc
    s_pos = np.asarray(data.star['pos'].in_units('kpc'))
    g_pos = np.asarray(data.gas['pos'].in_units('kpc'))
    s_r = np.sqrt((s_pos ** 2).sum(axis=1))
    g_r = np.sqrt((g_pos ** 2).sum(axis=1))
    # eliminate particles outside the aperture
    # Exclude BH sink particles (tform < 0): M*, SFR, and the in-aperture
    # star count all flow from s_in.
    is_bh = np.asarray(data.star['tform']) < 0
    s_in = (s_r < aperture_kpc) & ~is_bh
    g_in = g_r < aperture_kpc

    # --- stellar mass, BH mass ---
    s_mass = np.asarray(data.star['mass'].in_units('Msol'))
    bh_in = is_bh & (s_r < aperture_kpc)
    N_bh = int(bh_in.sum())
    M_bh = float(s_mass[bh_in].sum())
    Mstar = float(s_mass[s_in].sum())

    # --- SFR over the chosen window (current mass as massform proxy) ---
    age_yr = np.asarray(data.star['age'].in_units('yr'))
    young = s_in & (age_yr < sfr_window_yr)
    # SFR = total mass formed in the window / window duration
    # note this is 25Myr prior to the snapshot time
    # also using current mass, not massform since snapshot lacks this field
    SFR = float(s_mass[young].sum()) / sfr_window_yr

    # --- HI (atomic neutral hydrogen) ---
    g_mass = np.asarray(data.gas['mass'].in_units('Msol'))
    Z = np.asarray(data.gas['metals'], dtype=float)
    # to find the hydrogen mass fraction, subtract helium and metals
    # helium is assumed to be a constant mass fraction
    # metals are per-particle and can vary; they are already in mass fraction units
    X_H = np.clip(1.0 - he_fraction - Z, 0.0, 1.0)  
    # use pynbody's ionization equilibrium calculation when available, with a temperature-based fallback
    f_HI, hi_method = neutral_fraction(data.gas)
    # start with mass of each gas particle, multiply by its hydrogen mass fraction and neutral fraction, then sum over particles in the aperture
    MHI = float((g_mass[g_in] * X_H[g_in] * f_HI[g_in]).sum())

    return {
        "N_star_in_ap": int(s_in.sum()),
        "Mstar_Msol": Mstar,
        "is_dwarf": bool(Mstar < DWARF_MSTAR_MAX),
        "N_gas_in_ap": int(g_in.sum()),
        "MHI_Msol": MHI,
        "SFR_window_Myr": sfr_window_yr / 1e6,
        "SFR_Msol_per_yr": SFR,
        "hi_method": hi_method,
        "N_bh": N_bh,
        "M_bh_Msol": M_bh
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Reines BH-proposal census: M*, M_HI, SFR per Romulus halo")
    p.add_argument("--snapshot", action="append", required=True,
                   help="path to .tipsy snapshot (repeat per galaxy)")
    p.add_argument("--name", action="append", required=True,
                   help="short label (repeat; order must match --snapshot)")
    p.add_argument("--output", default="proposal_census.csv",
                   help="output CSV path (default: proposal_census.csv)")
    p.add_argument("--aperture-kpc", type=float, default=APERTURE_KPC_DEFAULT,
                   help=f"spherical aperture radius (default {APERTURE_KPC_DEFAULT} kpc)")
    p.add_argument("--sfr-window-myr", type=float, default=SFR_WINDOW_YR_DEFAULT / 1e6,
                   help="SFR averaging window in Myr (default 25, matches Fig 2)")
    p.add_argument("--he-fraction", type=float, default=HELIUM_MASS_FRACTION_DEFAULT,
                   help="helium mass fraction Y for X_H = 1 - Y - Z (default 0.24)")
    args = p.parse_args()

    if len(args.snapshot) != len(args.name):
        p.error("number of --snapshot and --name arguments must match")

    sfr_window_yr = args.sfr_window_myr * 1e6

    rows = []
    for snap, name in zip(args.snapshot, args.name):
        print(f"\n=== {name} ===")
        try:
            data = load_and_center(snap)
            print(f"  loaded: stars={len(data.star)}  gas={len(data.gas)}")
            q = compute_quantities(data, args.aperture_kpc,
                                   sfr_window_yr, args.he_fraction)
            q["name"] = name
            q["snapshot"] = snap
            q["aperture_kpc"] = args.aperture_kpc
            rows.append(q)
            print(f"  M*={q['Mstar_Msol']:.3e}  dwarf={q['is_dwarf']}  "
                  f"M_HI={q['MHI_Msol']:.3e}  "
                  f"SFR({q['SFR_window_Myr']:.0f}Myr)={q['SFR_Msol_per_yr']:.4f}  "
                  f"[{q['hi_method']}]")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # --- CSV ---
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    print(f"\nWrote {args.output}  ({len(rows)} galaxies)")

    # --- summary table ---
    if rows:
        print("\n" + "=" * 96)
        print(f"{'name':<8}{'M*[Msun]':>13}{'dwarf':>7}{'M_HI[Msun]':>14}"
              f"{'SFR[Msun/yr]':>15}{'N*':>9}{'Ngas':>8}{'N_bh':>6}{'M_bh[Msun]':>13}")
        print("-" * 96)
        for r in rows:
            print(f"{r['name']:<8}{r['Mstar_Msol']:>13.3e}{str(r['is_dwarf']):>7}"
                  f"{r['MHI_Msol']:>14.3e}{r['SFR_Msol_per_yr']:>15.4f}"
                  f"{r['N_star_in_ap']:>9d}{r['N_gas_in_ap']:>8d}"
                  f"{r['N_bh']:>6d}{r['M_bh_Msol']:>13.3e}")
        print("=" * 96)
        print(f"Dwarf threshold: M* < {DWARF_MSTAR_MAX:.1e} Msun.  "
              f"Aperture: {args.aperture_kpc:.0f} kpc.")
        print("M_HI is atomic HI (no H2). Confirm the prescription vs Sharma+2022.")


if __name__ == "__main__":
    main()
