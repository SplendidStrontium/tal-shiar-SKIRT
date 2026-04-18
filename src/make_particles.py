"""
make_particles.py — Tal Shiar SKIRT Pipeline, Step 1

Takes a Romulus zoom-in snapshot (.tipsy) and extracts star and gas particle
data into numpy arrays formatted for SKIRT radiative transfer.

What this script does:
    1. Loads the simulation snapshot with pynbody
    2. Centers the pynbody snapshot on the stellar center-of-mass
    3. Reorients the snapshot so the disk's angular momentum vector aligns
       with +z (face-on in the x-y plane). This makes downstream inclination
       angles physically meaningful: cos(i) = |v_hat . z_hat|.
    4. Converts all quantities to physical units (pc, Msol, km/s, yr)
    5. Applies a spatial cut to remove outlier particles
    6. Saves particle arrays as .npy files for SKIRT

Orientation correction (new in this version):
    The galaxy disk is not guaranteed to be aligned with the simulation's x-y
    plane. Without reorientation, the viewing directions produced by
    sampleOrientations.py have no well-defined inclination, and A(lambda, i)
    vs inclination plots are meaningless. We fix this by rotating the snapshot
    so the angular momentum vector of a disk tracer (cold gas preferred, young
    stars as fallback) points along +z BEFORE extracting particle arrays.

    Tracer selection is a cascading fallback:
        1. Cold gas (T < 3e4 K) within 10 kpc, if mass > 1e8 Msol
        2. Young stars (age < 1 Gyr) within 10 kpc, if mass > 1e7 Msol
        3. All stars within 10 kpc, with a loud warning (galaxy may be
           spheroidal and inclination interpretation is suspect)

What this script does NOT do (yet):
    - ageSmooth: a technique that splits each star particle into ~50 sub-particles
      with slightly different ages, to smooth out bursty star formation histories.
      The original NIHAO pipeline has this as an optional step. If needed later,
      refer to the ageSmooth() method in the original NIHAO makeParticles.py.
    - youngStars: separates stars younger than 10 Myr and assigns them
      MAPPINGS-III HII region SEDs instead of simple stellar SEDs, to model
      the fact that young stars are still embedded in birth clouds. Also creates
      "ghost" dust particles with negative mass to carve holes in the diffuse dust.
      If needed later, refer to the youngStars() method in the original NIHAO
      makeParticles.py.

Adapted from NIHAO-SKIRT-Pipeline/bin/makeParticles.py
Key differences from NIHAO version:
    - No halo catalogue lookup (Romulus data is already zoomed in on one halo)
    - Uses pynbody-native centering on the stellar center-of-mass
    - Applies orientation correction via pynbody.analysis.angmom.faceon before extraction
    - Field mapping: uses 'smooth' (not 'eps'), 'mass' (no 'massform'),
      'age' derived from 'tform'
"""

import pynbody
import pynbody.analysis
import numpy as np
import os
import argparse
import warnings
from timeit import default_timer as timer


# ---------------------------------------------------------------------------
# Tracer thresholds for orientation correction
# ---------------------------------------------------------------------------
# These are defaults chosen to catch common Romulus edge cases.
# - COLD_GAS_TEMP_K: upper temperature for "cold" gas (roughly the neutral/molecular phase)
# - TRACER_RADIUS_KPC: radial cut for angular momentum computation (0.1 R_vir proxy)
# - MIN_COLD_GAS_MASS / MIN_YOUNG_STAR_MASS: below these thresholds, fall back
#   to the next tracer. Romulus galaxies with strong BH feedback may have blown
#   out their cold gas, so the fallback is important.
COLD_GAS_TEMP_K = 3e4
YOUNG_STAR_AGE_YR = 1e9
TRACER_RADIUS_KPC = 10.0
MIN_COLD_GAS_MASS = 1e8
MIN_YOUNG_STAR_MASS = 1e7


def load_snapshot(filepath):
    """
    Load a tipsy snapshot with pynbody.

    pynbody automatically finds the .param file in the same directory,
    which provides cosmological parameters and unit conversions.
    """
    print(f"Loading snapshot: {filepath}")
    data = pynbody.load(filepath)
    data.physical_units()
    print(f"  Star particles: {len(data.star)}")
    print(f"  Gas particles:  {len(data.gas)}")
    return data


def center_snapshot(data):
    """
    Center the pynbody snapshot on the stellar center of mass.

    This is done on the pynbody SimSnap itself (not on extracted arrays) so
    that pynbody.analysis.angmom.faceon() downstream computes angular momentum in
    a frame where the galaxy is at the origin. faceon() assumes this.

    We use stars (not gas) because the stellar distribution better traces
    the galaxy center, especially in dwarfs where gas can be offset by feedback.
    """
    print("Centering snapshot on stellar density peak (shrinking-sphere)...")

    # Shrinking-sphere centering locks onto the main galaxy's densest peak,
    # ignoring satellites and extended halo stars. move_all=True ensures the
    # translation is applied to the parent snapshot, not just the stellar SubSnap.
    pynbody.analysis.halo.center(data.s, mode='ssc', move_all=True)

    # Verify centering propagated correctly
    com_check = (data.s['mass'].reshape(-1,1) * data.s['pos']).sum(axis=0) / data.s['mass'].sum()
    median_check = np.median(data.s['pos'], axis=0)
    print(f"  Post-centering stellar COM:    [{com_check[0]:+.3f}, {com_check[1]:+.3f}, {com_check[2]:+.3f}] kpc")
    print(f"  Post-centering stellar median: [{median_check[0]:+.3f}, {median_check[1]:+.3f}, {median_check[2]:+.3f}] kpc")
    if np.linalg.norm(median_check) > 0.5:
        print(f"  WARNING: stellar median is {np.linalg.norm(median_check):.2f} kpc from origin — centering may have failed")

    print("  Snapshot centered at origin.")


def select_orientation_tracer(data):
    """
    Pick the best tracer for defining the disk plane.

    Cascading fallback:
        1. Cold gas (T < COLD_GAS_TEMP_K) within TRACER_RADIUS_KPC
        2. Young stars (age < YOUNG_STAR_AGE_YR) within TRACER_RADIUS_KPC
        3. All stars within TRACER_RADIUS_KPC (with warning)

    Returns:
        tracer_subsnap: pynbody SubSnap to pass to faceon()
        tracer_name:    string describing the tracer (for logging/CSV)
    """
    print("Selecting orientation tracer...")

    # Compute radius in kpc for each gas/star particle
    gas_r_kpc = np.sqrt((data.gas['pos'].in_units('kpc') ** 2).sum(axis=1))
    star_r_kpc = np.sqrt((data.star['pos'].in_units('kpc') ** 2).sum(axis=1))

    # --- Attempt 1: cold gas ---
    gas_temp_k = data.gas['temp'].view(np.ndarray)
    cold_gas_mask = (gas_temp_k < COLD_GAS_TEMP_K) & (gas_r_kpc < TRACER_RADIUS_KPC)
    cold_gas_mass = float(data.gas['mass'].in_units('Msol')[cold_gas_mask].sum())
    print(f"  Cold gas (T<{COLD_GAS_TEMP_K:.0e}K, r<{TRACER_RADIUS_KPC}kpc): "
          f"{cold_gas_mask.sum()} particles, {cold_gas_mass:.2e} Msol")

    if cold_gas_mass > MIN_COLD_GAS_MASS:
        print(f"  -> Using cold gas as orientation tracer.")
        return data.gas[cold_gas_mask], 'cold_gas'

    # --- Attempt 2: young stars ---
    star_age_yr = data.star['age'].in_units('yr').view(np.ndarray)
    young_star_mask = (star_age_yr < YOUNG_STAR_AGE_YR) & (star_r_kpc < TRACER_RADIUS_KPC)
    young_star_mass = float(data.star['mass'].in_units('Msol')[young_star_mask].sum())
    print(f"  Young stars (age<{YOUNG_STAR_AGE_YR:.0e}yr, r<{TRACER_RADIUS_KPC}kpc): "
          f"{young_star_mask.sum()} particles, {young_star_mass:.2e} Msol")

    if young_star_mass > MIN_YOUNG_STAR_MASS:
        print(f"  -> Cold gas too sparse. Using young stars as orientation tracer.")
        return data.star[young_star_mask], 'young_stars'

    # --- Attempt 3: all stars (warn) ---
    all_star_mask = star_r_kpc < TRACER_RADIUS_KPC
    all_star_mass = float(data.star['mass'].in_units('Msol')[all_star_mask].sum())
    warnings.warn(
        f"Neither cold gas ({cold_gas_mass:.2e} Msol) nor young stars "
        f"({young_star_mass:.2e} Msol) meet the mass threshold for reliable "
        f"disk orientation. Falling back to all stars within {TRACER_RADIUS_KPC} kpc "
        f"({all_star_mass:.2e} Msol). Galaxy may be spheroidal, disturbed, or "
        f"quenched — inclination interpretation is suspect. Run galaxy_diagnostic.py "
        f"on this galaxy to assess whether it has a real disk.",
        RuntimeWarning,
    )
    return data.star[all_star_mask], 'all_stars_fallback'


def orient_faceon(data):
    """
    Rotate the snapshot so the tracer's angular momentum points along +z.

    After this call, the disk (if one exists) lies in the x-y plane, and any
    viewing direction v_hat has inclination cos(i) = |v_hat . z_hat|.

    pynbody.analysis.angmom.faceon operates in place on the parent snapshot when
    called on a SubSnap — all particles (not just the tracer subset) are
    rotated together.

    Returns:
        tracer_name: string identifying which tracer was used (for provenance)
    """
    tracer, tracer_name = select_orientation_tracer(data)
    print(f"Applying pynbody.analysis.angmom.faceon using tracer='{tracer_name}'...")

    # faceon rotates the parent snapshot in place. We pass move_all=True
    # (default) so the full data object — not just the tracer — is rotated.
    pynbody.analysis.angmom.faceon(tracer)

    print("  Snapshot reoriented: disk angular momentum now aligned with +z.")
    return tracer_name


def extract_star_properties(data):
    """
    Pull out all star particle properties that SKIRT needs,
    converting to physical units.

    Returns a dictionary of numpy arrays, each with shape (N,) or (N,3).

    Properties extracted:
        pos     - 3D position in parsecs (in disk-aligned frame)
        vel     - 3D velocity in km/s (in disk-aligned frame)
        mass    - particle mass in solar masses (used as both initial and current mass;
                  Romulus data does not have a separate 'massform' field)
        metals  - metallicity as a dimensionless mass fraction
        age     - stellar age in years (pynbody derives this from 'tform')
        smooth  - smoothing length in parsecs (pynbody derives this;
                  'eps' is all zeros/NaN in Romulus, do not use it)
    """
    print("Extracting star properties...")

    stars = {
        'x_pos':  np.float32(data.star['pos'].in_units('pc')[:, 0]),
        'y_pos':  np.float32(data.star['pos'].in_units('pc')[:, 1]),
        'z_pos':  np.float32(data.star['pos'].in_units('pc')[:, 2]),
        'x_vel':  np.float32(data.star['vel'].in_units('km s**-1')[:, 0]),
        'y_vel':  np.float32(data.star['vel'].in_units('km s**-1')[:, 1]),
        'z_vel':  np.float32(data.star['vel'].in_units('km s**-1')[:, 2]),
        'mass':   np.float32(data.star['mass'].in_units('Msol')),
        'metals': np.float32(data.star['metals']),
        'age':    np.float32(data.star['age'].in_units('yr')),
        'smooth': 2 * np.float32(data.star['smooth'].in_units('pc')),  # 2x smoothing length, following NIHAO convention
    }

    print(f"  Mass range: {stars['mass'].min():.2e} - {stars['mass'].max():.2e} Msol")
    print(f"  Age range:  {stars['age'].min():.2e} - {stars['age'].max():.2e} yr")

    return stars


def extract_gas_properties(data):
    """
    Pull out all gas/dust particle properties that SKIRT needs,
    converting to physical units.

    SKIRT uses gas particles to model the diffuse dust distribution.
    The dust mass at each particle is computed later from the gas mass
    and metallicity (metals act as a proxy for dust content).

    Properties extracted:
        pos      - 3D position in parsecs (in disk-aligned frame)
        mass     - particle mass in solar masses
        metals   - metallicity as a dimensionless mass fraction
        smooth   - smoothing length in parsecs
        temp     - temperature in Kelvin (used to exclude hot gas from dust modeling)
        density  - mass density in Msol/pc^3
    """
    print("Extracting gas properties...")

    gas = {
        'x_pos':   np.float32(data.gas['pos'].in_units('pc')[:, 0]),
        'y_pos':   np.float32(data.gas['pos'].in_units('pc')[:, 1]),
        'z_pos':   np.float32(data.gas['pos'].in_units('pc')[:, 2]),
        'mass':    np.float32(data.gas['mass'].in_units('Msol')),
        'metals':  np.float32(data.gas['metals']),
        'smooth':  2 * np.float32(data.gas['smooth'].in_units('pc')),  # 2x smoothing length
        'temp':    np.float32(data.gas['temp']),
        'density': np.float32(data.gas['rho'].in_units('Msol pc**-3')),
    }

    print(f"  Mass range: {gas['mass'].min():.2e} - {gas['mass'].max():.2e} Msol")
    print(f"  Temp range: {gas['temp'].min():.2e} - {gas['temp'].max():.2e} K")

    return gas


def spatial_cut(stars, gas, radius_pc):
    """
    Remove particles beyond a given radius from the origin.

    The Romulus zoom-in data is already focused on one halo, but there
    may be outlier particles at the edges that we don't want to include.
    This replaces the NIHAO approach of using a halo catalogue bounding box
    divided by 6 (an empirical calibration that doesn't apply here).

    Uses a spherical cut rather than a box cut — more physically appropriate
    for a galaxy.

    Args:
        stars:      dict of star property arrays
        gas:        dict of gas property arrays
        radius_pc:  maximum distance from origin in parsecs
    """
    # Compute radial distance for each particle
    star_r = np.sqrt(stars['x_pos']**2 + stars['y_pos']**2 + stars['z_pos']**2)
    gas_r = np.sqrt(gas['x_pos']**2 + gas['y_pos']**2 + gas['z_pos']**2)

    # Boolean masks: True for particles inside the radius
    star_mask = star_r < radius_pc
    gas_mask = gas_r < radius_pc

    n_stars_before = len(stars['x_pos'])
    n_gas_before = len(gas['x_pos'])

    # Apply mask to every array in the dictionaries
    for key in stars:
        stars[key] = stars[key][star_mask]
    for key in gas:
        gas[key] = gas[key][gas_mask]

    print(f"Spatial cut at {radius_pc:.0f} pc:")
    print(f"  Stars: {n_stars_before} -> {len(stars['x_pos'])} ({n_stars_before - len(stars['x_pos'])} removed)")
    print(f"  Gas:   {n_gas_before} -> {len(gas['x_pos'])} ({n_gas_before - len(gas['x_pos'])} removed)")


def save_particles(stars, gas, output_dir, tracer_name):
    """
    Save particle data as .npy files that downstream SKIRT scripts expect.

    Output format matches NIHAO-SKIRT convention:
        stars.npy — columns: x, y, z, smooth, vx, vy, vz, mass, metals, age
        gas.npy   — columns: x, y, z, smooth, mass, metals, temp
        gas_density.npy — gas density in its own file
        current_mass_stars.npy — stellar mass (same as mass for Romulus)

    Also writes orientation_info.txt recording which tracer was used to
    align the disk, for provenance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stars: 10 columns
    star_array = np.float32(np.c_[
        stars['x_pos'], stars['y_pos'], stars['z_pos'],
        stars['smooth'],
        stars['x_vel'], stars['y_vel'], stars['z_vel'],
        stars['mass'],
        stars['metals'],
        stars['age']
    ])

    # Gas: 7 columns
    gas_array = np.float32(np.c_[
        gas['x_pos'], gas['y_pos'], gas['z_pos'],
        gas['smooth'],
        gas['mass'],
        gas['metals'],
        gas['temp']
    ])

    # Gas density: separate file (same as NIHAO convention)
    gas_density_array = np.float32(np.c_[gas['density']])

    # Current mass: for Romulus, this is the same as mass
    # (NIHAO had a separate 'massform' for initial mass)
    current_mass_array = np.float32(np.c_[stars['mass']])

    np.save(os.path.join(output_dir, 'stars.npy'), star_array)
    np.save(os.path.join(output_dir, 'gas.npy'), gas_array)
    np.save(os.path.join(output_dir, 'gas_density.npy'), gas_density_array)
    np.save(os.path.join(output_dir, 'current_mass_stars.npy'), current_mass_array)

    # Provenance: record which tracer set the disk orientation
    with open(os.path.join(output_dir, 'orientation_info.txt'), 'w') as f:
        f.write(f"orientation_tracer: {tracer_name}\n")
        f.write(f"tracer_radius_kpc: {TRACER_RADIUS_KPC}\n")
        f.write(f"cold_gas_temp_k:   {COLD_GAS_TEMP_K}\n")
        f.write(f"young_star_age_yr: {YOUNG_STAR_AGE_YR}\n")
        f.write(f"convention: disk angular momentum L_hat aligned with +z_hat\n")
        f.write(f"inclination formula: cos(i) = |v_hat . z_hat|\n")

    print(f"\nSaved to {output_dir}:")
    print(f"  stars.npy              — {star_array.shape}")
    print(f"  gas.npy                — {gas_array.shape}")
    print(f"  gas_density.npy        — {gas_density_array.shape}")
    print(f"  current_mass_stars.npy — {current_mass_array.shape}")
    print(f"  orientation_info.txt   — tracer={tracer_name}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    start = timer()

    parser = argparse.ArgumentParser(description="Extract particle data for SKIRT")
    parser.add_argument("--snapshot", required=True,
                        help="Path to .tipsy snapshot file")
    parser.add_argument("--output", required=True,
                        help="Directory to save output .npy files")
    parser.add_argument("--radius", type=float, default=None,
                        help="Spatial cut radius in parsecs (optional; if not given, no cut is applied)")
    parser.add_argument("--skip-orient", action='store_true',
                        help="Skip orientation correction (for debugging only; "
                             "downstream inclination angles will be meaningless)")
    args = parser.parse_args()

    # Step 1: Load the simulation snapshot
    data = load_snapshot(args.snapshot)

    # Step 2: Center the pynbody snapshot (needed before faceon)
    center_snapshot(data)

    # Step 3: Orient the disk face-on (L_hat -> +z_hat)
    if args.skip_orient:
        warnings.warn("Skipping orientation correction. Downstream inclination "
                      "angles will NOT be physically meaningful.", RuntimeWarning)
        tracer_name = 'SKIPPED'
    else:
        tracer_name = orient_faceon(data)

    # Step 4: Extract and convert units (positions now in disk-aligned frame)
    stars = extract_star_properties(data)
    gas = extract_gas_properties(data)

    # Step 5: Spatial cut (optional)
    if args.radius is not None:
        spatial_cut(stars, gas, args.radius)

    # Step 6: Save
    save_particles(stars, gas, args.output, tracer_name)

    end = timer()
    print(f"\nDone in {end - start:.1f} seconds.")
