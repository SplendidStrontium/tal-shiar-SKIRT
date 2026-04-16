"""
make_particles.py — Tal Shiar SKIRT Pipeline, Step 1

Takes a Romulus zoom-in snapshot (.tipsy) and extracts star and gas particle
data into numpy arrays formatted for SKIRT radiative transfer.

What this script does:
    1. Loads the simulation snapshot with pynbody
    2. Converts all quantities to physical units (pc, Msol, km/s, yr)
    3. Centers the galaxy at the origin (SKIRT expects this)
    4. Applies a spatial cut to remove outlier particles
    5. Saves particle arrays as .npy files for SKIRT

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
    - Uses center of mass for centering instead of bounding box midpoint
    - Field mapping: uses 'smooth' (not 'eps'), 'mass' (no 'massform'), 
      'age' derived from 'tform'
"""

import pynbody
import numpy as np
import os
import argparse
from timeit import default_timer as timer


def load_snapshot(filepath):
    """
    Load a tipsy snapshot with pynbody.
    
    pynbody automatically finds the .param file in the same directory,
    which provides cosmological parameters and unit conversions.
    """
    print(f"Loading snapshot: {filepath}")
    data = pynbody.load(filepath)
    print(f"  Star particles: {len(data.star)}")
    print(f"  Gas particles:  {len(data.gas)}")
    return data


def extract_star_properties(data):
    """
    Pull out all star particle properties that SKIRT needs,
    converting to physical units.

    Returns a dictionary of numpy arrays, each with shape (N,) or (N,3).
    
    Properties extracted:
        pos     - 3D position in parsecs
        vel     - 3D velocity in km/s
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
        pos      - 3D position in parsecs
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


def compute_center_of_mass(stars):
    """
    Compute the mass-weighted center of position for the stellar component.
    
    This replaces the NIHAO approach of taking the midpoint of the bounding box
    extremes, which is not physically motivated — an asymmetric galaxy would
    have its center placed incorrectly.
    
    We use stars (not gas) because the stellar distribution better traces
    the galaxy center, especially in dwarfs where gas can be offset.
    
    Returns (x_center, y_center, z_center) in parsecs.
    """
    total_mass = np.sum(stars['mass'])
    x_center = np.sum(stars['x_pos'] * stars['mass']) / total_mass
    y_center = np.sum(stars['y_pos'] * stars['mass']) / total_mass
    z_center = np.sum(stars['z_pos'] * stars['mass']) / total_mass

    print(f"Center of mass: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f}) pc")

    return x_center, y_center, z_center


def center_particles(stars, gas, center):
    """
    Shift all particle positions so the galaxy center is at (0, 0, 0).
    SKIRT expects the source to be centered at the origin.
    """
    x_c, y_c, z_c = center

    stars['x_pos'] -= x_c
    stars['y_pos'] -= y_c
    stars['z_pos'] -= z_c

    gas['x_pos'] -= x_c
    gas['y_pos'] -= y_c
    gas['z_pos'] -= z_c

    print("Particles centered at origin.")


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


def save_particles(stars, gas, output_dir):
    """
    Save particle data as .npy files that downstream SKIRT scripts expect.
    
    Output format matches NIHAO-SKIRT convention:
        stars.npy — columns: x, y, z, smooth, vx, vy, vz, mass, metals, age
        gas.npy   — columns: x, y, z, smooth, mass, metals, temp
        gas_density.npy — gas density in its own file
        current_mass_stars.npy — stellar mass (same as mass for Romulus)
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

    print(f"\nSaved to {output_dir}:")
    print(f"  stars.npy            — {star_array.shape}")
    print(f"  gas.npy              — {gas_array.shape}")
    print(f"  gas_density.npy      — {gas_density_array.shape}")
    print(f"  current_mass_stars.npy — {current_mass_array.shape}")


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
    args = parser.parse_args()

    # Step 1: Load the simulation snapshot
    data = load_snapshot(args.snapshot)

    # Step 2: Extract and convert units
    stars = extract_star_properties(data)
    gas = extract_gas_properties(data)

    # Step 3: Center the galaxy at the origin using center of mass
    center = compute_center_of_mass(stars)
    center_particles(stars, gas, center)

    # Step 4: Spatial cut (optional)
    if args.radius is not None:
        spatial_cut(stars, gas, args.radius)

    # Step 5: Save
    save_particles(stars, gas, args.output)

    end = timer()
    print(f"\nDone in {end - start:.1f} seconds.")