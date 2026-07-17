"""
inspect_gas_temp.py — inspect gas particle temperature distribution
for a Romulus galaxy. Reports total mass, dust mass at the SKIRT cutoff,
and z-extent of the cold ISM.

Output drives the choice of MAX_DUST_TEMP_K in generate_ski.py: we want
a cutoff that captures the bulk of the dust-bearing gas without including
hot diffuse halo material.

Usage:
    Edit GALAXY_ID below and run.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Galaxy selection — change this for r107 / r320
# ---------------------------------------------------------------------------
GALAXY_ID = "r320"

# Reference cutoff for the dust-mass spot-check (matches MAX_DUST_TEMP_K in
# generate_ski.py). Update if you change the cutoff in the ski generator.
DUST_TEMP_CUTOFF_K = 30000

# ---------------------------------------------------------------------------

# gas.npy columns: [x, y, z, smooth, mass, metals, temp]
gas_path = f'/mnt/data0/pkrsnak/romulus/{GALAXY_ID}/gas.npy'
print(f"Loading {gas_path}")
g = np.load(gas_path)
x, y, z = g[:,0], g[:,1], g[:,2]
mass, metals, temp = g[:,4], g[:,5], g[:,6]

r = np.sqrt(x**2 + y**2 + z**2)
in_box = (np.abs(x) < 35000) & (np.abs(y) < 35000) & (np.abs(z) < 35000)  # pc, grid box is +/-35 kpc
cold = temp < DUST_TEMP_CUTOFF_K

print(f"\n=== {GALAXY_ID} gas summary ===")
print(f"Total gas particles:        {len(mass):>10d}, M = {mass.sum():.3e} Msun")
print(f"  in 35 kpc box:            {in_box.sum():>10d}, M = {mass[in_box].sum():.3e} Msun")
print(f"  cold (T<{DUST_TEMP_CUTOFF_K:.0e}):           {cold.sum():>10d}, M = {mass[cold].sum():.3e} Msun")
print(f"  cold AND in box:          {(cold & in_box).sum():>10d}, M = {mass[cold & in_box].sum():.3e} Msun")
print()
print(f"Temperature distribution:")
for tcut in [1e3, 3e3, 8e3, 3e4, 1e5, 1e6]:
    m = mass[temp < tcut].sum()
    pct = 100*m/mass.sum() if mass.sum() > 0 else 0.0
    print(f"  T < {tcut:.0e}:  {(temp<tcut).sum():>8d} particles, M = {m:.3e} Msun ({pct:.1f}%)")
print()
cold_in_box = cold & in_box
if cold_in_box.sum() > 0:
    dust_mass = 0.4 * (mass[cold_in_box] * metals[cold_in_box]).sum()
    print(f"Expected dust mass (dustFraction=0.4 x cold in-box metals): {dust_mass:.3e} Msun")
    print()
    print(f"Z-extent of cold in-box gas:")
    zcold = z[cold_in_box]
    print(f"  |z|_min: {np.abs(zcold).min():.1f} pc")
    print(f"  |z|_max: {np.abs(zcold).max():.1f} pc")
    print(f"  |z|_50%: {np.median(np.abs(zcold)):.1f} pc (half-thickness)")
    print(f"  |z|_90%: {np.percentile(np.abs(zcold), 90):.1f} pc")
else:
    print(f"WARNING: no cold gas particles in box — check temperature cutoff "
          f"or whether this galaxy has had its ISM blown out by feedback.")