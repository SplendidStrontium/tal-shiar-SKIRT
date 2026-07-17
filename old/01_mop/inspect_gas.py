"""
Inspect gas particle properties in detail.
For each field makeParticles.py needs, check if it exists and what it looks like.
"""
import pynbody
import numpy as np

data = pynbody.load('/mnt/data0/pkrsnak/romulus/r107.007779.tipsy')

print(f"Total gas particles: {len(data.gas)}")

# Fields that makeParticles.py needs for gas/dust:
# pos, smooth, mass, metals, temp, rho (density)
fields_to_check = {
    'pos':     'Position (SKIRT needs x,y,z in pc)',
    'mass':    'Mass (SKIRT needs Msol)',
    'metals':  'Metallicity (dimensionless fraction)',
    'temp':    'Temperature (K)',
    'rho':     'Density (SKIRT needs Msol/pc^3)',
    'smooth':  'Smoothing length (pc)',
    'eps':     'Gravitational softening (alternative to smooth)',
    'vel':     'Velocity (km/s)',
}

for field, description in fields_to_check.items():
    print(f"\n--- {field}: {description} ---")
    try:
        arr = data.gas[field]
        print(f"  Shape: {arr.shape}")
        print(f"  Units: {arr.units}")
        print(f"  Min:   {np.min(arr):.6e}")
        print(f"  Max:   {np.max(arr):.6e}")
        print(f"  Mean:  {np.mean(arr):.6e}")
        print(f"  First 3 values: {arr[:3]}")
    except Exception as e:
        print(f"  NOT AVAILABLE: {e}")