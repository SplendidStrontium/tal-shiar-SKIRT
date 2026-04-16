"""
Inspect star particle properties in detail.
For each field makeParticles.py needs, check if it exists and what it looks like.
"""
import pynbody
import numpy as np

data = pynbody.load('/mnt/data0/pkrsnak/romulus/r107.007779.tipsy')

print(f"Total star particles: {len(data.star)}")

# Fields that makeParticles.py needs for stars:
# pos, mass, massform, metals, age/tform, vel, smooth/eps
fields_to_check = {
    'pos':      'Position (SKIRT needs x,y,z in pc)',
    'mass':     'Current mass (SKIRT needs Msol)',
    'massform': 'Initial mass at formation (NIHAO uses this)',
    'metals':   'Metallicity (dimensionless fraction)',
    'age':      'Age - pynbody derived field (yr)',
    'tform':    'Formation time - raw field (yr)',
    'vel':      'Velocity (SKIRT needs km/s)',
    'smooth':   'Smoothing length (NIHAO uses this)',
    'eps':      'Gravitational softening (alternative to smooth)',
}

for field, description in fields_to_check.items():
    print(f"\n--- {field}: {description} ---")
    try:
        arr = data.star[field]
        print(f"  Shape: {arr.shape}")
        print(f"  Units: {arr.units}")
        print(f"  Min:   {np.min(arr):.6e}")
        print(f"  Max:   {np.max(arr):.6e}")
        print(f"  Mean:  {np.mean(arr):.6e}")
        print(f"  First 3 values: {arr[:3]}")
    except Exception as e:
        print(f"  NOT AVAILABLE: {e}")