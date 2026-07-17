"""
Inspect the overall structure of a Romulus tipsy snapshot.
What particle types exist, how many of each, what fields are available.
"""
import pynbody

data = pynbody.load('/mnt/data0/pkrsnak/romulus/r107.007779.tipsy')

print("=== Simulation Properties ===")
for key, val in data.properties.items():
    print(f"  {key}: {val}")

print(f"\n=== Particle Counts ===")
print(f"  Total particles: {len(data)}")
if hasattr(data, 'gas'):
    print(f"  Gas particles:  {len(data.gas)}")
if hasattr(data, 'star'):
    print(f"  Star particles: {len(data.star)}")
if hasattr(data, 'dm'):
    print(f"  DM particles:   {len(data.dm)}")

print(f"\n=== Star Loadable Keys ===")
print(data.star.loadable_keys())

print(f"\n=== Gas Loadable Keys ===")
print(data.gas.loadable_keys())

print(f"\n=== DM Loadable Keys ===")
print(data.dm.loadable_keys())