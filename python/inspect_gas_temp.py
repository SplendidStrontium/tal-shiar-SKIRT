import numpy as np

# gas.npy columns: [x, y, z, smooth, mass, metals, temp]
g = np.load('/mnt/data0/pkrsnak/romulus/r142/gas.npy')
x, y, z = g[:,0], g[:,1], g[:,2]
mass, metals, temp = g[:,4], g[:,5], g[:,6]

r = np.sqrt(x**2 + y**2 + z**2)
in_box = (np.abs(x) < 35000) & (np.abs(y) < 35000) & (np.abs(z) < 35000)  # pc, grid box is +/-35 kpc
cold = temp < 8000

print(f"Total gas particles:        {len(mass):>10d}, M = {mass.sum():.3e} Msun")
print(f"  in 35 kpc box:            {in_box.sum():>10d}, M = {mass[in_box].sum():.3e} Msun")
print(f"  cold (T<8000):            {cold.sum():>10d}, M = {mass[cold].sum():.3e} Msun")
print(f"  cold AND in box:          {(cold & in_box).sum():>10d}, M = {mass[cold & in_box].sum():.3e} Msun")
print()
print(f"Temperature distribution:")
for tcut in [1e3, 3e3, 8e3, 3e4, 1e5, 1e6]:
    m = mass[temp < tcut].sum()
    print(f"  T < {tcut:.0e}:  {(temp<tcut).sum():>8d} particles, M = {m:.3e} Msun ({100*m/mass.sum():.1f}%)")
print()
cold_in_box = cold & in_box
dust_mass = 0.4 * (mass[cold_in_box] * metals[cold_in_box]).sum()
print(f"Expected dust mass (dustFraction=0.4 x cold in-box metals): {dust_mass:.3e} Msun")
print(f"SKIRT reported:                                             5.66e5 Msun")
print()
print(f"Z-extent of cold in-box gas:")
zcold = z[cold_in_box]
print(f"  |z|_min: {np.abs(zcold).min():.1f} pc")
print(f"  |z|_max: {np.abs(zcold).max():.1f} pc")
print(f"  |z|_50%: {np.median(np.abs(zcold)):.1f} pc (half-thickness)")
print(f"  |z|_90%: {np.percentile(np.abs(zcold), 90):.1f} pc")