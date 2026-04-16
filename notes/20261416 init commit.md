# Thursday, April 16, 2026

## init commit
Starting creation and analysis of RT pipeline. 

## step 0: data inspection
Bringing over several scripts that allow inspection of the given data.

> python python/inspect_structure.py

Working on R107, we get the following:
=== Simulation Properties ===
  omegaM0: 0.3086               ## matter density fraction
  omegaL0: 0.6914               ## dark energy density fraction
  h: 0.6776931508813172         ## hubble parameter
  boxsize: 2.50e+04 kpc a       ## small, =16 kpc
  a: 0.9524351996399713         ## scale factor; gives redshift using z=(1/a); so z≈0.05
  time: 1.34e+01 s kpc km**-1

=== Particle Counts ===
  Total particles: 468580
  Gas particles:  52318
  Star particles: 416262
  DM particles:   0             ## dark matter stripped; SKIRT doesn't care anyway

=== Star Loadable Keys ===
['phi', 'mass', 'tform', 'eps', 'metals', 'vel', 'pos']

=== Gas Loadable Keys ===
['temp', 'phi', 'rho', 'mass', 'eps', 'metals', 'vel', 'pos']

=== DM Loadable Keys ===
['phi', 'mass', 'eps', 'vel', 'pos']

> python python/inspect_stars.py

> python python/inspect_gas.py

## step 1: make_particles.py
Drawing in original code from nihao-mod and adapting.

> python src/make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r107.007779.tipsy \
    --output /mnt/data0/pkrsnak/romulus/Particles/r107/