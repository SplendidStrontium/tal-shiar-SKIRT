# Tal Shiar SKIRT

Tal Shiar SKIRT is a pipeline that takes in data from hydrodynamic cosmological simulations and runs [SKIRT radiative transfer](https://github.com/SKIRT/SKIRT9), generating images similar to those that might be obtainable with a telescope. 

Workflow is heavily inspired by the [NIHAO-SKIRT-Pipeline](https://github.com/ntf229/NIHAO-SKIRT-Pipeline), adapted for initial use on zoom-in elements of the [Romulus simulation](https://mtremmel.github.io/research/romulus.html). Romulus has been tuned to exhibit high rates of BH feedback, so a number of parameters needed to be adjusted to account for less total ISM and less-disk-like galaxies. Temperature at which we assume dust can exist in hot gas also needed to be increased vs. the Camps 8000K used in NIHAO.

Initial run of project aims to study effect of dust and orientation on light attenuation on UV-optical-NIR wavelengths.

## Primary pipeline in /src:

1. make_particles.py         // create particle arrays for SKIRT
2. run_skirt_test.py         // test SKIRT set-up w/o running full SKIRT
3. run_skirt_production.py   // run SKIRT

## supplementary scripts in /python

1. inspect_structure.py      // inspect snapshot to analyze structure
2. galaxy_diagnostic.py      // compare snapshots side-by-side
3. inspect_stars.py          // inspect snapshot for star info
4. inspect_gas.py            // inspect snapshot for gas info
5. inspect_gas_temp.py       // inspect npy array for gas characteristics
6. generate_ski.py           // create XML file for SKIRT based on input parameters
7. plot_attenuation          // interpret SKIRT files
8. compare_galaxies          // compare galaxy attenuation curves
9. make_money_shots          // assign colors to SKIRT-assigned bands
