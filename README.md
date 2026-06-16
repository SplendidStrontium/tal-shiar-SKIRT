# Tal Shiar SKIRT

Tal Shiar SKIRT is a pipeline that takes in data from hydrodynamic cosmological simulations and runs [SKIRT radiative transfer](https://github.com/SKIRT/SKIRT9), generating images similar to those that might be obtainable with a telescope. 

Workflow is heavily inspired by the [NIHAO-SKIRT-Pipeline](https://github.com/ntf229/NIHAO-SKIRT-Pipeline), adapted for initial use on zoom-in elements of the [Romulus simulation](https://mtremmel.github.io/research/romulus.html). Romulus has been tuned to exhibit high rates of BH feedback, so a number of parameters needed to be adjusted to account for less total ISM and less-disk-like galaxies. Temperature at which we assume dust can exist in hot gas also needed to be increased vs. the Camps 8000K used in NIHAO.

Initial run of project, presented May 1, 2026, aimed to study effect of dust and orientation on light attenuation on UV-optical-NIR wavelengths.

Current run of project, slated to present July 1, 2026, analyzes B-K color of halos in order to make an apples-to-apples comparison to galaxies in the Nearby Galaxy Catalog.

## primary pipeline in /src:
1. make_particles.py         // create particle arrays for SKIRT
2. run_skirt_test.py         // test SKIRT set-up w/o running full SKIRT
3. run_skirt_production.py   // run SKIRT

## supplementary scripts in /python
1. inspect_structure.py      // inspect snapshot to analyze structure
3. inspect_stars.py          // inspect snapshot for star info
4. inspect_gas.py            // inspect snapshot for gas info
5. inspect_gas_temp.py       // inspect npy array for gas characteristics
6. generate_ski.py           // create XML file for SKIRT based on input parameters
9. make_money_shots          // assign colors to SKIRT-assigned bands
12. proposal_census.py       // read in NGC and analyze

## previously used files in /old
### scripts used for: A Matter of Perspective, presented May 1, 2026
1. plot_attenuation.py       // interpret SKIRT files for 12 inclinations
2. compare_galaxies          // compare galaxy attenuation curves
3. galaxy_diagnostic.py      // compare snapshots side-by-side
4. make_dust_comparison.py   // dust/nodust comparison
### scripts used for: Deciphering the Dwarfs, presenting July 1, 2026
1. detect_heavy_stars.py    // black holes are star particles with tform < 0