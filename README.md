# Tal Shiar SKIRT

Tal Shiar SKIRT is a pipeline that takes in data from hydrodynamic cosmological simulations and runs [SKIRT radiative transfer](https://github.com/SKIRT/SKIRT9), generating synthetic images and photometry. 

Workflow is heavily inspired by the [NIHAO-SKIRT-Pipeline](https://github.com/ntf229/NIHAO-SKIRT-Pipeline), adapted for initial use on zoom-in elements of the [Romulus simulation](https://mtremmel.github.io/research/romulus.html). 

## Completed Explorations

### "A Matter of Perspective", presented May 1, 2026
Initial run of project, designed to study effect of dust and orientation on light attenuation on UV-optical-NIR wavelengths. Largely a proof-of-concept exploration to establish functional SKIRT pipeline. (3 halos x 12 orientations)

### "Deciphering the Dwarfs", presented July 1, 2026
Analysis of B-K color of simulated halos in order to make an apples-to-apples comparison to galaxies in the Nearby Galaxy Catalog. Expanded pipeline to properly measure flux in appropriate bands. Used dustmaps to adjust NGC data to allow fair comparison. (15 simulated halos vs. ~200 NGC galaxies)

### "Twin Studies for Simulated Galaxies", presented July 29, 2026
How does changing BH physics change the evolution of a dwarf galaxy? A collection of zoom-in elements of ROMULUS, simulated to present day with our own recipe for BH physics. Constructed as a twin study, comparing and contrasting runs in matched pairs to isolate physics change. (5 families x 4 BH variants) (also referred to internally by shortname "Enterprise")

## Repo Layout

- **`/notes`** - diary summarizing work done, organized by date
- **`/old`** - scripts from older versions, included as reference only
- **`/python`** - helper/analysis scripts, data extraction, plot construction
- **`/src`** - particle prep, SKIRT runners, ski files

## Pipeline Shape
1. **make_particles.py**
   Takes one snapshot and halo ID and produces what SKIRT needs to run. pynbody loads snapshot, centers via shrinking-sphere, then rotates to face-on using a tracer's angular momentum (cold gas -> young stars -> all stars fallback). Extracts star/gas arrays, excludes BH sink particles while calculating angular momentum. Option to do a spherical spatial cut to exclude particles beyond given radius. Outputs stars.txt (old stars >= 10 Myr, FSPS SED assumes empty space around star), youngStars.txt (< 10 Myr, MAPPINGS SED assumes cocoon still exists around star), gas.txt (gas particles -> THEMIS dust), .npy arrays (legacy NIHAO convention, nothing in current pipeline consumes them), orientation_info.txt (which tracer set the frame), diagnostics.csv (one row per run)
2. **run_make_particles_all.py**
   Driver: loops calls of make_particles.py with fixed settings. Hardcoded config at start of script; HALO_ID hardcoded as 1 was personally verified with a diagnostic designed for this project, and snap glob also used for Enterprise; may need to be changed for future applications.
3. **run_skirt_test.py**
   Validates one run cheaply before committing hours to full SKIRT runs. Copies production dust/nodust .ski files to test_runs/, rewrites copies (photon count down to 1e6, pixels down, optionally fewer instruments, optionally maxLevel), symlinks particle files in, runs SKIRT twice per ski, first with -e emulation (catches schema and file-reference errors), then with timing. Finishes by projecting full-sweep cost, dust scaled linearly with photons, projects nodust flat, which underestimates because all photons still launched. Emulation proves the run can start, not that its output means anything.
4. **run_skirt_production.py**
   Runs SKIRT with no ski rewriting. .done_dust/.done_nodust markers written only on success so failed label retries on next sweep. Emulation canary gates each run before expensive part. Smallest family runs first. Detach mode re-execs itself under nohup with append-mode session-stamped log. 

## Python Scripts
1. **generate_ski_enterprise.py**
   Must be done before running SKIRT. Generates .ski files used by SKIRT to configure its simulated instrument. Changed to be a script after review of NIHAO, making it easier to change a .ski file from one line of code rather than hardcoding XML. TODO: files generated in working directory and then moved to where pipeline expects to find them. This should be changed, and possibly changed to output to data directory.
2. **extract_enterprise_colors.py**
   Parses every output SED, computes synthetic B and Ks, writes long-format enterprise_colors.csv

