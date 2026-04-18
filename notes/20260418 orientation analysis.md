# Saturday, April 18, 2026

## running analysis on romulus galaxies
> python galaxy_diagnostic.py \
        --snapshot /mnt/data0/pkrsnak/romulus/r107.007779.tipsy --name r107 \
        --snapshot /mnt/data0/pkrsnak/romulus/r142.007779.tipsy --name r142 \
        --snapshot /mnt/data0/pkrsnak/romulus/r320.007779.tipsy --name r320 \
        --output diagnostic_results/

## proceeding with r142 as our galaxy to analyze
diagnostic output revealed r142 as most disk-like, analyzing rotational dynamics and distribution of particles. Detailed diagnostic, including images of galaxies in both edge&face-on orientations, saved in personal Dropbox.

## deployed key fix to COM calculation
COM calculation was buggy for r107; this fix makes it more robust and ready to apply to r142 or any other galaxy.

## step 1: make_particles.py
> python make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r142.007779.tipsy \
    --output /mnt/data0/pkrsnak/romulus/r142 \
    --radius 30000

Adding a radius cut will help keep outer contaminants out and speed up SKIRT runs by excluding particles that do not notably contribute to the final image. Cut appears to have excluded ~1400 stars and about half the gas particles. 

## decisions about parameters
Moving forward with dust temperature max=8000K as per Camps and convention.
Dust fraction=0.4 as per NIHAO default
SKIRT template needs updating and adherence to the bands in which we're working.
