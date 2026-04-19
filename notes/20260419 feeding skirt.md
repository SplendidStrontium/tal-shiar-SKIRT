# Sunday, April 19, 2026

## correcting make_particles.py
Young stars need to be separated from population of ordinary stars; SKIRT expects this.

> python make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r142.007779.tipsy \
    --output /mnt/data0/pkrsnak/romulus/r142 \
    --radius 30000

## preparing to run SKIRT for real
SKIRT can take a long time, so we're going to do a low-count photon run on r142 at face-on orientation to ensure that logic holds.

> python run_skirt_test.py

Some errors here; SKIRT is not really expecting a no-dust run, so we have to edit our .ski file to omit the mediumSystem section. Consequently we have to rerun generate_ski.py.

This is also a diversion from the NIHAO setup; In NIHAO, the no-dust run was with tau=0. This version is completely different, no noise from photon transport, and it should run much faster as well.

Also worth noting is that SKIRT reads units in text files and expects Msun, not Msol.

Also an issue with ISM and temperature cutoff. Was using the 8K threshold used in NIHAO; this is totally inappropriate for Romulus. Need a higher temperature, otherwise there just isn't enough dust to get any meaningful result. 

## doing a dry-run to diagnose
> python run_skirt_production.py --dry-run

## run SKIRT
> python run_skirt_production.py

or, if you want to close your terminal:
> python run_skirt_production.py --detach
