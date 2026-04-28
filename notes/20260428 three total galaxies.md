# Tuesday, April 28, 2026

## make_particles.py for other galaxies

> python make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r107.007779.tipsy \
    --output /mnt/data0/pkrsnak/romulus/r107 \
    --radius 30000

> python make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r320.007779.tipsy \
    --output /mnt/data0/pkrsnak/romulus/r320 \
    --radius 30000

## generate_ski for new galaxies
Change line 26 and run for each.

## run_skirt parameterization
Change all occurences of r142 to use generalized ID set near top of code.

## run_skirt_test.py
Let's try r107.

> python run_skirt_test.py

## planning for time
These could take longer. Plan for it.

> python run_skirt_production.py --detach
