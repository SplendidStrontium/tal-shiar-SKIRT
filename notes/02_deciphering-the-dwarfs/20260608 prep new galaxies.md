# Monday, June 8, 2026

## analyze new galaxies re: proposal
As per discussed proposal, goal is to analyze snapshots to establish similiarity to actual data, in order to establish that we have an apples-to-apples comparison and argue for relevance of simulations to bring depth to findings on real-world galaxies.

## run proposal_census.py
> python proposal_census.py \
    --snapshot /mnt/data0/pkrsnak/romulus/r107.007779.tipsy --name r107 \
    --snapshot /mnt/data0/pkrsnak/romulus/r142.007779.tipsy --name r142 \
    --snapshot /mnt/data0/pkrsnak/romulus/r154.007779.tipsy --name r154 \
    --snapshot /mnt/data0/pkrsnak/romulus/r168.007779.tipsy --name r168 \
    --snapshot /mnt/data0/pkrsnak/romulus/r204.007779.tipsy --name r204 \
    --snapshot /mnt/data0/pkrsnak/romulus/r219.007779.tipsy --name r219 \
    --snapshot /mnt/data0/pkrsnak/romulus/r223.007779.tipsy --name r223 \
    --snapshot /mnt/data0/pkrsnak/romulus/r239.007779.tipsy --name r239 \
    --snapshot /mnt/data0/pkrsnak/romulus/r284.007779.tipsy --name r284 \
    --snapshot /mnt/data0/pkrsnak/romulus/r306.007779.tipsy --name r306 \
    --snapshot /mnt/data0/pkrsnak/romulus/r316.007779.tipsy --name r316 \
    --snapshot /mnt/data0/pkrsnak/romulus/r320.007779.tipsy --name r320 \
    --snapshot /mnt/data0/pkrsnak/romulus/r330.007779.tipsy --name r330 \
    --snapshot /mnt/data0/pkrsnak/romulus/r372.007779.tipsy --name r372 \
    --snapshot /mnt/data0/pkrsnak/romulus/r429.007779.tipsy --name r429 \
    --output proposal_census.csv

## preliminary results
Wrote proposal_census.csv  (15 galaxies)

=============================================================================
name         M*[Msun]  dwarf    M_HI[Msun]   SFR[Msun/yr]       N*    Ngas
-----------------------------------------------------------------------------
r107        1.630e+10  False     2.222e+08         0.0665   409210   17707
r142        8.553e+09  False     1.420e+09         1.2461   206019   74328
r154        1.650e+10  False     2.684e+08         1.3120   396678   20974
r168        6.592e+09  False     6.682e+08         0.4492   161969   36211
r204        4.146e+09  False     4.464e+08         0.4476   100424   27336
r219        6.080e+09  False     8.197e+08         0.8511   147193   45860
r223        6.015e+09  False     7.713e+08         1.0797   142654   37679
r239        4.310e+09  False     1.028e+09         1.0557   101446   45632
r284        2.933e+09   True     1.407e+08         0.0930    72957   11816
r306        3.539e+09  False     5.597e+08         0.3977    85366   32180
r316        1.993e+09   True     6.288e+08         0.4629    48018   33506
r320        3.880e+09  False     4.760e+08         0.6265    91463   26587
r330        2.085e+09   True     4.687e+08         0.2594    50568   29202
r372        2.363e+09   True     3.481e+08         0.2874    57458   19742
r429        1.813e+09   True     1.984e+08         0.0749    44256   15077
=============================================================================
Dwarf threshold: M* < 3.0e+09 Msun.  Aperture: 30 kpc.
M_HI is atomic HI (no H2). Confirm the prescription vs Sharma+2022.

## make_particles.py for new galaxies
a quick script and let's go
make sure you're in the right directory

> for g in r154 r168 r204 r219 r223 r239 r284 r306 r316 r330 r372 r429; do
  python src/make_particles.py \
    --snapshot /mnt/data0/pkrsnak/romulus/${g}.007779.tipsy \
    --output  /mnt/data0/pkrsnak/romulus/${g} \
    --radius 30000
done

## are there BHs in these galaxies?
Star particles that are heavier than they should be? Created detect_heavy_stars.py

## Yep, these look like black holes
They are not peeled off into data.bh, it seems. There are very heavy star particles with a negative tform, and these are almost certainly black holes.

r154  heavy>1e5:  2  max_M=1.43e+07  tform<0:  2  heavy&tform<0:  2  bh=0
r168  heavy>1e5:  1  max_M=1.41e+07  tform<0:  1  heavy&tform<0:  1  bh=0
r204  heavy>1e5:  1  max_M=6.87e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r219  heavy>1e5:  1  max_M=1.04e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r223  heavy>1e5:  0  max_M=6.36e+04  tform<0:  0  heavy&tform<0:  0  bh=0
r239  heavy>1e5:  1  max_M=2.05e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r284  heavy>1e5:  1  max_M=6.64e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r306  heavy>1e5:  1  max_M=1.01e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r316  heavy>1e5:  1  max_M=3.46e+06  tform<0:  1  heavy&tform<0:  1  bh=0
r330  heavy>1e5:  0  max_M=6.36e+04  tform<0:  0  heavy&tform<0:  0  bh=0
r372  heavy>1e5:  0  max_M=6.36e+04  tform<0:  0  heavy&tform<0:  0  bh=0
r429  heavy>1e5:  0  max_M=6.36e+04  tform<0:  0  heavy&tform<0:  0  bh=0

## revelation
I was getting ages for stars that were longer than the age of the universe; that mystery is now solved because tform is negative for BHs. Any calculation like age = t_now + |tform| was going to give values that don't make sense.

## @TODO 
- run detect_heavy_stars.py on galaxies I worked on before
- modify proposal_census.py to exclude BHs
- modify make_particles.py to exclude BHs
- run SKIRT on face-on galaxies, either w/ or w/o non-dwarfs
- 3e9 Msol as criteria for dwarf determined to be not important; remove it