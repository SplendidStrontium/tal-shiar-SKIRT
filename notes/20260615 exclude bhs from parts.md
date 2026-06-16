# Monday, June 15, 2026

## need to exclude BHs from star particles
When I worked on this last, I found that I was not excluding BHs from star particles, which may interfere with proper SKIRT runs.

## BH FIX: make_particles.py 
Print some information after loading the snapshot. Also create an array that tabulates whether something is a star or not.

Also adding a mask in select_orientation_tracer.

## BH FIX: proposal_census.py
Fixed and make table pretty.

================================================================================================
name         M*[Msun]  dwarf    M_HI[Msun]   SFR[Msun/yr]       N*    Ngas  N_bh   M_bh[Msun]
------------------------------------------------------------------------------------------------
r107        1.627e+10  False     2.222e+08         0.0665   409205   17707     5    2.809e+07
r142        8.549e+09  False     1.420e+09         1.2461   206018   74328     1    4.156e+06
r154        1.648e+10  False     2.684e+08         1.3120   396676   20974     2    1.538e+07
r168        6.578e+09  False     6.682e+08         0.4492   161968   36211     1    1.412e+07
r204        4.139e+09  False     4.464e+08         0.4476   100423   27336     1    6.871e+06
r219        6.079e+09  False     8.197e+08         0.8511   147192   45860     1    1.039e+06
r223        6.015e+09  False     7.713e+08         1.0797   142654   37679     0    0.000e+00
r239        4.308e+09  False     1.028e+09         1.0557   101445   45632     1    2.051e+06
r284        2.927e+09   True     1.407e+08         0.0930    72956   11816     1    6.640e+06
r306        3.538e+09  False     5.597e+08         0.3977    85365   32180     1    1.012e+06
r316        1.990e+09   True     6.288e+08         0.4629    48017   33506     1    3.457e+06
r320        3.876e+09  False     4.760e+08         0.6265    91461   26587     2    3.229e+06
r330        2.085e+09   True     4.687e+08         0.2594    50568   29202     0    0.000e+00
r372        2.363e+09   True     3.481e+08         0.2874    57458   19742     0    0.000e+00
r429        1.813e+09   True     1.984e+08         0.0749    44256   15077     0    0.000e+00
================================================================================================

## remake particles
> for g in r107 r142 r154 r168 r204 r219 r223 r239 r284 r306 r316 r320 r330 r372 r429; do
    echo "=== $g ==="
    python make_particles.py \
        --snapshot /mnt/data0/pkrsnak/romulus/${g}.007779.tipsy \
        --output  /mnt/data0/pkrsnak/romulus/${g}/ \
        --radius 30000
done 2>&1 | tee make_particles_batch.log
