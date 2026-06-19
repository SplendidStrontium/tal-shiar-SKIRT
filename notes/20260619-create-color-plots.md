# Friday, June 19, 2026

## compare_colors.py
Initial results point to very different populations; sims are quite a bit redder than the real halos, so let's diagnose and continue.

## is it the stellar populations that are too red, or inclusion of too much dust with the 30K hot gas cap?

> python extract_sim_colors.py --all
halo    mag_B   mag_Ks       BK  BK_nodust  internal_reddening flux_style
r107 5.433155 1.645892 3.787263   3.789876           -0.002613        fnu
r142 4.586804 1.509753 3.077050   2.658092            0.418958        fnu
r154 5.650265 1.145396 4.504869   3.079354            1.425514        fnu
r168 5.033962 2.120391 2.913571   2.773900            0.139671        fnu
r204 5.391932 2.406839 2.985093   2.699494            0.285598        fnu
r219 4.915393 1.934211 2.981182   2.712712            0.268470        fnu
r223 5.269657 1.748099 3.521557   2.604861            0.916696        fnu
r239 5.026388 1.990078 3.036310   2.422271            0.614039        fnu
r284 6.405272 3.277871 3.127401   3.082082            0.045319        fnu
r306 5.474001 2.527954 2.946047   2.648430            0.297616        fnu
r316 5.484996 3.040291 2.444705   2.385113            0.059591        fnu
r320 5.608248 2.235512 3.372737   2.467214            0.905523        fnu
r330 5.814842 3.164457 2.650385   2.599724            0.050661        fnu
r372 5.845481 3.023211 2.822270   2.673306            0.148964        fnu
r429 6.435801 3.472632 2.963169   2.875820            0.087349        fnu

wrote sim_colors.csv  (15/15 halos)
dust   B-K: median 2.99  range 2.44-4.50
nodust B-K: median 2.67  range 2.39-3.79
internal reddening from dust: median +0.27 mag in B-K
  -> compare nodust median to the observed median (~2.40):
     if nodust already ~observed, the +offset is the dust model;
     if nodust is still red, it's the stellar populations.

## still quite a bit too red, and sims are notably much more massive than the catalog given
But a rerun does not change the bottom bracket; nodust is the least red of the runs, and a run at 8K instead of 30K gas temp would only change the top limit, and is perhaps not a good guess for hot ROMULUS gas anyway.

## created plots, moving forward
(base) pkrsnak@hamilton:~/tal-shiar-SKIRT/python$ python compare_colors.py 
observed (clean):  n=221  median=2.40  range 1.47-4.04
simulated (dust):  n= 15  median=2.99  range 2.44-4.50
simulated (nodust):n= 15  median=2.67  range 2.39-3.79

KS two-sample:    D=0.731  p=4.26e-08
/home/pkrsnak/tal-shiar-SKIRT/python/compare_colors.py:63: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
  ad = stats.anderson_ksamp([obs, sim])
Anderson-Darling: A2=19.808  p=0.001
median offset (dust sim - obs): +0.59
KS (nodust vs obs): D=0.499  p=0.000957
median offset (nodust sim - obs): +0.28
--> dust-treatment bracket on sim median offset: +0.28 (nodust)  to  +0.59 (30,000 K dust)
NOTE: n_sim=15 -> modest power; read p-values with that in mind.

wrote bk_comparison.pdf (+ .png)
(base) pkrsnak@hamilton:~/tal-shiar-SKIRT/python$ python build_figures.py 
sim halos with mass: 15  (with-BH=11, no-BH=4)
observed clean: 221
wrote color_mass_diagram.pdf (+ .png)

BH split [dust (observable)]:
  with-BH n=11  median 3.04
  no-BH   n=4  median 2.89
  median difference (BH - noBH): +0.14
  KS: D=0.477  p=0.408
  permutation test on median diff: p=0.198

BH split [intrinsic (no dust)]:
  with-BH n=11  median 2.70
  no-BH   n=4  median 2.64
  median difference (BH - noBH): +0.06
  KS: D=0.295  p=0.895
  permutation test on median diff: p=0.56
  NOTE: n=11+4 is small; this test has low power and a non-detection is not evidence of 'no difference'.
wrote bh_split.pdf (+ .png)