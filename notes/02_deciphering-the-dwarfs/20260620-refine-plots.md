# Saturday, June 20, 2026

## splitting and refining plots
observed (clean):  n=221  median=2.40  range 1.47-4.04
simulated (dust):  n= 15  median=2.99  range 2.44-4.50
simulated (nodust):n= 15  median=2.67  range 2.39-3.79

KS two-sample:    D=0.731  p=4.26e-08
/home/pkrsnak/tal-shiar-SKIRT/python/compare_colors.py:65: UserWarning: p-value floored: true value smaller than 0.001. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)
  ad = stats.anderson_ksamp([obs, sim])
Anderson-Darling: A2=19.808  p=0.001
median offset (dust sim - obs): +0.59
KS (nodust vs obs): D=0.499  p=0.000957
median offset (nodust sim - obs): +0.28
--> dust-treatment bracket on sim median offset: +0.28 (nodust)  to  +0.59 (30,000 K dust)
NOTE: n_sim=15 -> modest power; read p-values with that in mind.

wrote bk_histogram.pdf (+ .png)
wrote bk_cdf.pdf (+ .png)

## producing plots poster vs. talk
KS/p value may go on slide deck, omitted from poster