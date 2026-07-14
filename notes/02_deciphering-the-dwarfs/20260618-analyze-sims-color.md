# Thursday, June 18, 2026

## SKIRT output must be analyzed for color

## install a library designed to analyze simulated photometry
> pip install pyphot

## run new scripts to analyze sim colors
this runs just on r284, verbose.
> python extract_sim_colors.py    

and this does everyone
> python extract_sim_colors.py --all

## how does SKIRT determine color?
When you run production, a simple .dat file for SED is generated. It has two columns and relays relative flux at various values of lambda. This information is NOT distance calibrated, so while B-K is meaningful, the raw data is not.

## A problem, and it's pretty big?
When we read the SED, there are not enough points to make our information meaninful. We need to rerun SKIRT with reconfigured ski files. Instead of a handful of points, we need many more in the range that interests us.

## corrections to generate_ski.py
more wavelengths

## ready to rerun
regenerated ski files with more wavelengths, ran some tests, ready to rerun

> python run_skirt_production.py --detach
> tail -f production_sweep.log

## then later when done:
> python extract_sim_colors.py --all

## results:
halo    mag_B   mag_Ks       BK flux_style
r107 5.433155 1.645892 3.787263        fnu
r142 4.586804 1.509753 3.077050        fnu
r154 5.650265 1.145396 4.504869        fnu
r168 5.033962 2.120391 2.913571        fnu
r204 5.391932 2.406839 2.985093        fnu
r219 4.915393 1.934211 2.981182        fnu
r223 5.269657 1.748099 3.521557        fnu
r239 5.026388 1.990078 3.036310        fnu
r284 6.405272 3.277871 3.127401        fnu
r306 5.474001 2.527954 2.946047        fnu
r316 5.484996 3.040291 2.444705        fnu
r320 5.608248 2.235512 3.372737        fnu
r330 5.814842 3.164457 2.650385        fnu
r372 5.845481 3.023211 2.822270        fnu
r429 6.435801 3.472632 2.963169        fnu

wrote sim_colors.csv  (15/15 halos)
B-K range: 2.44 to 4.50, median 2.99