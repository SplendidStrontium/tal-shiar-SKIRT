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