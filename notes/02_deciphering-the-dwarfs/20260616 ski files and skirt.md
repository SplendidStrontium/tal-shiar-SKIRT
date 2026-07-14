# Tuesday, June 16, 2026

## new ski files
Must rerun with inclinations corrected; last project was with changing the angle so now we're just doing face-on. 

Renamed old generate_ski.py to generate_ski_12_inc.py so that I could keep the old file and move forward into a new file for this project.

## ski files have moved
I created a directory because there are thirty of these and I would prefer to keep them separate.

## also redoing old skirt scripts

## no actually I'm going to make a directory of old things and move stuff in there

## run_skirt_test.py
> python run_skirt_test.py --photons 5e6

Results seem OK. 

## run_skirt_production.py
> python run_skirt_production.py --dry-run

> python run_skirt_production.py --detach

Monitor with:
> tail -f ~/tal-shiar-SKIRT/src/production_sweep.log