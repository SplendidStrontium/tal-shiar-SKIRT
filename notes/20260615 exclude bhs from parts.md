# Monday, June 15, 2026

## need to exclude BHs from star particles
When I worked on this last, I found that I was not excluding BHs from star particles, which may interfere with proper SKIRT runs.

## BH FIX: make_particles.py 
Print some information after loading the snapshot. Also create an array that tabulates whether something is a star or not.