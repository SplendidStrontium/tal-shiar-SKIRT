# Tal Shiar SKIRT

Tal Shiar SKIRT is a pipeline that takes in data from hydrodynamic cosmological simulations and runs [SKIRT radiative transfer](https://github.com/SKIRT/SKIRT9), generating images similar to those that might be obtainable with a telescope. 

Workflow is heavily inspired by the [NIHAO-SKIRT-Pipeline](https://github.com/ntf229/NIHAO-SKIRT-Pipeline), adapted for initial use on zoom-in elements of the [Romulus simulation](https://mtremmel.github.io/research/romulus.html). 

Initial run of project aims to study effect of dust and orientation on light attenuation on UV-optical-NIR wavelengths.

## Primary pipeline in /src:

1. make_particles.py 

## supplementary scripts in /python
