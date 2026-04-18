#!/usr/bin/env python3
"""
Generate Tal Shiar SKIRT ski files for r142.

Produces two files:
  r142_dust.ski   — ExtinctionOnly with THEMIS dust medium
  r142_nodust.ski — ExtinctionOnly with no medium (stellar-only baseline)

Both share identical sources, instruments, wavelength grid, and inclinations
so that F_dust / F_nodust is a clean per-orientation, per-wavelength ratio.

Inclinations: 12 values evenly spaced in i from 0° to 90° (deterministic grid
for A(lambda, i) curves — deliberately NOT arccos-uniform, which would be
appropriate for orientation-population sampling).

Usage:
    python generate_ski.py
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_INCLINATIONS = 12
NUM_PHOTONS_PRODUCTION = "1e7"   # override from driver for test runs
DISTANCE_MPC = 100

# Spatial grid — particle data cut at 30 kpc, give a little margin
GRID_HALF_KPC = 35          # -> ±35000 pc
FOV_KPC = 70                # -> 70000 pc across instrument
NUM_PIXELS = 500            # production; driver can override for test

# Wavelength range — UV through NIR
WL_MIN_MICRON = 0.1
WL_MAX_MICRON = 2.5
NUM_WAVELENGTHS = 80        # log-spaced SED resolution

# Grid refinement
MIN_LEVEL = 6
MAX_LEVEL = 8

# Dust
DUST_FRACTION = 0.4         # NIHAO mainstream
MAX_DUST_TEMP_K = 8000      # Camps+ convention
NUM_SILICATE_SIZES = 15
NUM_HYDROCARBON_SIZES = 15


# ---------------------------------------------------------------------------
# XML fragment builders
# ---------------------------------------------------------------------------

def inclinations_deg():
    """12 values evenly spaced in inclination from 0 to 90 degrees."""
    return np.linspace(0.0, 90.0, NUM_INCLINATIONS)


def instrument_block(inc_deg, idx):
    """
    One FullInstrument per inclination.

    FullInstrument records SED + datacube + broadband images in a single pass,
    so we don't need a separate SEDInstrument. Broadband grid is GALEX + SDSS
    + 2MASS only (dropping WISE/HERSCHEL — out of our 0.1-2.5 um range).
    """
    name = f"i{idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return f'''          <FullInstrument instrumentName="{name}" distance="{DISTANCE_MPC} Mpc" inclination="{inc_deg:.4f} deg" azimuth="0 deg" roll="0 deg" fieldOfViewX="{FOV_KPC*1000} pc" numPixelsX="{NUM_PIXELS}" centerX="0 pc" fieldOfViewY="{FOV_KPC*1000} pc" numPixelsY="{NUM_PIXELS}" centerY="0 pc" recordComponents="false" numScatteringLevels="0" recordPolarization="false" recordStatistics="false">
            <wavelengthGrid type="WavelengthGrid">
              <PredefinedBandWavelengthGrid includeGALEX="true" includeSDSS="true" include2MASS="true" includeWISE="false" includeHERSCHEL="false"/>
            </wavelengthGrid>
          </FullInstrument>'''


def all_instruments():
    incs = inclinations_deg()
    blocks = [instrument_block(inc, i) for i, inc in enumerate(incs)]
    return "\n".join(blocks)


def sources_block():
    """Two stellar populations: old (FSPS/Chabrier) + young (MAPPINGS)."""
    return f'''        <sources type="Source">
          <ParticleSource filename="stars.txt" importVelocity="true" importVelocityDispersion="false" useColumns="" sourceWeight="1" wavelengthBias="0.5">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <sedFamily type="SEDFamily">
              <FSPSSEDFamily imf="Chabrier"/>
            </sedFamily>
            <wavelengthBiasDistribution type="WavelengthDistribution">
              <LogWavelengthDistribution minWavelength="{WL_MIN_MICRON} micron" maxWavelength="{WL_MAX_MICRON} micron"/>
            </wavelengthBiasDistribution>
          </ParticleSource>
          <ParticleSource filename="youngStars.txt" importVelocity="true" importVelocityDispersion="false" useColumns="" sourceWeight="1" wavelengthBias="0.5">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <sedFamily type="SEDFamily">
              <MappingsSEDFamily/>
            </sedFamily>
            <wavelengthBiasDistribution type="WavelengthDistribution">
              <LogWavelengthDistribution minWavelength="{WL_MIN_MICRON} micron" maxWavelength="{WL_MAX_MICRON} micron"/>
            </wavelengthBiasDistribution>
          </ParticleSource>
        </sources>'''


def medium_system_with_dust():
    """
    THEMIS dust medium from gas.txt. dustFraction=0.4 applied via massFraction.
    maxTemperature=8000 K kills the hot ISM (Camps+ convention).

    NOTE: We keep PhotonPacketOptions but drop DustEmissionOptions and
    RadiationFieldOptions entirely — ExtinctionOnly mode doesn't use them.
    """
    grid_pc = GRID_HALF_KPC * 1000
    return f'''    <mediumSystem type="MediumSystem">
      <MediumSystem>
        <photonPacketOptions type="PhotonPacketOptions">
          <PhotonPacketOptions minWeightReduction="1e4" minScattEvents="0" pathLengthBias="0.5"/>
        </photonPacketOptions>
        <media type="Medium">
          <ParticleMedium filename="gas.txt" massFraction="{DUST_FRACTION}" importMetallicity="true" importTemperature="true" maxTemperature="{MAX_DUST_TEMP_K} K" importVelocity="false" importMagneticField="false" importVariableMixParams="false" useColumns="">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <materialMix type="MaterialMix">
              <ThemisDustMix numSilicateSizes="{NUM_SILICATE_SIZES}" numHydrocarbonSizes="{NUM_HYDROCARBON_SIZES}"/>
            </materialMix>
          </ParticleMedium>
        </media>
        <grid type="SpatialGrid">
          <PolicyTreeSpatialGrid minX="-{grid_pc} pc" maxX="{grid_pc} pc" minY="-{grid_pc} pc" maxY="{grid_pc} pc" minZ="-{grid_pc} pc" maxZ="{grid_pc} pc" treeType="OctTree">
            <policy type="TreePolicy">
              <DensityTreePolicy minLevel="{MIN_LEVEL}" maxLevel="{MAX_LEVEL}" maxDustFraction="1e-6" maxDustOpticalDepth="0" wavelength="0.55 micron" maxDustDensityDispersion="0" maxElectronFraction="1e-6" maxGasFraction="1e-6"/>
            </policy>
          </PolicyTreeSpatialGrid>
        </grid>
      </MediumSystem>
    </mediumSystem>'''


def medium_system_nodust():
    """
    No medium at all. SKIRT SourceSystem + no MediumSystem means stellar
    emission propagates unimpeded. F_nodust is the intrinsic stellar SED.
    """
    return '''    <mediumSystem type="MediumSystem">
      <MediumSystem>
      </MediumSystem>
    </mediumSystem>'''


def probe_system(with_dust):
    """
    Convergence probes only make sense when there's a medium.
    Keep them for the dust run, skip for no-dust.
    """
    if with_dust:
        return '''    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
          <ConvergenceInfoProbe probeName="spatial_convergence" wavelength="0.55 micron"/>
          <ConvergenceCutsProbe probeName="media_density_cuts"/>
        </probes>
      </ProbeSystem>
    </probeSystem>'''
    else:
        return '''    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
        </probes>
      </ProbeSystem>
    </probeSystem>'''


# ---------------------------------------------------------------------------
# Full ski file template
# ---------------------------------------------------------------------------

def build_ski(with_dust, num_photons=NUM_PHOTONS_PRODUCTION):
    label = "dust" if with_dust else "no-dust"
    medium = medium_system_with_dust() if with_dust else medium_system_nodust()
    probes = probe_system(with_dust)

    return f'''<?xml version='1.0' encoding='UTF-8'?>
<!-- Tal Shiar SKIRT: r142, {label}, ExtinctionOnly, UV-NIR, 12 inclinations -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="Tal Shiar pipeline">
  <MonteCarloSimulation userLevel="Regular" simulationMode="ExtinctionOnly" numPackets="{num_photons}">
    <random type="Random">
      <Random seed="0"/>
    </random>
    <units type="Units">
      <ExtragalacticUnits fluxOutputStyle="Frequency"/>
    </units>
    <cosmology type="Cosmology">
      <LocalUniverseCosmology/>
    </cosmology>
    <sourceSystem type="SourceSystem">
      <SourceSystem minWavelength="{WL_MIN_MICRON} micron" maxWavelength="{WL_MAX_MICRON} micron" sourceBias="0.5">
{sources_block()}
      </SourceSystem>
    </sourceSystem>
{medium}
    <instrumentSystem type="InstrumentSystem">
      <InstrumentSystem>
        <defaultWavelengthGrid type="WavelengthGrid">
          <LogWavelengthGrid minWavelength="{WL_MIN_MICRON} micron" maxWavelength="{WL_MAX_MICRON} micron" numWavelengths="{NUM_WAVELENGTHS}"/>
        </defaultWavelengthGrid>
        <instruments type="Instrument">
{all_instruments()}
        </instruments>
      </InstrumentSystem>
    </instrumentSystem>
{probes}
  </MonteCarloSimulation>
</skirt-simulation-hierarchy>
'''


# ---------------------------------------------------------------------------
# Write both files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    outdir = Path(".")

    dust_ski = build_ski(with_dust=True)
    nodust_ski = build_ski(with_dust=False)

    (outdir / "r142_dust.ski").write_text(dust_ski)
    (outdir / "r142_nodust.ski").write_text(nodust_ski)

    print(f"Wrote r142_dust.ski   ({len(dust_ski):,} bytes)")
    print(f"Wrote r142_nodust.ski ({len(nodust_ski):,} bytes)")
    print()
    print("Inclinations (deg):")
    for i, inc in enumerate(inclinations_deg()):
        print(f"  i{i:02d}: {inc:6.3f}")
