#!/usr/bin/env python3
"""
Generate Tal Shiar SKIRT ski files for the twelve-halo proposal sample.

For each halo in HALOS, produces two files:
  {halo}_dust.ski   — ExtinctionOnly with THEMIS dust medium
  {halo}_nodust.ski — NoMedium (stellar-only baseline, no transport)

Both share identical sources, instruments, wavelength grid, and inclinations
so that F_dust / F_nodust is a clean per-orientation, per-wavelength ratio.

Inclinations: face-on only (i=0) for the Reines-proposal B-K colors. The
inclination machinery is preserved — to recover the attenuation-vs-angle
sweep, set INCLINATIONS_DEG to list(np.linspace(0.0, 90.0, 12)).

Usage:
    python generate_ski.py        # writes 2 * len(HALOS) ski files into cwd
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample + run configuration
# ---------------------------------------------------------------------------

# The fifteen halos (BH-filtered particles regenerated). Edit if your
# intended sample differs.
HALOS = ["r107", "r142", "r154", "r168", "r204", "r219", "r223", "r239",
         "r284", "r306", "r316", "r320", "r330", "r372", "r429"]

# Face-on only for B-K. For the attenuation thread:
#   INCLINATIONS_DEG = list(np.linspace(0.0, 90.0, 12))
INCLINATIONS_DEG = [0.0]

NUM_PHOTONS_PRODUCTION = "5e7"   # dust run scales with this; nodust is NoMedium
DISTANCE_MPC = 100

# Spatial grid — particle data cut at 30 kpc, give a little margin
GRID_HALF_KPC = 35          # -> ±35000 pc
FOV_KPC = 70                # -> 70000 pc across instrument
NUM_PIXELS = 500            # production; driver can override for test

# Wavelength range — UV through NIR
WL_MIN_MICRON = 0.1
WL_MAX_MICRON = 2.5
NUM_WAVELENGTHS = 150        # log-spaced SED resolution

# Grid refinement
MIN_LEVEL = 6
MAX_LEVEL = 9       # was 8; increased to reduce noise

# Dust
DUST_FRACTION = 0.4         # NIHAO mainstream
MAX_DUST_TEMP_K = 30000      # adjusted for Romulus
NUM_SILICATE_SIZES = 15
NUM_HYDROCARBON_SIZES = 15


# ---------------------------------------------------------------------------
# XML fragment builders
# ---------------------------------------------------------------------------

def inclinations_deg():
    """Inclinations to render, in degrees (face-on only by default)."""
    return np.asarray(INCLINATIONS_DEG, dtype=float)


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

def sed_instrument_block(inc_deg, idx):
    """Spatially-integrated SED on the default fine LogWavelengthGrid.
    This is where the B-K colors come from; the FullInstrument keeps its
    band grid for the RGB images."""
    name = f"sed_i{idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return f'''          <SEDInstrument instrumentName="{name}" distance="{DISTANCE_MPC} Mpc" inclination="{inc_deg:.4f} deg" azimuth="0 deg" roll="0 deg" recordComponents="false" numScatteringLevels="0" recordPolarization="false" recordStatistics="false"/>'''

def all_instruments():
    incs = inclinations_deg()
    blocks = []
    for i, inc in enumerate(incs):
        blocks.append(instrument_block(inc, i))       # FullInstrument: band images (unchanged)
        blocks.append(sed_instrument_block(inc, i))   # SEDInstrument: fine SED for B-K
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
    maxTemperature={MAX_DUST_TEMP_K} K kills the hot ISM. Note this is
    raised vs the Camps+ 8000 K convention because Romulus's BH feedback
    leaves more of the ISM in a warm phase than NIHAO galaxies.

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
    Return empty string — no <mediumSystem> element at all.

    SKIRT v9 treats an empty <MediumSystem></MediumSystem> element as an
    error (it tries to instantiate default geometry requiring scaleLength).
    To run truly medium-free, the mediumSystem element must be absent from
    the ski entirely. F_nodust is then the intrinsic stellar SED with no
    radiative transfer.
    """
    return ""


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

def build_ski(galaxy_id, with_dust, num_photons=NUM_PHOTONS_PRODUCTION):
    label = "dust" if with_dust else "no-dust"
    medium = medium_system_with_dust() if with_dust else medium_system_nodust()
    mode = "ExtinctionOnly" if with_dust else "NoMedium"
    probes = probe_system(with_dust)
    n_inc = len(inclinations_deg())

    return f'''<?xml version='1.0' encoding='UTF-8'?>
<!-- Tal Shiar SKIRT: {galaxy_id}, {label}, {mode}, UV-NIR, {n_inc} inclination(s) -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="Tal Shiar pipeline">
  <MonteCarloSimulation userLevel="Regular" simulationMode="{mode}" numPackets="{num_photons}">
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
# Write all halos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    outdir = Path(".")

    for galaxy_id in HALOS:
        dust_ski = build_ski(galaxy_id, with_dust=True)
        nodust_ski = build_ski(galaxy_id, with_dust=False)

        dust_path = outdir / f"{galaxy_id}_dust.ski"
        nodust_path = outdir / f"{galaxy_id}_nodust.ski"
        dust_path.write_text(dust_ski)
        nodust_path.write_text(nodust_ski)

        print(f"[{galaxy_id}] wrote {dust_path.name} ({len(dust_ski):,} B)  "
              f"+ {nodust_path.name} ({len(nodust_ski):,} B)")

    print()
    print(f"Halos:            {len(HALOS)}  ->  {2 * len(HALOS)} ski files")
    print(f"Photons (dust):   {NUM_PHOTONS_PRODUCTION}")
    print("Inclinations (deg):", ", ".join(f"{x:.1f}" for x in inclinations_deg()))