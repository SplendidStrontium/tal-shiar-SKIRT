#!/usr/bin/env python3
"""
Generate Enterprise SKIRT ski files for the BH-variant comparison.

Adapted from the Tal Shiar generate_ski.py. For each of the 20 runs
(5 families x 4 BH-physics variants), produces two files:

  {run}_dust.ski   — ExtinctionOnly with THEMIS dust medium
  {run}_nodust.ski — NoMedium (stellar-only baseline, no transport)

40 ski files total. Both share identical sources, instruments, wavelength
grid, and inclinations so F_dust / F_nodust is a clean per-orientation,
per-wavelength ratio.

DELIBERATE DIFFERENCES from the Tal Shiar configuration (do not "fix"):

  1. MAX_DUST_TEMP_K = 8000, not 30000. Tal Shiar used 3e4 (set somewhat
     arbitrarily); Enterprise adopts the published Camps & Trayford
     convention. The sensitivity of retained dust mass to this choice is
     characterized in dust_threshold_scan.{csv,png} — median 32% of
     aperture dust mass lies between the two conventions, and the
     retained fraction differs systematically across BH variants
     (feedback pushes ISM into the 8e3-3e4 K warm phase).
  2. imf="Kroupa" in FSPSSEDFamily, not Chabrier. Matches ChaNGa's
     internal IMF and the Kroupa-like IMF baked into the MAPPINGS young
     star SEDs, making the two source populations self-consistent.
  3. INCLINATIONS_DEG = [0, 90]. Instruments are peel-off, so the
     edge-on view rides along in the same run at marginal cost; the
     edge-on minus face-on color isolates dust geometry per galaxy.
  4. NUM_WAVELENGTHS = 250 (was 150). Cheap insurance for band
     convolution; only affects binning, not photon count.
  5. Sample is 20 {family}_{variant} runs, not 15 Tal Shiar halos.

Photometric system decision (recorded here so it can't get lost):
    B and Ks are synthesized downstream by convolving the SEDInstrument
    output with Johnson B and 2MASS Ks response curves, reported in the
    VEGA system. Both bands are conventionally Vega; use Vega everywhere.

Usage:
    python generate_ski_enterprise.py       # writes 40 ski files into cwd
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample + run configuration
# ---------------------------------------------------------------------------

FAMILIES = ["r488", "r568", "r613", "r618", "r741"]
VARIANTS = ["noBH", "BH", "BH6", "BH8"]


def run_names():
    """20 run names matching the on-disk directory convention:
    noBH runs are the bare family name; others are family_variant."""
    names = []
    for fam in FAMILIES:
        for var in VARIANTS:
            names.append(fam if var == "noBH" else f"{fam}_{var}")
    return names


# Face-on + edge-on. Peel-off instruments: both views come from ONE run.
INCLINATIONS_DEG = [0.0, 90.0]

NUM_PHOTONS_PRODUCTION = "5e7"   # pending single-run timing test
DISTANCE_MPC = 100

# Spatial grid — particle data cut at 30 kpc, give a little margin
GRID_HALF_KPC = 35          # -> ±35000 pc
FOV_KPC = 70                # -> 70000 pc across instrument
NUM_PIXELS = 500            # production; driver can override for test

# Wavelength range — UV through NIR
WL_MIN_MICRON = 0.1
WL_MAX_MICRON = 2.5
NUM_WAVELENGTHS = 250        # log-spaced SED resolution (raised from 150)

# Grid refinement
MIN_LEVEL = 6
MAX_LEVEL = 9

# Dust
DUST_FRACTION = 0.4          # dust-to-metals, NIHAO mainstream
MAX_DUST_TEMP_K = 8000       # Camps & Trayford convention; see header
NUM_SILICATE_SIZES = 15
NUM_HYDROCARBON_SIZES = 15


# ---------------------------------------------------------------------------
# XML fragment builders
# ---------------------------------------------------------------------------

def inclinations_deg():
    return np.asarray(INCLINATIONS_DEG, dtype=float)


def instrument_block(inc_deg, idx):
    """
    One FullInstrument per inclination: SED + datacube + broadband images
    in a single pass. Broadband grid is GALEX + SDSS + 2MASS (0.1-2.5 um).
    """
    name = f"i{idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return f'''          <FullInstrument instrumentName="{name}" distance="{DISTANCE_MPC} Mpc" inclination="{inc_deg:.4f} deg" azimuth="0 deg" roll="0 deg" fieldOfViewX="{FOV_KPC*1000} pc" numPixelsX="{NUM_PIXELS}" centerX="0 pc" fieldOfViewY="{FOV_KPC*1000} pc" numPixelsY="{NUM_PIXELS}" centerY="0 pc" recordComponents="false" numScatteringLevels="0" recordPolarization="false" recordStatistics="false">
            <wavelengthGrid type="WavelengthGrid">
              <PredefinedBandWavelengthGrid includeGALEX="true" includeSDSS="true" include2MASS="true" includeWISE="false" includeHERSCHEL="false"/>
            </wavelengthGrid>
          </FullInstrument>'''


def sed_instrument_block(inc_deg, idx):
    """Spatially-integrated SED on the default fine LogWavelengthGrid.
    This is where B and Ks come from (convolution downstream, Vega system);
    the FullInstrument keeps its band grid for the RGB images."""
    name = f"sed_i{idx:02d}_{inc_deg:05.2f}deg".replace(".", "p")
    return f'''          <SEDInstrument instrumentName="{name}" distance="{DISTANCE_MPC} Mpc" inclination="{inc_deg:.4f} deg" azimuth="0 deg" roll="0 deg" recordComponents="false" numScatteringLevels="0" recordPolarization="false" recordStatistics="false"/>'''


def all_instruments():
    incs = inclinations_deg()
    blocks = []
    for i, inc in enumerate(incs):
        blocks.append(instrument_block(inc, i))
        blocks.append(sed_instrument_block(inc, i))
    return "\n".join(blocks)


def sources_block():
    """Two stellar populations: old (FSPS/Kroupa) + young (MAPPINGS).
    Kroupa matches ChaNGa internals and MAPPINGS' built-in IMF."""
    return f'''        <sources type="Source">
          <ParticleSource filename="stars.txt" importVelocity="true" importVelocityDispersion="false" useColumns="" sourceWeight="1" wavelengthBias="0.5">
            <smoothingKernel type="SmoothingKernel">
              <CubicSplineSmoothingKernel/>
            </smoothingKernel>
            <sedFamily type="SEDFamily">
              <FSPSSEDFamily imf="Kroupa"/>
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
    THEMIS dust medium from gas.txt. dustFraction=0.4 via massFraction.
    maxTemperature=8000 K: the Camps & Trayford published convention
    (Enterprise fiducial — see module header for why this differs from
    Tal Shiar's 3e4 K, and dust_threshold_scan.* for the sensitivity).
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
    """Empty string — mediumSystem element must be ABSENT for NoMedium
    (SKIRT v9 errors on an empty <MediumSystem/> element)."""
    return ""


def probe_system(with_dust):
    if with_dust:
        return '''    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
          <ConvergenceInfoProbe probeName="spatial_convergence" wavelength="0.55 micron"/>
          <ConvergenceCutsProbe probeName="media_density_cuts"/>
        </probes>
      </ProbeSystem>
    </probeSystem>'''
    return '''    <probeSystem type="ProbeSystem">
      <ProbeSystem>
        <probes type="Probe">
        </probes>
      </ProbeSystem>
    </probeSystem>'''


# ---------------------------------------------------------------------------
# Full ski file template
# ---------------------------------------------------------------------------

def build_ski(run_name, with_dust, num_photons=NUM_PHOTONS_PRODUCTION):
    label = "dust" if with_dust else "no-dust"
    medium = medium_system_with_dust() if with_dust else medium_system_nodust()
    mode = "ExtinctionOnly" if with_dust else "NoMedium"
    probes = probe_system(with_dust)
    n_inc = len(inclinations_deg())

    return f'''<?xml version='1.0' encoding='UTF-8'?>
<!-- Enterprise SKIRT: {run_name}, {label}, {mode}, UV-NIR, {n_inc} inclination(s), Kroupa, Tdust<{MAX_DUST_TEMP_K}K -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="Enterprise pipeline">
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
# Write all runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent / "ski_enterprise"
    outdir.mkdir(exist_ok=True)
    runs = run_names()

    for run_name in runs:
        dust_ski = build_ski(run_name, with_dust=True)
        nodust_ski = build_ski(run_name, with_dust=False)

        dust_path = outdir / f"{run_name}_dust.ski"
        nodust_path = outdir / f"{run_name}_nodust.ski"
        dust_path.write_text(dust_ski)
        nodust_path.write_text(nodust_ski)

        print(f"[{run_name}] wrote {dust_path.name} ({len(dust_ski):,} B)  "
              f"+ {nodust_path.name} ({len(nodust_ski):,} B)")

    print()
    print(f"Runs:             {len(runs)}  ->  {2 * len(runs)} ski files")
    print(f"Photons (dust):   {NUM_PHOTONS_PRODUCTION}")
    print(f"Dust temp cut:    {MAX_DUST_TEMP_K} K (Camps & Trayford)")
    print(f"IMF:              Kroupa (FSPS) + MAPPINGS")
    print("Inclinations (deg):", ", ".join(f"{x:.1f}" for x in inclinations_deg()))