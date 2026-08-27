# GOTCHAS — read this before touching the pipeline

Hard-won lessons from the Tal Shiar / Enterprise work. Each entry: the trap,
why it exists, the rule. If you only read one doc, read this one.

## Snapshots & simulation data

**Never derive filenames from `achOutName`.** The `achOutName` field in the
ChaNGa `.param` files contains stale pre-rename working names ("BHT", "BH2")
from before the runs were renamed. The on-disk names are the truth: directory
basename matches the variant (e.g. `r488_BH6` → `...HsbBH6.004096`).
**Rule: glob for `*.004096`; never parse `achOutName`.** Both
`run_make_particles_all.py` and the census scripts already do this.

**BH particles live in the star family with `tform < 0`.** ChaNGa/GASOLINE
stores BH sink particles as "stars" with negative formation time. Filter with
`tform < 0` **before** any stellar mass, SFR, age, or orientation
calculation. Two separate failure modes if you don't:
(1) SKIRT treats a 10^5–10^6 Msol sink as a stellar population and emits
photons from it; (2) one massive sink in the angular-momentum tracer can
drag the computed disk axis off-true, silently tilting the whole coordinate
frame so "face-on" isn't.

**BH6/BH8 use binary auxiliary arrays; noBH/BH use ASCII.** The two feedback
variants were run with `iBinaryOutput=1`. Reader code that assumes ASCII aux
files will fail (or worse, misread) on half the sample.

**The `dMaxGasMass` typo: do not "fix" it.** The noBH and BH param files
contain `"1.66431-11"` — missing the `e`. How ChaNGa parsed this is unknown.
**Rule: never edit the param files; ask Jillian how ChaNGa handled it** if
the value ever matters for analysis. ⚠ Status of that question: unresolved
as of Aug 2026.

**Halo 1 is the target dwarf in all 20 runs.** ⚠ Verified as the most
massive halo in each amiga catalog during the summer (confirm method with
Jillian). Sanity backstop: a wrong halo would show up as a wildly
discrepant `mstar` within a family in `products/diagnostics_all.csv`.

**pynbody can hang loading the halo catalogue's `.grp` array.** The ASCII
group-membership file is enormous. If a script hangs on `s.halos()`-adjacent
operations, the workaround used in the census scripts is to read halo
centers directly instead of loading full membership.

## Two different "cold gas" temperatures — they are not a contradiction

- **8,000 K** = the SKIRT dust threshold (`maxTemperature` in the ski
  files). Physics choice; Camps & Trayford published convention. Gas hotter
  than this hosts no dust in the model.
- **30,000 K** = the orientation-*tracer* selection cut in
  `make_particles.py`. Geometry choice; the tracer only needs enough
  disk-following gas to define a plane. Cutting the tracer at 8,000 K would
  starve it in the feedback-heated BH6/BH8 runs and trip the all-stars
  fallback for no benefit.

History: Tal Shiar used 30,000 K as the *dust* threshold (project-specific
choice). Enterprise moved to 8,000 K. Sensitivity is characterized in
`dust_threshold_scan.{csv,png}` (median 32% of aperture dust mass lies
between the two conventions, systematically different across BH variants).
If comparing against Tal Shiar numbers, treat the threshold as a bracketed
systematic.

## SKIRT

**SKIRT overwrites outputs by filename — convergence files are evidence,
archive them.** During the maxLevel resolution ladder (July 17), re-running
a test at a new maxLevel silently overwrote the previous level's
`*_spatial_convergence_convergence.dat`, nearly destroying the before/after
that justified the grid choice. The numbers that matter: at the lower
level, Z-axis optical depth gridded at **−77.2%** error; at maxLevel 11,
**+5.3%**, with dust mass fine at both levels and peak memory a trivial
569 MB. That pair is the citable justification for `maxLevel=11`.
`run_skirt_production.py` now copies each dust run's convergence file to a
timestamped name immediately after the run (`archive_convergence()`), so
this cannot recur — but the lesson generalizes: any SKIRT output you might
need as evidence, copy before re-running.

**Regenerating ski files does not update the ones the pipeline uses.**
`generate_ski_enterprise.py` lives in `python/` and writes to
`python/ski_enterprise/`, but the drivers read from `src/ski_enterprise/`.
If you regenerate and re-run production without copying, SKIRT silently
uses the OLD skis — no error anywhere. ⚠ TODO: repoint the generator's
output (see README TODO). Until then: regenerate → **copy to
`src/ski_enterprise/`** → verify with `grep numPackets` or similar.

**A NoMedium ski must OMIT the `<mediumSystem>` element entirely.** SKIRT
v9 errors on an empty `<MediumSystem/>`. The generator handles this
(`medium_system_nodust()` returns an empty string); don't "clean it up."

**Emulation (`skirt -e`) proves a run can start, not that its output means
anything.** It catches schema errors, unknown property names, and missing
particle files in seconds. It cannot catch under-resolved grids, photon
noise, or physically wrong parameters. That's what the convergence probe
and the test driver's timed real run are for.

**nodust runs are NOT nearly free.** NoMedium skips transport but still
launches every packet: measured 35.6 s at 1e6 photons → ~30 min at 5e7.
The test driver's cost projection assumes nodust is flat and therefore
underestimates. We knowingly kept 5e7 for nodust anyway, for consistency.

**gas.txt is 7 columns, no velocities.** The ski declares
`importVelocity="false"`, so SKIRT expects exactly x y z h M Z T. Any
column-count change must be made in the ski and the writer together.

## Photometry

**Published B−K comes from the SEDInstrument, never the FullInstrument.**
Each inclination has two instruments: FullInstrument (spatial datacube on a
~10-point broadband grid → RGB images) and SEDInstrument (spatially
integrated SED on the fine 250-point log grid). Filter convolution against
Johnson B / 2MASS Ks response curves is only valid on the fine grid;
`extract_enterprise_colors.py --all` hard-filters to `sed_` instruments for
exactly this reason. Note `--one` verbose mode shows Full-instrument rows
too — those numbers will disagree with the science values; that's expected.

**Everything is Vega.** Johnson B and 2MASS Ks are both conventionally
Vega-system; the pipeline uses Vega zero-points throughout. Don't mix in AB.

**Colors are exact; magnitudes are not distance-calibrated.** B−K is a flux
ratio so normalization cancels; the per-band mB/mK columns in
`enterprise_colors.csv` are not calibrated to the 100 Mpc instrument
distance. Use the colors, not the raw magnitudes.

## Figures & analysis

**RGB composites must share a fixed reference normalization.** Per-image
percentile normalization erases exactly the cross-variant color differences
the study measures. `make_rgb.py` uses a `--ref` anchor; keep it.

**Drop partial SFH bins.** Star-formation-history bins where
`t_right > t_now` produce a trailing cliff artifact. The SFH extraction
already drops them; if you rebin, do the same.

**The BH ↔ BH6 comparison is confounded — don't draw it.** BH and BH6
differ in `dBHSinkColdDen` (100 vs 0.2) AND `dEtaDiffusion`, so it is not a
single-variable pair. Clean axes: noBH↔BH (presence) and BH6↔BH8
(feedback efficiency, `dBHSinkFeedbackEff` 0.05 vs 0.005). Figures split
by clean axis on purpose; never imply cross-pair trajectories.

**Verify extraction-source consistency via the summary CSVs.** Mixed
`massform` sources bit us once; the extract scripts have cross-checks and a
`--force-mass` patch flag. ⚠ If that flag's details are fuzzy, see
`extract_enterprise_sfh.py`'s docstring.

## last updated:
2026-08-27 // init commit