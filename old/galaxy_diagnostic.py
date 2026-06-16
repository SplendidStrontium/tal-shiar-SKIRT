"""
galaxy_diagnostic.py — Tal Shiar SKIRT Pipeline, Disk-ness Diagnostic

For each Romulus galaxy (pre-reoriented via make_particles.py's logic, or
loaded fresh here), compute disk-ness metrics and produce face-on / edge-on
projection images so the user can judge which galaxy has the cleanest disk
structure for the orientation study.

For each galaxy, the script:
    1. Loads the snapshot with pynbody
    2. Centers on stellar center-of-mass
    3. Picks a disk tracer (cold gas -> young stars -> all stars fallback)
    4. Applies pynbody.analysis.faceon so L_hat -> +z
    5. Computes disk-ness metrics for two stellar populations:
         - All stars within R_vir
         - Stars within 0.1 R_vir (disk region only)
       Metrics reported per population:
         - kappa_rot (Sales+2012): fraction of KE in ordered co-rotation
         - v/sigma: mean v_phi over 3D velocity dispersion
         - stellar mass, gas mass (total + cold), SFR_100Myr
    6. Saves projection images:
         - face-on and edge-on stars (surface density)
         - face-on and edge-on gas (log column density, highlights dust layout)
    7. Writes per-galaxy multi-panel PDF with images + metrics
    8. Appends a row to a shared CSV for side-by-side comparison
    9. Prints a ranked summary table at the end

Usage:
    python galaxy_diagnostic.py \\
        --snapshot /path/to/gal1.tipsy --name gal1 \\
        --snapshot /path/to/gal2.tipsy --name gal2 \\
        --snapshot /path/to/gal3.tipsy --name gal3 \\
        --output diagnostic_results/

The --snapshot and --name flags can be repeated for multiple galaxies.
"""

import os
import csv
import argparse
import warnings
from timeit import default_timer as timer

import numpy as np
import pynbody
import pynbody.analysis

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Shared constants (kept identical to make_particles.py for consistency)
# ---------------------------------------------------------------------------
COLD_GAS_TEMP_K = 3e4
YOUNG_STAR_AGE_YR = 1e9
TRACER_RADIUS_KPC = 10.0
MIN_COLD_GAS_MASS = 1e8
MIN_YOUNG_STAR_MASS = 1e7

# Image parameters
IMAGE_WIDTH_KPC = 30.0   # projection width for all images
IMAGE_RESOLUTION = 500   # pixels per side

# SFR averaging window
SFR_WINDOW_YR = 1e8      # 100 Myr


# ===========================================================================
# Loading and orientation (mirrors make_particles.py)
# ===========================================================================

def load_and_orient(snapshot_path):
    """
    Load snapshot, center on stellar COM, and rotate face-on.
    Returns the oriented pynbody SimSnap and the tracer name used.
    """
    print(f"  Loading {snapshot_path}...")
    data = pynbody.load(snapshot_path)
    print(f"    Stars: {len(data.star)}, Gas: {len(data.gas)}")

    # Center on stellar center of mass
    pynbody.analysis.halo.center(data.s, mode='ssc')

    # Verify centering propagated to parent snapshot
    com_check = (data.s['mass'].reshape(-1,1) * data.s['pos']).sum(axis=0) / data.s['mass'].sum()
    print(f"    Post-centering stellar COM: [{com_check[0]:+.2f}, {com_check[1]:+.2f}, {com_check[2]:+.2f}] kpc")

    # Pick tracer
    gas_r_kpc = np.sqrt((data.gas['pos'].in_units('kpc') ** 2).sum(axis=1))
    star_r_kpc = np.sqrt((data.star['pos'].in_units('kpc') ** 2).sum(axis=1))

    gas_temp_k = data.gas['temp'].view(np.ndarray)
    cold_mask = (gas_temp_k < COLD_GAS_TEMP_K) & (gas_r_kpc < TRACER_RADIUS_KPC)
    cold_mass = float(data.gas['mass'].in_units('Msol')[cold_mask].sum())

    star_age_yr = data.star['age'].in_units('yr').view(np.ndarray)
    young_mask = (star_age_yr < YOUNG_STAR_AGE_YR) & (star_r_kpc < TRACER_RADIUS_KPC)
    young_mass = float(data.star['mass'].in_units('Msol')[young_mask].sum())

    if cold_mass > MIN_COLD_GAS_MASS:
        tracer = data.gas[cold_mask]
        tracer_name = 'cold_gas'
    elif young_mass > MIN_YOUNG_STAR_MASS:
        tracer = data.star[young_mask]
        tracer_name = 'young_stars'
    else:
        tracer = data.star[star_r_kpc < TRACER_RADIUS_KPC]
        tracer_name = 'all_stars_fallback'
        warnings.warn(
            f"  Galaxy has minimal cold gas ({cold_mass:.2e} Msol) and few young "
            f"stars ({young_mass:.2e} Msol). Using all inner stars as orientation "
            f"tracer. Disk assumption may not hold.",
            RuntimeWarning,
        )

    print(f"    Orientation tracer: {tracer_name}")
    pynbody.analysis.angmom.faceon(tracer)

    # Verify rotation propagated to parent snapshot (SubSnap caveat check)
    L = (tracer['mass'].reshape(-1,1) * np.cross(tracer['pos'], tracer['vel'])).sum(axis=0)
    L_hat = L / np.linalg.norm(L)
    print(f"    Post-rotation L̂ (tracer): [{L_hat[0]:+.3f}, {L_hat[1]:+.3f}, {L_hat[2]:+.3f}]")
    if abs(L_hat[2]) < 0.95:
        print(f"    WARNING: L̂_z = {L_hat[2]:.3f}, expected ~1.0 — rotation may not have propagated")
    print(f"    Snapshot reoriented (L_hat -> +z).")

    return data, tracer_name


# ===========================================================================
# Metrics
# ===========================================================================

def compute_rvir(data):
    """
    Quick virial-radius estimate: radius enclosing an overdensity of 200x
    the critical density. For a zoom-in with no halo catalogue, this is
    a reasonable proxy. We walk outward in spherical shells of all particles
    (DM + gas + stars) until the mean enclosed density drops below 200 * rho_crit.

    Returns R_vir in kpc.
    """
    # Cosmological critical density at z=0
    # rho_crit = 3 H0^2 / (8 pi G) ~ 1.4e-7 Msol / pc^3 for h=0.7
    # Convert to Msol/kpc^3: 1 kpc^3 = 1e9 pc^3 -> 1.4e2 Msol/kpc^3
    # But better to read h from the snapshot if possible
    # Romulus uses Planck-ish cosmology (dHubble0=2.894405 -> h~0.6777).
    # pynbody usually reads h from the .param file; this is just a fallback.
    try:
        h = float(data.properties.get('h', 0.6777))
    except Exception:
        h = 0.6777
    H0_kms_per_mpc = 100.0 * h
    G_kpc_km2_per_s2_per_msol = 4.30091e-6  # G in kpc (km/s)^2 / Msol
    # H0 in 1/s: (km/s/Mpc) / Mpc_in_km; easier to work in (km/s)^2 / kpc^2 units
    # rho_crit = 3 H0^2 / (8 pi G) in Msol / kpc^3
    H0_per_kpc = H0_kms_per_mpc / 1e3  # (km/s) / kpc
    rho_crit = 3 * H0_per_kpc**2 / (8 * np.pi * G_kpc_km2_per_s2_per_msol)  # Msol / kpc^3

    # Gather all particle positions + masses
    all_pos_kpc = np.vstack([
        np.asarray(data.dm['pos'].in_units('kpc')) if len(data.dm) > 0 else np.empty((0, 3)),
        np.asarray(data.gas['pos'].in_units('kpc')),
        np.asarray(data.star['pos'].in_units('kpc')),
    ])
    all_mass_msol = np.concatenate([
        np.asarray(data.dm['mass'].in_units('Msol')) if len(data.dm) > 0 else np.empty(0),
        np.asarray(data.gas['mass'].in_units('Msol')),
        np.asarray(data.star['mass'].in_units('Msol')),
    ])

    r_kpc = np.sqrt((all_pos_kpc ** 2).sum(axis=1))
    order = np.argsort(r_kpc)
    r_sorted = r_kpc[order]
    m_cum = np.cumsum(all_mass_msol[order])

    # Mean enclosed density at each radius
    # rho_enc(r) = M(<r) / (4/3 pi r^3)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_enc = m_cum / ((4.0 / 3.0) * np.pi * r_sorted**3)

    target = 200.0 * rho_crit
    # Find largest radius where rho_enc > target
    above = rho_enc > target
    if not above.any():
        # Fallback: use radius enclosing 90% of stellar mass * 10
        star_r = np.sqrt((np.asarray(data.star['pos'].in_units('kpc'))) ** 2).sum(axis=1) ** 0.5
        return float(np.percentile(star_r, 90) * 10.0)

    # Last True index
    idx = np.where(above)[0].max()
    rvir_kpc = float(r_sorted[idx])
    return rvir_kpc


def compute_disk_metrics(data, r_outer_kpc, label):
    """
    Compute kappa_rot and v/sigma for stars within r_outer_kpc.

    After faceon(), the disk lies in the x-y plane. For each star:
        - j_z = x * vy - y * vx                 (z-component of specific angular momentum)
        - R   = sqrt(x^2 + y^2)                 (cylindrical radius)
        - v_phi = j_z / R                       (tangential velocity)
        - K_rot contribution = 0.5 * m * v_phi^2  (only if co-rotating, i.e. j_z > 0)

    kappa_rot = K_rot / K_total
              = sum(0.5 m v_phi^2, co-rotating only) / sum(0.5 m |v|^2)

    v/sigma = <v_phi> / sigma_3D
        where sigma_3D^2 = var(vx) + var(vy) + var(vz)

    Returns dict of metrics.
    """
    pos_kpc = np.asarray(data.star['pos'].in_units('kpc'))
    vel_kms = np.asarray(data.star['vel'].in_units('km s**-1'))
    mass_msol = np.asarray(data.star['mass'].in_units('Msol'))

    r3d = np.sqrt((pos_kpc ** 2).sum(axis=1))
    in_region = r3d < r_outer_kpc

    if in_region.sum() < 10:
        return {
            f'{label}_N_stars':    int(in_region.sum()),
            f'{label}_Mstar_Msol': 0.0,
            f'{label}_kappa_rot':  np.nan,
            f'{label}_v_over_sigma': np.nan,
        }

    x, y, _ = pos_kpc[in_region].T
    vx, vy, vz = vel_kms[in_region].T
    m = mass_msol[in_region]

    R = np.sqrt(x**2 + y**2)
    # Guard against division by zero for stars near the z-axis
    R_safe = np.where(R > 1e-6, R, 1e-6)
    jz = x * vy - y * vx                 # kpc * km/s
    v_phi = jz / R_safe                  # km/s

    v2 = vx**2 + vy**2 + vz**2
    K_total = 0.5 * (m * v2).sum()

    # Co-rotating stars only (j_z > 0 since we aligned to +z)
    corot = jz > 0
    K_rot = 0.5 * (m[corot] * v_phi[corot] ** 2).sum()

    kappa_rot = float(K_rot / K_total) if K_total > 0 else np.nan

    # v/sigma: mean co-rotating v_phi over 3D dispersion
    mean_vphi = float(np.average(v_phi[corot], weights=m[corot])) if corot.any() else 0.0
    sigma_3d = float(np.sqrt(np.var(vx) + np.var(vy) + np.var(vz)))
    v_over_sigma = mean_vphi / sigma_3d if sigma_3d > 0 else np.nan

    return {
        f'{label}_N_stars':      int(in_region.sum()),
        f'{label}_Mstar_Msol':   float(m.sum()),
        f'{label}_kappa_rot':    kappa_rot,
        f'{label}_v_over_sigma': v_over_sigma,
    }


def compute_context_metrics(data, rvir_kpc):
    """
    Global context: total gas mass, cold gas mass, SFR averaged over 100 Myr.
    Masses are summed within R_vir.
    """
    gas_r = np.sqrt((np.asarray(data.gas['pos'].in_units('kpc'))) ** 2).sum(axis=1) ** 0.5
    star_r = np.sqrt((np.asarray(data.star['pos'].in_units('kpc'))) ** 2).sum(axis=1) ** 0.5

    in_gas = gas_r < rvir_kpc
    gas_temp = data.gas['temp'].view(np.ndarray)
    cold = (gas_temp < COLD_GAS_TEMP_K) & in_gas

    gas_mass_total = float(data.gas['mass'].in_units('Msol')[in_gas].sum())
    gas_mass_cold  = float(data.gas['mass'].in_units('Msol')[cold].sum())

    star_age_yr = data.star['age'].in_units('yr').view(np.ndarray)
    young = (star_age_yr < SFR_WINDOW_YR) & (star_r < rvir_kpc)
    m_young = float(data.star['mass'].in_units('Msol')[young].sum())
    sfr_msol_per_yr = m_young / SFR_WINDOW_YR

    return {
        'Mgas_total_Msol': gas_mass_total,
        'Mgas_cold_Msol':  gas_mass_cold,
        'SFR_100Myr_Msol_per_yr': sfr_msol_per_yr,
    }


# ===========================================================================
# Projections
# ===========================================================================

def _project_surface_density(pos_kpc, mass_msol, axis_pair, width_kpc, res):
    """
    Simple histogram-based surface density projection.

    axis_pair = (0, 1) -> project onto x-y (face-on after faceon())
    axis_pair = (0, 2) -> project onto x-z (edge-on)
    """
    a, b = axis_pair
    x = pos_kpc[:, a]
    y = pos_kpc[:, b]
    half = width_kpc / 2.0
    mask = (np.abs(x) < half) & (np.abs(y) < half)

    H, xedges, yedges = np.histogram2d(
        x[mask], y[mask],
        bins=res,
        range=[[-half, half], [-half, half]],
        weights=mass_msol[mask],
    )
    # Convert to surface density: Msol / kpc^2
    pixel_area_kpc2 = (width_kpc / res) ** 2
    sigma = H / pixel_area_kpc2
    return sigma.T  # transpose so imshow's y-axis matches our y-axis


def make_projection_images(data, out_prefix, width_kpc=IMAGE_WIDTH_KPC, res=IMAGE_RESOLUTION):
    """
    Produce 4 PNGs:
      {prefix}_faceon_stars.png
      {prefix}_edgeon_stars.png
      {prefix}_faceon_gas.png
      {prefix}_edgeon_gas.png

    Returns a dict mapping image kind -> 2D array (for reuse in the PDF).
    """
    star_pos = np.asarray(data.star['pos'].in_units('kpc'))
    star_mass = np.asarray(data.star['mass'].in_units('Msol'))
    gas_pos = np.asarray(data.gas['pos'].in_units('kpc'))
    gas_mass = np.asarray(data.gas['mass'].in_units('Msol'))

    images = {}
    specs = [
        ('faceon_stars', star_pos, star_mass, (0, 1), 'Stars (face-on)', 'inferno'),
        ('edgeon_stars', star_pos, star_mass, (0, 2), 'Stars (edge-on)', 'inferno'),
        ('faceon_gas',   gas_pos,  gas_mass,  (0, 1), 'Gas (face-on)',   'viridis'),
        ('edgeon_gas',   gas_pos,  gas_mass,  (0, 2), 'Gas (edge-on)',   'viridis'),
    ]

    for key, pos, mass, axes, title, cmap in specs:
        sigma = _project_surface_density(pos, mass, axes, width_kpc, res)
        images[key] = sigma

        fig, ax = plt.subplots(figsize=(6, 6))
        # Log-scale, avoid log(0)
        with np.errstate(divide='ignore'):
            log_sigma = np.log10(np.where(sigma > 0, sigma, np.nan))
        im = ax.imshow(
            log_sigma,
            origin='lower',
            extent=[-width_kpc/2, width_kpc/2, -width_kpc/2, width_kpc/2],
            cmap=cmap,
            aspect='equal',
        )
        ax.set_title(title)
        ax.set_xlabel('x [kpc]' if axes[0] == 0 else 'y [kpc]')
        ylabel = 'y [kpc]' if axes[1] == 1 else 'z [kpc]'
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$\log_{10}\ \Sigma\ [\mathrm{M}_\odot/\mathrm{kpc}^2]$')
        fig.tight_layout()
        fig.savefig(f'{out_prefix}_{key}.png', dpi=150)
        plt.close(fig)

    return images


def make_pdf_report(name, images, metrics, tracer_name, pdf_path,
                    width_kpc=IMAGE_WIDTH_KPC):
    """
    Single-page PDF with all four projections and a metrics text panel.
    """
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(12, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])

        specs = [
            ('faceon_stars', 'Stars (face-on)', 'inferno', (0, 0)),
            ('edgeon_stars', 'Stars (edge-on)', 'inferno', (0, 1)),
            ('faceon_gas',   'Gas (face-on)',   'viridis', (1, 0)),
            ('edgeon_gas',   'Gas (edge-on)',   'viridis', (1, 1)),
        ]
        for key, title, cmap, (row, col) in specs:
            ax = fig.add_subplot(gs[row, col])
            sigma = images[key]
            with np.errstate(divide='ignore'):
                log_sigma = np.log10(np.where(sigma > 0, sigma, np.nan))
            im = ax.imshow(
                log_sigma, origin='lower',
                extent=[-width_kpc/2, width_kpc/2, -width_kpc/2, width_kpc/2],
                cmap=cmap, aspect='equal',
            )
            ax.set_title(title)
            ax.set_xlabel('x [kpc]')
            ax.set_ylabel('y [kpc]' if 'faceon' in key else 'z [kpc]')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Metrics panel
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')
        lines = [
            f"Galaxy: {name}",
            f"Orientation tracer: {tracer_name}",
            f"R_vir (kpc): {metrics['Rvir_kpc']:.2f}",
            "",
            "Disk-ness metrics:",
            f"  All stars within R_vir:",
            f"    N = {metrics['allstars_N_stars']:,}, "
            f"M* = {metrics['allstars_Mstar_Msol']:.2e} Msol",
            f"    kappa_rot = {metrics['allstars_kappa_rot']:.3f}, "
            f"v/sigma = {metrics['allstars_v_over_sigma']:.3f}",
            f"  Stars within 0.1 R_vir:",
            f"    N = {metrics['inner_N_stars']:,}, "
            f"M* = {metrics['inner_Mstar_Msol']:.2e} Msol",
            f"    kappa_rot = {metrics['inner_kappa_rot']:.3f}, "
            f"v/sigma = {metrics['inner_v_over_sigma']:.3f}",
            "",
            "Context:",
            f"  Gas mass (total, within R_vir): {metrics['Mgas_total_Msol']:.2e} Msol",
            f"  Gas mass (cold, within R_vir):  {metrics['Mgas_cold_Msol']:.2e} Msol",
            f"  SFR (100 Myr avg):              {metrics['SFR_100Myr_Msol_per_yr']:.3f} Msol/yr",
            "",
            "Interpretation cues:",
            "  kappa_rot > 0.5  -> disk-dominated",
            "  kappa_rot ~ 0.3-0.5 -> mixed / weak disk",
            "  kappa_rot < 0.3  -> spheroidal or disturbed",
        ]
        ax_text.text(
            0.02, 0.98, '\n'.join(lines),
            transform=ax_text.transAxes,
            ha='left', va='top', family='monospace', fontsize=10,
        )

        fig.suptitle(f"Diagnostic: {name}", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        pdf.savefig(fig)
        plt.close(fig)


# ===========================================================================
# Per-galaxy driver
# ===========================================================================

def analyze_galaxy(snapshot_path, name, out_dir):
    """
    Full pipeline for one galaxy. Returns a dict of metrics suitable for
    writing to the shared CSV.
    """
    print(f"\n=== {name} ===")
    t0 = timer()

    data, tracer_name = load_and_orient(snapshot_path)

    print("  Estimating R_vir...")
    rvir_kpc = compute_rvir(data)
    print(f"    R_vir ~ {rvir_kpc:.2f} kpc")

    print("  Computing disk-ness metrics (both populations)...")
    metrics_all   = compute_disk_metrics(data, rvir_kpc,       label='allstars')
    metrics_inner = compute_disk_metrics(data, 0.1 * rvir_kpc, label='inner')
    ctx = compute_context_metrics(data, rvir_kpc)

    metrics = {
        'name': name,
        'snapshot': snapshot_path,
        'tracer': tracer_name,
        'Rvir_kpc': rvir_kpc,
        **metrics_all,
        **metrics_inner,
        **ctx,
    }

    print("  Generating projection images...")
    prefix = os.path.join(out_dir, name)
    images = make_projection_images(data, prefix)

    print("  Writing PDF report...")
    make_pdf_report(name, images, metrics, tracer_name,
                    pdf_path=f'{prefix}_diagnostic.pdf')

    dt = timer() - t0
    print(f"  Done ({dt:.1f}s): kappa_rot(inner)={metrics['inner_kappa_rot']:.3f}, "
          f"v/sigma(inner)={metrics['inner_v_over_sigma']:.3f}")
    return metrics


# ===========================================================================
# CSV + ranking
# ===========================================================================

CSV_COLUMNS = [
    'name', 'snapshot', 'tracer', 'Rvir_kpc',
    'allstars_N_stars', 'allstars_Mstar_Msol',
    'allstars_kappa_rot', 'allstars_v_over_sigma',
    'inner_N_stars', 'inner_Mstar_Msol',
    'inner_kappa_rot', 'inner_v_over_sigma',
    'Mgas_total_Msol', 'Mgas_cold_Msol', 'SFR_100Myr_Msol_per_yr',
]


def write_csv(all_metrics, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, '') for k in CSV_COLUMNS})


def print_ranking(all_metrics):
    """Rank by inner kappa_rot and print a neat table."""
    ordered = sorted(all_metrics, key=lambda m: -m['inner_kappa_rot']
                     if not np.isnan(m['inner_kappa_rot']) else 1)

    print("\n" + "=" * 78)
    print("RANKED SUMMARY (by inner kappa_rot, disk-like first)")
    print("=" * 78)
    header = f"{'rank':<5}{'name':<16}{'tracer':<18}{'k_rot_in':>10}{'v/s_in':>10}{'SFR':>12}"
    print(header)
    print("-" * len(header))
    for i, m in enumerate(ordered, start=1):
        print(
            f"{i:<5}{m['name']:<16}{m['tracer']:<18}"
            f"{m['inner_kappa_rot']:>10.3f}"
            f"{m['inner_v_over_sigma']:>10.3f}"
            f"{m['SFR_100Myr_Msol_per_yr']:>12.3f}"
        )
    print("=" * 78)
    print("Reminder: k_rot > 0.5 disk-dominated, < 0.3 spheroidal/disturbed.")
    print("Cross-check against the face-on/edge-on images -- a galaxy can have")
    print("decent k_rot and still look messy. Your eyes are part of the diagnostic.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Disk-ness diagnostic for Romulus galaxies",
    )
    parser.add_argument('--snapshot', action='append', required=True,
                        help='Path to .tipsy snapshot (repeat for multiple galaxies)')
    parser.add_argument('--name', action='append', required=True,
                        help='Short label for each galaxy (repeat; order must match --snapshot)')
    parser.add_argument('--output', required=True,
                        help='Output directory for PNGs, PDFs, and summary CSV')
    args = parser.parse_args()

    if len(args.snapshot) != len(args.name):
        parser.error("Number of --snapshot and --name arguments must match.")

    os.makedirs(args.output, exist_ok=True)

    start_all = timer()
    all_metrics = []
    for snap, nm in zip(args.snapshot, args.name):
        try:
            m = analyze_galaxy(snap, nm, args.output)
            all_metrics.append(m)
        except Exception as e:
            print(f"\n!! FAILED for {nm}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    csv_path = os.path.join(args.output, 'diagnostic_summary.csv')
    write_csv(all_metrics, csv_path)
    print(f"\nSummary CSV: {csv_path}")

    if all_metrics:
        print_ranking(all_metrics)

    print(f"\nTotal time: {timer() - start_all:.1f}s")
