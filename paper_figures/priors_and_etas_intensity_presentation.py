#%%
# =============================================================================
# priors_and_etas_intensity_presentation.py — presentation figures
# =============================================================================
# Two independent figures, both drawn with the same 'seismic' colormap used in
# prior_comparison.py:
#
#   1. ETAS conditional intensity immediately before vs. after a new
#      earthquake — the "bullseye" of triggered aftershock density that
#      appears at the epicenter the instant the event occurs. Synthetic toy
#      catalog, no data dependencies.
#      (from plot_scripts/etas_intensity_illustration.py)
#
#   2. Strip of the spatial priors on a common California grid, log10 scale,
#      one colorbar per prior. The last panel is a Uniform prior (constant
#      everywhere) to show that it carries no spatial information.
#      (from priors/scripts/compare_priors.py, "Figure 2")
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import SeismicPrior

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CMAP       = 'seismic'

#%%
# ── Figure 1: ETAS conditional intensity, before vs. after ──────────────────
rng = np.random.default_rng(7)

DOMAIN_KM = 40      # map extent, +/- km from center
GRID_N    = 300     # grid resolution per axis
N_HISTORY = 22      # number of historical background events
MU0       = 0.02    # constant background rate (events / km^2 / day)

# Omori-Utsu time decay: g(dt) = 1 / (dt + c)^p
C_OMORI, P_OMORI = 0.05, 1.1

# ETAS spatial kernel: f(r) = (q-1) / (pi * d^2) * (1 + (r/d)^2)^-q
# d grows with magnitude: d = D0 * 10^(ALPHA_D * (M - MC))
Q_KERNEL, D0, ALPHA_D = 1.6, 0.5, 0.35
MC     = 2.5        # completeness magnitude
K_PROD = 0.05       # productivity scale

MAINSHOCK_MAG = 5.5
MAINSHOCK_LOC = (0.0, 0.0)

# Synthetic catalog: scattered historical events over the past 30 days, plus
# one new mainshock at t = 0.
hist_x = rng.uniform(-DOMAIN_KM * 0.8, DOMAIN_KM * 0.8, N_HISTORY)
hist_y = rng.uniform(-DOMAIN_KM * 0.8, DOMAIN_KM * 0.8, N_HISTORY)
hist_m = rng.uniform(MC, 4.3, N_HISTORY)
hist_t = -rng.uniform(0.5, 30, N_HISTORY)   # days before the mainshock


def omori(dt):
    return 1.0 / (dt + C_OMORI) ** P_OMORI


def spatial_kernel(r, mag):
    d = D0 * 10 ** (ALPHA_D * (mag - MC))
    return (Q_KERNEL - 1) / (np.pi * d ** 2) * (1 + (r / d) ** 2) ** (-Q_KERNEL)


def conditional_intensity(xx, yy, t, events_x, events_y, events_m, events_t):
    lam = np.full_like(xx, MU0)
    for ex, ey, em, et in zip(events_x, events_y, events_m, events_t):
        if et >= t:
            continue
        r = np.hypot(xx - ex, yy - ey)
        lam += K_PROD * 10 ** (em - MC) * spatial_kernel(r, em) * omori(t - et)
    return lam


grid = np.linspace(-DOMAIN_KM, DOMAIN_KM, GRID_N)
xx, yy = np.meshgrid(grid, grid)

t_before, t_after = -1e-3, 1e-3   # an instant before / after the mainshock
lam_before = conditional_intensity(xx, yy, t_before, hist_x, hist_y, hist_m, hist_t)
lam_after = conditional_intensity(
    xx, yy, t_after,
    np.append(hist_x, MAINSHOCK_LOC[0]), np.append(hist_y, MAINSHOCK_LOC[1]),
    np.append(hist_m, MAINSHOCK_MAG),    np.append(hist_t, 0.0),
)

fig1, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True,
                          sharex=True, sharey=True)

norm = LogNorm(vmin=MU0 / 2, vmax=lam_after.max())

axes[0].pcolormesh(xx, yy, lam_before, norm=norm, cmap=CMAP, shading='auto')
axes[0].set_title('Before: background seismicity only', fontsize=12)

im1 = axes[1].pcolormesh(xx, yy, lam_after, norm=norm, cmap=CMAP, shading='auto')
axes[1].set_title('After: new earthquake ignites the aftershock kernel', fontsize=12)

for ax in axes:
    ax.scatter(hist_x, hist_y, s=15 + 8 * (hist_m - MC), facecolor='none',
               edgecolor='white', linewidth=0.8, label='Historical events')
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (km)', fontsize=11)
axes[0].set_ylabel('Northing (km)', fontsize=11)

# Yellow star — red would vanish into the dark-red peak of the 'seismic' map.
axes[0].scatter(*MAINSHOCK_LOC, marker='*', s=220, facecolor='none',
                edgecolor='yellow', linewidth=1.5, linestyle='--',
                label='Upcoming mainshock')
axes[1].scatter(*MAINSHOCK_LOC, marker='*', s=220, color='yellow',
                edgecolor='black', linewidth=0.8,
                label=f'M{MAINSHOCK_MAG} mainshock')
axes[0].legend(loc='upper left', fontsize=8, framealpha=0.85)
axes[1].legend(loc='upper left', fontsize=8, framealpha=0.85)

cbar = fig1.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label(r'Conditional intensity $\lambda(x,y,t)$  [events / km$^2$ / day]', fontsize=11)
fig1.suptitle('ETAS conditional intensity: before vs. after a new earthquake', fontsize=14)

out1 = os.path.join(OUTPUT_DIR, 'presentation_etas_intensity.png')
fig1.savefig(out1, dpi=200, bbox_inches='tight')
print(f'Saved: {out1}')
plt.show()

#%%
# ── Figure 2: strip of spatial priors on a common grid ──────────────────────
PRIOR_FILES = [
    ('GEAR1',        'GEAR1_prior.tt3'),
    ('NSHM',         'USGS_NSHM_prior.tt3'),
    ('Helmstetter',  'helmstetter_prior.tt3'),
    ('KDE_Seismicity', 'prior_seis_grid_US_Canada_filled.tt3'),
]
# Zoomed to California: clips off the empty strip at NSHM's western edge and
# the outer-edge artifacts of the other grids. [lon0, lon1, lat0, lat1]
EXTENT = [-126.5, -113.5, 31, 43]

# Percentile range of each panel's log10 values used for its colorbar
# (default is the full min/max). KDE's tails reach ~-100, which would wash out
# everything else.
CLIM_PCT = {'KDE_Seismicity': (10, 99.5)}

priors = {}
for label, fname in PRIOR_FILES:
    path = os.path.join(SeismicPrior.data_dir, fname)
    if os.path.exists(path):
        priors[label] = SeismicPrior.from_tt3(path)
    else:
        print(f'Warning: {fname} not found — skipping {label}.')

# Common grid at the finest native resolution among the priors.
dx = min(float(np.diff(p.lons).mean()) for p in priors.values())
dy = min(float(np.diff(p.lats).mean()) for p in priors.values())
lons_c = np.arange(EXTENT[0], EXTENT[1], dx)
lats_c = np.arange(EXTENT[2], EXTENT[3], dy)
LAT_C, LON_C = np.meshgrid(lats_c, lons_c, indexing='ij')
pts = np.stack([LAT_C.ravel(), LON_C.ravel()], axis=1)

# Interpolate each prior onto the common grid; cells outside its native extent
# get the mean in-bounds value (equal, low weight); renormalize to sum to 1.
log_grids = {}
for name, p in priors.items():
    interp = RegularGridInterpolator((p.lats, p.lons), p.grid,
                                     method='linear', bounds_error=False, fill_value=np.nan)
    g = interp(pts).reshape(LAT_C.shape)
    in_bounds = ((LAT_C >= p.lats[0]) & (LAT_C <= p.lats[-1]) &
                 (LON_C >= p.lons[0]) & (LON_C <= p.lons[-1]))
    g[~in_bounds] = np.nanmean(g[in_bounds])
    g /= np.nansum(g)
    log_grids[name] = np.where(g > 0, np.log10(g), np.nan)

# Uniform prior: constant on the same grid (each cell 1/N), so one flat color.
log_grids['Uniform'] = np.full(LAT_C.shape, np.log10(1 / LAT_C.size))

proj = ccrs.PlateCarree()
n = len(log_grids)
fig2, axes2 = plt.subplots(1, n, figsize=(3.6 * n, 4.0), subplot_kw={'projection': proj})

for ax, (name, g_log) in zip(axes2, log_grids.items()):
    ax.set_extent(EXTENT, crs=proj)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=9)
    if name == 'Uniform':
        # Constant field: give the colorbar a +/-1 decade range so the panel
        # shows one flat mid-scale color instead of an undefined range.
        vmin, vmax = g_log[0, 0] - 1, g_log[0, 0] + 1
    else:
        vmin, vmax = np.nanpercentile(g_log, CLIM_PCT.get(name, (0, 100)))
    im = ax.imshow(g_log, origin='lower', extent=EXTENT, transform=proj,
                   aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax)
    cb = fig2.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.06)
    cb.set_label('log₁₀(p)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

fig2.suptitle('Prior models — log₁₀ scale', fontsize=11, y=1.01)

out2 = os.path.join(OUTPUT_DIR, 'presentation_prior_strip.png')
fig2.savefig(out2, dpi=200, bbox_inches='tight')
print(f'Saved: {out2}')
plt.show()
