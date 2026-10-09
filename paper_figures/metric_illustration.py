#%%
# =============================================================================
# metric_illustration.py — conceptual figure: posterior confidence & coverage
# =============================================================================
# Two panels, same posterior (2024 Mendocino Triple Junction M7 mainshock,
# KDE_Seismicity prior, fixed trigger count), illustrating the two metrics:
#
#   (left)  posterior confidence — total posterior mass in cells whose density
#           is >= the density at the USGS location (the smallest HDR that just
#           touches USGS). Filled red = that region; gray contours = the rest
#           of the posterior; the blue line is the HDR boundary (the density
#           level at USGS).  Lower is better.
#   (right) posterior coverage — posterior mass within radius R of the USGS
#           location, with R = distance from USGS to the bEPIC MAP location.
#           Filled blue disc = that mass.  Higher is better.
#
# Both numbers in the panel titles come from benchmark.metrics
# (posterior_confidence_level, posterior_coverage), so the figure shows the
# same quantity the benchmark computes.
#
# Re-runs bEPIC for this one event via run_single_event_get_grid(), as in
# mtj_bayesian_location_example.py.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from priors import SeismicPrior
from benchmark import config
from benchmark.metrics import (posterior_confidence_level, posterior_coverage,
                               _haversine_km)
from benchmark.runner import run_single_event_get_grid

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MTJ_CASE   = 'MTJ_2024_M7'
EVENT_ID   = config.FOCUS_EVENTS_MAINSHOCK[MTJ_CASE]
N_TRIGGERS = 4
SIGMA_S    = 0.22
USGS_LAT, USGS_LON = 40.3740, -125.0217   # nc75095651

RUN_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', MTJ_CASE,
                        'run_files', f'{EVENT_ID}.run')
KDE_PATH = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{MTJ_CASE}.tt3')

# Palette shared with the other paper figures
C_CONF  = 'red'        # confidence region (posterior mass >= density at USGS)
C_COV   = 'darkblue'   # coverage disc
C_GRAY  = 'gray'

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

proj = ccrs.PlateCarree()


def _add_basemap(ax):
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightcyan', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

#%%
# ── Run bEPIC for this event/version ─────────────────────────────────────────
run_df = pd.read_csv(RUN_PATH)
run_df.columns = [c.replace(' ', '_') for c in run_df.columns]

focus_version = None
for v in sorted(run_df['version'].unique()):
    if len(run_df[run_df['version'] == v]) >= N_TRIGGERS:
        focus_version = v
        break

triggered = (run_df[run_df['version'] == (focus_version if focus_version is not None
                                          else run_df['version'].max())]
             .drop_duplicates(subset='station'))

params_kw = dict(config.BENCHMARK_PARAMS)
params_kw['sigma_s'] = SIGMA_S
_, odf, actual_v = run_single_event_get_grid(
    RUN_PATH, SeismicPrior.from_tt3(KDE_PATH), True, params_kw, focus_version=focus_version)

n = 2 * params_kw['grid_size'] + 1
lats_2d = odf['lat'].values.reshape(n, n)
lons_2d = odf['lon'].values.reshape(n, n)
post_2d = odf['post'].values.reshape(n, n)
post_2d = post_2d / post_2d.sum()

#%%
# ── Metrics ─────────────────────────────────────────────────────────────────
conf = posterior_confidence_level(odf, USGS_LAT, USGS_LON)

# Density at the grid cell nearest USGS (same cell the metric uses)
nearest = np.argmin(np.hypot(odf['lat'].values - USGS_LAT,
                             (odf['lon'].values - USGS_LON) * np.cos(np.radians(USGS_LAT))))
p_usgs = post_2d.ravel()[nearest]

# bEPIC MAP location and the USGS-to-MAP distance, used as the coverage radius
imap = np.argmax(post_2d.ravel())
MAP_LAT, MAP_LON = odf['lat'].values[imap], odf['lon'].values[imap]
R_KM = float(_haversine_km(USGS_LAT, USGS_LON, np.array([MAP_LAT]), np.array([MAP_LON]))[0])
cov = posterior_coverage(odf, USGS_LAT, USGS_LON, radii_km=R_KM)
print(f'confidence level = {conf:.3f}, MAP error = {R_KM:.1f} km, coverage = {cov:.3f}')

# Coverage disc as a geodesic circle (km -> degrees, cosine-corrected)
theta = np.linspace(0, 2 * np.pi, 361)
circ_lat = USGS_LAT + (R_KM / 111.19) * np.sin(theta)
circ_lon = USGS_LON + (R_KM / (111.19 * np.cos(np.radians(USGS_LAT)))) * np.cos(theta)

#%%
# ── Figure ──────────────────────────────────────────────────────────────────
# Zoom to where the posterior has mass (>0.5% of peak) plus the stations
mask = post_2d > 0.005 * post_2d.max()
pad = 0.25
lat_pad = 0.2
extent = [min(lons_2d[mask].min(), triggered['longitude'].min(), circ_lon.min()) - pad,
          max(lons_2d[mask].max(), triggered['longitude'].max(), circ_lon.max()) + pad,
          min(lats_2d[mask].min()-lat_pad, triggered['latitude'].min()-lat_pad, circ_lat.min()) - pad-lat_pad,
          max(lats_2d[mask].max()+lat_pad, triggered['latitude'].max()+lat_pad, circ_lat.max()) + pad+lat_pad]
# Figure height chosen so each map axes (0.44 of 12 in wide) fills its box
map_aspect = ((extent[1] - extent[0]) * np.cos(np.radians(USGS_LAT))) / (extent[3] - extent[2])
FIG_H = 0.44 * 12 / map_aspect / 0.78 + 1.0
gmax = post_2d.max()
rel = post_2d / gmax
# Evenly spaced gray reference contours, all relative to the posterior peak
level = p_usgs / gmax   # density at USGS, relative to the peak
gray_levels = np.linspace(0.1, 1.0, 10)
gray_levels = gray_levels[gray_levels > level]
# Low-density contours: fractions of the USGS density, to show mass below it
low_levels = level * np.array([0.1, 0.3, 0.6])

AX_POS = {
    'conf': [0.04, 0.12, 0.44, 0.78],
    'cov':  [0.53, 0.12, 0.44, 0.78],
}

fig = plt.figure(figsize=(12, 5))
axes = {}
for key in ('conf', 'cov'):
    ax = fig.add_axes(AX_POS[key], projection=proj)
    ax.set_extent(extent, crs=proj)
    _add_basemap(ax)
    # Posterior contours (gray) common to both panels
    ax.contour(lons_2d, lats_2d, rel, levels=gray_levels, colors=C_GRAY,
               linewidths=1.5, transform=proj, zorder=5.5)
    ax.contour(lons_2d, lats_2d, rel, levels=low_levels, colors=C_GRAY,
               linewidths=1., linestyles='dashed', alpha=0.8, transform=proj, zorder=5.5)
    ax.scatter(triggered['longitude'], triggered['latitude'], marker='^', s=50,
               facecolor='white', edgecolor='black', linewidths=0.8,
               transform=proj, zorder=6)
    ax.plot(USGS_LON, USGS_LAT, marker='*', markersize=18, color='gold',
            markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (key == 'conf')
    axes[key] = ax

# -- Left: posterior confidence ---------------------------------------------
ax = axes['conf']
ax.contourf(lons_2d, lats_2d, rel, levels=[level, 1.0 + 1e-9], colors=[C_CONF],
            alpha=0.45, transform=proj, zorder=4)
ax.contour(lons_2d, lats_2d, rel, levels=[level], colors=[C_CONF],
           linewidths=2.5, transform=proj, zorder=5)
ax.set_title(f'Posterior confidence = {conf:.2f}')
ax.legend(handles=[
    Patch(facecolor=C_CONF, alpha=0.45, edgecolor=C_CONF,
          label='Posterior mass with density $\\geq$ density at USGS'),
    Line2D([], [], color=C_GRAY, lw=1, label='Posterior contours'),
    Line2D([], [], color=C_GRAY, lw=1, ls='--', label='Low-density contours (below USGS density)'),
], loc='lower left', fontsize=9, framealpha=0.9)

# -- Right: posterior coverage ----------------------------------------------
ax = axes['cov']
ax.fill(circ_lon, circ_lat, facecolor=C_COV, alpha=0.30, edgecolor='none',
        transform=proj, zorder=4)
ax.plot(circ_lon, circ_lat, color=C_COV, lw=2.2, transform=proj, zorder=5)
ax.plot([USGS_LON, MAP_LON], [USGS_LAT, MAP_LAT], color='black', lw=1.5,
        ls='--', transform=proj, zorder=7)
ax.plot(MAP_LON, MAP_LAT, marker='o', markersize=9, color=C_CONF,
        markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=8)
ax.text(0.5 * (USGS_LON + MAP_LON), 0.5 * (USGS_LAT + MAP_LAT) - 0.06,
        f'R = {R_KM:.0f} km', ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5),
        transform=proj, zorder=9)
ax.set_title(f'Posterior coverage = {cov:.2f}')
ax.legend(handles=[
    Patch(facecolor=C_COV, alpha=0.30, edgecolor=C_COV,
          label='Posterior mass within R of USGS'),
    Line2D([], [], color=C_GRAY, lw=1, label='Posterior contours'),
    Line2D([], [], color=C_GRAY, lw=1, ls='--', label='Low-density contours (below USGS density)'),
], loc='lower left', fontsize=9, framealpha=0.9)

# Shared markers
fig.legend(handles=[
    Line2D([], [], marker='*', ls='', markersize=14, color='gold', markeredgecolor='black',
           label='USGS location'),
    Line2D([], [], marker='o', ls='', markersize=8, color=C_CONF, markeredgecolor='black',
           label='bEPIC MAP location'),
    Line2D([], [], marker='^', ls='', markersize=8, markerfacecolor='white',
           markeredgecolor='black', label='triggered station'),
], loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.savefig(os.path.join(OUTPUT_DIR, 'metric_illustration.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'metric_illustration.pdf'), bbox_inches='tight')
plt.show()
