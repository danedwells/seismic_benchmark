#%%
# =============================================================================
# mtj_bayesian_location_example.py — paper figure: prior/likelihood/posterior
# =============================================================================
# Illustrates the Bayesian location problem bEPIC solves: 3 horizontal panels
# (prior, likelihood, posterior) as line contours over a shared map extent,
# for one example event — the 2024 Mendocino Triple Junction M7 mainshock —
# located with the KDE_Seismicity prior at a fixed, moderate trigger count
# (so the likelihood alone is still ambiguous and the prior visibly matters).
#
# Re-runs bEPIC for this single event via benchmark.runner's
# run_single_event_get_grid() (the same single-event diagnostic entry point
# used by benchmark/plots.py's plot_posterior_grid() and plot_scripts/
# plot_comparison.py's "Figure 15") to get the prior/likelihood/posterior
# grids — these aren't saved anywhere by the main benchmark run, only the
# scalar posterior/exp/like point estimates are.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import SeismicPrior
from benchmark import config
from benchmark.runner import run_single_event_get_grid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MTJ_CASE  = 'MTJ_2024_M7'
EVENT_ID  = config.FOCUS_EVENTS_MAINSHOCK[MTJ_CASE]  # nc75095651 — the M7 mainshock itself
N_TRIGGERS = 4     # plot the version at which this many triggers have reported
                   # (deliberately modest — with many more triggers the
                   # likelihood alone would already pin the location down,
                   # which defeats the point of illustrating the prior's role)
SIGMA_S = 0.22     # tuned case-study value (time_dependent_scripts/case_studies.py)

RUN_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', MTJ_CASE,
                         'run_files', f'{EVENT_ID}.run')
KDE_PATH = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{MTJ_CASE}.tt3')

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
# ── Resolve the trigger version to plot ──────────────────────────────────────
run_df = pd.read_csv(RUN_PATH)
run_df.columns = [c.replace(' ', '_') for c in run_df.columns]

focus_version = None
for v in sorted(run_df['version'].unique()):
    if len(run_df[run_df['version'] == v]) >= N_TRIGGERS:
        focus_version = v
        break
if focus_version is None:
    print(f'Event has fewer than {N_TRIGGERS} triggers — using the last available version.')

triggered = (run_df[run_df['version'] == (focus_version if focus_version is not None else run_df['version'].max())]
             .drop_duplicates(subset='station'))

MAINSHOCK_LAT, MAINSHOCK_LON = 40.3740, -125.0217  # nc75095651, see config_california.py FOCUS_EVENTS_MAINSHOCK

#%%
# ── Run bEPIC for this one event/version with the KDE_Seismicity prior ──────
kde_prior = SeismicPrior.from_tt3(KDE_PATH)

params_kw = dict(config.BENCHMARK_PARAMS)
params_kw['sigma_s'] = SIGMA_S

t_out, odf, actual_v = run_single_event_get_grid(
    RUN_PATH, kde_prior, True, params_kw, focus_version=focus_version,
)
print(f'Plotting version {actual_v} ({len(triggered)} stations triggered)')

grid_width = 2 * params_kw['grid_size'] + 1
lats_2d  = odf['lat'].values.reshape(grid_width, grid_width)
lons_2d  = odf['lon'].values.reshape(grid_width, grid_width)
prior_2d = odf['prior'].values.reshape(grid_width, grid_width)
like_2d  = odf['like'].values.reshape(grid_width, grid_width)
post_2d  = odf['post'].values.reshape(grid_width, grid_width)

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    'prior':      [0.03, 0.08, 0.30, 0.80],
    'likelihood': [0.36, 0.08, 0.30, 0.80],
    'posterior':  [0.69, 0.08, 0.30, 0.80],
}

pad = 0.3
extent = [
    float(lons_2d.min()) - pad, float(lons_2d.max()) + pad,
    float(lats_2d.min()) - pad, float(lats_2d.max()) + pad,
]

panels = [
    ('prior',      'Prior  (KDE Seismicity)', prior_2d, 'Reds'),
    ('likelihood', 'Likelihood',              like_2d,  'Reds'),
    ('posterior',  'Posterior',               post_2d,  'Reds'),
]

fig = plt.figure(figsize=(15, 5.5))

for key, title, grid_2d, cmap in panels:
    ax = fig.add_axes(AX_POS[key], projection=proj)
    ax.set_extent(extent, crs=proj)
    _add_basemap(ax)

    gmax = grid_2d.max()
    if gmax > 0:
        ax.contour(lons_2d, lats_2d, grid_2d / gmax,
                   levels=np.linspace(0.1, 1.0, 10),
                   cmap=cmap, transform=proj, zorder=4, linewidths=1.2)

    ax.scatter(triggered['longitude'], triggered['latitude'],
               marker='^', s=50, facecolor='white', edgecolor='black',
               linewidths=0.8, transform=proj, zorder=5, label='triggered station')
    ax.plot(MAINSHOCK_LON, MAINSHOCK_LAT, marker='*', markersize=16, color='gold',
            markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=6,
            label='USGS location')

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.left_labels  = (key == 'prior')

    ax.set_title(title)

    # 50 km scale bar, bottom-left corner — prior panel only.
    if key == 'prior':
        lon_min, lon_max, lat_min, lat_max = extent
        lat_mid   = (lat_min + lat_max) / 2
        scale_km  = 50
        scale_deg = scale_km / (111.32 * np.cos(np.radians(lat_mid)))
        x0 = lon_min + 0.08 * (lon_max - lon_min)
        y0 = lat_min + 0.08 * (lat_max - lat_min)
        x1 = x0 + scale_deg
        tick_h = 0.02 * (lat_max - lat_min)
        for xs, ys in (([x0, x1], [y0, y0]),
                       ([x0, x0], [y0 - tick_h, y0 + tick_h]),
                       ([x1, x1], [y0 - tick_h, y0 + tick_h])):
            ax.plot(xs, ys, color='black', linewidth=2,
                    transform=proj, zorder=10, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, y0 + tick_h + 0.02 * (lat_max - lat_min), f'{scale_km} km',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                transform=proj, zorder=10)

axes = [fig.axes[i] for i in range(3)]
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

#fig.suptitle(f'Bayesian location — MTJ M7.0 mainshock  ({len(triggered)} stations triggered)', y=1.0)

fig.savefig(os.path.join(OUTPUT_DIR, 'mtj_bayesian_location_example.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'mtj_bayesian_location_example.pdf'), bbox_inches='tight')
plt.show()

# %%
