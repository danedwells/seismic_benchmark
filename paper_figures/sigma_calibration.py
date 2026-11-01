#%%
# =============================================================================
# sigma_calibration.py — paper figure: calibrating the travel-time sigma_s
# =============================================================================
# (a-c) MTJ M7 mainshock likelihood at 4 triggers (as in mtj_bayesian_location_example.py)
#       for three sigma_s: the default, a wider one, and the calibrated 0.22.
# (d)   Sum of log(likelihood at USGS location) vs sigma_s over the Ridgecrest
#       case-study sequence (Uniform prior, 4 triggers) — the sweep peaks at 0.22.
# =============================================================================
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import SeismicPrior
from benchmark import config
from benchmark.plots import load_final_values
from benchmark.runner import run_single_event_get_grid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

MTJ_CASE   = 'MTJ_2024_M7'
EVENT_ID   = config.FOCUS_EVENTS_MAINSHOCK[MTJ_CASE]
N_TRIGGERS = 4
MAINSHOCK_LAT, MAINSHOCK_LON = 40.3740, -125.0217

SIGMAS = [(0.35, 'Default'), (1.0, 'Too wide'), (0.22, 'Optimal')]   # panels a, b, c

SWEEP_CASE = 'Ridgecrest'
SWEEP_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', SWEEP_CASE,
                          'output', 'time_independent')
COLUMN     = 'like_val_at_usgs'

RUN_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', MTJ_CASE,
                        'run_files', f'{EVENT_ID}.run')
KDE_PATH = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{MTJ_CASE}.tt3')

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()

#%%
# ── Likelihood grids for each sigma_s ───────────────────────────────────────
run_df = pd.read_csv(RUN_PATH)
run_df.columns = [c.replace(' ', '_') for c in run_df.columns]
focus_version = next((v for v in sorted(run_df['version'].unique())
                      if len(run_df[run_df['version'] == v]) >= N_TRIGGERS), None)
triggered = (run_df[run_df['version'] == (focus_version if focus_version is not None
                                          else run_df['version'].max())]
             .drop_duplicates(subset='station'))

kde_prior = SeismicPrior.from_tt3(KDE_PATH)
grids = []
for sigma, _ in SIGMAS:
    params_kw = dict(config.BENCHMARK_PARAMS)
    params_kw['sigma_s'] = sigma
    _, odf, _ = run_single_event_get_grid(RUN_PATH, kde_prior, True, params_kw,
                                          focus_version=focus_version)
    w = 2 * params_kw['grid_size'] + 1
    grids.append((odf['lon'].values.reshape(w, w), odf['lat'].values.reshape(w, w),
                  odf['like'].values.reshape(w, w)))

#%%
# ── Log-likelihood sweep ─────────────────────────────────────────────────────
sweep = {}
for entry in sorted(os.listdir(SWEEP_DIR)):
    m = re.match(r'^sig_(\d+(?:\.\d+)?)$', entry)
    if not m:
        continue
    d = os.path.join(SWEEP_DIR, entry)
    trig_dirs = [t for t in os.listdir(d) if t.startswith('max_trigs')]
    vals = load_final_values(os.path.join(d, trig_dirs[0], 'uniform_benchmark_results.csv'),
                             COLUMN, n_trigs=N_TRIGGERS)
    if vals is not None and len(vals):
        sweep[float(m.group(1))] = float(np.log(np.clip(vals, 1e-30, None)).sum())
xs = sorted(sweep)
ys = [sweep[x] for x in xs]
best = xs[int(np.argmax(ys))]
print('best sigma_s:', best)

#%%
# ── Figure ───────────────────────────────────────────────────────────────────
lons0, lats0, _ = grids[0]
pad = 0.3
extent = [lons0.min() - pad, lons0.max() + pad, lats0.min() - pad, lats0.max() + pad]

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.12, wspace=0.12)

for i, ((sigma, label), (lons, lats, like)) in enumerate(zip(SIGMAS, grids)):
    ax = fig.add_subplot(gs[0, i], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightcyan', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)
    ax.contour(lons, lats, like / like.max(), levels=np.linspace(0.1, 1.0, 10),
               cmap='Reds', transform=proj, zorder=4, linewidths=1.2)
    ax.scatter(triggered['longitude'], triggered['latitude'], marker='^', s=50,
               facecolor='white', edgecolor='black', linewidths=0.8, transform=proj,
               zorder=5, label='triggered station')
    ax.plot(MAINSHOCK_LON, MAINSHOCK_LAT, marker='*', markersize=16, color='gold',
            markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=6,
            label='USGS location')
    ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    ax.set_title(f'({"abc"[i]}) {label}: $\\sigma_s$ = {sigma:g} s')
    if i == 0:
        ax.legend(loc='lower left', fontsize=9)

axd = fig.add_subplot(gs[1, :])
axd.plot(xs, ys, '-o', color='tab:blue', ms=6, zorder=3)
axd.scatter([best], [sweep[best]], marker='*', s=250, color='tab:red', zorder=4,
            label=f'best $\\sigma_s$ = {best:g} s')
for i, (sigma, _) in enumerate(SIGMAS):
    axd.axvline(sigma, color='gray', ls='--', lw=1, zorder=1)
    axd.text(sigma, 1.01, f'({"abc"[i]})', transform=axd.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=11, fontweight='bold')
axd.set_xlabel('$\\sigma_s$ (s)')
axd.set_ylabel('Σ log(likelihood at USGS location)')
axd.text(0.97, 0.93, f'(d) Log-likelihood vs $\\sigma_s$ - {N_TRIGGERS} triggers',
         transform=axd.transAxes, ha='right', va='top', fontsize=13, fontweight='bold')
axd.grid(True, alpha=0.3)
axd.legend()

fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'sigma_calibration.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'sigma_calibration.pdf'), bbox_inches='tight')
plt.show()
