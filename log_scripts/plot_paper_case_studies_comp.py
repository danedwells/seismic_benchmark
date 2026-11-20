"""
Paper map figure (same style as the figures from run_error_histograms_comp.py) comparing three versions of one sequence:

  EPIC     the pure EPIC replay from the log files   output/log_results_<SEQUENCE>_nS4.csv  (run == 'epic')
  Control  the Control EPIC replay from the log files      output/log_results_<SEQUENCE>_nS4.csv  (run == 'control')
  bEPIC    bEPIC re-run: ETAS prior, station mask, calibrated sigma   output/<SEQUENCE>_station_masking.csv (alert rows)
           (run_station_masking.py)

(a) map of the locations with arrows to the USGS location, (b) location-error and (c) magnitude-error histograms.
A version whose results file is missing gets a warning and is drawn as an empty placeholder, so the figure can be made before
all results exist.  
"""
#%%
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark.config_california as config
import parsing_tools

SEQUENCE = os.environ.get('SEQUENCE', 'Ridgecrest')     # 'MTJ' | 'Ridgecrest'
CASE_KEY = {'Ridgecrest': 'Ridgecrest', 'MTJ': 'MTJ_2024_M7'}[SEQUENCE]
MAINSHOCK_ID = config.FOCUS_EVENTS_MAINSHOCK[CASE_KEY]   # starred on the map
ALERTS_ONLY = True                                       # first version with nS >= N_TRIGS that EPIC alerted on (False: first version, alerted or not)
N_TRIGS = 4                                              # station count of the log results (the re-run's alert uses the same)
OUT = Path(__file__).resolve().parent / 'output'


RUNS = {'epic': 'gray', 'control': 'blue', 'rerun': 'red'}
LABELS = {'epic': 'EPIC', 'control': 'Control', 'rerun': 'ETAS'}

#%% Load the three result sets (empty placeholder + warning when a file is missing)
COLS = ['anss_id', 'lat', 'lon', 'usgs_lat', 'usgs_lon', 'loc_err_km', 'mag_err']

def load(which, path, run=None):
    if not path.exists():
        warnings.warn(f'{LABELS[which]}: results not found ({path.name}); plotting a placeholder')
        return pd.DataFrame(columns=COLS)
    versions = pd.read_csv(path)
    if run is not None:
        versions = versions[versions.run == run]
    return parsing_tools.pick_version(versions, N_TRIGS, ALERTS_ONLY)[COLS]

results = {
    'epic':    load('epic', OUT / f'log_results_{SEQUENCE}_nS{N_TRIGS}.csv', 'epic'),
    'control': load('control', OUT / f'log_results_{SEQUENCE}_nS{N_TRIGS}.csv', 'control'),
    'rerun':   load('rerun', OUT / f'{SEQUENCE}_station_masking.csv'),
}
for w, r in results.items():
    print(f'{LABELS[w]}: {len(r)} events')

#%% Paper-style figure (same layout/style as run_error_histograms_comp.py)
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()
SIZE_SCALE = dict(zip(RUNS, (1.6, 1.3, 1.1)))   # decreasing with draw order so overlapping dots read as concentric
BUF = 0.1                                        # degrees added around the plotted locations
MAP_PCT = {'Ridgecrest': 96}.get(SEQUENCE, 100)  # middle percentile range of the plotted points that sets the map extent
MAP_W, MAP_H_MIN, MAP_H_MAX = 9.0, 4.5, 7.5      # map panel (inches): width, and the limits of its height
HIST_W, GAP = 2.5, 0.6                           # width of each histogram column; gap before each column
MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.9, 0.2, 1.0, 0.4

# -- Map extent from every plotted point; with no results at all, from the case-study search bounds --
lats = pd.concat([r[c] for r in results.values() for c in ('lat', 'usgs_lat')]).astype(float)
lons = pd.concat([r[c] for r in results.values() for c in ('lon', 'usgs_lon')]).astype(float)
if lats.empty:
    lon_min, lon_max, lat_min, lat_max = config.CASE_STUDIES[CASE_KEY]['bounds']
    extent = [lon_min, lon_max, lat_min, lat_max]
else:
    lo, hi = (100 - MAP_PCT) / 2, 100 - (100 - MAP_PCT) / 2
    extent = [np.percentile(lons, lo) - BUF, np.percentile(lons, hi) + BUF, np.percentile(lats, lo) - BUF, np.percentile(lats, hi) + BUF]
aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
map_h = float(np.clip(MAP_W / aspect, MAP_H_MIN, MAP_H_MAX))
map_w = MAP_W if MAP_W / aspect <= MAP_H_MAX else map_h * aspect
box = map_w / map_h
if aspect < box:
    half = 0.5 * box * (extent[3] - extent[2])
    c = 0.5 * (extent[0] + extent[1]); extent[0], extent[1] = c - half, c + half
else:
    half = 0.5 * (extent[1] - extent[0]) / box
    c = 0.5 * (extent[2] + extent[3]); extent[2], extent[3] = c - half, c + half

fig_w = MARGIN_L + map_w + 2 * (GAP + HIST_W) + MARGIN_R
fig_h = MARGIN_B + map_h + MARGIN_T
def rect(x, w):                                  # inches -> figure fractions, spanning the map's vertical range
    return [x / fig_w, MARGIN_B / fig_h, w / fig_w, map_h / fig_h]
fig = plt.figure(figsize=(fig_w, fig_h))

def stacked_hists(x, values, bins, fmt, xlabel, tag):
    """One histogram per run, stacked, shared x-axis: run name, then its median (number only) directly below."""
    left, bottom, width, height = rect(x, HIST_W)
    each = height / len(RUNS)
    axes = []
    for i, (which, color) in enumerate(RUNS.items()):
        ax = fig.add_axes([left, bottom + height - (i + 1) * each, width, each], sharex=axes[0] if axes else None)
        axes.append(ax)
        stats = values[which]
        if len(stats):
            ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7)
            med = np.median(stats)
            ax.axvline(med, ymax=0.55, color=color, ls='--', lw=1.5)      # stops above the bars, below the labels
            ax.annotate(fmt(med), xy=(0.03, 0.96), xytext=(0, -17), xycoords='axes fraction', textcoords='offset points',
                        ha='left', va='top', fontsize=11, color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.25, 'results pending', transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color=color, alpha=0.6, style='italic')
        ax.annotate(LABELS[which], xy=(0.03, 0.96), xycoords='axes fraction', ha='left', va='top',
                    fontsize=13, color=color, fontweight='bold')
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        if i < len(RUNS) - 1:
            ax.tick_params(labelbottom=False, bottom=False)
    axes[-1].set_xlabel(xlabel)
    ymax = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(0, 2.0 * ymax)   # headroom: bars top out at half height so the name/median labels sit clear above them
    axes[0].text(-0.2, 0.87, tag, transform=axes[0].transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')
    return axes

# -- (b) location error --
x_b = MARGIN_L + map_w + GAP
axes_b = stacked_hists(x_b, {w: r.loc_err_km.astype(float).values for w, r in results.items()}, np.logspace(-1, 3, 20),
                       lambda m: f'{m:.1f} km', 'Location error (km)', '(b)')
axes_b[-1].set_xscale('log')
axes_b[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

# -- (c) magnitude error (shared bins) --
mag_vals = {w: r.mag_err.astype(float).dropna().values for w, r in results.items()}
mag_all = np.concatenate(list(mag_vals.values()))
mag_bins = np.linspace(mag_all.min(), mag_all.max(), 41) if len(mag_all) else np.linspace(-2, 2, 41)
axes_c = stacked_hists(x_b + HIST_W + GAP, mag_vals, mag_bins,
                       lambda m: f'{m:+.2f}'.replace('-', '−'), 'Magnitude − USGS mag', '(c)')
for ax in axes_c:
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)

# -- (a) map of locations, colored by run --
ax_map = fig.add_axes(rect(MARGIN_L, map_w), projection=proj)
ax_map.set_extent(extent, crs=proj)
ax_map.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

for which, color in RUNS.items():                # arrows from each run's location to the USGS location
    m = results[which].astype({'lat': float, 'lon': float, 'usgs_lat': float, 'usgs_lon': float})
    ax_map.quiver(m.lon.values, m.lat.values, (m.usgs_lon - m.lon).values, (m.usgs_lat - m.lat).values,
                  angles='xy', scale_units='xy', scale=1, color=color, alpha=0.3, width=0.0025,
                  headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
for which, color in RUNS.items():                # empty runs still get their legend entry
    m = results[which].astype({'lat': float, 'lon': float})
    ax_map.scatter(m.lon, m.lat, color=color, s=30 * SIZE_SCALE[which], alpha=0.5,
                   edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=LABELS[which])
# Mainshock: each run's location as a large star in the run's color; the stars shrink in draw order so they nest
for i, (which, color) in enumerate(RUNS.items()):
    ms = results[which][results[which].anss_id == MAINSHOCK_ID].astype({'lat': float, 'lon': float})
    ax_map.scatter(ms.lon, ms.lat, color=color, s=300 + 150 * (len(RUNS) - 1 - i), edgecolors='black', linewidths=1.0,
                   marker='*', transform=proj, zorder=8 + i)
usgs_pts = pd.concat([r[['anss_id', 'usgs_lat', 'usgs_lon']] for r in results.values()]).drop_duplicates('anss_id').astype(
    {'usgs_lat': float, 'usgs_lon': float})
ax_map.scatter(usgs_pts.usgs_lon, usgs_pts.usgs_lat, marker='x', c='black', s=35, linewidths=1.1,
               transform=proj, zorder=6, alpha=0.5, label='ANSS locations')
# True (ANSS) mainshock location: yellow star on top of the run stars, with a legend entry
true_ms = usgs_pts[usgs_pts.anss_id == MAINSHOCK_ID]
ax_map.scatter(true_ms.usgs_lon, true_ms.usgs_lat, marker='*', c='yellow', s=400,
               edgecolors='black', linewidths=0.8, transform=proj, zorder=20, label='ANSS mainshock')

ax_map.set_xticks(np.linspace(extent[0], extent[1], 5), crs=proj)
ax_map.set_yticks(np.linspace(extent[2], extent[3], 5), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
ax_map.text(0.98, 0.02, f'Locations at {N_TRIGS} stations', transform=ax_map.transAxes,
            ha='right', va='bottom', fontsize=13, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

out_stem = Path(__file__).resolve().parent / 'figures' / f'{SEQUENCE}_location_hist_{N_TRIGS}_paper_comp'
fig.savefig(out_stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
print('Saved', out_stem, f'(figure {fig_w:.1f} x {fig_h:.1f} in, map {map_w:.1f} x {map_h:.1f} in)')
