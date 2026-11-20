"""
Paper map figure (same style as plot_paper_case_studies_comp.py) comparing three versions of one California event set.
Pick the set with the DATASET environment variable (one figure per set):

  benchmark   the California benchmark catalog (~700 events)
  large       the test suite / large events

  EPIC     pure EPIC run                  output/<DATASET>_epic.csv      (placeholder, not yet produced)
  Control  the Control EPIC replay        output/<DATASET>_control.csv   (placeholder, not yet produced)
  ETAS     bEPIC re-run: ETAS prior, station mask, calibrated sigma   output/<DATASET>_rerun.csv   (placeholder, not yet produced)

(a) map of the locations with arrows to the USGS location, (b) inset zoom on the MTJ, (c) location-error and (d) magnitude-error histograms.
A version whose results file is missing gets a warning and is drawn as an empty placeholder, so the figure can be made before
the results exist.  Each file needs the columns anss_id, lat, lon, usgs_lat, usgs_lon, loc_err_km, mag_err.
"""
#%%
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import ConnectionPatch
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'output'
CA_EXTENT = [-125.0, -113.5, 32.0, 42.5]         # lon_min, lon_max, lat_min, lat_max; used until there are results

DATASET = os.environ.get('DATASET', 'benchmark')  # 'benchmark' | 'large'
assert DATASET in ('benchmark', 'large'), DATASET

RUNS = {'epic': 'gray', 'control': 'blue', 'rerun': 'red'}
LABELS = {'epic': 'EPIC', 'control': 'Control', 'rerun': 'ETAS'}

#%% Load the three result sets (empty placeholder + warning when a file is missing)
COLS = ['anss_id', 'lat', 'lon', 'usgs_lat', 'usgs_lon', 'loc_err_km', 'mag_err']

def load(which, path, rows=None):
    if not path.exists():
        warnings.warn(f'{LABELS[which]}: results not found ({path.name}); plotting a placeholder')
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    return (df if rows is None else df[rows(df)])[COLS].reset_index(drop=True)

results = {
    'epic':    load('epic', OUT / f'{DATASET}_epic.csv'),
    'control': load('control', OUT / f'{DATASET}_control.csv'),
    'rerun':   load('rerun', OUT / f'{DATASET}_rerun.csv'),
}
#%% Dummy events for styling the figure before any results exist (benchmark only).  Comment out the next line to turn off.
DUMMY = True
if DUMMY and DATASET == 'benchmark':
    rng = np.random.default_rng(0)
    bg = pd.read_parquet(ROOT / 'data' / 'california' / 'reference' / 'background_seismicity.parquet')
    bg = bg[bg.mag >= 3.1].sample(300, random_state=0).reset_index(drop=True)      # USGS background events stand in for the catalog
    for which, median_km in zip(RUNS, (12, 9, 6)):                                  # fake median location error per run
        if len(results[which]):
            continue
        err = rng.lognormal(np.log(median_km), 0.9, len(bg))                         # km
        az = rng.uniform(0, 2 * np.pi, len(bg))
        results[which] = pd.DataFrame({
            'anss_id': np.arange(len(bg)), 'usgs_lat': bg.latitude, 'usgs_lon': bg.longitude,
            'lat': bg.latitude + err * np.cos(az) / 111.0,
            'lon': bg.longitude + err * np.sin(az) / (111.0 * np.cos(np.radians(bg.latitude))),
            'loc_err_km': err, 'mag_err': rng.normal(-0.3 * (which == 'epic'), 0.4, len(bg))})
    warnings.warn('DUMMY events (USGS background seismicity + random offsets) are plotted; not real results')
dummy_used = DUMMY and DATASET == 'benchmark'     # stamps 'PLACEHOLDER DATA' on the map
for w, r in results.items():
    print(f'{LABELS[w]}: {len(r)} events')

#%% Paper-style figure (same layout/style as run_error_histograms_comp.py)
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()
SIZE_SCALE = dict(zip(RUNS, (1.6, 1.3, 1.1)))   # decreasing with draw order so overlapping dots read as concentric
BUF = 0.1                                        # degrees added around the plotted locations
MAP_PCT = 100  # middle percentile range of the plotted points that sets the map extent
MAP_W, MAP_H_MIN, MAP_H_MAX = 9.0, 5.0, 9.0      # map panel (inches): width, and the limits of its height
HIST_W, GAP = 2.5, 0.5                           # width of each histogram column; gap before each column
MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.9, 0.1, 0.95, 0.15

# -- Map extent from every plotted point; with no results at all, from CA_EXTENT --
lats = pd.concat([r[c] for r in results.values() for c in ('lat', 'usgs_lat')]).astype(float)
lons = pd.concat([r[c] for r in results.values() for c in ('lon', 'usgs_lon')]).astype(float)
if lats.empty:
    extent = list(CA_EXTENT)
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

IN_H, IN_GAP = 2.6, 0.3                          # inset map height above (b)+(c), and the gap between it and the histograms
HIST_H = map_h - IN_H - IN_GAP                   # histogram columns are compressed to leave room for the inset
fig_w = MARGIN_L + map_w + 2 * (GAP + HIST_W) + MARGIN_R
fig_h = MARGIN_B + map_h + MARGIN_T
def rect(x, w):                                  # inches -> figure fractions, spanning the map's vertical range
    return [x / fig_w, MARGIN_B / fig_h, w / fig_w, map_h / fig_h]
fig = plt.figure(figsize=(fig_w, fig_h))

INSET_BOUNDS = (-127.5, -123.5, 39.5, 41.5)      # lon_min, lon_max, lat_min, lat_max of the MTJ zoom (before aspect fit)
LARGE_ERR_KM = 100                               # vertical line in (b)
def in_mtj(r):                                   # events whose ANSS location is inside the MTJ zoom box
    return ((r.usgs_lon.astype(float).between(*INSET_BOUNDS[:2])) & (r.usgs_lat.astype(float).between(*INSET_BOUNDS[2:])))
mtj = {w: r[in_mtj(r)] for w, r in results.items()}

def stacked_hists(x, values, mtj_values, bins, fmt, xlabel, tag):
    """One histogram per run, stacked, shared x-axis: run name, then its median (number only) directly below.
    Filled bars: all events.  Outline + dotted median + second number: the MTJ subset.  Both are fractions of their own total."""
    left, bottom, width, height = x / fig_w, MARGIN_B / fig_h, HIST_W / fig_w, HIST_H / fig_h
    each = height / len(RUNS)
    axes = []
    for i, (which, color) in enumerate(RUNS.items()):
        ax = fig.add_axes([left, bottom + height - (i + 1) * each, width, each], sharex=axes[0] if axes else None)
        axes.append(ax)
        stats = values[which]
        if len(stats):
            ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7, weights=np.full(len(stats), 1 / len(stats)))
            med = np.median(stats)
            ax.axvline(med, ymax=0.55, color=color, ls='--', lw=1.5)      # stops above the bars, below the labels
            ax.annotate(fmt(med), xy=(0.03, 0.96), xytext=(0, -17), xycoords='axes fraction', textcoords='offset points',
                        ha='left', va='top', fontsize=11, color=color, fontweight='bold')
            sub = mtj_values[which]
            if len(sub):
                ax.hist(sub, bins=bins, histtype='step', color=color, lw=1.6, weights=np.full(len(sub), 1 / len(sub)))
                msub = np.median(sub)
                ax.axvline(msub, ymax=0.55, color=color, ls=':', lw=1.8)
                ax.annotate('MTJ ' + fmt(msub), xy=(0.03, 0.96), xytext=(0, -31), xycoords='axes fraction',
                            textcoords='offset points', ha='left', va='top', fontsize=9, color=color, fontweight='bold')
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
        a.set_ylim(0, 1.65 * ymax)   # headroom: bars top out just below the name/median labels
    axes[0].text(-0.2, 0.87, tag, transform=axes[0].transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')
    return axes

# -- (b) location error --
LOC_BINS = np.logspace(-1, 3, 20)
x_b = MARGIN_L + map_w + GAP
axes_b = stacked_hists(x_b, {w: r.loc_err_km.astype(float).values for w, r in results.items()},
                       {w: r.loc_err_km.astype(float).values for w, r in mtj.items()}, LOC_BINS,
                       lambda m: f'{m:.1f} km', 'Location error (km)', '(c)')
for ax in axes_b:
    ax.axvline(LARGE_ERR_KM, color='k', ls='-', lw=0.9, alpha=0.6)
axes_b[0].annotate('large error', xy=(LARGE_ERR_KM, 0.96), xycoords=('data', 'axes fraction'), xytext=(3, 0),
                   textcoords='offset points', ha='left', va='top', fontsize=9, alpha=0.8, rotation=90)
axes_b[-1].set_xscale('log')
axes_b[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

# -- (c) magnitude error (shared bins) --
mag_vals = {w: r.mag_err.astype(float).dropna().values for w, r in results.items()}
mag_mtj = {w: r.mag_err.astype(float).dropna().values for w, r in mtj.items()}
mag_all = np.concatenate(list(mag_vals.values()))
mag_bins = np.linspace(mag_all.min(), mag_all.max(), len(LOC_BINS)) if len(mag_all) else np.linspace(-2, 2, len(LOC_BINS))   # same number of bins as (b)
axes_c = stacked_hists(x_b + HIST_W + GAP, mag_vals, mag_mtj, mag_bins,
                       lambda m: f'{m:+.2f}'.replace('-', '−'), 'Magnitude − USGS mag', '(d)')
for ax in axes_c:
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)

# -- (a) map of locations, colored by run --
ax_map = fig.add_axes(rect(MARGIN_L, map_w), projection=proj)
ax_map.set_extent(extent, crs=proj)

# Inset: zoom on the MTJ region, above the histograms and spanning both columns
in_w = 2 * HIST_W + GAP
ax_in = fig.add_axes([x_b / fig_w, (MARGIN_B + map_h - IN_H) / fig_h, in_w / fig_w, IN_H / fig_h], projection=proj)
in_h = IN_H
in_ext = list(INSET_BOUNDS)                      # widen one side so the extent matches the inset's shape (no shrinking)
if (in_ext[1] - in_ext[0]) / (in_ext[3] - in_ext[2]) < in_w / in_h:
    c, half = 0.5 * (in_ext[0] + in_ext[1]), 0.5 * (in_ext[3] - in_ext[2]) * in_w / in_h
    in_ext[0], in_ext[1] = c - half, c + half
else:
    c, half = 0.5 * (in_ext[2] + in_ext[3]), 0.5 * (in_ext[1] - in_ext[0]) * in_h / in_w
    in_ext[2], in_ext[3] = c - half, c + half
ax_in.set_extent(in_ext, crs=proj)
for sp in ax_in.spines.values():
    sp.set_linewidth(1.5)
ax_in.text(0.015, 0.97, '(b)', transform=ax_in.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), zorder=10)
for (cx, cy), (fx, fy) in (((in_ext[1], in_ext[3]), (0, 1)), ((in_ext[1], in_ext[2]), (0, 0))):   # box corners -> inset corners
    fig.add_artist(ConnectionPatch(xyA=(cx, cy), coordsA='data', axesA=ax_map, xyB=(fx, fy), coordsB='axes fraction', axesB=ax_in,
                                   color='k', lw=0.8, alpha=0.35, zorder=1))
ax_map.plot([in_ext[0], in_ext[1], in_ext[1], in_ext[0], in_ext[0]], [in_ext[2], in_ext[2], in_ext[3], in_ext[3], in_ext[2]],
            color='k', lw=1.0, transform=proj, zorder=7)    # where the inset is looking

usgs_pts = pd.concat([r[['anss_id', 'usgs_lat', 'usgs_lon']] for r in results.values()]).drop_duplicates('anss_id').astype(
    {'usgs_lat': float, 'usgs_lon': float})
for ax in (ax_map, ax_in):
    ax.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)
    for which, color in RUNS.items():            # arrows from each run's location to the USGS location
        m = results[which].astype({'lat': float, 'lon': float, 'usgs_lat': float, 'usgs_lon': float})
        ax.quiver(m.lon.values, m.lat.values, (m.usgs_lon - m.lon).values, (m.usgs_lat - m.lat).values,
                  angles='xy', scale_units='xy', scale=1, color=color, alpha=0.3, width=0.0025,
                  headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
    for which, color in RUNS.items():            # empty runs still get their legend entry
        m = results[which].astype({'lat': float, 'lon': float})
        ax.scatter(m.lon, m.lat, color=color, s=30 * SIZE_SCALE[which], alpha=0.5,
                   edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=LABELS[which])
    ax.scatter(usgs_pts.usgs_lon, usgs_pts.usgs_lat, marker='x', c='black', s=35, linewidths=1.1,
               transform=proj, zorder=6, alpha=0.5, label='ANSS locations')

ax_map.set_xticks(np.linspace(extent[0], extent[1], 5), crs=proj)
ax_map.set_yticks(np.linspace(extent[2], extent[3], 5), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
if dummy_used:
    ax_map.text(0.5, 0.97, 'PLACEHOLDER DATA', transform=ax_map.transAxes, ha='center', va='top', fontsize=18, fontweight='bold',
                color='firebrick', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='firebrick'), zorder=10)
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

out_stem = Path(__file__).resolve().parent / 'figures' / f'california_{DATASET}_paper_comp'
fig.savefig(out_stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
print('Saved', out_stem, f'(figure {fig_w:.1f} x {fig_h:.1f} in, map {map_w:.1f} x {map_h:.1f} in)')
