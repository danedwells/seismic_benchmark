"""
Comparison of location and magnitude error histograms for one sequence, overlaid (see
run_error_histograms.py for the single-run version):
  EPIC     pure EPIC replay logs   (log_files/<SEQUENCE>-selected/<SEQUENCE>_EPIC_50_25_grid)
  Control  Control EPIC replay logs
  ETAS     bEPIC re-run (ETAS prior, station mask, calibrated sigma): output/<SEQUENCE>_station_masking.csv, alert rows
           (run_station_masking.py); missing file -> empty placeholder
The sigma_only / sigma_and_prior logs are still parsed into log_results_*.csv (used by make_log_table.py and
improvement_map_logs.py) but are not plotted.

For the log runs, reads every <anss_id>.log in a run folder (parsing_tools.py), finds the EPIC event that matches the
USGS/ANSS event (within MAX_ORIGIN_DIFF_S and 100 km), saves every version of that event whose station count (nS, the E:I: column) reaches N_TRIGS (epic_alerted = 1 if EPIC alerted
at that version), and plots the first of them that alerted (ALERTS_ONLY; parsing_tools.pick_version).  Events without one are skipped.
USGS solutions are cached to a CSV so re-runs don't re-query ComCat.
"""
#%%
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark.usgs import get_usgs_event
import benchmark.config_california as config
import parsing_tools

# Sequence -> case-study key in config.FOCUS_EVENTS_MAINSHOCK.  The figure size follows the map extent (see the paper-style cell).
SEQUENCE = os.environ.get('SEQUENCE', 'MTJ')     # 'MTJ' | 'Ridgecrest' | 'ElMayor' | 'Ferndale'
CASE_KEY = {'Ridgecrest': 'Ridgecrest', 'MTJ': 'MTJ_2024_M7', 'ElMayor': 'ElMayor', 'Ferndale': 'Ferndale'}[SEQUENCE]
MAINSHOCK_ID = config.FOCUS_EVENTS_MAINSHOCK[CASE_KEY]   # starred on the map
RUNS = {'epic': 'gray', 'control': 'blue', 'rerun': 'red'}          # plotted, in draw order
LABELS = {'epic': 'EPIC', 'control': 'Control', 'rerun': 'ETAS'}   # names shown on the paper figure
LOG_RUNS = {'epic': 'EPIC_50_25_grid', 'control': 'control', 'sigma_only': 'sigma_only', 'sigma_and_prior': 'sigma_and_prior'}   # run -> log folder suffix

N_TRIGS = 4
MAX_ORIGIN_DIFF_S = 30          # match log event to USGS origin time within this
MIN_USGS_MAG = 3.5            # only compare events with USGS magnitude >= this (None = all)
ALERTS_ONLY = True              # plot the first version with nS >= N_TRIGS that EPIC alerted on (False: first version, alerted or not)
USGS_CACHE = Path(__file__).resolve().parent / 'output' / f'usgs_cache_{SEQUENCE}.csv'

#%% Parse logs: one row (event, version) per E:I: line with eventid > 0
all_logs = {}
for which, folder in LOG_RUNS.items():
    run_dir = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{folder}'
    run_dir = run_dir / 'epic' if (run_dir / 'epic').is_dir() else run_dir    # some runs still have the nested folder
    if not run_dir.is_dir():
        raise FileNotFoundError(f'No logs for {SEQUENCE}: {run_dir} does not exist')
    all_logs[which] = {p.stem: parsing_tools.read_ei(p) for p in sorted(run_dir.glob('*.log'))}
    logs = all_logs[which]
    print(f'{which}: {len(logs)} logs; {sum((d.nS >= N_TRIGS).any() for d in logs.values())} reach nS >= {N_TRIGS}; {sum((d.Ast == 1).any() for d in logs.values())} contain an alerted event')

#%% USGS solutions (cached)
def fetch_usgs(anss_id):
    try:
        g = get_usgs_event(anss_id)
        lon, lat, dep = g['geometry']['coordinates']
        return dict(anss_id=anss_id, lat=lat, lon=lon, mag=g['properties']['mag'],
                    time=pd.to_datetime(g['properties']['time'], unit='ms'))
    except Exception as e:
        print(f'USGS lookup failed for {anss_id}: {e}')

usgs = pd.read_csv(USGS_CACHE, parse_dates=['time']) if USGS_CACHE.exists() else pd.DataFrame(
    columns=['anss_id', 'lat', 'lon', 'mag', 'time'])
todo = sorted({i for logs in all_logs.values() for i, d in logs.items() if (d.nS >= N_TRIGS).any()} - set(usgs.anss_id))
if todo:
    with ThreadPoolExecutor(8) as ex:
        new = [r for r in ex.map(fetch_usgs, todo) if r]
    usgs = pd.concat([usgs, pd.DataFrame(new)], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.set_index('anss_id')

#%% Errors of every version with nS >= N_TRIGS (one row per event and version, like run_station_masking.py)
all_versions = {}
for which, logs in all_logs.items():
    rows = []
    for anss_id, log in logs.items():
        if log.empty or anss_id not in usgs.index:
            continue
        usgs_event = usgs.loc[anss_id]
        if MIN_USGS_MAG is not None and not usgs_event.mag >= MIN_USGS_MAG:
            continue
        # Several events can be in one log; take the EPIC event that matches USGS, then all its versions with enough stations
        eventid = parsing_tools.match_event(log, usgs_event.time, usgs_event.lat, usgs_event.lon, max_dt_s=MAX_ORIGIN_DIFF_S)
        if eventid is None:
            continue
        event_versions = log[(log.eventid == eventid) & (log.nS >= N_TRIGS)]
        for _, v in event_versions.iterrows():
            rows.append(dict(anss_id=anss_id, ver=v.ver, nS=v.nS, epic_alerted=int(v.Ast == 1), usgs_mag=usgs_event.mag,
                             loc_err_km=gps2dist_azimuth(usgs_event.lat, usgs_event.lon, v.lat, v.lon)[0] / 1e3,
                             mag=v.mag, mag_err=v.mag - usgs_event.mag, lat=v.lat, lon=v.lon,
                             usgs_lat=usgs_event.lat, usgs_lon=usgs_event.lon))
    all_versions[which] = pd.DataFrame(rows)
    print(f'{which}: {all_versions[which].anss_id.nunique()} events, {len(all_versions[which])} versions with nS >= {N_TRIGS}'
          + (f' and USGS M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''))

# Every version of every log run, for tables and other scripts (they pick the version with parsing_tools.pick_version)
pd.concat([v.assign(run=w) for w, v in all_versions.items()]).to_csv(
    Path(__file__).resolve().parent / 'output' / f'log_results_{SEQUENCE}_nS{N_TRIGS}.csv', index=False)

# bEPIC re-run: every version too (run_station_masking.py)
rerun_csv = Path(__file__).resolve().parent / 'output' / f'{SEQUENCE}_station_masking.csv'
if rerun_csv.exists():
    all_versions['rerun'] = pd.read_csv(rerun_csv)
else:
    warnings.warn(f'ETAS: {rerun_csv.name} not found; plotting a placeholder')
    all_versions['rerun'] = pd.DataFrame(columns=all_versions['control'].columns)

# The version plotted for each event: the first with nS >= N_TRIGS, alerted only if ALERTS_ONLY
results = {which: parsing_tools.pick_version(all_versions[which], N_TRIGS, ALERTS_ONLY) for which in RUNS}
for which, r in results.items():
    print(f'{which}: {len(r)} events plotted')
    print(r[['loc_err_km', 'mag_err']].describe())

#%% Shared magnitude-error bins for the stacked histograms in the figure below
mag_all = pd.concat([r.mag_err for r in results.values()])
mag_bins = np.linspace(mag_all.min(), mag_all.max(), 41)

#%% Paper-style figure (same layout/style as paper_figures/location_hist_case_study.py):
# (a) map of EPIC locations per run with arrows to the USGS location, (b) stacked location-error histograms, (c) magnitude errors.
# Sizes are in inches and follow the map extent: the map is MAP_W wide and as tall as its extent needs (within MAP_H_MIN..MAP_H_MAX),
# the histogram columns sit to its right, and the figure is as wide/tall as that takes.  Works for any sequence without hand-tuning.
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()
SIZE_SCALE = dict(zip(RUNS, (1.6, 1.3, 1.1)))   # decreasing with draw order so overlapping dots read as concentric
BUF = 0.1                                        # degrees added around the plotted locations
MAP_PCT = float(os.environ.get('MAP_PCT', {'Ridgecrest': 96}.get(SEQUENCE, 100)))  # middle percentile range of the plotted points that sets the map extent: 100 = every point,
                                                 # e.g. 96 lets the farthest 2% on each side run off the map edge (histograms still include them)
MAP_W, MAP_H_MIN, MAP_H_MAX = 9.0, 4.5, 7.5      # map panel (inches): width, and the limits of its height
HIST_W, GAP = 2.5, 0.6                           # width of each histogram column; gap before each column (room for the (b)/(c) labels)
MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.9, 0.2, 1.0, 0.4

# -- Map extent from every plotted point (EPIC and USGS locations), then padded to the map box's aspect --
lats = pd.concat([r[c] for r in results.values() for c in ('lat', 'usgs_lat')])
lons = pd.concat([r[c] for r in results.values() for c in ('lon', 'usgs_lon')])
lo, hi = (100 - MAP_PCT) / 2, 100 - (100 - MAP_PCT) / 2
extent = [np.percentile(lons, lo) - BUF, np.percentile(lons, hi) + BUF, np.percentile(lats, lo) - BUF, np.percentile(lats, hi) + BUF]
aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])                 # axes are equal-aspect in degrees
map_h = float(np.clip(MAP_W / aspect, MAP_H_MIN, MAP_H_MAX))
map_w = MAP_W if MAP_W / aspect <= MAP_H_MAX else map_h * aspect           # tall regions: narrower map instead of a huge one
box = map_w / map_h
if aspect < box:                                                           # pad whichever side is short so the extent fills the box
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

def stacked_hists(x, values, bins, fmt, xlabel, tag, log=False):
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
axes_b = stacked_hists(x_b, {w: r.loc_err_km.values for w, r in results.items()}, np.logspace(-1, 3, 20),
                       lambda m: f'{m:.1f} km', 'Location error (km)', '(b)')
axes_b[-1].set_xscale('log')
axes_b[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

# -- (c) magnitude error (shared bins) --
axes_c = stacked_hists(x_b + HIST_W + GAP, {w: r.mag_err.dropna().values for w, r in results.items()}, mag_bins,
                       lambda m: f'{m:+.2f}'.replace('-', '−'), 'EPIC mag − USGS mag', '(c)')
for ax in axes_c:
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)

# -- (a) combined map of EPIC locations, colored by run --
ax_map = fig.add_axes(rect(MARGIN_L, map_w), projection=proj)
ax_map.set_extent(extent, crs=proj)
ax_map.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

# Arrows from each run's location to the USGS location, colored to match that run's points
for which, color in RUNS.items():
    m = results[which]
    ax_map.quiver(m.lon.values, m.lat.values, (m.usgs_lon - m.lon).values, (m.usgs_lat - m.lat).values,
                  angles='xy', scale_units='xy', scale=1, color=color, alpha=0.3, width=0.0025,
                  headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
for which, color in RUNS.items():
    m = results[which]
    ax_map.scatter(m.lon, m.lat, color=color, s=30 * SIZE_SCALE[which], alpha=0.5,
                   edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=LABELS[which])
# Mainshock: each run's location as a large star in the run's color, drawn above everything else.
# Runs often land on the same spot, so the stars shrink in draw order and nest instead of hiding each other.
for i, (which, color) in enumerate(RUNS.items()):
    ms = results[which][results[which].anss_id == MAINSHOCK_ID]
    ax_map.scatter(ms.lon, ms.lat, color=color, s=300 + 150 * (len(RUNS) - 1 - i), edgecolors='black', linewidths=1.0,
                   marker='*', transform=proj, zorder=8 + i)
usgs_pts = pd.concat([r[['anss_id', 'usgs_lat', 'usgs_lon']] for r in results.values()]).drop_duplicates('anss_id')
ax_map.scatter(usgs_pts.usgs_lon, usgs_pts.usgs_lat, marker='x', c='black', s=35, linewidths=1.1,
               transform=proj, zorder=6, alpha=0.5, label='ANSS locations')
# True (USGS) mainshock location: yellow star on top of the run stars, with a legend entry
if MAINSHOCK_ID in usgs.index:
    ax_map.scatter(usgs.loc[MAINSHOCK_ID, 'lon'], usgs.loc[MAINSHOCK_ID, 'lat'], marker='*', c='yellow', s=400,
                   edgecolors='black', linewidths=0.8, transform=proj, zorder=20, label='ANSS mainshock')

ax_map.set_xticks(np.linspace(extent[0], extent[1], 5), crs=proj)
ax_map.set_yticks(np.linspace(extent[2], extent[3], 5), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
ax_map.text(0.98, 0.02, f'EPIC locations at {N_TRIGS} stations', transform=ax_map.transAxes,
            ha='right', va='bottom', fontsize=13, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

out_stem = Path(__file__).resolve().parent / 'figures' / f'{SEQUENCE}_location_hist_{N_TRIGS}_comp'
fig.savefig(out_stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
print('Saved', out_stem, f'(figure {fig_w:.1f} x {fig_h:.1f} in, map {map_w:.1f} x {map_h:.1f} in)')

# %%
