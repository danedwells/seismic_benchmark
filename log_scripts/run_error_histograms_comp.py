"""
Comparison of location and magnitude error histograms across the control, sigma_only and
sigma_and_prior EPIC replay runs of one sequence, overlaid (see run_error_histograms.py
for the single-run version).

Reads every <anss_id>.log in a run folder, takes the first EPIC version whose
station count (nS, the E:I: column) reaches N_TRIGS, and compares its location and
magnitude to the USGS/ANSS solution.  Logs with no event reaching N_TRIGS are skipped.
USGS solutions are cached to a CSV so re-runs don't re-query ComCat.
"""
#%%
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.usgs import get_usgs_event

SEQUENCE = 'MTJ' #'Ridgecrest' # 'MTJ'
RUNS = {'control': 'gray', 'sigma_only': 'blue', 'sigma_and_prior': 'red'}

N_TRIGS = 4
MAX_ORIGIN_DIFF_S = 30          # match log event to USGS origin time within this
ALPHA = 0.3
MIN_USGS_MAG = 4.            # only compare events with USGS magnitude >= this (None = all)
ALERTS_ONLY = True              # keep only E:I: lines with Ast == 1 (alert actually sent)
AST_COL = 42                    # index of the Ast column in the E:I: line
USGS_CACHE = Path(__file__).resolve().parent / f'usgs_cache_{SEQUENCE}.csv'

#%% Parse logs: one row (event, version) per E:I: line with eventid > 0
def parse_log(path, alerts_only=ALERTS_ONLY):
    rows = []
    for line in path.open(errors='replace'):
        _, _, msg = line.partition('| INFO | E:I:')
        if msg.startswith('F:'):       # 'E:I:F:' = the event's first alert version (same columns as E:I:)
            msg = msg[2:]
        f = msg.split()
        if len(f) < 17 or not f[0].lstrip('-').isdigit() or int(f[0]) <= 0:
            continue
        if alerts_only and (len(f) <= AST_COL or f[AST_COL] != '1'):
            continue
        rows.append((int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[5]),
                     pd.Timestamp(f[6]), int(f[15]), int(f[16])))
    return pd.DataFrame(rows, columns=['eventid', 'ver', 'lat', 'lon', 'mag', 'time', 'nT', 'nS'])

all_logs = {}
for which in RUNS:
    run_dir = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{which}' / 'epic'
    all_logs[which] = {p.stem: parse_log(p) for p in sorted(run_dir.glob('*.log'))}
    logs = all_logs[which]
    print(f'{which}: {len(logs)} logs; {sum(len(d) > 0 for d in logs.values())} contain an alerted event')

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
todo = sorted({i for logs in all_logs.values() for i, d in logs.items() if len(d)} - set(usgs.anss_id))
if todo:
    with ThreadPoolExecutor(8) as ex:
        new = [r for r in ex.map(fetch_usgs, todo) if r]
    usgs = pd.concat([usgs, pd.DataFrame(new)], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.set_index('anss_id')

#%% Errors at the first version with nS >= N_TRIGS
results = {}
for which, logs in all_logs.items():
    out = []
    for anss_id, d in logs.items():
        if not len(d) or anss_id not in usgs.index:
            continue
        u = usgs.loc[anss_id]
        if MIN_USGS_MAG is not None and not u.mag >= MIN_USGS_MAG:
            continue
        d = d[d.nS >= N_TRIGS]
        if d.empty:
            continue
        # Several events can be alerted in one log; take the one whose origin matches USGS
        dt = (d.time - u.time).abs().dt.total_seconds()
        d = d[dt == dt.min()]
        if dt.min() > MAX_ORIGIN_DIFF_S:
            continue
        r = d.sort_values('ver').iloc[0]
        out.append(dict(anss_id=anss_id, ver=r.ver, nS=r.nS, usgs_mag=u.mag,
                        loc_err_km=gps2dist_azimuth(u.lat, u.lon, r.lat, r.lon)[0] / 1e3,
                        mag_err=r.mag - u.mag, lat=r.lat, lon=r.lon, usgs_lat=u.lat, usgs_lon=u.lon))
    results[which] = pd.DataFrame(out)
    print(f'{which}: {len(results[which])} events with a matched version at nS >= {N_TRIGS}'
          + (f' and USGS M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''))
    print(results[which][['loc_err_km', 'mag_err']].describe())

#%% Plot
# Shared bins so the three runs are directly comparable
loc_all = pd.concat([r.loc_err_km for r in results.values()])
mag_all = pd.concat([r.mag_err for r in results.values()])
loc_bins = np.logspace(np.log10(loc_all.min()), np.log10(loc_all.max()), 41)
mag_bins = np.linspace(mag_all.min(), mag_all.max(), 41)

BIN_OFFSET = 0.15   # per-run shift, as a fraction of bin width, so overlapping bars stay distinguishable
BAR_WIDTH = 0.7     # bar width as a fraction of bin width

def offset_hist(ax, values, bins, k, color, label, log=False):
    """Histogram drawn as bars shifted by k*BIN_OFFSET bin widths (in log10 space if log)."""
    counts, edges = np.histogram(values, bins=bins)
    u = np.log10(edges) if log else edges
    w = np.diff(u)
    left = u[:-1] + (0.5 - BAR_WIDTH / 2 + k * BIN_OFFSET) * w
    right = left + BAR_WIDTH * w
    if log:
        left, right = 10 ** left, 10 ** right
    ax.bar(left, counts, width=right - left, align='edge', color=color, alpha=ALPHA, label=label)

fig, (ax_loc, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.5))
for i, (which, color) in enumerate(RUNS.items()):
    res = results[which]
    k = i - (len(RUNS) - 1) / 2
    label = f'{which} (n={len(res)})'
    offset_hist(ax_loc, res.loc_err_km, loc_bins, k, color, label, log=True)
    offset_hist(ax_mag, res.mag_err, mag_bins, k, color, label)
    for ax, med in ((ax_loc, res.loc_err_km.abs().median()), (ax_mag, res.mag_err.abs().median())):
        unit = ' km' if ax is ax_loc else ''
        ax.text(0.02, 0.97 - 0.07 * i, f'{which}: median |err| = {med:.2f}{unit}', transform=ax.transAxes,
                color=color, ha='left', va='top', fontsize=9)
for ax in (ax_loc, ax_mag):
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)   # headroom for annotations/legend
ax_loc.set_xscale('log')
ax_loc.set_xlabel('Location error vs USGS (km)'); ax_loc.set_ylabel('Events')
mag_note = f', M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''
ax_loc.set_title(f'Location error (nS >= {N_TRIGS}{mag_note})')
ax_loc.legend(loc='upper right')
ax_mag.axvline(0, color='k', ls='--')
ax_mag.set_xlabel('EPIC mag - USGS mag'); ax_mag.set_ylabel('Events')
ax_mag.set_title(f'Magnitude error (nS >= {N_TRIGS}{mag_note})')
ax_mag.legend(loc='upper right')
fig.tight_layout()
out_path = Path(__file__).resolve().parent / 'figures' / f'{SEQUENCE}_error_hist_{N_TRIGS}_comp.png'
fig.savefig(out_path, dpi=150)
print('Saved', out_path)

#%% Paper-style figure (same layout/style as paper_figures/location_hist_case_study.py):
# (a) map of EPIC locations per run with arrows to the USGS location, (b) stacked location-error histograms
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()
SIZE_SCALE = dict(zip(RUNS, (1.6, 1.3, 1.1)))   # decreasing with draw order so overlapping dots read as concentric
BUF = 0.2                                     # degrees added around the posterior locations
MIN_MAP_ASPECT = 6.3 / 7.3                      # map box width/height; tall regions are widened east-west
AX_POS = {'map': [0.05, 0.08, 0.45, 0.86], 'hists': [0.58, 0.183, 0.17, 0.654], 'mags': [0.82, 0.183, 0.17, 0.654]}

fig = plt.figure(figsize=(14, 8.5))

# -- Right: stacked location-error histograms, one per run, shared x-axis --
bins = np.logspace(-1, 3, 20)
hl, hb, hw, hh = AX_POS['hists']
h_each = hh / len(RUNS)
hist_axes = []
for i, (which, color) in enumerate(RUNS.items()):
    ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each], sharex=hist_axes[0] if hist_axes else None)
    hist_axes.append(ax)
    stats = results[which].loc_err_km.values
    if len(stats):
        ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7)
        median_err = np.median(stats)
        ax.axvline(median_err, ymax=0.55, color=color, ls='--', lw=1.5)   # stops above the bars, below the labels
        ax.text(0.03, 0.70, f'median: {median_err:.1f} km', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.text(0.03, 0.92, which, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, color=color, fontweight='bold')
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(RUNS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
hist_axes[-1].set_xscale('log')
hist_axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
hist_axes[-1].set_xlabel('Location error (km)')
_y_max = max(a.get_ylim()[1] for a in hist_axes)
for a in hist_axes:
    a.set_ylim(0, 2.0 * _y_max)   # headroom: bars top out at half height so the name/median labels sit clear above them
hist_axes[0].text(-0.2, 0.87, '(b)', transform=hist_axes[0].transAxes,
                  ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- Middle-right: stacked magnitude-error histograms (same style, shared bins) --
ml, mb, mw, mh = AX_POS['mags']
m_each = mh / len(RUNS)
mag_axes = []
for i, (which, color) in enumerate(RUNS.items()):
    ax = fig.add_axes([ml, mb + mh - (i + 1) * m_each, mw, m_each], sharex=mag_axes[0] if mag_axes else None)
    mag_axes.append(ax)
    stats = results[which].mag_err.values
    if len(stats):
        ax.hist(stats, bins=mag_bins, rwidth=0.9, color=color, alpha=0.7)
        median_err = np.median(stats)
        ax.axvline(median_err, ymax=0.55, color=color, ls='--', lw=1.5)
        ax.text(0.03, 0.70, f'median: {median_err:+.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.text(0.03, 0.92, which, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, color=color, fontweight='bold')
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(RUNS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
mag_axes[-1].set_xlabel('EPIC mag − USGS mag')
_y_max = max(a.get_ylim()[1] for a in mag_axes)
for a in mag_axes:
    a.set_ylim(0, 2.0 * _y_max)
mag_axes[0].text(-0.2, 0.87, '(c)', transform=mag_axes[0].transAxes,
                 ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- Left: combined map of EPIC locations, colored by run --
ax_map = fig.add_axes(AX_POS['map'], projection=proj)
all_lats = pd.concat([r.lat for r in results.values()])
all_lons = pd.concat([r.lon for r in results.values()])
extent = [all_lons.min() - BUF, all_lons.max() + BUF, all_lats.min() - BUF, all_lats.max() + BUF]
if (extent[1] - extent[0]) < MIN_MAP_ASPECT * (extent[3] - extent[2]):
    _half = 0.5 * MIN_MAP_ASPECT * (extent[3] - extent[2])
    _lon_c = 0.5 * (extent[0] + extent[1])
    extent[0], extent[1] = _lon_c - _half, _lon_c + _half
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
                   edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=which)
usgs_pts = pd.concat([r[['anss_id', 'usgs_lat', 'usgs_lon']] for r in results.values()]).drop_duplicates('anss_id')
ax_map.scatter(usgs_pts.usgs_lon, usgs_pts.usgs_lat, marker='x', c='black', s=35, linewidths=1.1,
               transform=proj, zorder=6, alpha=0.5, label='USGS (true)')

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
print('Saved', out_stem)

# %%
