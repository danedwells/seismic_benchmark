#%%
# =============================================================================
# improvement_map.py — paper figure: where does sigma calibration + ETAS help?
# =============================================================================
# California benchmark catalog binned into PIXEL_DEG pixels. Each pixel shows the
# median (or mean) change in MAP location error at the first location with >= TRIGGER_NUMBER stations,
#   calibrated (sigma_s = 0.35 + dynamic ETAS prior) minus
#   baseline   (sigma_s = 1.0 (EPIC default) + KDE_Seismicity prior),
# so negative (blue) = improvement. Pixels with fewer than MIN_EVENTS are blank.
#
# Second figure (improvement_map_MTJ): same quantity zoomed on the Mendocino Triple Junction
# (plus the active area to the southeast). Benchmark events are added to the MTJ and Ferndale
# case-study events for statistics. NOTE (presentation talking point, not shown in the figure):
# the case studies were run with the tuned sigma_s = 0.22, not the benchmark's 0.35, so the
# calibrated side of the case-study events uses sigma_s = 0.22 + dynamic ETAS. Their baselines:
#   Ferndale: sigma_s = 1.0 + KDE (same as the benchmark baseline)
#   MTJ:      sigma_s = 1.0 + KDE  (case_studies run with SIGMA_S = 1.0, KDE only)
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

from benchmark import config
from benchmark.runner import load_reference_catalog, rows_at_n_stations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output')

MAX_TRIGS      = 10
TRIGGER_NUMBER = 4                                    # stations (net.sta) from the .run files
RUN_DIR        = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
PIXEL_DEG      = 0.25
EXTENT         = [-125.5, -114, 32, 42.5]     # California (lon0, lon1, lat0, lat1)
MIN_EVENTS     = 2
STAT           = 'median'      # 'median' | 'mean'
RELATIVE       = True          # True: percent change instead of km

BASE_CSV  = os.path.join(RESULTS_DIR, 'time_independent', 'sig_1.0', f'max_trigs_{MAX_TRIGS}',
                         'kde_seismicity_benchmark_results.csv')
CALIB_CSV = os.path.join(RESULTS_DIR, 'time_dependent', 'sig_0.35', f'max_trigs_{MAX_TRIGS}',
                         'etas_dynamic_benchmark_results.csv')
CATALOG   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelsize': 11, 'figure.dpi': 150,
                     'savefig.transparent': True})
proj = ccrs.PlateCarree()

#%%
# ── Per-event error difference ───────────────────────────────────────────────
def errors_at_n(path, run_dir=RUN_DIR):
    df = rows_at_n_stations(pd.read_csv(path), run_dir, TRIGGER_NUMBER)   # first location with >= N stations
    return df.set_index(df['event_id'].astype(str))['map_err_km']

cat = load_reference_catalog(CATALOG)
cat['event_id'] = cat['event_id'].astype(str)
ev = pd.concat({'base': errors_at_n(BASE_CSV), 'calib': errors_at_n(CALIB_CSV)}, axis=1).dropna()
ev = ev.join(cat.set_index('event_id')[['usgs_lat', 'usgs_lon']], how='inner')
ev['delta'] = (100 * (ev.calib - ev.base) / ev.base) if RELATIVE else (ev.calib - ev.base)
ev_all = ev
ev = ev[ev.usgs_lon.between(*EXTENT[:2]) & ev.usgs_lat.between(*EXTENT[2:])].copy()
print(f'{len(ev)} events; median error {ev.base.median():.1f} -> {ev.calib.median():.1f} km; '
      f'{(ev.calib < ev.base).mean():.0%} improved')

#%%
# ── Pixelate and plot ────────────────────────────────────────────────────────
def plot_improvement_map(ev, extent, pixel_deg, min_events, figsize, out_name, annotate=False):
    ev = ev.copy()
    lon_edges = np.arange(extent[0], extent[1] + pixel_deg, pixel_deg)
    lat_edges = np.arange(extent[2], extent[3] + pixel_deg, pixel_deg)
    ev['ix'] = np.digitize(ev.usgs_lon, lon_edges) - 1
    ev['iy'] = np.digitize(ev.usgs_lat, lat_edges) - 1

    grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    for (iy, ix), g in ev.groupby(['iy', 'ix']):
        if len(g) >= min_events:
            grid[iy, ix] = g.delta.median() if STAT == 'median' else g.delta.mean()

    lim  = 1.0 if not RELATIVE else 50   # fixed color limit (km or %); saturates beyond
    unit = '%' if RELATIVE else 'km'
    fig  = plt.figure(figsize=figsize)
    ax   = fig.add_subplot(projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightcyan', zorder=0)
    cmap, norm = plt.get_cmap('RdBu_r'), TwoSlopeNorm(0, -lim, lim)
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid), cmap=cmap,
                         norm=norm, transform=proj, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=3)
    ax.scatter(ev.usgs_lon, ev.usgs_lat, s=3, color='k', alpha=0.3, transform=proj, zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.7, pad=0.08, extend='both')
    cb.set_label(f'{STAT.capitalize()} change in location error ({unit})')
    # "better / worse" read-out under the colorbar, each end in its own color (blue = error reduced)
    for txt, x, ha, v in (('\u2190 better', 0.0, 'left', -0.8 * lim), ('worse \u2192', 1.0, 'right', 0.8 * lim)):
        cb.ax.annotate(txt, xy=(x, 0), xycoords='axes fraction', xytext=(0, -48), textcoords='offset points',
                       ha=ha, va='top', fontsize=16, fontweight='bold', color=cmap(norm(v)))
    if annotate:   # reduction in the median error over all events plotted (positive = error reduced)
        # (not the median of per-event changes: ~25% of benchmark events are unchanged, which pins that at 0)
        km  = ev.base.median() - ev.calib.median()
        pct = 100 * km / ev.base.median()
        # colored by where this change falls on the colorscale (an improvement of pct is a change of -pct)
        ax.text(0.98, 0.97, f'Median improvement:\n{pct:.0f}%  ({km:.1f} km)', transform=ax.transAxes,
                ha='right', va='top', fontsize=14, fontweight='bold', color=cmap(norm(-(pct if RELATIVE else km))),
                bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.92, edgecolor='none'), zorder=10)   # dark box so pale blues stay readable
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.pdf'), bbox_inches='tight')
    plt.show()


plot_improvement_map(ev, EXTENT, PIXEL_DEG, MIN_EVENTS, (10, 11.5), 'improvement_map', annotate=True)

#%%
# ── MTJ zoom: benchmark events + MTJ_2024_M7 and Ferndale case studies ───────
MTJ_EXTENT     = [-126.0, -122.0, 38.4, 41.2]
MTJ_PIXEL_DEG  = 0.1
MTJ_MIN_EVENTS = 2
CS_DIR   = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies')
CS_DATA  = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies')
CS_ETAS  = os.path.join('time_dependent', 'sig_0.22', 'max_trigs_10', 'fb1_fp0_mc-fixed3_mref3_bw4',
                        'etas_dynamic_benchmark_results.csv')
# name -> (baseline KDE csv, catalog file); calibrated side is always sigma_s = 0.22 + ETAS
CASES = {
    'Ferndale':    (os.path.join('time_independent', 'sig_1.0',  'max_trigs_10', 'kde_seismicity_benchmark_results.csv'),
                    'Ferndale_2022_catalog.parquet'),
    'MTJ_2024_M7': (os.path.join('time_independent', 'sig_1.0',  'max_trigs_10', 'kde_seismicity_benchmark_results.csv'),
                    'mtj_2024_m7.csv'),
}
parts = [ev_all]
for name, (base_rel, cat_file) in CASES.items():
    cs_cat = pd.read_parquet(os.path.join(CS_DATA, name, cat_file)) if cat_file.endswith('.parquet') \
        else pd.read_csv(os.path.join(CS_DATA, name, cat_file))
    cs_cat = cs_cat.rename(columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'})
    cs_cat['event_id'] = cs_cat['event_id'].astype(str)
    e = pd.concat({'base':  errors_at_n(os.path.join(CS_DIR, name, 'output', base_rel), os.path.join(CS_DATA, name, 'run_files')),
                   'calib': errors_at_n(os.path.join(CS_DIR, name, 'output', CS_ETAS), os.path.join(CS_DATA, name, 'run_files'))}, axis=1).dropna()
    e = e.join(cs_cat.set_index('event_id')[['usgs_lat', 'usgs_lon']], how='inner')
    e['delta'] = (100 * (e.calib - e.base) / e.base) if RELATIVE else (e.calib - e.base)
    print(f'{name}: {len(e)} events; median error {e.base.median():.1f} -> {e.calib.median():.1f} km')
    parts.append(e)
ev_mtj = pd.concat(parts)
ev_mtj = ev_mtj[ev_mtj.usgs_lon.between(*MTJ_EXTENT[:2]) & ev_mtj.usgs_lat.between(*MTJ_EXTENT[2:])]
print(f'MTJ zoom: {len(ev_mtj)} events; median error {ev_mtj.base.median():.1f} -> {ev_mtj.calib.median():.1f} km')

plot_improvement_map(ev_mtj, MTJ_EXTENT, MTJ_PIXEL_DEG, MTJ_MIN_EVENTS, (11, 8.5), 'improvement_map_MTJ', annotate=True)

#%%
# ── Tighter zoom on the MTJ itself (no southeast cluster) ────────────────────
ZOOM_EXTENT = [-125.7, -123.8, 40.0, 40.8]
ev_zoom = ev_mtj[ev_mtj.usgs_lon.between(*ZOOM_EXTENT[:2]) & ev_mtj.usgs_lat.between(*ZOOM_EXTENT[2:])]
print(f'MTJ tight zoom: {len(ev_zoom)} events; median error {ev_zoom.base.median():.1f} -> {ev_zoom.calib.median():.1f} km')
plot_improvement_map(ev_zoom, ZOOM_EXTENT, 0.05, MTJ_MIN_EVENTS, (11, 7), 'improvement_map_MTJ_zoom', annotate=True)
