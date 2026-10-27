#%%
# =============================================================================
# time_independent_comparison.py — paper figures: posterior location errors of
# the time-independent priors, drawn over each prior's density
# =============================================================================
# Two figures with identical styling, one per entry in FIGURES below:
#   - Ridgecrest 2019 case study
#   - California benchmark (large catalog)
#
# One map panel per prior (ETAS excluded for now). Each panel shows the prior
# density as a faded background, the MAP location of every event at
# TRIGGER_NUMBER triggers colored/sized by location-error bin, a thin line from
# each estimate to its USGS location, and (optionally) the station network.
#
# Adapted from plot_scripts/plot_comparison.py, "Figure 9"
# (error_map_{N}trigs_map.png).
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from benchmark import config
from benchmark.runner import load_reference_catalog
from priors import SeismicPrior

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'california')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_TRIGS      = 10     # which run's CSVs to read
TRIGGER_NUMBER = 4      # trigger count at which locations are compared
SHOW_STATIONS  = True   # triggered (orange) / active-untriggered (gray) stations
PRIOR_NAMES    = ['Gear1', 'NSHM', 'Helmstetter', 'KDE_Seismicity', 'Uniform']

column_lat = 'posterior_lat'
column_lon = 'posterior_lon'
column_err = 'map_err_km'

# Error bins (km): colors and marker sizes.
ERROR_BINS = [0, 10, 25, 50, 100, np.inf]
BIN_LABELS = ['< 10 km', '10–25 km', '25–50 km', '50–100 km', '≥ 100 km']
BIN_COLORS = ['#2196F3', '#66BB6A', '#FFA726', '#EF5350', '#7B1FA2']
BIN_SIZES  = [12, 28, 50, 75, 105]

# One entry per figure. The map extent is 'extent' [lon0, lon1, lat0, lat1] if
# given, else the data extent padded by 'buf' degrees.
FIGURES = [
    {
        'tag':      'Ridgecrest',
        'title':    'Ridgecrest 2019',
        'static':   os.path.join(RESULTS_DIR, 'case_studies', 'Ridgecrest', 'output',
                                 'time_independent', f'max_trigs_{MAX_TRIGS}'),
        'dynamic':  os.path.join(RESULTS_DIR, 'case_studies', 'Ridgecrest', 'output',
                                 'time_dependent', f'max_trigs_{MAX_TRIGS}'),
        'catalog':  os.path.join(DATA_DIR, 'case_studies', 'Ridgecrest', 'Ridgecrest_2019_catalog.parquet'),
        'run_dir':  os.path.join(DATA_DIR, 'case_studies', 'Ridgecrest', 'run_files'),
        'stations': os.path.join(DATA_DIR, 'case_studies', 'Ridgecrest', 'station_availability_cache.parquet'),
        'extent':   None,
        'buf':      0.5,
    },
    {
        'tag':      'benchmark',
        'title':    'California benchmark',
        'static':   os.path.join(RESULTS_DIR, 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}'),
        'dynamic':  os.path.join(RESULTS_DIR, 'output', 'time_dependent', f'max_trigs_{MAX_TRIGS}'),
        'catalog':  os.path.join(DATA_DIR, 'reference', 'bEPIC_testing_catalog.txt'),
        'run_dir':  os.path.join(DATA_DIR, 'run_files'),
        'stations': os.path.join(DATA_DIR, 'reference', 'station_availability_cache.parquet'),
        # Prior grids span -129..-112 E, 30..51 N; north cut to California.
        'extent':   [-129, -112, 30, 42.5],
        'buf':      None,
    },
]

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

#%%
# ── Helpers ─────────────────────────────────────────────────────────────────
def load_catalog(path):
    """USGS reference locations as (event_id, usgs_lat, usgs_lon); parquet, csv or bEPIC .txt."""
    if path.endswith('.txt'):
        cat = load_reference_catalog(path)
    else:
        read = pd.read_parquet if path.endswith('.parquet') else pd.read_csv
        cat = read(path).rename(columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'})
    cat = cat[['event_id', 'usgs_lat', 'usgs_lon']].copy()
    cat['event_id'] = cat['event_id'].astype(str)
    return cat


ETAS_BG_CACHE = os.path.join(OUTPUT_DIR, '.cache', 'row1.pkl')   # written by prior_comparison.py


def load_etas_background():
    """ETAS prior with the free_background spatial mu (panel (c) of prior_comparison.png), or None if not cached.

    It is the prior evaluated at the end of the inversion window, so it is dominated by the time-invariant
    spatial background; it is not the prior bEPIC used for any particular event.
    """
    if not os.path.exists(ETAS_BG_CACHE):
        print(f'  ETAS background cache not found (run prior_comparison.py first): {ETAS_BG_CACHE}')
        return None
    with open(ETAS_BG_CACHE, 'rb') as f:
        return pickle.load(f)[2]      # (kde_prior, prior_flat, prior_spatial, ...)


def load_prior_grid(name, tag):
    """Cached .tt3 for the background shading, or None (Uniform has no file)."""
    if name == 'ETAS':
        return load_etas_background()
    fn = f'kde_seismicity_{tag}.tt3' if name == 'KDE_Seismicity' else config.PRIOR_FILENAMES.get(name)
    if fn is None:
        return None
    path = os.path.join(SeismicPrior.data_dir, fn)
    return SeismicPrior.from_tt3(path) if os.path.exists(path) else None


def load_stations(cfg, event_ids):
    """Unique (triggered, untriggered) station positions across the events shown.

    Triggered = stations in the first version of each event that has exactly
    TRIGGER_NUMBER stations. Untriggered = availability-cache stations not triggered.
    """
    cols = ['station', 'network', 'longitude', 'latitude']
    frames = []
    for eid in sorted(event_ids):
        path = os.path.join(cfg['run_dir'], f'{eid}.run')
        if not os.path.exists(path):
            continue
        run = pd.read_csv(path)
        run.columns = [c.replace(' ', '_') for c in run.columns]
        counts = run.groupby('version').size()
        versions = counts[counts == TRIGGER_NUMBER].index
        if len(versions):
            frames.append(run[run['version'] == versions[0]][cols])
    trig = pd.concat(frames).drop_duplicates(subset=['station', 'network']) if frames else pd.DataFrame(columns=cols)

    if not os.path.exists(cfg['stations']):
        return trig, pd.DataFrame(columns=cols)
    avail = pd.read_parquet(cfg['stations'])
    avail = avail[avail['event_id'].astype(str).isin(event_ids)][cols].drop_duplicates(subset=['station', 'network'])
    untrig = avail.merge(trig[['station', 'network']], on=['station', 'network'],
                         how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
    return trig, untrig


def plot_prior_density(ax, prior, extent, proj, pad=0.5):
    """Faded log-scaled prior density clipped to the map extent."""
    lon_mask = (prior.lons >= extent[0] - pad) & (prior.lons <= extent[1] + pad)
    lat_mask = (prior.lats >= extent[2] - pad) & (prior.lats <= extent[3] + pad)
    if not (lon_mask.any() and lat_mask.any()):
        return
    grid = prior.grid[np.ix_(lat_mask, lon_mask)].astype(float)
    pos = grid[grid > 0]
    if len(pos) == 0:
        return
    vmin, vmax = np.percentile(pos, 5), np.percentile(pos, 99)
    if not (0 < vmin < vmax):
        return
    lons2d, lats2d = np.meshgrid(prior.lons[lon_mask], prior.lats[lat_mask])
    ax.pcolormesh(lons2d, lats2d, grid, transform=proj, cmap='YlOrBr', shading='auto',
                  alpha=0.35, norm=LogNorm(vmin=vmin, vmax=vmax), zorder=2)


def results_csv(cfg, name):
    """Results CSV for a prior: ETAS is the time-dependent run, the rest are time-independent."""
    if name == 'ETAS':
        return os.path.join(cfg['dynamic'], 'etas_dynamic_benchmark_results.csv')
    return os.path.join(cfg['static'], f'{name.lower()}_benchmark_results.csv')


def plot_comparison(cfg, names=None, out_prefix='time_independent_comparison'):
    """One map panel per prior in `names` (default PRIOR_NAMES), 3 columns, legend in the next free cell.
    ETAS is shaded with its free_background spatial prior (benchmark region only; the cache is for California)."""
    names = names or PRIOR_NAMES
    # -- Load results at TRIGGER_NUMBER, one DataFrame per prior ---------------
    map_data = {}
    for name in names:
        csv = results_csv(cfg, name)
        if not os.path.exists(csv):
            print(f'  [{cfg["tag"]}/{name}] CSV not found — skipping: {csv}')
            continue
        df = pd.read_csv(csv)
        df['event_id'] = df['event_id'].astype(str)
        sub = df[df['n_trigs'] == TRIGGER_NUMBER][['event_id', column_lat, column_lon, column_err]].dropna()
        if len(sub):
            map_data[name] = sub
    if not map_data:
        print(f'  [{cfg["tag"]}] no data — skipping figure')
        return

    ref_catalog = load_catalog(cfg['catalog'])
    all_pts = pd.concat(map_data.values())
    extent = cfg['extent'] or [all_pts[column_lon].min() - cfg['buf'], all_pts[column_lon].max() + cfg['buf'],
                               all_pts[column_lat].min() - cfg['buf'], all_pts[column_lat].max() + cfg['buf']]

    if SHOW_STATIONS:
        trig_stas, untrig_stas = load_stations(cfg, set(all_pts['event_id']))

    # -- Figure: 3 columns, the cell after the last panel holds the legend ----
    # Panels are equal-aspect in degrees, so size the figure from the map extent (no dead space between rows/title).
    proj = ccrs.PlateCarree()
    ncols = 3
    nrows = int(np.ceil((len(map_data) + 1) / ncols))
    panel_w = 15 * 0.30
    panel_h = panel_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig_h = nrows * (panel_h + 0.45) + 0.8      # + panel titles, x tick labels and the suptitle
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, fig_h), subplot_kw={'projection': proj})
    fig.subplots_adjust(left=0.05, right=0.99, wspace=0.05, hspace=0.16 * 4 / fig_h * 2,
                        top=1 - 0.65 / fig_h, bottom=0.45 / fig_h)
    axes = axes.flatten()

    for ax, (name, sub) in zip(axes, map_data.items()):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
        ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='#aaaaaa', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)

        prior = load_prior_grid(name, cfg['tag'])
        if prior is not None:
            plot_prior_density(ax, prior, extent, proj)

        # Line from each estimate to its USGS location
        matched = sub.merge(ref_catalog, on='event_id', how='inner')
        if not matched.empty:
            n = len(matched)
            seg_lons, seg_lats = np.full(n * 3, np.nan), np.full(n * 3, np.nan)
            seg_lons[0::3], seg_lons[1::3] = matched['usgs_lon'].values, matched[column_lon].values
            seg_lats[0::3], seg_lats[1::3] = matched['usgs_lat'].values, matched[column_lat].values
            ax.plot(seg_lons, seg_lats, color='black', linewidth=0.5, alpha=0.35,
                    transform=proj, zorder=4)
            ax.scatter(matched['usgs_lon'], matched['usgs_lat'], c='gray', s=10, alpha=0.2,
                       transform=proj, zorder=4)

        if SHOW_STATIONS:
            ax.scatter(untrig_stas['longitude'], untrig_stas['latitude'], s=10, color='lightgray',
                       marker='v', edgecolors='gray', linewidths=0.3, transform=proj, zorder=3)
            ax.scatter(trig_stas['longitude'], trig_stas['latitude'], s=15, color='orange',
                       marker='v', edgecolors='darkorange', linewidths=0.3, transform=proj, zorder=3)

        # Estimates, one scatter per error bin
        for i, (lo, hi) in enumerate(zip(ERROR_BINS[:-1], ERROR_BINS[1:])):
            pts = sub[(sub[column_err] >= lo) & (sub[column_err] < hi)]
            if len(pts):
                ax.scatter(pts[column_lon], pts[column_lat], c=BIN_COLORS[i], s=BIN_SIZES[i],
                           alpha=0.75, linewidths=0.3, edgecolors='white',
                           transform=proj, zorder=5)

        # Count of purple (>= 100 km) circles visible on this panel
        in_view = sub[sub[column_lon].between(*extent[:2]) & sub[column_lat].between(*extent[2:])]
        n_large = int((in_view[column_err] >= ERROR_BINS[-2]).sum())
        ax.text(0.97, 0.97, f'{ERROR_BINS[-2]:g}+ km errors: {n_large}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, zorder=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_title(name)

    # -- Shared ticks: labels only on the left column / bottom row ----------
    lon_ticks = [t for t in MaxNLocator(nbins=5).tick_values(extent[0], extent[1]) if extent[0] <= t <= extent[1]]
    lat_ticks = [t for t in MaxNLocator(nbins=5).tick_values(extent[2], extent[3]) if extent[2] <= t <= extent[3]]
    for idx, ax in enumerate(axes[:len(map_data)]):
        col = idx % 3
        ax.set_xticks(lon_ticks, crs=proj)
        ax.set_yticks(lat_ticks, crs=proj)
        ax.set_extent(extent, crs=proj)   # set_xticks can widen the view to include ticks
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=8, labelleft=(col == 0), labelbottom=(idx + 3 >= len(map_data)))

    # -- Legend in the unused cell(s) ----------------------------------------
    handles = [plt.scatter([], [], c=BIN_COLORS[i], s=BIN_SIZES[i], label=BIN_LABELS[i],
                           edgecolors='white', linewidths=0.3) for i in range(len(BIN_LABELS))]
    if SHOW_STATIONS:
        handles += [plt.scatter([], [], c='orange', s=15, marker='v', label='Triggered',
                                edgecolors='darkorange', linewidths=0.3),
                    plt.scatter([], [], c='lightgray', s=10, marker='v', label='Active (untriggered)',
                                edgecolors='gray', linewidths=0.3)]
    legend_ax = axes[len(map_data)]
    legend_ax.axis('off')
    legend_ax.legend(handles=handles, loc='center', fontsize=11, title='Location error', title_fontsize=12)
    for ax in axes[len(map_data) + 1:]:
        ax.set_visible(False)

    fig.suptitle(f'Posterior location errors at {TRIGGER_NUMBER} triggers — {cfg["title"]}', fontsize=14,
                 y=1 - 0.1 / fig_h)

    out = os.path.join(OUTPUT_DIR, f'{out_prefix}_{cfg["tag"]}')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f'Saved: {out}.png')
    plt.show()

#%%
# Usage: python time_independent_comparison.py [ridgecrest] [benchmark] [kde_vs_etas]   (default: all three)
which = [a.lower() for a in sys.argv[1:]] or ['ridgecrest', 'benchmark', 'kde_vs_etas']

if 'ridgecrest' in which:
    plot_comparison(FIGURES[0])

#%%
if 'benchmark' in which:
    plot_comparison(FIGURES[1])

#%%
# KDE_Seismicity vs ETAS only (California benchmark): two panels + legend
if 'kde_vs_etas' in which:
    plot_comparison(FIGURES[1], names=['KDE_Seismicity', 'ETAS'], out_prefix='kde_vs_etas_comparison')
