#%%
# =============================================================================
# improvement_map_logs.py — copy of paper_figures/improvement_map.py using the EPIC log results
# =============================================================================
# Same map, but the per-event errors come from the log-file replays instead of the benchmark CSVs:
#   calibrated = sigma_and_prior run (sigma_s calibration + ETAS prior)
#   baseline   = control run
# Source: output/log_results_<SEQUENCE>_nS<N>.csv (written by run_error_histograms_comp.py; one row per
# event and run: first alerted version with >= N stations, USGS M >= 3.5). Events missing from either run are dropped.
# Each pixel shows the median change in location error, so negative (blue) = improvement.
#
# TODO (PLACEHOLDER DATA): the MTJ + Ridgecrest replays above are only a stand-in. The real comparison is
# the CONTROL run vs the SIGMA_AND_PRIOR run, both re-run WITH station masking, on the full California
# benchmark dataset. Those log files don't exist yet. When they do: produce the per-event results
# (run_error_histograms_comp.py or equivalent), then update SEQUENCES / LOG_CSVS below to point at them.
# Until then the numbers on this figure are NOT the paper result.
# =============================================================================
import os

import numpy as np
from benchmark import parsing_tools
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm


OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SEQUENCES      = ['MTJ', 'Ridgecrest']                # TODO: placeholder; replace with the masked California-benchmark log results (see header)
N_TRIGS        = 4                                    # stations; matches the log_results_*_nS<N>.csv files
PIXEL_DEG      = 0.25
EXTENT         = [-125.5, -114, 32, 42.5]     # California (lon0, lon1, lat0, lat1)
MIN_EVENTS     = 2
STAT           = 'median'      # 'median' | 'mean'
RELATIVE       = True          # True: percent change instead of km
PLACEHOLDER    = True          # stamp 'PLACEHOLDER DATA' on the figure; set False once the masked California-benchmark logs are in

LOG_CSVS  = [os.path.join(OUTPUT_DIR, 'output', f'log_results_{seq}_nS{N_TRIGS}.csv') for seq in SEQUENCES]

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelsize': 11, 'figure.dpi': 150,
                     'savefig.transparent': True})
proj = ccrs.PlateCarree()

#%%
# ── Per-event error difference ───────────────────────────────────────────────
ev = pd.concat([pd.read_csv(p) for p in LOG_CSVS])
ev = parsing_tools.pick_version(ev, N_TRIGS)       # first alerted version with nS >= N_TRIGS, per event and run
ev = ev.pivot(index='anss_id', columns='run', values='loc_err_km').dropna(subset=['control', 'sigma_and_prior'])
ev = ev.rename(columns={'control': 'base', 'sigma_and_prior': 'calib'})
ev = ev.join(pd.concat([pd.read_csv(p) for p in LOG_CSVS]).drop_duplicates('anss_id').set_index('anss_id')[['usgs_lat', 'usgs_lon']])
ev['delta'] = (100 * (ev.calib - ev.base) / ev.base) if RELATIVE else (ev.calib - ev.base)
ev = ev[ev.usgs_lon.between(*EXTENT[:2]) & ev.usgs_lat.between(*EXTENT[2:])].copy()
print(f'{len(ev)} events; median error {ev.base.median():.1f} -> {ev.calib.median():.1f} km; '
      f'{(ev.calib < ev.base).mean():.0%} improved')

#%%
# ── Pixelate and plot ────────────────────────────────────────────────────────
def plot_improvement_map(ev, extent, pixel_deg, min_events, figsize, out_name, annotate=False, inset=False):
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
    if inset:   # small colorbar + histogram of the per-event changes, inside the map over the ocean (lower left)
        cax = ax.inset_axes([0.04, 0.14, 0.32, 0.022], zorder=10)
        hax = ax.inset_axes([0.04, 0.20, 0.32, 0.12],  zorder=10)
        cb  = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both')
        cb.ax.tick_params(labelsize=10)
        cb.set_label(f'{STAT.capitalize()} change in location error ({unit})', fontsize=10)
        vals = np.clip(ev.delta.values, -lim, lim)    # one value per event; beyond +-lim saturates, like the colorbar
        edges = np.linspace(-lim, lim, 21)
        counts, _ = np.histogram(vals, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hax.bar(centers, counts, width=np.diff(edges), color=cmap(norm(centers)), edgecolor='k', linewidth=0.3)
        hax.axvline(0, color='k', linewidth=1.2)
        hax.axvline(np.median(vals), color='k', linestyle='--', linewidth=1.2)
        hax.set_xlim(-lim, lim)
        hax.set_xticks([])
        hax.set_yticks([])
        hax.set_ylabel('Events', fontsize=10)
        hax.patch.set_alpha(0.85)
        hax.spines[['top', 'right']].set_visible(False)
    else:
        cb = fig.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.7, pad=0.08, extend='both')
        cb.set_label(f'{STAT.capitalize()} change in location error ({unit})')
    # "better / worse" read-out under the colorbar, each end in its own color (blue = error reduced)
    for txt, x, ha, v in (('\u2190 better', 0.0, 'left', -0.8 * lim), ('worse \u2192', 1.0, 'right', 0.8 * lim)):
        cb.ax.annotate(txt, xy=(x, 0), xycoords='axes fraction', xytext=(0, -56 if inset else -48), textcoords='offset points',
                       ha=ha, va='top', fontsize=13 if inset else 16, fontweight='bold', color=cmap(norm(v)))
    if annotate:   # reduction in the median error over all events plotted (positive = error reduced)
        # (not the median of per-event changes: ~25% of benchmark events are unchanged, which pins that at 0)
        km  = ev.base.median() - ev.calib.median()
        pct = 100 * km / ev.base.median()
        # colored by where this change falls on the colorscale (an improvement of pct is a change of -pct)
        ax.text(0.98, 0.97, f'Median improvement:\n{pct:.0f}%  ({km:.1f} km)', transform=ax.transAxes,
                ha='right', va='top', fontsize=14, fontweight='bold', color=cmap(norm(-(pct if RELATIVE else km))),
                bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.92, edgecolor='none'), zorder=10)   # dark box so pale blues stay readable
    if PLACEHOLDER:
        ax.text(0.02, 0.97, 'PLACEHOLDER DATA', transform=ax.transAxes, ha='left', va='top', fontsize=18,
                fontweight='bold', color='firebrick', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                edgecolor='firebrick'), zorder=10)
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.pdf'), bbox_inches='tight')
    plt.show()


plot_improvement_map(ev, EXTENT, PIXEL_DEG, MIN_EVENTS, (10, 11.5), 'improvement_map_logs', annotate=True, inset=True)
