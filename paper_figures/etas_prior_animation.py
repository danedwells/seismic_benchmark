#%%
# =============================================================================
# etas_prior_animation.py — presentation animation: the dynamic ETAS prior
# updating as California benchmark events happen.
# =============================================================================
# Replays the benchmark's time-dependent ETAS prior (EtasPriorUpdater, same
# parameters/config as time_dependent_scripts/run_benchmarks.py). Every event
# in the reference catalog is fed to the updater in time order at its USGS
# location (as in the benchmark), but frames are only drawn for events with
# USGS M > MIN_MAG. Two frames per drawn event:
#   1. "before": the prior bEPIC would use for that event (evaluated at the event time)
#   2. "after":  the prior one minute later, with the event added to the catalog
# Color is the raw prior density on a fixed log scale. The limits are set from the
# time-independent background (free_background spatial mu field): the floor is the background's
# dim end and the ceiling sits VMAX_FACTOR above the background's brightest cell, so the
# background stays visible but dim and recent events saturate the colormap.
#
# To keep it photosensitivity-friendly: the color scale is fixed for the whole animation, the colormap is
# capped below its near-white end (CMAP_TOP), and consecutive frames are cross-faded in log space
# (FADE_STEPS interpolated frames) instead of cutting, so no part of the map jumps in brightness.
#
# Output: etas_prior_animation.mp4 (best for PowerPoint) and .gif
# =============================================================================
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import EtasPriorUpdater, SeismicPrior
import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import load_reference_catalog

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

MIN_MAG   = 4.0
N_EVENTS  = None          # None = all events above MIN_MAG; an int renders just the first N (for testing)
EXTENT    = [-125.5, -114, 32, 42.5]
FPS       = 24
HOLD      = 2             # display frames each prior grid is held
FADE_STEPS = 4            # cross-fade display frames between consecutive grids
CMAP_TOP  = 0.8           # fraction of 'inferno' used (stops before the bright yellow-white end)
TRAIL     = 8             # number of previous events kept on the map (fading)
VMIN_PCT   = 10           # color floor: this percentile of the background prior
VMAX_FACTOR = 30          # color ceiling: this multiple of the background's maximum
AFTER_DT  = pd.Timedelta(minutes=1)
OUT_NAME  = 'etas_prior_animation'

config.ETAS_INVERSION_CONFIG['bw_sq'] = 4
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion',
                                  f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"])}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input',
                                  f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"])}.csv')
CATALOG_PATH       = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')

#%%
# ── Build the updater and replay the catalog ─────────────────────────────────
hist = pd.read_csv(HISTORICAL_CATALOG, index_col=0, dtype={'url': str, 'alert': str})
hist['time'] = pd.to_datetime(hist['time'], format='ISO8601', utc=True).dt.tz_convert(None)
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, catalog_df=hist,
                                               **config.ETAS_UPDATER_CONFIG, spatial_factor=None)

# Background-only prior (mu field, no events) on the same grid, for the ratio
bg_prior = SeismicPrior.from_etas(lats=updater.grid_lats_masked, lons=updater.grid_lons_masked,
                                  lambda_grid=np.asarray(updater._bg_field, dtype=float),
                                  forecast_time=pd.Timestamp('2019-01-01'), metadata=dict(updater.metadata_base),
                                  bounds=updater.bounds, out_of_bounds_fill=updater.out_of_bounds_fill)

cat = load_reference_catalog(CATALOG_PATH)
cat['time'] = pd.to_datetime(cat['usgs_time'], utc=True).dt.tz_convert(None)
cat = cat.sort_values('time').reset_index(drop=True)

frames = []     # dicts: grid, time, lat, lon, mag, stage, k (event number)
k = 0
for ev in cat.itertuples():
    big = ev.usgs_mag > MIN_MAG and (N_EVENTS is None or k < N_EVENTS)
    if big:
        pre = updater.update(ev.time)
        lats, lons = pre.lats, pre.lons
        sel_y = (lats >= EXTENT[2]) & (lats <= EXTENT[3])
        sel_x = (lons >= EXTENT[0]) & (lons <= EXTENT[1])
        frames.append(dict(grid=pre.grid[np.ix_(sel_y, sel_x)].astype(np.float32), stage='before'))
    updater.append_events(pd.DataFrame([{'time': ev.time, 'latitude': ev.usgs_lat,
                                         'longitude': ev.usgs_lon, 'magnitude': ev.usgs_mag}]))
    if big:
        post = updater.update(ev.time + AFTER_DT)
        frames.append(dict(grid=post.grid[np.ix_(sel_y, sel_x)].astype(np.float32), stage='after'))
        for fr in frames[-2:]:
            fr.update(time=ev.time, lat=ev.usgs_lat, lon=ev.usgs_lon, mag=ev.usgs_mag, k=k)
        k += 1
        if k % 20 == 0:
            print(f'  {k} events rendered into prior grids')
print(f'{len(frames)} frames from {k} events with M > {MIN_MAG}')

plot_lats, plot_lons = lats[sel_y], lons[sel_x]
all_vals = np.concatenate([f['grid'].ravel() for f in frames])
bg_ext = bg_prior.grid[np.ix_(sel_y, sel_x)]
norm = LogNorm(vmin=np.percentile(bg_ext[bg_ext > 0], VMIN_PCT), vmax=bg_ext.max() * VMAX_FACTOR)

events_shown = pd.DataFrame([dict(k=f['k'], lat=f['lat'], lon=f['lon'], mag=f['mag'])
                             for f in frames if f['stage'] == 'after'])

# Display sequence: hold each grid, then cross-fade (log space) into the next one
LOG_FLOOR = norm.vmin / 10
def blend(a, b, t):
    return np.exp((1 - t) * np.log(np.maximum(a, LOG_FLOOR)) + t * np.log(np.maximum(b, LOG_FLOOR))).astype(np.float32)

seq = []     # (grid, metadata frame)
for i, fr in enumerate(frames):
    seq += [(fr['grid'], fr)] * HOLD
    if i + 1 < len(frames):
        nxt = frames[i + 1]
        seq += [(blend(fr['grid'], nxt['grid'], (j + 1) / (FADE_STEPS + 1)), nxt) for j in range(FADE_STEPS)]

brightness = lambda g: float(np.mean(np.clip(norm(g), 0, 1)))
for name, grids in [('raw frames', [f['grid'] for f in frames]), ('cross-faded', [g for g, _ in seq])]:
    b = np.array([brightness(g) for g in grids])
    print(f'{name}: max frame-to-frame change in mean map brightness {np.abs(np.diff(b)).max():.4f} '
          f'(mean level {b.mean():.3f})')

#%%
# ── Draw ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8, 7), facecolor='black')
ax = fig.add_axes([0.02, 0.02, 0.86, 0.96], projection=proj)
ax.set_extent(EXTENT, crs=proj)
ax.set_facecolor('black')
ax.add_feature(cfeature.LAND,      facecolor='#15151a', zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor='black',   zorder=0)

img = ax.imshow(frames[0]['grid'], origin='lower', cmap=ListedColormap(plt.cm.inferno(np.linspace(0, CMAP_TOP, 256))), norm=norm, transform=proj,
                extent=[plot_lons[0], plot_lons[-1], plot_lats[0], plot_lats[-1]],
                interpolation='nearest', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#aaaaaa', zorder=3)
ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='#666666', zorder=3)

trail = ax.scatter([], [], s=[], facecolors='none', edgecolors='cyan', linewidths=1.2, transform=proj, zorder=5)
star  = ax.scatter([], [], marker='*', s=260, c='cyan', edgecolors='white', linewidths=1.0, transform=proj, zorder=6)
title = ax.text(0.03, 0.97, '', transform=ax.transAxes, ha='left', va='top', color='white',
                fontsize=15, fontweight='bold', zorder=7)
sub   = ax.text(0.03, 0.915, '', transform=ax.transAxes, ha='left', va='top', color='#dddddd',
                fontsize=12, zorder=7)

cax = fig.add_axes([0.91, 0.25, 0.02, 0.5])
cb = fig.colorbar(img, cax=cax)
cb.set_ticks([])
cb.ax.minorticks_off()
cax.text(0.5, 1.03, 'high', transform=cax.transAxes, ha='center', va='bottom', color='white', fontsize=11)
cax.text(0.5, -0.03, 'low', transform=cax.transAxes, ha='center', va='top', color='white', fontsize=11)
cax.text(-0.9, 0.5, 'ETAS prior', transform=cax.transAxes, ha='center', va='center',
         rotation=90, color='white', fontsize=11)


def draw(i):
    grid, f = seq[i]
    img.set_data(grid)
    past = events_shown[events_shown.k < f['k']].tail(TRAIL)
    trail.set_offsets(np.c_[past.lon, past.lat] if len(past) else np.empty((0, 2)))
    trail.set_sizes(20 * 3.0 ** (past.mag.values - MIN_MAG))
    trail.set_alpha(0.5)
    star.set_offsets([[f['lon'], f['lat']]])
    star.set_visible(True)
    tag = 'Prior at event time' if f['stage'] == 'before' else 'Event added → prior updated'
    title.set_text(f"{pd.Timestamp(f['time']).strftime('%Y-%m-%d %H:%M')}   M{f['mag']:.1f}")
    sub.set_text(tag)
    return img, trail, star, title, sub


anim = animation.FuncAnimation(fig, draw, frames=len(seq), blit=False)
mp4 = os.path.join(OUTPUT_DIR, f'{OUT_NAME}.mp4')
anim.save(mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                              extra_args=['-pix_fmt', 'yuv420p']), dpi=120)
print('saved', mp4)

# GIF (palette-optimized, narrower) from the mp4
gif = os.path.join(OUTPUT_DIR, f'{OUT_NAME}.gif')
subprocess.run(['ffmpeg', '-y', '-i', mp4, '-vf',
                f'fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                gif], check=True, capture_output=True)
print('saved', gif)
