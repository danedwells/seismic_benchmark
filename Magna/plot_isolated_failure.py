#%%
# =============================================================================
# plot_isolated_failure.py — what the ETAS prior does at a badly located isolated event
# =============================================================================
# One event of the isolated Wasatch Front test (run_isolated.py), at TRIGGER_NUMBER triggers, with the ETAS prior as
# the colored background (log10 of prior / its median: white = the typical, flat level, red = bumps above it), the likelihood as
# black contours (50% / 90% of its peak), the triggered stations, the USGS location, where Uniform (likelihood only)
# puts the event, and where ETAS puts it. Same settings as the run: env switches in variant.py.
# Usage: MAGNA_NOCUT=1 MAGNA_BW_SQ=16 MAGNA_ISO_MAG=2.5 MAGNA_PRIOR_ALPHA=1 MAGNA_FLOOR_FRAC=0.005 python Magna/plot_isolated_failure.py [event_id | worst]
# =============================================================================
import contextlib
import copy
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import config_california as config
from benchmark import runner as benchmark_runner
from benchmark.metrics import _haversine_km
from benchmark.runner import run_single_event_get_grid
from faults import draw_wasatch
from variant import BW_SQ, CONTEXT, ISO_TAG, PRIOR_ALPHA, FLOOR_FRAC, PRIOR_TAG

MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(MAGNA_DIR, 'data')
ISO_DIR   = os.path.join(DATA_DIR, f'wasatch_isolated{ISO_TAG}')
RES_DIR   = os.path.join(MAGNA_DIR, 'results', f'output_isolated{ISO_TAG}{PRIOR_TAG}')
FIG_DIR   = os.path.join(MAGNA_DIR, 'results', 'figures')
TRIGGER_NUMBER = 4
BOUNDS = (-114.5, -109.0, 37.0, 42.5)
PARAMS = copy.deepcopy(config.BENCHMARK_PARAMS); PARAMS['sigma_s'] = 0.22

ETAS_CFG = copy.deepcopy(config.ETAS_INVERSION_CONFIG); ETAS_CFG.update({'mc': 2.5, 'm_ref': 2.5})
if BW_SQ is not None:
    ETAS_CFG['bw_sq'] = BW_SQ
ETAS_JSON = os.path.join(DATA_DIR, 'etas_inversion', f'parameters_{config.etas_output_id(CONTEXT, ETAS_CFG)}.json')
ETAS_HIST = os.path.join(DATA_DIR, 'etas_inversion', 'input', f'catalog_{config.etas_catalog_tag(CONTEXT, ETAS_CFG)}.csv')

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.dpi': 150, 'savefig.transparent': True})
proj = ccrs.PlateCarree()

#%%
cat = pd.read_parquet(os.path.join(ISO_DIR, 'catalog.parquet')).set_index('id')
arg = sys.argv[1] if len(sys.argv) > 1 else 'worst'
if arg == 'worst':                    # largest ETAS error at TRIGGER_NUMBER triggers
    e = pd.read_csv(os.path.join(RES_DIR, 'etas_benchmark_results.csv')).query('n_trigs == @TRIGGER_NUMBER').set_index('event_id')
    arg = e['map_err_km'].idxmax()
eid = arg
us = cat.loc[eid, ['latitude', 'longitude']].astype(float).values
t = pd.Timestamp(cat.loc[eid, 'time']) - pd.Timedelta(seconds=1)          # as run_isolated.py: strictly before the event
run_path = os.path.join(ISO_DIR, 'run_files', f'{eid}.run')

# ETAS prior exactly as in run_isolated.py (inverted background, causal to t, tempered + floored)
hist = pd.read_csv(ETAS_HIST, index_col=0, parse_dates=['time'])
benchmark_runner.repair_inversion_json_paths(ETAS_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=ETAS_JSON, catalog_df=hist,
                                               **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS}, spatial_factor=None)
import json
from etas.intensity import _compute_background_field
cj = json.load(open(ETAS_JSON)); ev = pd.read_csv(cj['fn_ip'], index_col=0, parse_dates=['time']); ev = ev[ev['time'] < t]
bg = _compute_background_field(updater.grid_lats_masked, updater.grid_lons_masked, ev['latitude'].values, ev['longitude'].values,
                               ev['P_background'].values, cj['bw_sq'], cj['timewindow_length'])
updater._bg_field = bg / bg.sum() * updater._bg_field.sum()
prior = updater.update(t)
g = prior.grid ** PRIOR_ALPHA if PRIOR_ALPHA != 1 else prior.grid
g = np.maximum(g, FLOOR_FRAC * g.max()); prior.grid = g / g.sum()

run = pd.read_csv(run_path)
# as the benchmark: "n triggers" = rows in a version (a station can trigger on more than one channel); first version with that many rows
VERSION = next(v for v in sorted(run['version'].unique()) if (run['version'] == v).sum() >= TRIGGER_NUMBER)

def locate(use_prior):
    with contextlib.redirect_stdout(io.StringIO()):
        _, o, _ = run_single_event_get_grid(run_path, prior, use_prior, dict(PARAMS), focus_version=VERSION)
    return o
o = locate(True)
size = int(round(np.sqrt(len(o))))
lat, lon, like, post, pri = (o[c].values.reshape(size, size) for c in ('lat', 'lon', 'like', 'post', 'prior'))
i_post = o['post'].values.argmax(); i_like = o['like'].values.argmax()
etas_pt = (o.lat.values[i_post], o.lon.values[i_post]); uni_pt = (o.lat.values[i_like], o.lon.values[i_like])
err = lambda p: float(_haversine_km(us[0], us[1], p[0], p[1]))
print(f'{eid} M{cat.loc[eid, "mag"]:.1f} {cat.loc[eid, "time"]}: ETAS {err(etas_pt):.0f} km, Uniform (likelihood only) {err(uni_pt):.0f} km')

stn = run[run['version'] == VERSION].drop_duplicates('station')

#%%
pts = np.array([us, etas_pt, uni_pt] + list(zip(stn['latitude'], stn['longitude'])))
pad = 0.25
ext = [pts[:, 1].min() - pad, pts[:, 1].max() + pad, pts[:, 0].min() - pad, pts[:, 0].max() + pad]
W, H = 8.2, 7.4
_a = W / H; _w, _h = ext[1] - ext[0], ext[3] - ext[2]                       # equal-aspect box so the map fills the axes
if _w / _h < _a:
    c = (ext[0] + ext[1]) / 2; ext[0], ext[1] = c - _a * _h / 2, c + _a * _h / 2
else:
    c = (ext[2] + ext[3]) / 2; ext[2], ext[3] = c - _w / _a / 2, c + _w / _a / 2

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.01, 0.01, 0.98, 0.98], projection=proj)
ax.set_extent(ext, crs=proj)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=-1)
med = np.median(pri)
ratio = np.log10(pri / med)                                                  # 0 = the typical (flat) level
ax.pcolormesh(lon, lat, ratio, cmap='Reds', norm=Normalize(0, max(1.0, ratio.max())), transform=proj, shading='auto', zorder=0)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='lightcyan', edgecolor='gray', linewidth=0.4, zorder=2)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='k', zorder=2)
draw_wasatch(ax, proj, zorder=3)
ax.contour(lon, lat, like / like.max(), levels=[0.5, 0.9], colors='k', linewidths=[1.0, 1.8], linestyles='-', transform=proj, zorder=4)
ax.scatter(stn['longitude'], stn['latitude'], marker='^', s=110, facecolor='white', edgecolor='k', linewidths=1.2, transform=proj, zorder=6)
for p, color, mk, lab in ((uni_pt, 'gray', 's', 'Uniform'), (etas_pt, 'red', 'o', 'ETAS')):
    ax.plot([us[1], p[1]], [us[0], p[0]], color=color, lw=1.8, ls='--', transform=proj, zorder=5)
    ax.plot(p[1], p[0], mk, ms=13, mfc=color, mec='k', mew=1.2, transform=proj, zorder=7)
ax.plot(us[1], us[0], '*', ms=26, mfc='gold', mec='k', mew=1.2, transform=proj, zorder=8)
handles = [Line2D([], [], marker='*', ls='', ms=16, mfc='gold', mec='k', label='USGS location'),
           Line2D([], [], marker='^', ls='', ms=10, mfc='white', mec='k', label=f'triggered stations ({TRIGGER_NUMBER} triggers)'),
           Line2D([], [], color='k', lw=1.8, label='likelihood (50%, 90% of peak)'),
           Line2D([], [], marker='s', ls='', ms=9, mfc='gray', mec='k', label=f'Uniform: {err(uni_pt):.0f} km'),
           Line2D([], [], marker='o', ls='', ms=9, mfc='red', mec='k', label=f'ETAS: {err(etas_pt):.0f} km')]
ax.legend(handles=handles, loc='lower left', fontsize=10, framealpha=0.92, edgecolor='none')
i_e = np.argmin(np.hypot(lat - etas_pt[0], lon - etas_pt[1])); i_t = np.argmin(np.hypot(lat - us[0], lon - us[1]))
bump_e, bump_t = pri.ravel()[i_e] / med, pri.ravel()[i_t] / med             # prior at the ETAS pick / at the true location, vs typical
ax.text(0.98, 0.98, f'ETAS prior (white = typical level, red = higher)\nat the ETAS pick: x{bump_e:.1f}   at the true location: x{bump_t:.1f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none'))
fig.savefig(os.path.join(FIG_DIR, f'isolated_failure_{eid}{ISO_TAG}{PRIOR_TAG}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, f'isolated_failure_{eid}{ISO_TAG}{PRIOR_TAG}.pdf'), bbox_inches='tight')
