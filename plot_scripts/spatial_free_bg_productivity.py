#%%
# =============================================================================
# spatial_free_bg_productivity.py — visualize free_background / free_productivity
# =============================================================================
# Three diagnostics comparing the flat-mu / population-law ETAS prior against
# the spatially-varying fields now available in
# data/etas_inversion/parameters_benchmark.json (free_background=True,
# free_productivity=True, store_spatial_fields=True):
#
#   1. Background field mu(x, y) vs the flat scalar mu — shows where
#      free_background's locally-smoothed background rate departs from
#      uniform.
#   2. Per-event fitted source_kappa, plotted directly (it's a per-event EM
#      estimate driven by each event's own observed aftershock count, not a
#      smooth function of magnitude — see the note in cell 2 for why a ratio
#      to the population law isn't a useful comparison here).
#   3. "Local productivity tendency" — a Nadaraya-Watson kernel-smoothed
#      average of log10(source_kappa), i.e. cell 2's scatter smoothed into a
#      continuous field. Diagnostic only: EtasPriorUpdater.update() does not
#      use a smoothed kappa field anywhere (see _compute_smoothed_field's
#      docstring in etas_2/etas/intensity.py for why productivity isn't
#      currently modeled as a spatial field at runtime).
#   4. Full evaluated prior (flat vs spatial) at the inversion's
#      timewindow_end — the combined, end-to-end effect on what bEPIC
#      actually sees.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage

from etas.intensity import _compute_smoothed_field
from priors import EtasPriorUpdater
from benchmark import config

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make sure these files are up to date
_ETAS_ID       = config.etas_output_id('benchmark')
INVERSION_JSON = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', f'parameters_{_ETAS_ID}.json')
SOURCES_CSV    = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', f'sources_{_ETAS_ID}.csv')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion')

proj = ccrs.PlateCarree()


def _add_basemap(ax):
    ax.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray',   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightyellow', zorder=0)


#%%
# ── Build updater with both spatial fields enabled ───────────────────────────
_spatial_cfg = dict(config.ETAS_UPDATER_CONFIG)
_spatial_cfg['use_spatial_background']   = True
_spatial_cfg['use_spatial_productivity'] = False

updater = EtasPriorUpdater.from_inversion_json(
    json_path = INVERSION_JSON,
    **_spatial_cfg,
)
theta = updater.theta
print(updater)

#%%
# ── 1. Spatial background field mu(x, y) vs flat mu ──────────────────────────
flat_mu  = 10 ** theta['log10_mu']
bg_field = updater._bg_field
ratio_bg = bg_field / flat_mu

# Many far-field grid points underflow to exact float32 zero under the tight
# Gaussian bandwidth (bw_sq) — add a floor before log10 so those don't show
# up as -inf and blow out the color scale.
eps = 1e-12

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': proj})
for ax in axes:
    _add_basemap(ax)

sc0 = axes[0].scatter(updater.grid_lons_masked, updater.grid_lats_masked,
                       c=np.log10(bg_field + eps), cmap='viridis', s=12,
                       transform=proj)
axes[0].set_title('Spatial background  log10 mu(x, y)')
plt.colorbar(sc0, ax=axes[0], shrink=0.7, label='log10 mu')

sc1 = axes[1].scatter(updater.grid_lons_masked, updater.grid_lats_masked,
                       c=np.log10(ratio_bg + eps), cmap='RdBu_r', s=12,
                       vmin=-3, vmax=3, transform=proj)
axes[1].set_title('log10( mu(x,y) / flat mu )')
plt.colorbar(sc1, ax=axes[1], shrink=0.7, label='log10 ratio')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'free_background_spatial.png'), dpi=150)
plt.show()

#%%
# ── 2. Per-event productivity: fitted source_kappa ────────────────────────────
# NOTE: source_kappa is a per-event EM estimate (update_source_kappa in
# etas_2/etas/inversion.py: kappa_new = kappa_old * l_hat / G), driven by how
# many aftershocks THAT SPECIFIC event triggered in the data — not a smooth
# function of magnitude. It spans ~60 orders of magnitude (many events
# triggered zero observed aftershocks and converge toward 0) with a median
# around 1e-26, so comparing it against the population-law k0*exp(a*(m-mc))
# via a ratio just amplifies per-event sampling noise into a meaningless
# signal. Plotting log10(source_kappa) directly (as for mu(x,y) above) is
# the informative view — same reasoning as the background field.
src = pd.read_csv(SOURCES_CSV)

vlo, vhi = np.percentile(np.log10(src['source_kappa'] + eps), [5, 95])

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})
_add_basemap(ax)

sc = ax.scatter(src['longitude'], src['latitude'],
                 c=np.log10(src['source_kappa'] + eps), cmap='viridis',
                 vmin=vlo, vmax=vhi, s=10, alpha=0.6, transform=proj)
ax.set_title('log10( fitted source_kappa )  [5th-95th pct color range]')
plt.colorbar(sc, ax=ax, shrink=0.7, label='log10 source_kappa')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'free_productivity_spatial.png'), dpi=150)
plt.show()

#%%
# ── 3. Local productivity tendency: kernel-smoothed log10(source_kappa) ──────
# Nadaraya-Watson kernel average (see _compute_smoothed_field in
# etas_2/etas/intensity.py) — smooths cell 2's discrete per-event scatter
# into a continuous field so areas of systematically higher/lower fitted
# productivity are visible independent of exactly where events happened to
# occur. Smoothing is done in log10 space so the handful of very large
# kappa outliers don't dominate the local average the way they would in
# linear space.
#
# bw_sq here is reused from the inversion's free_background bandwidth —
# there's no fitted bandwidth for productivity smoothing (update_source_kappa
# has no spatial kernel at all), so this is a free diagnostic parameter, not
# a model-fitted one. Reconsider it before reading too much into fine
# spatial structure.
with open(INVERSION_JSON) as _fh:
    bw_sq = json.load(_fh)['bw_sq']

# Productivity specific
bandwidth = 20

log_kappa_smooth = _compute_smoothed_field(
    updater.grid_lats_masked, updater.grid_lons_masked,
    src['latitude'].values, src['longitude'].values,
    np.log10(src['source_kappa'].values + eps),
    bandwidth,
    # values passed in are already log10-space, so the no-nearby-data
    # fallback must be too (log10(eps) = -12) — the function's own default
    # (1e-12) is meant for linear-space callers like _compute_background_field
    # and would read as ~0, i.e. near the TOP of this color scale, not the
    # bottom, if left unset here.
    eps=np.log10(eps),
)

# Extra Gaussian pass on top of the NW-smoothed field above. grid_lats_masked /
# grid_lons_masked are a polygon-masked flatten of a regular meshgrid (see
# EtasPriorUpdater.from_inversion_json), not stored as 2D, so reconstruct the
# grid, smooth, and re-flatten. Smoothed sum/weight (rather than filling
# outside-mask cells with 0 and smoothing directly) keeps the field from
# darkening near the polygon edge, where a plain gaussian_filter would blend
# in the artificial zeros.
grid_spacing_deg = config.ETAS_UPDATER_CONFIG['grid_spacing']
sigma_gridpts = 3  # ~0.3 deg smoothing radius at the 0.1 deg grid spacing

uniq_lats = np.unique(updater.grid_lats_masked)
uniq_lons = np.unique(updater.grid_lons_masked)
lat_idx = np.searchsorted(uniq_lats, updater.grid_lats_masked)
lon_idx = np.searchsorted(uniq_lons, updater.grid_lons_masked)

field_2d = np.full((uniq_lats.size, uniq_lons.size), np.nan)
field_2d[lat_idx, lon_idx] = log_kappa_smooth
valid = ~np.isnan(field_2d)

smoothed_sum = ndimage.gaussian_filter(np.where(valid, field_2d, 0.0),
                                        sigma=sigma_gridpts, mode='constant', cval=0.0)
smoothed_weight = ndimage.gaussian_filter(valid.astype(float),
                                           sigma=sigma_gridpts, mode='constant', cval=0.0)
# smoothed_weight is ~0 far outside the polygon mask (no valid neighbors
# within sigma_gridpts) — those cells 0/0 to NaN, which is fine since only
# the masked-point values re-extracted below are ever used.
with np.errstate(invalid='ignore'):
    field_2d_smooth = smoothed_sum / smoothed_weight

log_kappa_smooth = field_2d_smooth[lat_idx, lon_idx]

vlo2, vhi2 = np.percentile(log_kappa_smooth, [5, 95])

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})
_add_basemap(ax)

sc = ax.scatter(updater.grid_lons_masked, updater.grid_lats_masked,
                 c=log_kappa_smooth, cmap='viridis',
                 vmin=vlo2, vmax=vhi2, s=12, transform=proj)
ax.set_title('Local productivity tendency  —  smoothed log10(source_kappa)')
plt.colorbar(sc, ax=ax, shrink=0.7, label='smoothed log10 source_kappa')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'free_productivity_smoothed.png'), dpi=150)
plt.show()

#%%
# ── 4. Full prior comparison: flat vs spatial, at timewindow_end ─────────────
forecast_time = pd.Timestamp(updater.metadata_base['timewindow_end'])

flat_cfg = dict(config.ETAS_UPDATER_CONFIG)
flat_cfg['use_spatial_background']   = False
flat_cfg['use_spatial_productivity'] = False
flat_updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, **flat_cfg)

prior_flat    = flat_updater.update(forecast_time)
prior_spatial = updater.update(forecast_time)

log_flat    = np.log10(prior_flat.grid + 1e-12)
log_spatial = np.log10(prior_spatial.grid + 1e-12)

# Share one color scale between the two density panels — with independent
# auto-scaling, the same out_of_bounds_fill constant (flat outside the ETAS
# polygon, which only covers CA + coastal OR) renders as different colors in
# each panel and makes the visual comparison misleading.
vlo, vhi = np.percentile(np.concatenate([log_flat.ravel(), log_spatial.ravel()]),
                          [1, 99.5])

# Zoom to the ETAS polygon's extent (with padding) rather than the full
# bEPIC search bounds — most of the latter is flat out_of_bounds_fill and
# just wastes frame area.
_pad = 1.0
_extent = [
    updater.grid_lons_masked.min() - _pad, updater.grid_lons_masked.max() + _pad,
    updater.grid_lats_masked.min() - _pad, updater.grid_lats_masked.max() + _pad,
]

fig, axes = plt.subplots(1, 3, figsize=(19, 6), subplot_kw={'projection': proj})
for ax in axes:
    ax.set_extent(_extent, crs=proj)
    _add_basemap(ax)

flat_mu = axes[0].pcolormesh(prior_flat.lons, prior_flat.lats, log_flat,
                    cmap='viridis', vmin=vlo, vmax=vhi, transform=proj)
axes[0].set_title('Flat mu')

spatial_axis = axes[1].pcolormesh(prior_spatial.lons, prior_spatial.lats, log_spatial,
                    cmap='viridis', vmin=vlo, vmax=vhi, transform=proj)
axes[1].set_title('Spatial mu')

diff = log_spatial - log_flat
pcm = axes[2].pcolormesh(prior_flat.lons, prior_flat.lats, diff,
                          cmap='RdBu_r', vmin=-3, vmax=3, transform=proj)
axes[2].set_title(f'log10 ratio  (spatial / flat) @ {forecast_time.date()}')
plt.colorbar(flat_mu, ax = axes[0], shrink=0.7)
plt.colorbar(spatial_axis, ax=axes[1], shrink=0.7)
plt.colorbar(pcm, ax=axes[2], shrink=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'free_bg_productivity_prior_comparison.png'), dpi=150)
plt.show()

# %%

