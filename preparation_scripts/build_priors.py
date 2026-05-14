#%%
# =============================================================================
# build_priors.py  —  one-time prior construction
# =============================================================================
# Builds and caches all prior .tt3 files from their raw source data.
#
# Run this script before run_benchmarks.py or case_studies.py.
# Re-run whenever source data changes or construction params are updated.
#
# Usage:
#   cd seismic_benchmark
#   python preparation_scripts/build_priors.py
# =============================================================================

# TODO - fix the construction of the KDE in benchmark/priors.py
# TODO - fix the consturction of the KDE in priors/src/prior_models.py
# TODO - test amy's implementation (here) vs adaptive (also here, also in above files)

import os
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde
import scipy.linalg
from scipy.stats._stats import gaussian_kernel_estimate
from datetime import datetime, timezone
from math import radians, sin, cos, pi

from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark import config
from benchmark import priors as utils

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

# Build all static priors - no variation based on time or anything

utils.build_and_cache_priors(cache_paths, data_dir, config.PRIOR_CONSTRUCTION_PARAMS)



#%% Do the kde seismicity here
# Build the Smoothed KDE seismicity maps - need to end before the applicable study.
print("Building smooth seismicity maps with KDE for 4 use case")


PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDE_CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'kde_catalogs', 'kde_base_seismicity.parquet')
_kde_construction = {
    **config.PRIOR_CONSTRUCTION_PARAMS,
    'kde_seismicity_params': config.KDE_SEISMICITY_PARAMS,
}

# All known contexts: name → cutoff date (exclusive upper bound for seismicity).
# The benchmark cutoff is the first event in the reference catalog 
# is a safe conservative value; adjust if you know the exact first event time).
CONTEXTS = {
    'benchmark':  '2018-09-30T00:00:00',   # first event in bEPIC reference catalog
    'Ridgecrest': '2019-07-04T17:00:00',   # Ridgecrest mainshock origin time
    'Ferndale':   '2022-12-20T10:00:00',   # Ferndale mainshock origin time
    'ElMayor':    '2010-04-04T22:00:00',   # El Mayor-Cucapah mainshock origin time
}

# Extracted from a function 
BUILD_CONTEXTS = list(CONTEXTS.keys())   # all four; edit to build a subset
context_name = BUILD_CONTEXTS[3]
cutoff = CONTEXTS[context_name]
construction_params = _kde_construction
force_rebuild=True
force_redownload=False
kde_start = config.KDE_START_DATE
kde_catalog_path = KDE_CATALOG_PATH


if construction_params is None:
    construction_params = {}

bounds     = construction_params.get('bounds', (-129, -112, 30, 51))
oob_fills  = construction_params.get('out_of_bounds_fill', {})
target_res = construction_params.get('target_resolution_deg', {})

tt3_path = os.path.join(data_dir, f'kde_seismicity_{context_name}.tt3')
cutoff_date = cutoff

os.makedirs(os.path.dirname(kde_catalog_path), exist_ok=True)

start_year  = pd.Timestamp(kde_start).year
end_year    = pd.Timestamp('today').year
kde_min_mag = 3.0
print(f"KDE Min Mag: {kde_min_mag}")

catalog = load_background_seismicity(
    cache_path    = kde_catalog_path,
    bounds        = bounds,
    start_year    = start_year,
    end_year      = end_year,
    min_mag       = kde_min_mag,
    force_refresh = force_redownload,
)

# Filter to strictly before the context cutoff
cutoff_ts = pd.Timestamp(cutoff_date)
if cutoff_ts.tzinfo is None:
    cutoff_ts = cutoff_ts.tz_localize('UTC')
else:
    cutoff_ts = cutoff_ts.tz_convert('UTC')
catalog = catalog[catalog['time'] < cutoff_ts]
catalog = catalog[catalog['mag'] >= kde_min_mag]

if len(catalog) == 0:
    raise ValueError(
        f"No events in KDE catalog before {cutoff_ts.date()} "
        f"(context: '{context_name}')."
    )


#%%
grid_size          = 200
bw_method          = 'scott'
# adaptive_alpha     = 0.5   # adaptive KDE — commented out
out_of_bounds_fill = 0.0001

lons_pts = catalog['longitude'].values.astype(float)
lats_pts = catalog['latitude'].values.astype(float)

if bounds is None:
    pad = 1.0
    bounds = (
        float(lons_pts.min()) - pad, float(lons_pts.max()) + pad,
        float(lats_pts.min()) - pad, float(lats_pts.max()) + pad,
    )
lon_min, lon_max, lat_min, lat_max = bounds

# Flat-Earth projection anchored at (lon_min, lat_min), matching Amy's AMY_construct_static_prior.py
_ref_lat_rad = radians(lat_min)
_R           = 6378137.0        # WGS84 semi-major axis (m)
_ff          = 1.0 / 298.257    # WGS84 flattening
_mpd         = _R * (1 - _ff * sin(_ref_lat_rad)**2) * pi / 180  # m per degree

def _to_km(lon, lat):
    x_km = (lon - lon_min) * _mpd * cos(_ref_lat_rad) / 1000
    y_km = (lat - lat_min) * _mpd / 1000
    return x_km, y_km

if np.isscalar(grid_size):
    nx, ny = int(grid_size), int(grid_size)
else:
    nx, ny = int(grid_size[0]), int(grid_size[1])

lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_min, lat_max, ny)
#%%

# from KDEpy import TreeKDE                    # adaptive KDE — commented out
# from sklearn.neighbors import KernelDensity  # adaptive KDE — commented out

grid_lons = lons
grid_lats = lats
# pilot_bw = bw_method   # adaptive KDE — commented out
# alpha = adaptive_alpha # adaptive KDE — commented out

pts_x_km, pts_y_km = _to_km(lons_pts, lats_pts)
points = np.column_stack([pts_x_km, pts_y_km])   # (N, 2) in km
n, d = points.shape

# --- Adaptive KDE (commented out — restore to switch back) ---
# # Scott's rule: n^(-1/(d+4)); Silverman: (n*(d+2)/4)^(-1/(d+4))
# if pilot_bw == 'scott':
#     h_global = n ** (-1.0 / (d + 4))
# elif pilot_bw == 'silverman':
#     h_global = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
# else:
#     h_global = float(pilot_bw)
# print(h_global)
#
# # Stage 1: pilot density at data points via ball-tree KDE — O(N log N)
# # vs. scipy gaussian_kde which is O(N²) and dominates runtime for large N
# f_pilot = np.exp(
#     KernelDensity(kernel='gaussian', bandwidth=h_global)
#     .fit(points)
#     .score_samples(points)
# )
#
# # Stage 2: local bandwidth per point
# g = np.exp(np.mean(np.log(np.clip(f_pilot, 1e-30, None))))
# bw_per_point = h_global * (g / f_pilot) ** alpha  # shape (N,)

#%%
# Grid setup (shared between KDE implementations)
xx, yy = np.meshgrid(grid_lons, grid_lats)
gx_km, gy_km = _to_km(xx.ravel(), yy.ravel())
grid_pts = np.column_stack([gx_km, gy_km])

# --- Adaptive Stage 3 (commented out — restore to switch back) ---
# density = (TreeKDE('gaussian', bw=bw_per_point)
#             .fit(points)
#             .evaluate(grid_pts))
# grid = density.reshape(xx.shape)
# grid = np.clip(grid, 0.0, None)

# --- Amy-equivalent: Scott's rule + k.factor=0.05, diagonal covariance ---


k = gaussian_kde(points.T, bw_method='scott')   # gaussian_kde expects (d, N)
output_dtype = np.common_type(k.covariance, grid_pts)
data_cho_cov = scipy.linalg.cholesky(k._data_covariance, lower=True)
k.factor = 0.05
cho_cov = (data_cho_cov * k.factor).astype(np.float64)
weights = np.ones(n)
cho_cov[0, 1] = 0
cho_cov[1, 0] = 0
density = gaussian_kernel_estimate['double'](
    points, weights[:, None], grid_pts, cho_cov, output_dtype
)
grid = density.reshape(xx.shape)
grid = np.clip(grid, 0.0, None)

#%%
if out_of_bounds_fill is not None:
    lons, lats, grid = SeismicPrior._expand_to_bounds(
        lons, lats, grid, bounds, out_of_bounds_fill)

grid = grid / np.nansum(grid)
adaptive_alpha = False
metadata = {
    'bw_method':      str(bw_method),
    'adaptive':       True,
    'adaptive_alpha': adaptive_alpha,
    'grid_size':      [nx, ny],
    'n_events':       int(len(lons_pts)),
    'bounds':         list(bounds),
    'generated_at':   datetime.now(timezone.utc).isoformat(),
}

name = f'kde_seismicity_{context_name}'
p = SeismicPrior(name=name, lons=lons, lats=lats, grid=grid, metadata=metadata)

#%%

res = target_res.get('KDE_Seismicity')
if res is not None:
    p = p.resample(res)
    print(f"  KDE_Seismicity ({context_name}): resampled to {res}° "
            f"({len(p.lons)}×{len(p.lats)} cells)")
    
p.to_tt3(tt3_path)
print(f"KDE_Seismicity ({context_name}): {len(catalog):,} events → "
        f"{os.path.basename(tt3_path)}")



#%%



# Map extent: (lon_min, lon_max, lat_min, lat_max).  None = auto from prior grid.
EXTENT = (-129, -112, 30, 51)

def _plot_prior(prior, label, extent):
    proj     = ccrs.PlateCarree()
    log_grid = np.log10(np.where(prior.grid > 0, prior.grid, np.nan))
    panels   = [
        (prior.grid, 'linear', 'YlOrRd', False),
        (log_grid,   'log₁₀',  'viridis', True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': proj})
    for ax, (data, scale, cmap, is_log) in zip(axes, panels):
        ax.set_extent(extent, crs=proj)
        pcm = ax.pcolormesh(prior.lons, prior.lats, data,
                            cmap=cmap, transform=proj, shading='auto')
        plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.04, fraction=0.04,
                     label='log₁₀ p' if is_log else 'p')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='0.3')
        ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='0.5')
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor='0.4')
        ax.gridlines(draw_labels=True, linewidth=0.3, color='0.6', linestyle='--',
                     x_inline=False, y_inline=False)
        ax.set_title(f'{label}\n{scale}', fontsize=9)
    plt.tight_layout()
    plt.show()


plot_extent = EXTENT if EXTENT is not None else (
    p.lons.min(), p.lons.max(), p.lats.min(), p.lats.max()
)

# Path to the .tt3 file to examine.  Set context_name to one of CONTEXTS keys.
KDE_FILE = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{context_name}.tt3')

_plot_prior(p, os.path.basename(KDE_FILE), plot_extent)

# ---------------------------------------------------------------------------
# Comparison: Smooth_seismicity prior
# ---------------------------------------------------------------------------

_ss_fname = config.PRIOR_FILENAMES.get('Smooth_seismicity')
_ss_path  = os.path.join(SeismicPrior.data_dir, _ss_fname) if _ss_fname else None

if _ss_path and os.path.exists(_ss_path):
    ss = SeismicPrior.from_tt3(_ss_path)
    _plot_prior(ss, os.path.basename(_ss_path), plot_extent)
else:
    print(f"Smooth_seismicity file not found: {_ss_path}")
    print("Run time_independent_scripts/build_priors.py first.")


# %%
