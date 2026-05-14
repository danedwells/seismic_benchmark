#%%
# =============================================================================
# examine_kde_prior.py  —  build and/or inspect KDE seismicity .tt3 files
# =============================================================================
# Two independent cells:
#
#   Cell 1 (this cell + Build cell): build any missing .tt3 files.
#           Set BUILD = True, pick contexts, run.
#
#   Cell 2 (Plot cell): load a single .tt3 and show a diagnostic map.
#           Set KDE_FILE and run.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import SeismicPrior
from benchmark import config
from benchmark.priors import build_or_load_kde_prior

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

#%%
# # ---------------------------------------------------------------------------
# # Build KDE .tt3 files
# # ---------------------------------------------------------------------------
# # BUILD            — run this cell at all
# # FORCE_REBUILD    — overwrite existing .tt3 files (re-fit the KDE)
# # FORCE_REDOWNLOAD — re-download the base parquet from USGS
# #                    (independent of FORCE_REBUILD; use when the download
# #                     previously timed out or you want a fresher catalog)

# BUILD             = True
# FORCE_REBUILD     = True
# FORCE_REDOWNLOAD  = False

# # Contexts to build — any subset of CONTEXTS keys.
# BUILD_CONTEXTS = list(CONTEXTS.keys())   # all four; edit to build a subset

# if BUILD:
#     for ctx_name in BUILD_CONTEXTS:
#         cutoff = CONTEXTS[ctx_name]

#         print(f"\n── Building KDE prior: {ctx_name}  (cutoff {cutoff}) ──")

#         build_or_load_kde_prior(
#             context_name        = ctx_name,
#             cutoff_date         = cutoff,
#             kde_catalog_path    = KDE_CATALOG_PATH,
#             data_dir            = SeismicPrior.data_dir,
#             construction_params = _kde_construction,
#             kde_start           = config.KDE_START_DATE,
#             force_rebuild       = FORCE_REBUILD,
#             force_redownload    = FORCE_REDOWNLOAD,
#         )
#     print("\nDone.")

#%%
# ---------------------------------------------------------------------------
# ── CONFIGURE plot ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Path to the .tt3 file to examine.  Set context_name to one of CONTEXTS keys.
context_name = 'Ferndale'
KDE_FILE = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{context_name}.tt3')

# Map extent: (lon_min, lon_max, lat_min, lat_max).  None = auto from prior grid.
EXTENT = (-129, -112, 30, 51)

# ---------------------------------------------------------------------------
# Load prior
# ---------------------------------------------------------------------------

p = SeismicPrior.from_tt3(KDE_FILE)

print(f"Loaded: {os.path.basename(KDE_FILE)}")
print(f"  Grid shape : {p.grid.shape}  ({len(p.lats)} lat × {len(p.lons)} lon)")
print(f"  Lon range  : {p.lons.min():.2f} → {p.lons.max():.2f}")
print(f"  Lat range  : {p.lats.min():.2f} → {p.lats.max():.2f}")
print(f"  Grid sum   : {p.grid.sum():.6f}  (should be ~1.0)")
print(f"  Non-zero   : {(p.grid > 0).sum():,} / {p.grid.size:,} cells")
print(f"  Value range: {p.grid.min():.3e} → {p.grid.max():.3e}")

# ---------------------------------------------------------------------------
# Figure: raw density (left) + log₁₀ density (right)
# ---------------------------------------------------------------------------

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
