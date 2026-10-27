#%%
# =============================================================================
# plot_priors.py — KDE seismicity prior vs ETAS background, side by side
# =============================================================================
# (a) KDE_Seismicity prior (data/priors/kde_seismicity_Magna.tt3)
# (b) ETAS spatial background: the free_background field from the ETAS inversion
#     (what the ETAS prior reduces to when no recent events are in the catalog,
#     e.g. at the first event of the sequence)
#
# Both are shown as probability density (per deg^2, integrating to 1 over their
# grid) on one shared log10 color scale, with the training earthquakes on top,
# so they can be compared directly. Reference only — needs run_benchmarks.py to
# have built the KDE .tt3 and the ETAS inversion first.
#
# Usage: python Magna/plot_priors.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import config_california as config
from benchmark.runner import repair_inversion_json_paths
from faults import draw_wasatch, WASATCH_HANDLE
from variant import BW_SQ, MAX_LON, CONTEXT, OUT_SUFFIX, ETAS_SUFFIX   # env switches, see variant.py

# ---------------------------------------------------------------------------
# Config — must match Magna/run_benchmarks.py
# ---------------------------------------------------------------------------
MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(MAGNA_DIR, 'data')
FIG_DIR   = os.path.join(MAGNA_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEQUENCE_START = '2020-03-18T13:00:00'
BOUNDS  = (-114.5, -109.0, 37.0, 42.5)   # (lon_min, lon_max, lat_min, lat_max)
M_MIN   = 2.5                             # magnitude of completeness used for the background

KDE_TT3   = os.path.join(DATA_DIR, 'priors', f'kde_seismicity_{CONTEXT}.tt3')
ETAS_DIR  = os.path.join(DATA_DIR, 'etas_inversion')
ETAS_CFG  = {**config.ETAS_INVERSION_CONFIG, 'mc': M_MIN, 'm_ref': M_MIN, **({'bw_sq': BW_SQ} if BW_SQ is not None else {})}   # as in run_benchmarks.py
ETAS_JSON = os.path.join(ETAS_DIR, f'parameters_{config.etas_output_id(CONTEXT, ETAS_CFG)}.json')
ETAS_HIST = os.path.join(ETAS_DIR, 'input', f'catalog_{config.etas_catalog_tag(CONTEXT, ETAS_CFG)}.csv')
BACKGROUND_PATH = os.path.join(DATA_DIR, f'background_earthquakes_M{M_MIN:g}.parquet')
CATALOG_PATH    = os.path.join(DATA_DIR, 'Magna_2020_catalog.parquet')

EXTENT   = (-113.0, -111.0, 39.0, 42.5)    # map extent [lon0, lon1, lat0, lat1], zoomed on the Wasatch fault trace (full region: BOUNDS)
DECADES  = 4         # color range below the maximum density (orders of magnitude)
NEAR_KM  = 50        # radius for the "prior mass near Magna" summary
PRIOR_ALPHA = 0.5     # tempering exponent used in run_benchmarks.py (1 = none)
FLOOR_FRAC  = 0.05    # floor used in run_benchmarks.py: min = this fraction of each prior's max (None = off)

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                     'figure.dpi': 150, 'savefig.transparent': True})

#%%
# ---------------------------------------------------------------------------
# Load: KDE prior, ETAS background prior, training earthquakes, mainshock
# ---------------------------------------------------------------------------
kde = SeismicPrior.from_tt3(KDE_TT3)

repair_inversion_json_paths(ETAS_JSON)
updater = EtasPriorUpdater.from_inversion_json(
    json_path      = ETAS_JSON,
    catalog_df     = pd.read_csv(ETAS_HIST, index_col=0, parse_dates=['time']),
    **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS},
    spatial_factor = None,
)
bg_field = updater._bg_field
if bg_field is None:                     # free_background=False: constant background
    bg_field = np.full(len(updater.grid_lats_masked), 10 ** updater.theta['log10_mu'])
etas_bg = SeismicPrior.from_etas(
    lats=updater.grid_lats_masked, lons=updater.grid_lons_masked, lambda_grid=bg_field,
    forecast_time=pd.Timestamp(SEQUENCE_START), bounds=BOUNDS,
    out_of_bounds_fill=config.ETAS_UPDATER_CONFIG['out_of_bounds_fill'])

for _p in (kde, etas_bg):                 # same adjustment as run_benchmarks.adjust_prior
    _g = _p.grid ** PRIOR_ALPHA if PRIOR_ALPHA != 1 else _p.grid
    if FLOOR_FRAC is not None:
        _g = np.maximum(_g, FLOOR_FRAC * _g.max())
    _p.grid = _g / _g.sum()

quakes = pd.read_parquet(BACKGROUND_PATH)
quakes['time'] = pd.to_datetime(quakes['time'], utc=True).dt.tz_convert(None)
quakes = quakes[(quakes['time'] < pd.Timestamp(SEQUENCE_START)) & (quakes['mag'] >= M_MIN)]
if MAX_LON is not None:
    quakes = quakes[quakes['longitude'] <= MAX_LON]

catalog = pd.read_parquet(CATALOG_PATH)
main = catalog.loc[catalog['mag'].idxmax()]


def density(prior):
    """Prior as probability density per deg^2 (integrates to 1 over its grid)."""
    cell = np.abs(np.median(np.diff(prior.lons)) * np.median(np.diff(prior.lats)))
    d = prior.grid / cell
    return d / (d.sum() * cell)


def mass_near(prior, lat, lon, radius_km):
    """Fraction of the prior's probability mass within radius_km of (lat, lon)."""
    LON, LAT = np.meshgrid(prior.lons, prior.lats)
    dx = (LON - lon) * 111.32 * np.cos(np.radians(lat))
    dy = (LAT - lat) * 110.57
    return prior.grid[np.hypot(dx, dy) <= radius_km].sum() / prior.grid.sum()


panels = [('KDE_Seismicity', kde, 'darkblue'),
          ('ETAS background', etas_bg, 'red')]
log_dens = [np.log10(np.maximum(density(p), 1e-300)) for _, p, _ in panels]   # zero cells -> floor
# Diverging about the uniform density over the grid (1/area): red = more probable than uniform, blue = less
uni = [np.log10(1 / (np.ptp(p.lons) * np.ptp(p.lats))) for _, p, _ in panels]
vmax = max(d.max() for d in log_dens)
vmin = min(uni) - DECADES / 2
dens = [np.clip(d, vmin, None) for d in log_dens]                               # clip to the color range

print(f"Training earthquakes plotted: {len(quakes)}  (M>={M_MIN}, lon<={MAX_LON})")
for (title, p, _), ld in zip(panels, log_dens):
    i = np.abs(p.lats - main.latitude).argmin(); j = np.abs(p.lons - main.longitude).argmin()
    print(f"{title:40s} density at Magna / max = {10 ** (ld[i, j] - ld.max()):.3g}; "
          f"mass within {NEAR_KM} km of Magna = {100 * mass_near(p, main.latitude, main.longitude, NEAR_KM):.1f}%")

#%%
# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(8.8, 9.6), subplot_kw={'projection': proj})
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.03, wspace=0.04)

for ax, (title, p, color), d, u in zip(axes, panels, dens, uni):
    ax.set_extent(EXTENT, crs=proj)
    pcm = ax.pcolormesh(p.lons, p.lats, d, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=u, vmin=vmin, vmax=vmax),
                        transform=proj, shading='auto', zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'),
                   facecolor='none', edgecolor='gray', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=2)
    ax.scatter(quakes['longitude'], quakes['latitude'], s=8, c='k', alpha=0.6, linewidths=0,
               transform=proj, zorder=4, label=f'M\u2265{M_MIN:g} earthquakes used (n={len(quakes)})')
    if MAX_LON is not None:
        ax.axvline(MAX_LON, color='k', linestyle='--', linewidth=1.0, zorder=3,
                   label=f'seismicity cut ({MAX_LON}\u00b0)')
    draw_wasatch(ax, proj, zorder=5)
    ax.scatter(main.longitude, main.latitude, marker='*', s=350, c='gold', edgecolors='k',
               linewidths=1, transform=proj, zorder=6, label=f'Magna M{main.mag:.1f}')
    # Lat/lon ticks commented out for the presentation version (restore for the paper):
    # ax.set_xticks(np.linspace(EXTENT[0], EXTENT[1], 5), crs=proj)
    # ax.set_yticks(np.linspace(EXTENT[2], EXTENT[3], 5), crs=proj)
    # ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f'))
    # ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f'))
    ax.set_title(title, color=color, fontsize=14, fontweight='bold')   # panel label, colored by prior

_h, _l = axes[0].get_legend_handles_labels()
axes[0].legend(_h + [WASATCH_HANDLE], _l + [WASATCH_HANDLE.get_label()], loc='lower left', fontsize=9, framealpha=0.9, edgecolor='none')
# (Colorbar dropped for the presentation version: red = more probable than uniform, blue = less.)
# cax = fig.add_axes([0.25, 0.07, 0.5, 0.025])
# cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
# cb.set_label(f'log$_{{10}}$ probability density (per deg\u00b2), shared scale, {DECADES} decades')

fig.savefig(os.path.join(FIG_DIR, f'priors_kde_vs_etas_background{OUT_SUFFIX}{ETAS_SUFFIX}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, f'priors_kde_vs_etas_background{OUT_SUFFIX}{ETAS_SUFFIX}.pdf'), bbox_inches='tight')
plt.show()
