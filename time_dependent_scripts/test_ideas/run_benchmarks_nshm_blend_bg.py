#%%
# =============================================================================
# run_benchmarks_nshm_blend_bg.py  —  California benchmark with a 50/50 NSHM + ETAS blend as the ETAS background-rate grid
# =============================================================================
# Same setup as time_dependent_scripts/run_benchmarks.py (bw_sq = 4, no tempering, USGS locations fed back to ETAS
# after each event, sigma_s = 0.35), run for three priors in ONE causal pass over the events so they share the ETAS
# updater and the station-availability mask:
#   Uniform       — no prior
#   ETAS          — dynamic ETAS prior with the inverted (spatially smoothed) background grid  [= the existing run]
#   ETAS_NSHMblend — the same ETAS prior with updater._bg_field replaced by a BLEND_W / (1 - BLEND_W) mix of the NSHM
#                   moment-rate shape and the inverted ETAS background, each normalized to sum 1 first, then rescaled to
#                   the same total background rate. Triggering parameters are left as inverted.
# NSHM cells outside its grid (~3% of the ETAS grid) get the median NSHM value, so no cell has zero background.
# Results (separate folder): results/california/output/time_dependent_nshm_blend_bg/sig_0.35/max_trigs_10/{name}_benchmark_results.csv
# =============================================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner
import benchmark.config_california as config
from benchmark.runner import BenchmarkRunner, runner_results_to_df, make_epic_params, load_station_availability_cache
from benchmark.time_dependent_helpers import run_trigger_time

BW_SQ   = 4
SIGMA_S = 0.35
BLEND_W = 0.5      # weight on the NSHM; ETAS gets 1 - BLEND_W
CAP_PCT = None     # DIAGNOSTIC: clip each ETAS prior to this percentile of its in-bounds cells, then renormalize. None = off.
# DIAGNOSTIC search-grid settings (all three priors): a small grid centred on the first trigger that, once MIGRATE_MIN_STATIONS
# stations have reported, re-centres on the posterior MAP after every version. GRID_KM = None keeps the config default.
GRID_KM          = 50
MIGRATE_MIN_STATIONS = 3
# DIAGNOSTIC station-activity mask: for the events probed by preparation_scripts/probe_station_availability_mtj.py (true location
# in the MTJ box), replace their entry in the station-availability cache with the probed active stations. The main cache only
# has BK/NC stations that triggered, so it cannot tell the mask that the Mendocino network stayed silent. False = main cache.
MTJ_PROBE = True
config.ETAS_INVERSION_CONFIG['bw_sq'] = BW_SQ
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S
if GRID_KM is not None:
    config.BENCHMARK_PARAMS.update(grid_km=GRID_KM, migrate_grid=True, migrate_grid_min_stations=MIGRATE_MIN_STATIONS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet')
RUN_DIR      = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
_ETAS_ID     = config.ETAS_INVERSION_CONFIG['id']
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', f'parameters_{config.etas_output_id(_ETAS_ID)}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input', f'catalog_{config.etas_catalog_tag(_ETAS_ID)}.csv')
MAX_TRIGS  = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent_nshm_blend_bg', f'sig_{SIGMA_S:g}', f'max_trigs_{MAX_TRIGS}'
                          + (f'_grid{GRID_KM}_migrate{MIGRATE_MIN_STATIONS}' if GRID_KM is not None else '')
                          + ('_mtjprobe' if MTJ_PROBE else '')
                          + (f'_cap{CAP_PCT}' if CAP_PCT is not None else ''))
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%
catalog_df = benchmark_runner.load_reference_catalog(CATALOG_PATH)
station_availability = load_station_availability_cache(STATION_AVAIL_CACHE) if os.path.exists(STATION_AVAIL_CACHE) else None
if MTJ_PROBE:
    _probe = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_mtj_probe.parquet'))
    _probe = _probe[_probe['active']].drop(columns=['triggered', 'active', 'client'])
    station_availability = dict(station_availability or {})
    for _eid, _grp in _probe.groupby('event_id'):
        station_availability[str(_eid)] = _grp.drop(columns='event_id').reset_index(drop=True)
    print(f'MTJ probe inventory used for {_probe.event_id.nunique()} events')
_usgs = (catalog_df[['event_id', 'usgs_time', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
         .rename(columns={'usgs_time': 'time', 'usgs_lat': 'latitude', 'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
         .set_index('event_id'))

hist = pd.read_csv(HISTORICAL_CATALOG, index_col=0, dtype={'url': str, 'alert': str})
hist['time'] = pd.to_datetime(hist['time'], format='ISO8601', utc=True).dt.tz_convert(None)
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, catalog_df=hist, **config.ETAS_UPDATER_CONFIG, spatial_factor=None)

# NSHM background on the ETAS grid
orig_bg = updater._bg_field.copy()
nshm_dir = os.path.join(SeismicPrior.data_dir, 'USGS_NSHM_data')
nshm = SeismicPrior.from_nshm(os.path.join(nshm_dir, 'gridded_moment_rates.xyz'),
                              fault_data_path=os.path.join(nshm_dir, 'fault_moment_rates.xyz'),
                              bounds=config.PRIOR_CONSTRUCTION_PARAMS['bounds'])
f = RegularGridInterpolator((nshm.lats, nshm.lons), nshm.grid, bounds_error=False, fill_value=np.nan)(
    np.column_stack([updater.grid_lats_masked, updater.grid_lons_masked]))
print(f'NSHM background: {np.isnan(f).mean():.1%} of the ETAS grid outside the NSHM grid (filled with the median)')
f = np.where(np.isnan(f), np.nanmedian(f), f)
nshm_bg = (BLEND_W * f / f.sum() + (1 - BLEND_W) * orig_bg / orig_bg.sum())
nshm_bg /= nshm_bg.sum()


#%%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Background rate maps, each normalized to sum 1 so the panels share one log colour scale
lons, lats = updater.grid_lons_masked, updater.grid_lats_masked
panels = [('(a) ETAS inverted background', orig_bg / orig_bg.sum()),
          ('(b) NSHM', f / f.sum()),
          (f'(c) Blend ({BLEND_W:.0%} NSHM)', nshm_bg / nshm_bg.sum())]
vmax = max(v.max() for _, v in panels)
norm = mcolors.LogNorm(vmin=vmax * 1e-3, vmax=vmax)
extent = [lons.min(), lons.max(), lats.min(), lats.max()]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
for ax, (title, v) in zip(axes, panels):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    sc = ax.scatter(lons, lats, c=v, s=3, marker='s', norm=norm, cmap='viridis', transform=ccrs.PlateCarree(), rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(np.ceil(extent[0]), extent[1], 3), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.tick_params(labelsize=8)
axes[0].set_yticks(np.arange(np.ceil(extent[2]), extent[3], 3), crs=ccrs.PlateCarree())
axes[0].yaxis.set_major_formatter(LatitudeFormatter())
fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04, aspect=50, label='Fraction of total background rate per cell')
FIG_DIR = os.path.join(PROJECT_ROOT, 'paper_figures')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'nshm_blend_background.{ext}'), dpi=300, bbox_inches='tight')
plt.show()

#%%
def cap_prior(prior):
    """Clip cells above the CAP_PCT-th percentile of in-bounds cells (out-of-bounds cells are the constant minimum fill)."""
    if CAP_PCT is None:
        return prior
    g = np.asarray(prior.grid, dtype=float)
    g = np.minimum(g, np.percentile(g[g > 100 * g.min()], CAP_PCT))
    prior.grid = g / g.sum()
    return prior

event_ids = sorted((p.stem for p in Path(RUN_DIR).glob('*.run')), key=lambda e: run_trigger_time(e, RUN_DIR))
t0 = pd.Timestamp(run_trigger_time(event_ids[0], RUN_DIR), unit='s')
p0 = updater.update(t0)
runners = {name: BenchmarkRunner(prior=p0, params=make_epic_params(p0, name != 'Uniform', config.BENCHMARK_PARAMS), run_dir=RUN_DIR,
                                 catalog_df=catalog_df, station_availability=station_availability)
           for name in ('Uniform', 'ETAS', 'ETAS_NSHMblend')}

print(f'running {len(event_ids)} events x 3 priors')
for i, eid in enumerate(event_ids, 1):
    t = pd.Timestamp(run_trigger_time(eid, RUN_DIR), unit='s')
    for name, bg in (('ETAS', orig_bg), ('ETAS_NSHMblend', nshm_bg)):
        updater._bg_field = bg
        runners[name].update_prior(cap_prior(updater.update(t)))
    for runner in runners.values():
        runner.run_event(eid)
    if int(eid) in _usgs.index:                 # USGS location (not the bEPIC estimate) goes into ETAS, once per event
        r = _usgs.loc[int(eid)]
        updater.append_events(pd.DataFrame([{'time': r['time'], 'latitude': r['latitude'], 'longitude': r['longitude'], 'magnitude': r['magnitude']}]))
    if i % 25 == 0:
        print(f'  {i}/{len(event_ids)} events done', flush=True)

for name, runner in runners.items():
    out = os.path.join(OUTPUT_DIR, f'{name.lower()}_benchmark_results.csv')
    runner_results_to_df(runner).to_csv(out, index=False)
    print(f'{name} -> {out}')
