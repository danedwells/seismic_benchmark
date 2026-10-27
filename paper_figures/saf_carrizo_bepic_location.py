#%%
# =============================================================================
# saf_carrizo_bepic_location.py — where would bEPIC (with the benchmark's priors) place the hypothetical Carrizo M7.0?
# Synthetic P picks at the 4 closest stations (bEPIC's own ak135 travel-time table, depth 8 km), once noise-free and
# NOISE_REAL times with Gaussian pick noise; located with the Uniform, KDE_Seismicity and ETAS priors.
# Writes saf_carrizo_bepic_locations.csv (used by saf_carrizo_eew_warning_time.py).
# =============================================================================
import contextlib
import io
import os
import tempfile

import numpy as np
import pandas as pd

import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import BenchmarkRunner, get_unique_stations, make_epic_params
from bEPIC import EPIC_locate_prelim
from priors import SeismicPrior, EtasPriorUpdater

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR      = os.path.join(OUTPUT_DIR, 'saf_carrizo')    # figures are saved in this subfolder
os.makedirs(FIG_DIR, exist_ok=True)
RUN_DIR      = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')

EV_LAT, EV_LON, DEPTH_KM = 35.25, -119.80, 8.0      # depth = bEPIC's search depth
N_TRIGS      = 4
PICK_SIGMA_S = 0.3      # Gaussian pick noise for the ensemble
NOISE_REAL   = 30
T0           = 1.7e9    # arbitrary origin time (unix s)
ETAS_DATE    = pd.Timestamp('2008-01-01')

INVERSION_JSON = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion',
                              f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"])}.json')
HIST_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input',
                            f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"])}.csv')

#%%
# Priors
hist = pd.read_csv(HIST_CATALOG, index_col=0, dtype={'url': str, 'alert': str})
hist['time'] = pd.to_datetime(hist['time'], format='ISO8601', utc=True).dt.tz_convert(None)
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, catalog_df=hist, **config.ETAS_UPDATER_CONFIG)
kde = SeismicPrior.from_tt3(os.path.join(SeismicPrior.data_dir, 'kde_seismicity_benchmark.tt3'))
PRIORS = {'Uniform': (kde, False), 'KDE_Seismicity': (kde, True), 'ETAS': (updater.update(ETAS_DATE), True)}

# Synthetic picks at the closest stations, from bEPIC's own travel-time table
stations = get_unique_stations(RUN_DIR)
dx = (stations['longitude'] - EV_LON) * 111.32 * np.cos(np.radians(EV_LAT))
dy = (stations['latitude'] - EV_LAT) * 110.57
stations['hypo_km'] = np.sqrt(dx ** 2 + dy ** 2 + DEPTH_KM ** 2)
tt = EPIC_locate_prelim._get_travel_time_function(DEPTH_KM)
stations['tt_s'] = tt(stations['hypo_km'].values)
used = stations.nsmallest(N_TRIGS, 'tt_s').reset_index(drop=True)
print(used[['station', 'network', 'hypo_km', 'tt_s']].round(2).to_string(index=False))


def locate(prior, use_prior, noise, run_dir):
    picks = used.assign(**{'trigger time': T0 + used['tt_s'] + noise})
    pd.DataFrame({'version': N_TRIGS - 1, 'order': np.arange(1, N_TRIGS + 1), 'station': picks['station'], 'channel': 'HHZ',
                  'network': picks['network'], 'location': '--', 'longitude': picks['longitude'],
                  'latitude': picks['latitude'], 'trigger time': picks['trigger time'], 'tterr': 0.0,
                  'logPd': 0.0}).sort_values('trigger time').assign(order=np.arange(1, N_TRIGS + 1)) \
        .to_csv(os.path.join(run_dir, 'synthetic.run'), index=False)
    params = make_epic_params(prior, use_prior, config.BENCHMARK_PARAMS)
    runner = BenchmarkRunner(prior=prior, params=params, run_dir=run_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run_event('synthetic')
    t = runner.results[('synthetic', N_TRIGS - 1)]
    err = float(np.hypot((t.posterior_lon - EV_LON) * 111.32 * np.cos(np.radians(EV_LAT)),
                         (t.posterior_lat - EV_LAT) * 110.57))
    return t.posterior_lat, t.posterior_lon, err


rng = np.random.default_rng(0)
rows = []
with tempfile.TemporaryDirectory() as run_dir:
    for k in range(NOISE_REAL + 1):
        noise = np.zeros(N_TRIGS) if k == 0 else rng.normal(0, PICK_SIGMA_S, N_TRIGS)
        for name, (prior, use_prior) in PRIORS.items():
            lat, lon, err = locate(prior, use_prior, noise, run_dir)
            rows.append(dict(realization=k, prior=name, lat=lat, lon=lon, error_km=err))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(FIG_DIR, 'saf_carrizo_bepic_locations.csv'), index=False)

#%%
print('\nNoise-free picks:')
print(res[res.realization == 0].round(3).to_string(index=False))
print(f'\nWith {PICK_SIGMA_S} s pick noise ({NOISE_REAL} realizations): location error (km)')
print(res[res.realization > 0].groupby('prior')['error_km'].describe(percentiles=[.5, .9]).round(1).to_string())
