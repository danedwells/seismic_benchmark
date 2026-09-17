#%%
# =============================================================================
# sigma_calibration.py — sweep bEPIC's sigma_s (likelihood-only, Uniform prior)
# =============================================================================
# Calibrating sigma_s is a likelihood-only question (posterior ∝ likelihood
# when use_prior=False), so this only ever runs the Uniform prior — no other
# prior's .tt3 grid values are needed, just NSHM's grid geometry (shared
# across regions/case studies, so no region-specific prior cache is needed).
#
# Writes one CSV per sigma_s value to sig_{value}/max_trigs_{N}/, same layout
# the existing plot_scripts/*_sigma_s.py scripts already discover and plot.
#
# Works for any region or case study — set TARGET_KIND / TARGET below.
# =============================================================================

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'
from concurrent.futures import ProcessPoolExecutor

from priors import SeismicPrior
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import make_epic_params, load_station_availability_cache
from benchmark.usgs import download_case_study_catalog

#%%
# ---------------------------------------------------------------------------
# Select what to calibrate
# ---------------------------------------------------------------------------
TARGET_KIND = 'region'        # 'region' | 'case_study'
TARGET      = 'cascadia'      # 'california'|'cascadia' (region), or a
                               # benchmark.config.CASE_STUDIES key (case_study)

SIGMA_VALUES = [0.6, 0.8, 0.9, 1.0, 1.2, 1.5]   # sweep — edit as needed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir     = SeismicPrior.data_dir            # priors/data/
MAX_TRIGS    = config.BENCHMARK_PARAMS['max_trigs']

# NSHM's .tt3 is shared across every region/case study (not region-specific) —
# used only for grid geometry since use_prior=False ignores its values.
nshm_path = os.path.join(data_dir, config.PRIOR_FILENAMES['NSHM'])

if TARGET_KIND == 'region':
    RUN_DIR         = os.path.join(PROJECT_ROOT, 'data', TARGET, 'run_files')
    STATION_AVAIL   = os.path.join(PROJECT_ROOT, 'data', TARGET, 'reference', 'station_availability_cache.parquet')
    BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', TARGET, 'output', 'time_independent')

    if TARGET == 'california':
        catalog_path = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
        catalog_df   = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None
    elif TARGET == 'cascadia':
        catalog_path = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'cascadia_test_catalog.csv')
        catalog_df   = benchmark_runner.load_reference_catalog_usgs(catalog_path) if os.path.exists(catalog_path) else None
    else:
        raise ValueError(f"Unknown region '{TARGET}' — expected 'california' or 'cascadia'")

    ref_df = catalog_df.rename(columns={
        'id':        'event_id',
        'latitude':  'usgs_lat',
        'longitude': 'usgs_lon',
    })[['event_id', 'usgs_lat', 'usgs_lon']]

elif TARGET_KIND == 'case_study':
    cs              = config.CASE_STUDIES[TARGET]
    CS_DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', TARGET)
    RUN_DIR         = os.path.join(CS_DATA_DIR, 'run_files')
    STATION_AVAIL   = os.path.join(CS_DATA_DIR, 'station_availability_cache.parquet')
    BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', TARGET, 'output', 'time_independent')

    catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)
    ref_df = catalog_df.rename(columns={
        'id':        'event_id',
        'latitude':  'usgs_lat',
        'longitude': 'usgs_lon',
    })[['event_id', 'usgs_lat', 'usgs_lon']]

else:
    raise ValueError(f"Unknown TARGET_KIND '{TARGET_KIND}' — expected 'region' or 'case_study'")

_avail = (
    load_station_availability_cache(STATION_AVAIL)
    if os.path.exists(STATION_AVAIL) else None
)

#%%
# ---------------------------------------------------------------------------
# Sweep sigma_s — one process per value, Uniform prior only
# ---------------------------------------------------------------------------

def _run_one(sigma):
    prior       = SeismicPrior.from_tt3(nshm_path)
    params_dict = {**config.BENCHMARK_PARAMS, 'sigma_s': sigma}   # local copy — no shared mutable state
    params      = make_epic_params(prior, False, params_dict, station_inventory=_avail)
    out_dir     = os.path.join(BASE_OUTPUT_DIR, f'sig_{sigma}', f'max_trigs_{MAX_TRIGS}')
    return benchmark_runner.run_prior(params, 'Uniform', ref_df, RUN_DIR, out_dir)

with ProcessPoolExecutor(max_workers=len(SIGMA_VALUES)) as ex:
    futures = [ex.submit(_run_one, s) for s in SIGMA_VALUES]
    for f in futures:
        f.result()  # re-raise any worker exception

# %%
