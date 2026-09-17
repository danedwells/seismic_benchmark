#%%
# =============================================================================
# run_cascadia.py  —  bEPIC static-prior benchmark for Cascadia
# Prerequisite: run preparation_scripts/build_priors.py (shared static priors)
# and preparation_scripts/build_priors_cascadia.py (Cascadia KDE_Seismicity)
# first to build the .tt3 cache files.
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

# Custom repository imports
from priors import SeismicPrior

from benchmark import runner as benchmark_runner
from benchmark import config_cascadia as config
from benchmark.runner import (load_station_availability_cache,
                             make_epic_params)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir  # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'run_files')
EDT_SIGMA_S    = config.BENCHMARK_PARAMS['edt_sigma_s']

#SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
SIGMA_S = 0.60
config.BENCHMARK_PARAMS['sigma_s']      = SIGMA_S
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']

OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'cascadia', 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')

os.makedirs(OUTPUT_DIR,  exist_ok=True)


# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog. Cascadia .run files are named by ANSS event id
# (str), not the postgres int ids the CA bEPIC_testing_catalog.txt uses, so
# load via load_reference_catalog_usgs() — see time_dependent_scripts/run_cascadia.py.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'cascadia_test_catalog_west.csv')
catalog_df = benchmark_runner.load_reference_catalog_usgs(catalog_path) if os.path.exists(catalog_path) else None

# Build reference catalog before job_args so it can be passed to each worker.
ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

# Cascadia-specific KDE_Seismicity cache, built by preparation_scripts/build_priors_cascadia.py.
cache_paths['KDE_Seismicity'] = os.path.join(data_dir, 'kde_seismicity_cascadia.tt3')

_avail = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# ── 1. Create reference locations ─────────────────────────────────────────
ref_dir = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference')

# ---------------------------------------------------------
# Construct parameters and run (one thread per prior)
#----------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor

priors_to_run = ['KDE_Seismicity', 'Uniform']#, 'NSHM', 'KDE_Seismicity', 'Helmstetter', 'Uniform']

def _run_one(name, path):
    if path is not None:
        prior = SeismicPrior.from_tt3(path)
        use_prior = True
    else:  # Uniform
        path = cache_paths['NSHM']
        prior = SeismicPrior.from_tt3(path)
        use_prior = False
    params = make_epic_params(prior, use_prior, config.BENCHMARK_PARAMS, station_inventory=_avail)
    return benchmark_runner.run_prior(params, name, ref_df, RUN_DIR, OUTPUT_DIR)

with ProcessPoolExecutor(max_workers=len(priors_to_run)) as ex:
    futures = [ex.submit(_run_one, name, cache_paths[name]) for name in priors_to_run]
    for f in futures:
        f.result()  # re-raise any worker exception



# %%
