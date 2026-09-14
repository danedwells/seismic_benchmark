#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner (static priors)
# =============================================================================
# Runs bEPIC across all static spatial priors for a predefined aftershock
# sequence, mirroring run_benchmarks.py.
#
# Prerequisites
# -------------
#   preparation_scripts/case_study_preparation.py  — download catalog + .run files
#   preparation_scripts/build_priors.py            — build .tt3 prior cache
#
# Usage:
#   Set ACTIVE_CASE_STUDY to one of the keys in CASE_STUDIES, flip the
#   control flags, then run cells in order (or execute the whole script).
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

# Custom repository imports
from priors import SeismicPrior

from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import load_station_availability_cache, make_epic_params


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir            # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}
CASE_STUDIES = config.CASE_STUDIES
# --- Select active case study --- (override with CASE_STUDY env var)

DEFAULT_CASE_STUDY = "Ferndale"
ACTIVE_CASE_STUDY = os.environ.get('CASE_STUDY', DEFAULT_CASE_STUDY)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')
cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

# Params 
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
SIGMA_S = 0.22
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S

# Directories
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'california', 'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_independent', f'max_trigs_{MAX_TRIGS}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

cache_paths['KDE_Seismicity'] = os.path.join(data_dir, f'kde_seismicity_{ACTIVE_CASE_STUDY}.tt3')

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)
print(f"{len(catalog_df)} events in {cs['name']} catalog.")
print(catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']].head())

# ------------------------------------------------------------------------------
# ── 1. Run bEPIC across priors ────────────────────────────────────────────
# ------------------------------------------------------------------------------

# Build reference catalog before job_args so it can be passed to each worker.
cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

# Get the station availability inventory from (preparation_scripts/build_station_availability.py)
_avail = (load_station_availability_cache(AVAIL_CACHE)
          if os.path.exists(AVAIL_CACHE) else None)
if _avail:
    print("Station availability cache loaded")

#%%
from concurrent.futures import ProcessPoolExecutor

priors_to_run = ['Gear1']#, 'NSHM', 'KDE_Seismicity', 'Helmstetter', 'Uniform']

def _run_one(name, path):
    if path is not None:
        prior = SeismicPrior.from_tt3(path)
        use_prior = True
    else:  # Uniform
        path = cache_paths['NSHM']
        prior = SeismicPrior.from_tt3(path)
        use_prior = False
    params = make_epic_params(prior, use_prior, config.BENCHMARK_PARAMS, station_inventory=_avail)
    return benchmark_runner.run_prior(params, name, cs_ref_df, CS_RUN_DIR, CS_OUTPUT_DIR)

with ProcessPoolExecutor(max_workers=len(priors_to_run)) as ex:
    futures = [ex.submit(_run_one, name, cache_paths[name]) for name in priors_to_run]
    for f in futures:
        f.result()  # re-raise any worker exception
