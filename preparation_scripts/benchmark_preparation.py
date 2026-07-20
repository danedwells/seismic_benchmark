# =============================================================================
# Download and cache background_seismicity
# =============================================================================
#%%
import os


from pathlib import Path

# Custom repository imports
from priors import SeismicPrior
from benchmark.background import load_background_seismicity

from benchmark import config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir  # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
SEIS_CACHE   = os.path.join('/home/a01738353/2024_NEHRP/RECAST/case_studies', 'background_seismicity.parquet')



bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = (-120, -115, 32, 38),
    start_year  = 2000,
    end_year    = 2018,
    min_mag     = 2.0,
    force_refresh= True
)

#%%