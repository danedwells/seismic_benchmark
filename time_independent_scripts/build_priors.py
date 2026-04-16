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
#   python scripts/build_priors.py
# =============================================================================

from priors import SeismicPrior
from benchmark import priors as utils
from benchmark import config
import os

data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

utils.build_and_cache_priors(cache_paths, data_dir, config.PRIOR_CONSTRUCTION_PARAMS)
