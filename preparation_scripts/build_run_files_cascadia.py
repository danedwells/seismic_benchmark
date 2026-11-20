#%%
# =============================================================================
# cascadia_preparation.py — Cascadia benchmark data preparation
# =============================================================================
# Downloads the Cascadia benchmark catalog and builds .run trigger files for
# every event in it. This is the Cascadia-region equivalent of the main
# California benchmark's data/run_files/ (consumed by
# time_independent_scripts/run_benchmarks.py) -- unlike that CA catalog,
# which comes from real historical EPIC deployment logs, Cascadia has no
# such archive, so its benchmark set has to be synthesized from USGS phase
# data. This reuses the exact same download/build pipeline
# preparation_scripts/case_study_preparation.py uses for case studies
# (download_case_study_catalog / build_run_files_for_case_study), but this
# is NOT a case study -- it's a broad multi-event test set, not a single
# aftershock sequence (see benchmark/config_cascadia.py's
# BENCHMARK_CATALOG_CONFIG vs. CASE_STUDIES).
#
# Re-run with REDOWNLOAD=True to refresh the catalog, or REBUILD_RUN_FILES=True
# to force-rebuild the .run trigger files.
#
# Usage:
#   cd seismic_benchmark
#   python preparation_scripts/cascadia_preparation.py
# =============================================================================

import os
from benchmark import config_cascadia as config
from benchmark.usgs import download_case_study_catalog, build_run_files_for_case_study

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHMARK = config.BENCHMARK_CATALOG_CONFIG  # defined in benchmark/config_cascadia.py

# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------
REDOWNLOAD        = True  # re-download the catalog even if a cache already exists
REBUILD_RUN_FILES = True  # rebuild .run files even if they already exist

# ---------------------------------------------------------------------------
# Prepare the Cascadia benchmark catalog + run files
# ---------------------------------------------------------------------------
CS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference')
CS_RUN_DIR  = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'run_files')
os.makedirs(CS_DATA_DIR, exist_ok=True)
os.makedirs(CS_RUN_DIR, exist_ok=True)

print(f"\n=== {BENCHMARK['name']} ===")

# 1. Download (or load cached) USGS catalog
catalog_df = download_case_study_catalog(
    BENCHMARK,
    cache_dir  = CS_DATA_DIR,
    REDOWNLOAD = REDOWNLOAD,
)
print(f"  {len(catalog_df)} events in catalog.")

# 2. Build .run trigger files from USGS phase data
existing = [f for f in os.listdir(CS_RUN_DIR) if f.endswith('.run')]
if REBUILD_RUN_FILES or not existing:
    print(f"  Building .run files → {CS_RUN_DIR}")
    build_run_files_for_case_study(
        catalog_df    = catalog_df,
        run_dir       = CS_RUN_DIR,
        max_dist_deg  = 5.0,
        skip_existing = not REBUILD_RUN_FILES,
    )
else:
    print(f"  {len(existing)} .run files already present — skipping "
          f"(set REBUILD_RUN_FILES=True to force rebuild).")

print("\nCascadia benchmark preparation complete.")
