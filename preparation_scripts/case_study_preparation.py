#%%
# =============================================================================
# case_study_preparation.py  —  one-time case study data preparation
# =============================================================================
# Downloads USGS event catalogs and builds .run trigger files for all
# predefined case studies. Run this before any case_studies.py workflow script.
#
# Re-run with REDOWNLOAD=True to refresh catalogs, or REBUILD_RUN_FILES=True
# to force-rebuild the .run trigger files.
#
# Usage:
#   cd seismic_benchmark
#   python preparation_scripts/case_study_preparation.py
# =============================================================================

import os
from benchmark import config
from benchmark.usgs import download_case_study_catalog, build_run_files_for_case_study

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASE_STUDIES = config.CASE_STUDIES  # defined in benchmark/config.py

# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------
REDOWNLOAD        = False  # re-download catalogs even if a cache already exists
REBUILD_RUN_FILES = False  # rebuild .run files even if they already exist

# ---------------------------------------------------------------------------
# Prepare all case studies
# ---------------------------------------------------------------------------
for name, cs in CASE_STUDIES.items():
    print(f"\n=== {cs['name']} ===")

    CS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'case_studies', name)
    CS_RUN_DIR  = os.path.join(CS_DATA_DIR, 'run_files')
    os.makedirs(CS_RUN_DIR, exist_ok=True)

    # 1. Download (or load cached) USGS catalog
    catalog_df = download_case_study_catalog(
        cs,
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

print("\nCase study preparation complete.")
