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
from benchmark.usgs import download_case_study_catalog, build_run_files_for_case_study

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Case study definitions — single authoritative source shared by all workflows
# ---------------------------------------------------------------------------
CASE_STUDIES = {
    'Ridgecrest': {
        'name':      'Ridgecrest 2019',
        'starttime': '2019-07-04T17:00:00',
        'endtime':   '2019-08-07T00:00:00',
        'bounds':    (-118.5, -116.5, 35.0, 36.5),
        'min_mag':   3.0,
    },
    'Ferndale': {
        'name':      'Ferndale 2022',
        'starttime': '2022-12-20T10:00:00',
        'endtime':   '2023-01-20T00:00:00',
        'bounds':    (-127.0, -122.5, 39, 41.0),
        'min_mag':   3.0,
    },
    'ElMayor': {
        'name':      'El Mayor-Cucapah 2010',
        'starttime': '2010-04-04T22:00:00',
        'endtime':   '2010-05-04T00:00:00',
        'bounds':    (-117.0, -114.5, 31.5, 33.5),
        'min_mag':   3.0,
    },
}

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
