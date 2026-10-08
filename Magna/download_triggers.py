#%%
# =============================================================================
# download_triggers.py — Magna, UT (March 2020) catalog + .run trigger files
# =============================================================================
# Downloads the USGS catalog for the 2020-03-18 Magna M5.7 sequence and builds
# one bEPIC .run trigger file per event from USGS phase picks (UU stations,
# coordinates from IRIS/EarthScope). Mirrors
# preparation_scripts/case_study_preparation.py, self-contained in Magna/.
#
# Usage:  python Magna/download_triggers.py
# =============================================================================
import os

from benchmark.usgs import download_case_study_catalog, build_run_files_for_case_study

MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(MAGNA_DIR, 'data')
RUN_DIR   = os.path.join(DATA_DIR, 'run_files')

CS = {
    'name':      'Magna 2020',
    'starttime': '2020-03-18T13:00:00',     # mainshock 13:09 UTC
    'endtime':   '2020-04-18T00:00:00',
    'bounds':    (-112.6, -111.5, 40.3, 41.2),   # (lon_min, lon_max, lat_min, lat_max)
    'min_mag':   3.0,
}

REDOWNLOAD        = False   # re-download the catalog even if cached
REBUILD_RUN_FILES = False   # rebuild .run files even if they already exist

#%%
os.makedirs(RUN_DIR, exist_ok=True)

catalog_df = download_case_study_catalog(CS, cache_dir=DATA_DIR, REDOWNLOAD=REDOWNLOAD)
print(f"{len(catalog_df)} events in catalog.")
print(catalog_df.sort_values('mag', ascending=False).head())

written = build_run_files_for_case_study(
    catalog_df    = catalog_df,
    run_dir       = RUN_DIR,
    max_dist_deg  = 5.0,
    skip_existing = not REBUILD_RUN_FILES,
)

#%%
missing = sorted(set(catalog_df['id']) - set(written))
print(f"\n{len(written)}/{len(catalog_df)} events have a .run file in {RUN_DIR}")
if missing:
    print(f"{len(missing)} events skipped (no ComCat phases, or <2 stations with coordinates):")
    print('  ' + ', '.join(missing))
