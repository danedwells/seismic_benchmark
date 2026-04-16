#%%
# =============================================================================
# build_initial_prior.py — ETAS inversion → initial time-dependent prior
# =============================================================================
# Runs the ETAS parameter inversion on a seismicity catalog and stores the
# result in seismic_benchmark/data/etas_inversion/.  The output JSON is then
# consumed by EtasPriorUpdater.from_inversion_json() at runtime to produce
# fast real-time prior updates as new events arrive.
#
# All inversion parameters live in benchmark/config.py (ETAS_INVERSION_CONFIG).
# Edit that dict, not this script, to change the inversion settings.
#
# Catalog requirements
# --------------------
# The catalog must be a CSV with columns:
#   id, latitude, longitude, time (datetime), magnitude
# and must be complete above ETAS_INVERSION_CONFIG['mc'].
# The bEPIC testing catalog (bEPIC_testing_catalog.txt) is NOT suitable —
# it covers only 237 target events, not the full seismicity needed for
# robust parameter estimation.  Use a complete ANSS/ComCat extract instead
# (see CATALOG_PATH below).
#
# Usage
# -----
#   cd seismic_benchmark
#   python time_dependent_scripts/build_initial_prior.py
#
# Or run cell-by-cell in VS Code / Jupyter.
# =============================================================================

import logging
import os
import json

import pandas as pd

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation

from benchmark import config

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory — inversion results (parameters_benchmark.json, etc.)
# are written here and read back by EtasPriorUpdater.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONFIGURE: catalog ────────────────────────────────────────────────────────
# Point this at a complete seismicity catalog in ETAS CSV format:
#   columns: id, latitude, longitude, time, magnitude
#
# The example below references the etas_2 example catalog.  To use a catalog
# within seismic_benchmark instead, change CATALOG_PATH accordingly and make
# sure the file is in ETAS format (or handled by load_catalog() below).

CATALOG_PATH = os.path.join(
    PROJECT_ROOT, '..', 'etas_2', 'input_data', 'example_catalog.csv'
)

# Set to True to re-run the inversion even if a result already exists.
FORCE_RERUN = False

#%%
# ── Catalog loader ────────────────────────────────────────────────────────────

def load_catalog(path):
    """
    Load a seismicity catalog into the format expected by ETASParameterCalculation.

    Accepts two formats:
      ETAS format   — CSV with columns: id, latitude, longitude, time, magnitude
      bEPIC format  — tab-separated catalog from load_reference_catalog()
                      (columns: ANSS ID, ANSS date, ANSS lat, ANSS lon,
                       ANSS depth, ANSS mag, postgres id, …)

    Returns a DataFrame with columns: id, latitude, longitude, time, magnitude.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found: {path}")

    # Detect format by peeking at header
    with open(path) as fh:
        header = fh.readline()

    if 'ANSS' in header or '\t' in header:
        # bEPIC reference catalog format
        raw = pd.read_csv(path, sep='\t')
        df = pd.DataFrame({
            'id':        raw['postgres id'].astype(str),
            'latitude':  raw['ANSS lat'],
            'longitude': raw['ANSS lon'],
            'time':      pd.to_datetime(raw['ANSS date'], format='%Y-%m-%d-%H:%M:%S.%f-GMT'),
            'magnitude': raw['ANSS mag'],
        })
        print(f"  Detected bEPIC catalog format ({len(df)} events).")
        print("  WARNING: bEPIC testing catalogs are sparse (~237 events).")
        print("  ETAS inversion works best with a complete seismicity catalog (thousands of events).")
    else:
        # Standard ETAS CSV format
        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=['time'],
            dtype={'url': str, 'alert': str},
        )
        df = df.reset_index().rename(columns={'index': 'id'})
        if 'id' not in df.columns:
            df.insert(0, 'id', range(len(df)))
        print(f"  Detected ETAS catalog format ({len(df)} events).")

    required = {'latitude', 'longitude', 'time', 'magnitude'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    mc = config.ETAS_INVERSION_CONFIG['mc']
    n_before = len(df)
    df = df[df['magnitude'] >= mc].copy()
    print(f"  After mc={mc} filter: {len(df)} / {n_before} events retained.")

    return df

#%%
# ── Check for existing result ─────────────────────────────────────────────────

inversion_id  = config.ETAS_INVERSION_CONFIG['id']
output_json   = os.path.join(OUTPUT_DIR, f'parameters_{inversion_id}.json')

if os.path.exists(output_json) and not FORCE_RERUN:
    print(f"Inversion output already exists: {output_json}")
    print("Set FORCE_RERUN = True to re-run.")
else:

    # ── Load catalog ──────────────────────────────────────────────────────────

    print(f"Loading catalog from:\n  {os.path.abspath(CATALOG_PATH)}")
    catalog_df = load_catalog(CATALOG_PATH)
    print(f"  Time range: {catalog_df['time'].min()} → {catalog_df['time'].max()}")
    print(f"  Mag  range: {catalog_df['magnitude'].min():.1f} → {catalog_df['magnitude'].max():.1f}")


    # ── Build inversion metadata dict ─────────────────────────────────────────
    # Merge config params with runtime paths.  ETASParameterCalculation accepts
    # a 'catalog' DataFrame directly, so no temp file is needed.

    inversion_metadata = dict(config.ETAS_INVERSION_CONFIG)
    inversion_metadata['catalog']   = catalog_df
    inversion_metadata['data_path'] = OUTPUT_DIR + os.sep   # store_results needs trailing sep

    # Remove fn_catalog if present — we're passing the DataFrame directly
    inversion_metadata.pop('fn_catalog', None)

    print("\nInversion configuration:")
    _display = {k: v for k, v in inversion_metadata.items()
                if k not in ('catalog', 'theta_0', 'shape_coords')}
    for k, v in _display.items():
        print(f"  {k}: {v}")
    print(f"  theta_0: {config.ETAS_INVERSION_CONFIG['theta_0']}")
    print(f"  shape_coords: {len(config.ETAS_INVERSION_CONFIG['shape_coords'])} polygon vertices")

    # ── Run inversion ─────────────────────────────────────────────────────────

    set_up_logger(level=logging.INFO)

    print(f"\nRunning ETAS inversion (id='{inversion_id}')…")
    print("This typically takes several minutes.\n")

    # This comes directly from the runnable_code/invert_etas.py script within the etas/ repository.
    calculation = ETASParameterCalculation(inversion_metadata)
    calculation.prepare()
    parameters  = calculation.invert()
    calculation.store_results(OUTPUT_DIR + os.sep, store_pij=False)

    print(f"\nInversion complete.")
    print(f"Output: {output_json}")

#%%
# ── Report results ────────────────────────────────────────────────────────────

with open(output_json) as fh:
    result = json.load(fh)

print("\nInverted ETAS parameters (final_parameters):")
for k, v in result.get('final_parameters', {}).items():
    if v is not None:
        print(f"  {k:15s}: {v:.6f}")
    else:
        print(f"  {k:15s}: None")

print(f"\nForecast origin (timewindow_end): {result.get('timewindow_end')}")
print(f"Iterations:                       {result.get('n_iterations')}")

#%%
# ── Verify: build an EtasPriorUpdater from the output ────────────────────────
# Quick sanity check — load the result and evaluate a baseline prior.

import pandas as pd
from priors import EtasPriorUpdater

print("\nVerifying output with EtasPriorUpdater…")
updater = EtasPriorUpdater.from_inversion_json(
    json_path  = output_json,
    catalog_df = load_catalog(CATALOG_PATH),
    **config.ETAS_UPDATER_CONFIG,
)
print(updater)

forecast_time = pd.Timestamp(result['timewindow_end'])
prior = updater.update(forecast_time)
print(f"Baseline prior at {forecast_time}:")
print(f"  grid shape : {prior.grid.shape}")
print(f"  grid max   : {prior.grid.max():.4e}")
print(f"  grid sum   : {prior.grid.sum():.6f}  (should be 1.0)")
print(f"\nAll done.  Pass this path to EtasPriorUpdater.from_inversion_json():")
print(f"  {output_json}")
