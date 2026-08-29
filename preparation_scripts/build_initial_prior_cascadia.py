#%%
# =============================================================================
# build_initial_prior_cascadia.py — Cascadia ETAS inversion → initial prior
# =============================================================================
# Cascadia-region counterpart to build_initial_prior.py. Mirrors its logic
# exactly but imports benchmark.config_cascadia instead of benchmark.config,
# so the California/main-benchmark inversion outputs are never touched by
# runs of this script (and vice versa) -- config_cascadia.py has its own,
# wider ETAS spatial polygon (reaches to ~51N) since config.py's polygon
# caps at ~44N and doesn't cover most of the Pacific Northwest.
#
# Writes to data/cascadia/etas_inversion/ (parallel to the main benchmark's
# data/etas_inversion/), so the two regions' outputs never collide.
#
# ETAS and declustering
# ---------------------
# Do NOT pre-decluster the catalog.  ETAS explicitly models clustered
# seismicity; the pij responsibility matrix internally separates background
# from aftershock-triggered events during inversion.
#
# Usage
# -----
#   cd seismic_benchmark
#   python preparation_scripts/build_initial_prior_cascadia.py
#
# Or run cell-by-cell in VS Code / Jupyter.
# =============================================================================

import logging
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation

from priors import EtasPriorUpdater
from benchmark import config_cascadia as config
from benchmark.background import load_background_seismicity

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'etas_inversion')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIR  = os.path.join(OUTPUT_DIR, 'input')
os.makedirs(INPUT_DIR, exist_ok=True)

# Shared catalog cache — downloaded once, filtered per context.
SHARED_CATALOG_CACHE = os.path.join(INPUT_DIR, 'etas_base_seismicity.parquet')

# ── Per-context cutoff dates ───────────────────────────────────────────────────
# timewindow_end for each inversion = the moment auxiliary/training data ends.
# Events strictly before this timestamp are used for parameter estimation.
#
# 'cascadia' is a whole-region background fit (not tied to a single
# aftershock sequence's case-study benchmark -- MTJ_2024_M7 is benchmarked
# through the California polygon in config.py/CASE_STUDIES on purpose).
# Cutoff chosen as the MTJ 2024 mainshock origin time so this region-level
# prior is directly comparable to that case study's forecast time if needed
# later -- adjust freely if a different cutoff makes more sense.
CONTEXTS = {
    'cascadia': '2024-12-04T22:00:00',
}

# Subset to invert; edit to rebuild only specific contexts.
BUILD_CONTEXTS = list(CONTEXTS.keys())

# Set True to re-run even if parameters_{name}.json already exists.
FORCE_RERUN = True
SAVE_SPATIAL = True

#%%
# ── Download shared catalog (once) ────────────────────────────────────────────
# Covers the full span from auxiliary_start through the latest context cutoff.
# load_background_seismicity() uses recursive bisection to stay under the
# USGS 20,000-event limit, caching to SHARED_CATALOG_CACHE as parquet.

_aux_start  = config.ETAS_INVERSION_CONFIG['auxiliary_start']
_m_ref      = config.ETAS_INVERSION_CONFIG['m_ref']

# Derive lon/lat bounds from the shape polygon for the USGS bounding box query.
_shape_lats = [pt[0] for pt in config.ETAS_INVERSION_CONFIG['shape_coords']]
_shape_lons = [pt[1] for pt in config.ETAS_INVERSION_CONFIG['shape_coords']]
_query_bounds = (
    min(_shape_lons) - 0.5,   # lon_min
    max(_shape_lons) + 0.5,   # lon_max
    min(_shape_lats) - 0.5,   # lat_min
    max(_shape_lats) + 0.5,   # lat_max
)

_start_year = pd.Timestamp(_aux_start).year
_end_year   = max(pd.Timestamp(cutoff).year for cutoff in CONTEXTS.values())

print(f"Loading shared Cascadia seismicity catalog  M≥{_m_ref}  "
      f"{_start_year}–{_end_year}  bounds={_query_bounds}")

raw_catalog = load_background_seismicity(
    cache_path    = SHARED_CATALOG_CACHE,
    bounds        = _query_bounds,
    start_year    = _start_year,
    end_year      = _end_year,
    min_mag       = _m_ref,
    force_refresh = False,
)

# Normalise to ETAS column format: id, latitude, longitude, time, magnitude.
raw_catalog = raw_catalog.copy()
raw_catalog.insert(0, 'id', range(len(raw_catalog)))
raw_catalog = raw_catalog.rename(columns={'mag': 'magnitude'})
raw_catalog['time'] = pd.to_datetime(raw_catalog['time'], utc=True).dt.tz_convert(None)
raw_catalog = raw_catalog[['id', 'latitude', 'longitude', 'time', 'magnitude']]
raw_catalog = raw_catalog[raw_catalog['magnitude'] >= _m_ref].reset_index(drop=True)

print(f"  Shared catalog: {len(raw_catalog):,} events  "
      f"{raw_catalog['time'].min().date()} → {raw_catalog['time'].max().date()}")

#%%
# ── Magnitude distribution (shared catalog) ───────────────────────────────────
# Verify that the catalog appears complete down to m_ref.

bins = np.arange(
    np.floor(raw_catalog['magnitude'].min() * 10) / 10,
    raw_catalog['magnitude'].max() + 0.15,
    0.1,
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw_catalog['magnitude'], bins=bins, color='steelblue',
        edgecolor='white', linewidth=0.4)
ax.axvline(_m_ref, color='crimson', linestyle='--', linewidth=1.5,
           label=f'$M_{{ref}}$ = {_m_ref}')
ax.set_yscale('log')
ax.set_xlabel('Magnitude')
ax.set_ylabel('Count')
ax.set_title('Magnitude distribution — shared Cascadia ETAS catalog (verify completeness above $M_{ref}$)')
ax.legend()
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'magnitude_distribution_shared.png')
fig.savefig(fig_path, dpi=150)
plt.show()
print(f"  Magnitude distribution saved: {fig_path}")

#%%
# ── Per-context inversion loop ────────────────────────────────────────────────

for context_name in BUILD_CONTEXTS:
    cutoff_str  = CONTEXTS[context_name]
    tagged_id   = config.etas_output_id(context_name, config.ETAS_INVERSION_CONFIG)
    output_json = os.path.join(OUTPUT_DIR, f'parameters_{tagged_id}.json')

    print(f"\n{'='*70}")
    print(f"Context: {context_name}  (cutoff: {cutoff_str})")
    print(f"{'='*70}")

    if os.path.exists(output_json) and not FORCE_RERUN:
        print(f"  Output already exists: {output_json}")
        print("  Set FORCE_RERUN = True to re-run.")
        continue

    # -- Filter shared catalog to strictly before the cutoff ------------------
    cutoff_ts = pd.Timestamp(cutoff_str)
    catalog_df = (
        raw_catalog[raw_catalog['time'] < cutoff_ts]
        .reset_index(drop=True)
        .copy()
    )
    catalog_df['id'] = range(len(catalog_df))   # re-index after filter

    print(f"  Catalog after cutoff filter: {len(catalog_df):,} events  "
          f"({catalog_df['time'].min().date()} → {catalog_df['time'].max().date()})")

    if len(catalog_df) < 500:
        print(f"  WARNING: only {len(catalog_df)} events — parameter estimates "
              f"may be unreliable.  Consider lowering m_ref or extending auxiliary_start.")

    catalog_tag = config.etas_catalog_tag(context_name, config.ETAS_INVERSION_CONFIG)
    catalog_csv = os.path.join(INPUT_DIR, f'catalog_{catalog_tag}.csv')
    catalog_df.to_csv(catalog_csv, index=False)
    print(f"  Catalog saved: {catalog_csv}")

    # -- Build inversion metadata ---------------------------------------------
    inversion_metadata = dict(config.ETAS_INVERSION_CONFIG)
    inversion_metadata['timewindow_end'] = cutoff_str
    inversion_metadata['id']             = tagged_id
    inversion_metadata['catalog']        = catalog_df
    inversion_metadata['data_path']      = OUTPUT_DIR + os.sep
    inversion_metadata.pop('fn_catalog', None)
    print("Free_background: ", inversion_metadata['free_background'])
    print("Free_productivity: ", inversion_metadata['free_productivity'])

    print(f"\n  Inversion configuration:")
    _display = {k: v for k, v in inversion_metadata.items()
                if k not in ('catalog', 'theta_0', 'shape_coords')}
    for k, v in _display.items():
        print(f"    {k}: {v}")
    print(f"    theta_0: {config.ETAS_INVERSION_CONFIG['theta_0']}")

    # -- Run inversion --------------------------------------------------------
    set_up_logger(level=logging.INFO)

    print(f"\n  Running ETAS inversion (id='{tagged_id}')…")
    print("  This typically takes several minutes.\n")

    calculation = ETASParameterCalculation(inversion_metadata)
    calculation.prepare()
    parameters  = calculation.invert()
    calculation.store_results(OUTPUT_DIR + os.sep, store_pij=False, store_spatial_fields=SAVE_SPATIAL)

    print(f"\n  Inversion complete → {output_json}")

    # -- Quick verification via EtasPriorUpdater ------------------------------
    print(f"\n  Verifying with EtasPriorUpdater…")
    with open(output_json) as fh:
        result = json.load(fh)

    print("  Inverted ETAS parameters:")
    for k, v in result.get('final_parameters', {}).items():
        if v is not None:
            print(f"    {k:15s}: {v:.6f}")
        else:
            print(f"    {k:15s}: None")

    hist = pd.read_csv(catalog_csv, parse_dates=['time'])
    hist['time'] = pd.to_datetime(hist['time']).dt.tz_localize(None)

    updater = EtasPriorUpdater.from_inversion_json(
        json_path  = output_json,
        catalog_df = hist,
        **config.ETAS_UPDATER_CONFIG,
    )
    forecast_time = pd.Timestamp(result['timewindow_end'])
    prior = updater.update(forecast_time)
    print(f"  Baseline prior at {forecast_time}:")
    print(f"    grid shape : {prior.grid.shape}")
    print(f"    grid max   : {prior.grid.max():.4e}")
    print(f"    grid sum   : {prior.grid.sum():.6f}  (should be ~1.0)")

print(f"\n{'='*70}")
print("All contexts complete.")
print(f"Outputs in: {OUTPUT_DIR}")
for context_name in BUILD_CONTEXTS:
    tagged_id = config.etas_output_id(context_name, config.ETAS_INVERSION_CONFIG)
    p = os.path.join(OUTPUT_DIR, f'parameters_{tagged_id}.json')
    status = 'OK' if os.path.exists(p) else 'MISSING'
    print(f"  [{status}]  parameters_{tagged_id}.json")
print(f"{'='*70}")
