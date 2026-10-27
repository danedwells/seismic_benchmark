#%%
# =============================================================================
# download_isolated.py — isolated M>=3 events along the Wasatch Front + their .run trigger files
# =============================================================================
# Tests the priors on events where the ETAS *background* matters (no recent nearby seismicity), unlike the Magna
# aftershock sequence where the triggered terms dominate. Selection (ComCat, eventtype=earthquake):
#   M >= MIN_MAG, START..END, inside BOX and within MAX_FAULT_KM of the Wasatch fault trace,
#   isolated = no M>=2.5 event within ISO_KM in the previous ISO_DAYS days (M>=2.5 catalog: data/background_earthquakes_M2.5.parquet).
# Outputs: data/wasatch_isolated/{catalog.parquet, run_files/*.run}  (.run files from USGS phase picks, as download_triggers.py)
#
# Usage: python Magna/download_isolated.py        (MAGNA_ISO_MAG=2.5 for the larger M>=2.5, west-of-trace set)
# =============================================================================
import io
import os
import sys

import numpy as np
import pandas as pd
import requests

from benchmark.metrics import _haversine_km
from benchmark.usgs import build_run_files_for_case_study
from faults import wasatch_fault, east_of_wasatch
from variant import ISO_MIN_MAG, ISO_TAG   # MAGNA_ISO_MAG=2.5 -> its own data/wasatch_isolated_M2.5

MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(MAGNA_DIR, 'data', f'wasatch_isolated{ISO_TAG}')
RUN_DIR   = os.path.join(OUT_DIR, 'run_files')
os.makedirs(RUN_DIR, exist_ok=True)

MIN_MAG, START, END = ISO_MIN_MAG, '2000-01-01', '2020-03-18'      # END = Magna mainshock day: the priors are built from earlier seismicity
BOX = dict(minlongitude=-112.8, maxlongitude=-111.0, minlatitude=39.5, maxlatitude=42.0)
MAX_FAULT_KM = 30
WEST_ONLY    = ISO_TAG != ''     # new (non-default magnitude) sets: only events west of the fault trace, so no picks are fetched for the rest
ISO_KM, ISO_DAYS = 20, 14

#%%
resp = requests.get('https://earthquake.usgs.gov/fdsnws/event/1/query', timeout=120, params={
    'format': 'csv', 'eventtype': 'earthquake', 'orderby': 'time-asc', 'starttime': START, 'endtime': END,
    'minmagnitude': MIN_MAG, **BOX})
resp.raise_for_status()
ev = pd.read_csv(io.StringIO(resp.text))[['id', 'time', 'latitude', 'longitude', 'depth', 'mag']]
ev['time'] = pd.to_datetime(ev['time'], utc=True).dt.tz_convert(None)

trace = np.vstack([np.array(g.coords) for gg in wasatch_fault() for g in (gg.geoms if hasattr(gg, 'geoms') else [gg])])
ev['d_fault_km'] = [_haversine_km(la, lo, trace[:, 1], trace[:, 0]).min() for la, lo in zip(ev.latitude, ev.longitude)]

bg = pd.read_parquet(os.path.join(MAGNA_DIR, 'data', 'background_earthquakes_M2.5.parquet'))
bg['time'] = pd.to_datetime(bg['time'], utc=True).dt.tz_convert(None)
def n_prior(r):
    m = (bg['time'] < r.time) & (bg['time'] >= r.time - pd.Timedelta(days=ISO_DAYS))
    return int((_haversine_km(r.latitude, r.longitude, bg.latitude[m].values, bg.longitude[m].values) < ISO_KM).sum())
ev['n_prior'] = [n_prior(r) for r in ev.itertuples()]

ev['east'] = east_of_wasatch(ev.latitude.values, ev.longitude.values) == 1
sel = ev[(ev.d_fault_km <= MAX_FAULT_KM) & (ev.n_prior == 0) & ~(ev.east & WEST_ONLY)].reset_index(drop=True)
print(f'{len(ev)} M>={MIN_MAG:g} events in the box; {(ev.d_fault_km <= MAX_FAULT_KM).sum()} within {MAX_FAULT_KM} km of the fault; '
      f'{len(sel)} also isolated ({ISO_KM} km / {ISO_DAYS} d)')

#%%
written = build_run_files_for_case_study(sel, RUN_DIR, max_dist_deg=5.0, skip_existing=True)
sel['has_run'] = sel['id'].isin(written)
sel.to_parquet(os.path.join(OUT_DIR, 'catalog.parquet'), index=False)
print(f"\n{sel.has_run.sum()}/{len(sel)} events have a .run file")
print(sel[['id', 'time', 'mag', 'd_fault_km', 'has_run']].to_string(index=False))
