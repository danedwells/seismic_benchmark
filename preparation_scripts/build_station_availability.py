#!/usr/bin/env python3
"""
Pre-compute per-event active station availability for the benchmark.

For each event in the benchmark catalog:
  1. Query ShakeAlert networks for stations installed within SEARCH_RADIUS_KM
     of the USGS epicenter at the time of the event.
  2. All triggered stations (from the .run file) are included unconditionally —
     they had data by definition.
  3. Each non-triggered candidate is probed: request PROBE_DURATION_S of ??Z
     waveform data starting at the first trigger time.  Stations that return
     any data are considered "active" for this event.
  4. Results are accumulated and written to a parquet cache.

The cache is loaded at benchmark runtime and used to set
EPIC_PARAMS.station_inventory per event, enabling the activity eligibility
check to use actual data availability rather than a proxy inventory.

Output
------
data/reference/station_availability_cache.parquet
  Columns: event_id (str), station (str), network (str),
           longitude (float64), latitude (float64)

Usage
-----
    cd seismic_benchmark
    python preparation_scripts/build_station_availability.py

The script is resumable: events already in the cache are skipped.
Reduce MAX_WORKERS or increase INTER_EVENT_SLEEP_S if IRIS rate-limits you.
"""

import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE         = Path(__file__).parent.parent   # seismic_benchmark/

# Main benchmark
RUN_DIR      = HERE / 'data' / 'run_files'
CATALOG_PATH = HERE / 'data' / 'reference' / 'bEPIC_testing_catalog.txt'
OUTPUT_PATH  = HERE / 'data' / 'reference' / 'station_availability_cache.parquet'

# CASE_STUDY = 'ElMayor'

# RUN_DIR      = HERE / 'data' / 'case_studies' / f'{CASE_STUDY}' / 'run_files'
# #CATALOG_PATH     = HERE / 'data' / 'case_studies' / f'{CASE_STUDY}' / f'{CASE_STUDY}_2019_catalog.parquet'
# CATALOG_PATH     = HERE / 'data' / 'case_studies' / f'{CASE_STUDY}' / f'El_Mayor-Cucapah_2010_catalog.parquet'
# #CATALOG_PATH     = HERE / 'data' / 'case_studies' / f'{CASE_STUDY}' / f'{CASE_STUDY}_2022_catalog.parquet'
# OUTPUT_PATH  = HERE / 'data' / 'case_studies' / f'{CASE_STUDY}' / 'station_availability_cache.parquet'

# ShakeAlert contributing networks (western US)
# TODO - from benchmark catalog - replace with full list
SHAKEALERT_NETWORKS = 'AZ,BC,BK,CC,CE,CI,CN,IU,NC,NN,NP,SB,SN,UO,US,UW'

FDSN_CLIENT         = 'EARTHSCOPE'
SEARCH_RADIUS_KM    = 300.0  # generous upper bound on R_MAX; runtime filtering
                              # by actual R_MAX happens inside EPIC_locate_prelim
PROBE_DURATION_S    = 1.0    # waveform window length (seconds)
MAX_WORKERS         = 10     # concurrent threads for station probes per event
INTER_EVENT_SLEEP_S = 0.2    # courtesy pause between events
FLUSH_EVERY         = 50     # write parquet to disk every N events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_catalog_date(s):
    """Parse '2018-09-30-14:41:29.510-GMT' → UTCDateTime."""
    s = str(s).replace('-GMT', '')
    parts = s.split('-', 3)   # ['2018', '09', '30', '14:41:29.510']
    return UTCDateTime(f"{parts[0]}-{parts[1]}-{parts[2]}T{parts[3]}")


def load_reference_catalog(path):
    """Return dict {postgres_id (str): (lat, lon, UTCDateTime)}."""
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip() for c in df.columns] # get rid of whitespaces in col names
    out = {}
    for _, row in df.iterrows():
        pid = str(row['postgres id']).strip()
        out[pid] = (
            float(row['ANSS lat']),
            float(row['ANSS lon']),
            _parse_catalog_date(row['ANSS date']),
        )
    return out

def load_case_study_catalog(path):
    """Return dict {event_id (str): (lat, lon, UTCDateTime)} from a case-study parquet."""
    df = pd.read_parquet(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row['id'])] = (
            float(row['latitude']),
            float(row['longitude']),
            UTCDateTime(row['time'].isoformat()),
        )
    return out

def event_info_from_run(run_path):
    """
    Return (t_first_trig, triggered_df) for a .run file.

    t_first_trig  — UTCDateTime of the earliest trigger across all versions
    triggered_df  — unique triggered stations: station, network, longitude, latitude
    """
    df = pd.read_csv(run_path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    t_first = UTCDateTime(float(df['trigger_time'].min()))
    triggered = (df[['station', 'network', 'longitude', 'latitude']]
                 .drop_duplicates(subset=['station', 'network'])
                 .reset_index(drop=True))
    return t_first, triggered


def query_candidates(client, lat, lon, t_event, radius_km, networks):
    """
    Query FDSN for ShakeAlert stations installed within radius_km of
    (lat, lon) at t_event.  Returns DataFrame: station, network, longitude, latitude.
    """
    radius_deg = radius_km / 111.0
    try:
        inv = client.get_stations(
            network=networks,
            station='*',
            channel='??Z',
            latitude=lat,
            longitude=lon,
            maxradius=radius_deg,
            starttime=t_event,
            endtime=t_event,
            level='station',
        )
    except Exception as exc:
        print(f"  get_stations failed ({type(exc).__name__}): {exc}")
        return pd.DataFrame(columns=['station', 'network', 'longitude', 'latitude'])

    rows = []
    for net in inv:
        for sta in net:
            rows.append({'station': sta.code, 'network': net.code,
                         'longitude': sta.longitude, 'latitude': sta.latitude})
    return pd.DataFrame(rows, columns=['station', 'network', 'longitude', 'latitude'])


def _probe_one(fdsn_client_name, net, sta, t_start, duration):
    """Return True if the station returns any ??Z trace in the probe window."""
    try:
        client = Client(fdsn_client_name)  # own Client per thread — libmseed isn't thread-safe
        st = client.get_waveforms(
            network=net, station=sta, location='*', channel='??Z',
            starttime=t_start, endtime=t_start + duration,
        )
        return len(st) > 0
    except Exception:
        return False


def probe_active(fdsn_client_name, candidates_df, t_probe, max_workers, duration):
    """
    Probe candidate stations in parallel.
    Returns the subset of candidates_df for which waveform data was found.
    """
    if candidates_df.empty:
        return candidates_df.copy()

    active_idx = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_probe_one, fdsn_client_name, row.network, row.station,
                        t_probe, duration): idx
            for idx, row in candidates_df.iterrows()
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                if fut.result():
                    active_idx.append(idx)
            except Exception:
                pass

    return candidates_df.loc[sorted(active_idx)].reset_index(drop=True)


def _flush(chunks, existing_df, path):
    """Append chunks to existing cache DataFrame and write parquet."""
    if not chunks:
        return existing_df
    new_df = pd.concat(chunks, ignore_index=True)
    combined = (pd.concat([existing_df, new_df], ignore_index=True)
                if not existing_df.empty else new_df)
    combined.to_parquet(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FLUSH_EVERY = 1
def main():
    client = Client(FDSN_CLIENT)

    # Resumability: skip events already in the cache
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        done_ids = set(existing['event_id'].astype(str).unique())
        print(f"Resuming: {len(done_ids)} events already cached.")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    # Get catalogs of events - need station availability per event
    catalog   = load_reference_catalog(CATALOG_PATH)
    #catalog = load_case_study_catalog(CATALOG_PATH)
    run_files = sorted(RUN_DIR.glob('*.run'))
    pending   = [f.stem for f in run_files if f.stem not in done_ids]
    print(f"{len(pending)} events to process ({len(run_files)} total).\n")

    chunks = []

    for i, eid in enumerate(pending):
        if eid not in catalog:
            print(f"  [{i+1}/{len(pending)}] {eid}: not in reference catalog — skipping")
            continue

        lat, lon, t_event = catalog[eid]
        t_first, triggered = event_info_from_run(RUN_DIR / f'{eid}.run')
        triggered_set = set(zip(triggered['station'], triggered['network']))

        candidates = query_candidates(client, lat, lon, t_event,
                                      SEARCH_RADIUS_KM, SHAKEALERT_NETWORKS)

        is_triggered = candidates.apply(
            lambda r: (r['station'], r['network']) in triggered_set, axis=1
        )
        trig_in_radius   = candidates[is_triggered].reset_index(drop=True)
        untrig_in_radius = candidates[~is_triggered].reset_index(drop=True)

        active_untrig = probe_active(FDSN_CLIENT, untrig_in_radius, t_first,
                                     MAX_WORKERS, PROBE_DURATION_S)

        event_active = pd.concat([trig_in_radius, active_untrig], ignore_index=True)
        event_active.insert(0, 'event_id', eid)
        chunks.append(event_active)

        print(
            f"  [{i+1}/{len(pending)}] {eid}"
            f"  candidates={len(candidates)}"
            f"  triggered={len(trig_in_radius)}"
            f"  probed={len(untrig_in_radius)}"
            f"  active={len(event_active)}"
        )

        if (i + 1) % FLUSH_EVERY == 0:
            existing = _flush(chunks, existing, OUTPUT_PATH)
            chunks = []
            print(f"  → flushed to {OUTPUT_PATH.name}\n")

        time.sleep(INTER_EVENT_SLEEP_S)

    _flush(chunks, existing, OUTPUT_PATH)
    print(f"\nDone. Cache written to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
