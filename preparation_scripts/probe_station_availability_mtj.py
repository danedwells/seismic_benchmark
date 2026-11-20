#!/usr/bin/env python3
"""
Isolated station-availability probe for the MTJ box (diagnostic).

build_station_availability.py probes waveform availability on EARTHSCOPE only, so BK/NC stations
(archived at NCEDC) only appear in the cache when they triggered. This script redoes the probe for
just the MTJ region:

  events    — benchmark events whose ANSS (true) location is inside BOX
  stations  — every ShakeAlert-network ??Z station installed at event time inside BOX
  probe     — 1 s of ??Z data at the first-trigger time, tried on NCEDC then EARTHSCOPE;
              a station is active if either returns data. Triggered stations are active by definition.

Writes data/california/reference/station_availability_mtj_probe.parquet — a separate file; the main
cache is not touched. One row per (event, candidate station), columns:
  event_id, station, network, longitude, latitude, triggered, active, client
Filter to active == True (and drop triggered/client) for a cache-compatible frame.

Usage (resumable; one flush per event):
    cd seismic_benchmark
    python preparation_scripts/probe_station_availability_mtj.py
    N_EVENTS=3 python preparation_scripts/probe_station_availability_mtj.py     # smoke test, 3 events
"""

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

HERE         = Path(__file__).parent.parent
RUN_DIR      = HERE / 'data' / 'california' / 'run_files'
CATALOG_PATH = HERE / 'data' / 'california' / 'reference' / 'bEPIC_testing_catalog.txt'
OUTPUT_PATH  = HERE / 'data' / 'california' / 'reference' / 'station_availability_mtj_probe.parquet'

BOX = dict(minlongitude=-125.0, maxlongitude=-122.0, minlatitude=38.4, maxlatitude=41.2)
NETWORKS      = 'AZ,BC,BK,CC,CE,CI,CN,IU,NC,NN,NP,SB,SN,UO,US,UW'
PROBE_CLIENTS = ('NCEDC', 'EARTHSCOPE')     # tried in order
PROBE_DURATION_S = 1.0
MAX_WORKERS   = 10
N_EVENTS      = int(os.environ['N_EVENTS']) if 'N_EVENTS' in os.environ else None


def load_catalog(path):
    """{postgres_id (str): (lat, lon, UTCDateTime)} from the bEPIC testing catalog."""
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for _, r in df.iterrows():
        y, mo, d, hms = str(r['ANSS date']).replace('-GMT', '').split('-', 3)
        out[str(r['postgres id']).strip()] = (float(r['ANSS lat']), float(r['ANSS lon']), UTCDateTime(f'{y}-{mo}-{d}T{hms}'))
    return out


def in_box(lat, lon):
    return BOX['minlatitude'] <= lat <= BOX['maxlatitude'] and BOX['minlongitude'] <= lon <= BOX['maxlongitude']


def first_trigger_and_stations(run_path):
    """(UTCDateTime of earliest trigger, set of triggered (station, network))."""
    df = pd.read_csv(run_path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return UTCDateTime(float(df['trigger_time'].min())), set(zip(df['station'], df['network']))


def candidates_in_box(client, t_event):
    """Stations installed at t_event inside BOX: station, network, longitude, latitude."""
    inv = client.get_stations(network=NETWORKS, station='*', channel='??Z', starttime=t_event, endtime=t_event,
                              level='station', **BOX)
    rows = [{'station': s.code, 'network': n.code, 'longitude': s.longitude, 'latitude': s.latitude}
            for n in inv for s in n]
    return pd.DataFrame(rows, columns=['station', 'network', 'longitude', 'latitude']).drop_duplicates(['station', 'network'])


def probe_one(net, sta, t_start):
    """Name of the first client that returns ??Z data for the station, else None."""
    for name in PROBE_CLIENTS:
        try:
            st = Client(name).get_waveforms(network=net, station=sta, location='*', channel='??Z',
                                            starttime=t_start, endtime=t_start + PROBE_DURATION_S)
            if len(st) > 0:
                return name
        except Exception:
            pass
    return None


def main():
    catalog = load_catalog(CATALOG_PATH)
    events = [e for e, (lat, lon, _) in catalog.items() if in_box(lat, lon) and (RUN_DIR / f'{e}.run').exists()]
    events.sort(key=lambda e: catalog[e][2])
    done = set()
    existing = pd.DataFrame()
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        done = set(existing['event_id'].astype(str))
    pending = [e for e in events if e not in done][:N_EVENTS]
    print(f'{len(events)} events with a true location in the box; {len(done)} already probed; running {len(pending)}\n')

    client = Client('EARTHSCOPE')   # station metadata only
    for i, eid in enumerate(pending, 1):
        _, _, t_event = catalog[eid]
        t_first, trig = first_trigger_and_stations(RUN_DIR / f'{eid}.run')
        cand = candidates_in_box(client, t_event)
        cand['triggered'] = [(s, n) in trig for s, n in zip(cand.station, cand.network)]

        clients = {}
        probe_rows = cand[~cand.triggered]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(probe_one, r.network, r.station, t_first): idx for idx, r in probe_rows.iterrows()}
            for f in as_completed(futs):
                clients[futs[f]] = f.result()
        cand['client'] = [('triggered' if t else clients.get(idx)) for idx, t in zip(cand.index, cand.triggered)]
        cand['active'] = cand['client'].notna()
        cand.insert(0, 'event_id', eid)

        existing = pd.concat([existing, cand], ignore_index=True) if len(existing) else cand
        existing.to_parquet(OUTPUT_PATH, index=False)
        by = cand[cand.active].client.value_counts().to_dict()
        print(f'  [{i}/{len(pending)}] {eid}  candidates={len(cand)}  triggered={int(cand.triggered.sum())}  '
              f'active={int(cand.active.sum())}  via {by}', flush=True)
        time.sleep(0.2)
    print(f'\nDone. {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
