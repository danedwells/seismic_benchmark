"""
Shared parsing tools for EPIC replay logs (one <anss_id>.log per replayed ComCat event).

A log can hold several real EPIC events (an aftershock sequence, or one event "split" in two), so the
solution for a ComCat event is found in two steps (find_solution):
  1. match_event:        pick the EPIC event that is the ComCat event: an alerted version within
                         MAX_DT_S of the ComCat origin time AND within MAX_KM of its location; if several
                         EPIC events qualify, the one with the closest origin time.
  2. first_at_stations:  take that EPIC event's first alerted version with at least n_stations stations.

Replaces the per-script parse_log copies and the matching in python-scripts-selected/parse_etas_logs.py (kept
unchanged as the reference).  All times are naive UTC.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MAX_DT_S, MAX_KM = 30, 100

# Fields of an E:I: line after the log prefix (Amy's EI_cols without the u1-u4 prefix tokens)
EI_COLS = ['eventid', 'ver', 'lat', 'lon', 'depth', 'mag', 'time', 'latu', 'lonu', 'depu', 'magu', 'timeu',
           'lk', 'nTb', 'nSb', 'nT', 'nS', 'ave', 'rms', 'fitOK', 'splitOK', 'near', 'statrig', 'active',
           'inact', 'nsta', 'percent', 'prcntOK', 'mindist', 'maxdist', 'distOK', 'azspan', 'MOK', 'nSOK',
           'LOK', 'Tdif', 'tpave', 'pdave', 'TF', 'TOK', 'AZOK', 'AOK', 'Ast', 'alert_time']


def _utc(t):
    """Timestamp -> naive UTC."""
    t = pd.Timestamp(t)
    return t.tz_convert('UTC').tz_localize(None) if t.tzinfo else t


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _lines(path, tag):
    """Token lists of every log line containing `tag` (text after the tag)."""
    for line in Path(path).open(errors='replace'):
        _, sep, msg = line.partition(tag)
        if sep:
            yield msg.split()


def read_ei(path):
    """
    One row per (eventid, ver) E:I: line, including the E:I:F: line (an event's first alert version; flagged in
    `first_alert`).  Only real events (eventid > 0; negative ids are proto events).  Lines with an unexpected
    number of fields are skipped.  Columns are EI_COLS, numeric except `time` and `alert_time` (datetimes).
    """
    rows = []
    for f in _lines(path, '| INFO | E:I:'):
        first = bool(f) and f[0] == 'F:'
        if first:
            f = f[1:]
        if len(f) == len(EI_COLS) and f[0].lstrip('-').isdigit() and int(f[0]) > 0:
            rows.append(f + [first])
    df = pd.DataFrame(rows, columns=EI_COLS + ['first_alert'])
    for c in EI_COLS:
        df[c] = pd.to_datetime(df[c]) if c in ('time', 'alert_time') else pd.to_numeric(df[c], errors='coerce')
    return df.astype({'eventid': int, 'ver': int, 'nT': int, 'nS': int, 'Ast': int}) if len(df) else df


def read_eit(path):
    """
    One row per trigger per (eventid, ver) E:I:T: line, columns from the log's own header, numeric where possible,
    plus `t` = trigger time as epoch seconds (UTC).  Only real events (eventid > 0).
    """
    hdr, rows = None, []
    for f in _lines(path, '| INFO | E:I:T:'):
        if f and f[0] == 'H:':
            hdr = hdr or f[1:]
        elif hdr and len(f) >= len(hdr) and f[0].lstrip('-').isdigit() and int(f[0]) > 0:
            rows.append(f[:len(hdr)])
    df = pd.DataFrame(rows, columns=hdr)
    for c in df.columns:
        num = pd.to_numeric(df[c], errors='coerce')
        if c != 'trigger_time' and num.notna().all():
            df[c] = num
    if len(df):
        df['t'] = pd.to_datetime(df['trigger_time']).dt.tz_localize('UTC').map(lambda x: x.timestamp())
    return df


def match_event(ei, ref_time, ref_lat, ref_lon, max_dt_s=MAX_DT_S, max_km=MAX_KM, alerted_only=True):
    """
    EPIC eventid that corresponds to the reference (ComCat) event, or None.  A version qualifies if its origin time
    is within max_dt_s and its location within max_km of the reference; among qualifying versions the one with the
    smallest |origin time difference| decides.  alerted_only restricts to alerted versions (Ast == 1).
    """
    if alerted_only:
        ei = ei[ei.Ast == 1]
    if ei.empty:
        return None
    dt = (ei.time - _utc(ref_time)).dt.total_seconds().abs()
    km = _haversine_km(ref_lat, ref_lon, ei.lat.values, ei.lon.values)
    ok = (dt <= max_dt_s) & (km <= max_km)
    return int(ei.eventid[dt[ok].idxmin()]) if ok.any() else None


def first_at_stations(ei, eventid, n_stations, alerted_only=True):
    """First version (row) of EPIC event `eventid` with nS >= n_stations (alerted only if alerted_only), or None."""
    ev = ei[(ei.eventid == eventid) & (ei.nS >= n_stations)]
    if alerted_only:
        ev = ev[ev.Ast == 1]
    return ev.sort_values('ver').iloc[0] if len(ev) else None


def find_solution(path, ref_time, ref_lat, ref_lon, n_stations, **match_kw):
    """EPIC solution (row of read_ei) for a reference event at n_stations stations, or None (no log, no match, too few stations)."""
    if not Path(path).exists():
        return None
    ei = read_ei(path)
    eventid = match_event(ei, ref_time, ref_lat, ref_lon, **match_kw)
    return None if eventid is None else first_at_stations(ei, eventid, n_stations, match_kw.get('alerted_only', True))


def summarize(path, ref_time, ref_lat, ref_lon, **match_kw):
    """
    Per-log status for the ComCat event the log was replayed for:
      log (file exists), detected (>= 1 real event), alerted (>= 1 real event alerted), num_alert_events (distinct
      alerted EPIC events; > 1 means a split or a neighbouring event), matched_eventid (the alerted EPIC event that
      matches the reference, or None).  `alerted` alone does NOT mean the target event alerted; use matched_eventid.
    """
    out = dict(log=Path(path).exists(), detected=False, alerted=False, num_alert_events=0, matched_eventid=None)
    if not out['log']:
        return out
    ei = read_ei(path)
    out.update(detected=len(ei) > 0, alerted=bool((ei.Ast == 1).any()) if len(ei) else False,
               num_alert_events=int(ei[ei.Ast == 1].eventid.nunique()) if len(ei) else 0,
               matched_eventid=match_event(ei, ref_time, ref_lat, ref_lon, **match_kw) if len(ei) else None)
    return out
