"""
Shared parsing tools for EPIC replay logs (one <anss_id>.log per replayed ComCat event).

A log can hold several real EPIC events (an aftershock sequence, or one event "split" in two), so the
solution for a ComCat event is found in two steps (find_solution):
  1. match_event:        pick the EPIC event that is the ComCat event: any version (alerted or not) within
                         MAX_DT_S of the ComCat origin time AND within MAX_KM of its location; the closest in
                         space and time wins.  A match that never alerted is still a match (summarize records
                         it as 'matched_not_alerted', formally a missed event).
  2. first_at_stations:  take that EPIC event's first alerted version with at least n_stations stations.

summarize / status_table give every ComCat event one status, so missed events are counted, not dropped.

Counts (A. Williamson): nT = number of triggers, nS = number of stations.  A station can have up to two triggers (e.g. broadband
HH plus strong-motion HN) and every trigger counts equally in EPIC's location and magnitude.  An alert needs at least 4 stations
(up to 8 triggers).  If several stations report at once the first alerted version can already have more than 4 stations although
a viable solution existed at 4, so "at n stations" here means "the first alerted version with nS >= n"; read r.nS for the actual count.

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
        # Get the first line (has E:I:F)
        if first:
            f = f[1:]
        # Exactly 44 fields, event id is an integer, event id is positive. Reject Trigger rows (T is not an integer)
        if len(f) == len(EI_COLS) and f[0].lstrip('-').isdigit() and int(f[0]) > 0:
            rows.append(f + [first])
    df = pd.DataFrame(rows, columns=EI_COLS + ['first_alert'])
    for c in EI_COLS:
        df[c] = pd.to_datetime(df[c]) if c in ('time', 'alert_time') else pd.to_numeric(df[c], errors='coerce')
    return df.astype({'eventid': int, 'ver': int, 'nT': int, 'nS': int, 'Ast': int}) if len(df) else df


def read_eit(path):
    """
    One row per trigger (not per station: a station can appear twice, e.g. HH and HN) per (eventid, ver) E:I:T: line, columns from the log's own header, numeric where possible,
    plus `t` = trigger time as epoch seconds (UTC).  Only real events (eventid > 0).
    """
    hdr, rows = None, []
    # Here we are getting Trigger lines
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


def match_event(ei, ref_time, ref_lat, ref_lon, max_dt_s=MAX_DT_S, max_km=MAX_KM):
    """
    EPIC eventid that corresponds to the reference (ComCat) event, or None.  A version qualifies if its origin time
    is within max_dt_s and its location within max_km of the reference.  Among qualifying versions the closest in
    space and time wins (smallest dt/max_dt_s + km/max_km); whether the event alerted plays no role (Amy Williamson:
    matching is retrospective, closest in space and time within the tolerances).
    """
    if ei.empty:
        return None
    dt = (ei.time - _utc(ref_time)).dt.total_seconds().abs()
    km = pd.Series(_haversine_km(ref_lat, ref_lon, ei.lat.values, ei.lon.values), index=ei.index)
    hit = (dt <= max_dt_s) & (km <= max_km)
    if not hit.any():
        return None
    score = (dt / max_dt_s + km / max_km)[hit]
    return int(ei.eventid[score.idxmin()])


def first_at_stations(ei, eventid, n_stations, alerted_only=True):
    """
    First version (row) of EPIC event `eventid` with nS (stations, not triggers) >= n_stations, alerted only if alerted_only, or None.
    Alerts need >= 4 stations, so with alerted_only any n_stations <= 4 gives the first alerted version, whose nS can exceed n_stations.
    """
    ev = ei[(ei.eventid == eventid) & (ei.nS >= n_stations)]
    if alerted_only:
        ev = ev[ev.Ast == 1]
    return ev.sort_values('ver').iloc[0] if len(ev) else None


def find_solution(path, ref_time, ref_lat, ref_lon, n_stations, alerted_only=True, **match_kw):
    """EPIC solution (row of read_ei) for a reference event at n_stations stations, or None (see summarize for why)."""
    if not Path(path).exists():
        return None
    ei = read_ei(path)
    eventid = match_event(ei, ref_time, ref_lat, ref_lon, **match_kw)
    return None if eventid is None else first_at_stations(ei, eventid, n_stations, alerted_only)


def summarize(path, ref_time, ref_lat, ref_lon, n_stations=None, **match_kw):
    """
    Status of the ComCat event a log was replayed for (a dict; status_table stacks these).  `status` is one of
      no_log                    no log file
      not_detected              the log has no real EPIC event
      unmatched                 EPIC events exist, but none within max_dt_s / max_km of the ComCat event (missed)
      matched_not_alerted       the matching EPIC event never alerted
      matched_too_few_stations  it alerted, but no alerted version has >= n_stations stations (only if n_stations given)
      ok                        matched, alerted (and reached n_stations if given): find_solution returns a row
    Other fields: detected, alerted (ANY event in the log alerted; does not mean the target did), matched_eventid,
    matched_alerted, num_alert_events (distinct alerted EPIC events), unmatched_eventids (all other real EPIC events
    in the log) and unmatched_alerted_eventids (those that alerted: splits / neighbouring events / false alerts).
    """
    out = dict(status='no_log', detected=False, alerted=False, matched_eventid=None, matched_alerted=False,
               num_alert_events=0, unmatched_eventids=[], unmatched_alerted_eventids=[])
    if not Path(path).exists():
        return out
    ei = read_ei(path)
    out.update(status='not_detected', detected=len(ei) > 0)
    if ei.empty:
        return out
    alerted_ids = set(ei[ei.Ast == 1].eventid)
    eventid = match_event(ei, ref_time, ref_lat, ref_lon, **match_kw)
    others = sorted(set(ei.eventid) - {eventid})
    out.update(alerted=bool(alerted_ids), num_alert_events=len(alerted_ids), matched_eventid=eventid,
               unmatched_eventids=others, unmatched_alerted_eventids=[e for e in others if e in alerted_ids])
    if eventid is None:
        out['status'] = 'unmatched'
        return out
    out['matched_alerted'] = eventid in alerted_ids
    if not out['matched_alerted']:
        out['status'] = 'matched_not_alerted'
    elif n_stations is not None and first_at_stations(ei, eventid, n_stations) is None:
        out['status'] = 'matched_too_few_stations'
    else:
        out['status'] = 'ok'
    return out


def status_table(log_dir, ref, n_stations=None, **match_kw):
    """
    summarize() for every reference event.  `ref` is a DataFrame with columns anss_id, time, lat, lon; the log of each
    is log_dir/<anss_id>.log.  Count the outcomes with status_table(...).status.value_counts().
    """
    return pd.DataFrame([dict(anss_id=r.anss_id, **summarize(Path(log_dir) / f'{r.anss_id}.log', r.time, r.lat, r.lon,
                                                              n_stations, **match_kw)) for r in ref.itertuples()])
