"""
Unit tests for benchmark/background.py.

The USGS-hitting internals (_count_events, _fetch_chunk) are monkeypatched
out everywhere so no network call is ever made. What's tested is the pure
control-flow logic: the recursive time-range bisection in _download_range
(the mechanism that keeps requests under USGS's 20,000-event cap), and the
cache-hit / cache-miss branching in load_background_seismicity.
"""
import pandas as pd
import pytest

from benchmark import background as bg


# ---------------------------------------------------------------------------
# _download_range — recursive bisection
# ---------------------------------------------------------------------------

def _fake_counts(count_map):
    """count_map: {(starttime, endtime): count}. Falls back to 0 if the
    exact bisected range wasn't pre-registered (shouldn't happen with the
    fixed 2020-01-01..2020-01-02 midpoint used below)."""
    def _stub(bounds, starttime, endtime, min_mag):
        return count_map.get((starttime, endtime), 0)
    return _stub


def test_download_range_no_split_when_under_limit(monkeypatch):
    fetch_calls = []
    monkeypatch.setattr(bg, '_count_events',
                        _fake_counts({('2020-01-01', '2020-01-02'): 500}))
    monkeypatch.setattr(bg, '_fetch_chunk',
                        lambda bounds, s, e, m: fetch_calls.append((s, e)) or
                        pd.DataFrame({'time': [], 'latitude': [], 'longitude': [],
                                      'depth': [], 'mag': []}))
    monkeypatch.setattr(bg.time, 'sleep', lambda s: None)

    chunks = []
    bg._download_range((-120, -119, 36, 37), '2020-01-01', '2020-01-02', 2.5, chunks)

    assert fetch_calls == [('2020-01-01', '2020-01-02')]
    assert len(chunks) == 1


def test_download_range_zero_events_fetches_nothing(monkeypatch):
    fetch_calls = []
    monkeypatch.setattr(bg, '_count_events', lambda *a, **k: 0)
    monkeypatch.setattr(bg, '_fetch_chunk', lambda *a, **k: fetch_calls.append(1))
    chunks = []
    bg._download_range((-120, -119, 36, 37), '2020-01-01', '2020-01-02', 2.5, chunks)
    assert fetch_calls == []
    assert chunks == []


def test_download_range_splits_when_over_limit(monkeypatch):
    """A range reporting > MAX_EVENTS_PER_REQUEST must bisect in time and
    fetch each half separately, never issuing a single over-limit request."""
    starttime, endtime = '2020-01-01T00:00:00', '2020-01-03T00:00:00'
    midpoint = pd.Timestamp(starttime) + (pd.Timestamp(endtime) - pd.Timestamp(starttime)) / 2
    mid_str = midpoint.strftime('%Y-%m-%dT%H:%M:%S')

    counts = {
        (starttime, endtime):  bg.MAX_EVENTS_PER_REQUEST + 1,   # too many -> split
        (starttime, mid_str):  100,                             # first half: fine
        (mid_str, endtime):    200,                              # second half: fine
    }
    monkeypatch.setattr(bg, '_count_events', _fake_counts(counts))

    fetch_calls = []
    def _fetch(bounds, s, e, m):
        fetch_calls.append((s, e))
        return pd.DataFrame({'time': [], 'latitude': [], 'longitude': [],
                             'depth': [], 'mag': []})
    monkeypatch.setattr(bg, '_fetch_chunk', _fetch)
    monkeypatch.setattr(bg.time, 'sleep', lambda s: None)

    chunks = []
    bg._download_range((-120, -119, 36, 37), starttime, endtime, 2.5, chunks)

    # Never fetched the full over-limit range directly -- only the two halves.
    assert (starttime, endtime) not in fetch_calls
    assert set(fetch_calls) == {(starttime, mid_str), (mid_str, endtime)}
    assert len(chunks) == 2


def test_download_range_recurses_until_under_limit(monkeypatch):
    """Nested splits: even after one bisection, a still-too-large half must
    split again rather than being fetched directly."""
    starttime, endtime = '2020-01-01T00:00:00', '2020-01-05T00:00:00'
    mid1 = pd.Timestamp(starttime) + (pd.Timestamp(endtime) - pd.Timestamp(starttime)) / 2
    mid1_str = mid1.strftime('%Y-%m-%dT%H:%M:%S')
    mid2 = pd.Timestamp(starttime) + (pd.Timestamp(mid1_str) - pd.Timestamp(starttime)) / 2
    mid2_str = mid2.strftime('%Y-%m-%dT%H:%M:%S')

    counts = {
        (starttime, endtime):    bg.MAX_EVENTS_PER_REQUEST + 100,  # split
        (starttime, mid1_str):   bg.MAX_EVENTS_PER_REQUEST + 1,    # still too big -> split again
        (mid1_str, endtime):     50,                                # fine
        (starttime, mid2_str):   10,                                 # fine
        (mid2_str, mid1_str):    10,                                 # fine
    }
    monkeypatch.setattr(bg, '_count_events', _fake_counts(counts))

    fetch_calls = []
    def _fetch(bounds, s, e, m):
        fetch_calls.append((s, e))
        return pd.DataFrame({'time': [], 'latitude': [], 'longitude': [],
                             'depth': [], 'mag': []})
    monkeypatch.setattr(bg, '_fetch_chunk', _fetch)
    monkeypatch.setattr(bg.time, 'sleep', lambda s: None)

    chunks = []
    bg._download_range((-120, -119, 36, 37), starttime, endtime, 2.5, chunks)

    assert (starttime, mid1_str) not in fetch_calls  # over-limit range never fetched directly
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# load_background_seismicity — cache-hit / cache-miss branching
# ---------------------------------------------------------------------------

def test_load_background_seismicity_uses_cache_without_downloading(tmp_path, monkeypatch):
    cache_path = tmp_path / 'bg.parquet'
    expected = pd.DataFrame({
        'time':      pd.to_datetime(['2020-01-01T00:00:00Z']),
        'latitude':  [37.0], 'longitude': [-120.0],
        'depth':     [5.0],  'mag':       [3.5],
    })
    expected.to_parquet(cache_path, index=False)

    def _boom(*a, **k):
        raise AssertionError("download_background_seismicity should not be called on cache hit")
    monkeypatch.setattr(bg, 'download_background_seismicity', _boom)

    df = bg.load_background_seismicity(str(cache_path), (-121, -119, 36, 38), 2019, 2020)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


def test_load_background_seismicity_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / 'bg.parquet'
    pd.DataFrame({'time': pd.to_datetime(['2020-01-01T00:00:00Z']),
                 'latitude': [37.0], 'longitude': [-120.0],
                 'depth': [5.0], 'mag': [3.5]}).to_parquet(cache_path, index=False)

    calls = []
    def _fake_download(bounds, start_year, end_year, min_mag, cache_path=None):
        calls.append((start_year, end_year))
        return pd.DataFrame({'time': [], 'latitude': [], 'longitude': [],
                             'depth': [], 'mag': []})
    monkeypatch.setattr(bg, 'download_background_seismicity', _fake_download)

    bg.load_background_seismicity(str(cache_path), (-121, -119, 36, 38), 2019, 2020,
                                  force_refresh=True)
    assert calls == [(2019, 2020)]


def test_load_background_seismicity_missing_cache_downloads(tmp_path, monkeypatch):
    cache_path = tmp_path / 'does_not_exist.parquet'
    calls = []
    def _fake_download(bounds, start_year, end_year, min_mag, cache_path=None):
        calls.append(True)
        return pd.DataFrame({'time': [], 'latitude': [], 'longitude': [],
                             'depth': [], 'mag': []})
    monkeypatch.setattr(bg, 'download_background_seismicity', _fake_download)

    bg.load_background_seismicity(str(cache_path), (-121, -119, 36, 38), 2019, 2020)
    assert calls == [True]
