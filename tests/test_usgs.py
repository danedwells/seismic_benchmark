"""
Unit tests for benchmark/usgs.py.

Network-calling functions (get_usgs_event, get_phases_df, get_station_coords,
the USGS/IRIS HTTP requests) are monkeypatched out everywhere, so these tests
never touch the network. What's tested is the pure parsing and control-flow
logic: channel-string parsing, phase filtering, deduplication, and the
version-assignment algorithm in build_run_file_from_usgs, plus the
cache-hit path of download_case_study_catalog and the skip_existing /
error-isolation behaviour of build_run_files_for_case_study.
"""
import pandas as pd
import pytest

from benchmark import usgs


# ---------------------------------------------------------------------------
# _parse_channel
# ---------------------------------------------------------------------------

def test_parse_channel_with_location_code():
    net, sta, cha, loc = usgs._parse_channel('NC SAO HHZ --')
    assert (net, sta, cha, loc) == ('NC', 'SAO', 'HHZ', '--')


def test_parse_channel_without_location_code_defaults():
    net, sta, cha, loc = usgs._parse_channel('NC SAO HHZ')
    assert (net, sta, cha, loc) == ('NC', 'SAO', 'HHZ', '--')


def test_parse_channel_strips_whitespace():
    net, sta, cha, loc = usgs._parse_channel('  NC SAO HHZ 00  ')
    assert (net, sta, cha, loc) == ('NC', 'SAO', 'HHZ', '00')


# ---------------------------------------------------------------------------
# build_run_file_from_usgs — version assignment / filtering logic
# ---------------------------------------------------------------------------

def _phases_row(net, sta, phase, iso_time, dist, residual=0.1, cha='HHZ', loc='--'):
    return {
        'Channel':      f'{net} {sta} {cha} {loc}',
        'Distance':     dist,
        'Phase':        phase,
        'Arrival Time': iso_time,
        'Residual':     residual,
    }


def _station_coords_stub(coords_by_station):
    def _stub(network, station, channel, origin_time_iso):
        return coords_by_station.get(station, (None, None))
    return _stub


def test_build_run_file_writes_expected_versions(tmp_path, monkeypatch):
    phases_df = pd.DataFrame([
        _phases_row('NC', 'AAA', 'P', '2020-01-01T00:00:01Z', dist=1.0),
        _phases_row('NC', 'AAA', 'S', '2020-01-01T00:00:05Z', dist=1.0),   # filtered: not 'P'
        _phases_row('NC', 'BBB', 'P', '2020-01-01T00:00:02Z', dist=2.0),
        _phases_row('NC', 'CCC', 'P', '2020-01-01T00:00:03Z', dist=10.0),  # filtered: too far
    ])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)
    monkeypatch.setattr(usgs, 'get_station_coords',
                        _station_coords_stub({'AAA': (37.0, -120.0), 'BBB': (38.0, -121.0)}))

    out_path = tmp_path / 'evt.run'
    ok = usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path),
                                       max_dist_deg=5.0)
    assert ok is True

    out_df = pd.read_csv(out_path)
    # AAA arrives first (version 0, 1 row); BBB adds a new station (version 1, 2 rows).
    assert set(out_df['version']) == {0, 1}
    assert len(out_df[out_df['version'] == 0]) == 1
    assert len(out_df[out_df['version'] == 1]) == 2
    assert 'CCC' not in set(out_df['station'])


def test_build_run_file_deduplicates_repeated_channel_keeps_first_arrival(tmp_path, monkeypatch):
    phases_df = pd.DataFrame([
        _phases_row('NC', 'AAA', 'P', '2020-01-01T00:00:01Z', dist=1.0, residual=0.1),
        _phases_row('NC', 'AAA', 'P', '2020-01-01T00:00:09Z', dist=1.0, residual=0.9),  # dup, later
        _phases_row('NC', 'BBB', 'P', '2020-01-01T00:00:02Z', dist=2.0),
    ])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)
    monkeypatch.setattr(usgs, 'get_station_coords',
                        _station_coords_stub({'AAA': (37.0, -120.0), 'BBB': (38.0, -121.0)}))

    out_path = tmp_path / 'evt.run'
    usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path))

    out_df = pd.read_csv(out_path)
    aaa_rows = out_df[out_df['station'] == 'AAA']
    assert aaa_rows['tterr'].nunique() == 1
    assert aaa_rows['tterr'].iloc[0] == pytest.approx(0.1)  # first arrival kept, not the dup


def test_build_run_file_no_phases_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: None)
    out_path = tmp_path / 'evt.run'
    ok = usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path))
    assert ok is False
    assert not out_path.exists()


def test_build_run_file_empty_after_filter_returns_false(tmp_path, monkeypatch):
    phases_df = pd.DataFrame([_phases_row('NC', 'AAA', 'S', '2020-01-01T00:00:01Z', dist=1.0)])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)
    out_path = tmp_path / 'evt.run'
    ok = usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path))
    assert ok is False


def test_build_run_file_fewer_than_two_resolved_stations_returns_false(tmp_path, monkeypatch):
    """Only one station resolves coordinates via IRIS -> insufficient triggers."""
    phases_df = pd.DataFrame([
        _phases_row('NC', 'AAA', 'P', '2020-01-01T00:00:01Z', dist=1.0),
        _phases_row('NC', 'BBB', 'P', '2020-01-01T00:00:02Z', dist=2.0),
    ])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)
    # BBB fails to resolve (None, None)
    monkeypatch.setattr(usgs, 'get_station_coords',
                        _station_coords_stub({'AAA': (37.0, -120.0)}))
    out_path = tmp_path / 'evt.run'
    ok = usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path))
    assert ok is False


def test_build_run_file_custom_phases_filter(tmp_path, monkeypatch):
    phases_df = pd.DataFrame([
        _phases_row('NC', 'AAA', 'S', '2020-01-01T00:00:01Z', dist=1.0),
        _phases_row('NC', 'BBB', 'S', '2020-01-01T00:00:02Z', dist=2.0),
    ])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)
    monkeypatch.setattr(usgs, 'get_station_coords',
                        _station_coords_stub({'AAA': (37.0, -120.0), 'BBB': (38.0, -121.0)}))
    out_path = tmp_path / 'evt.run'
    ok = usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path),
                                       phases_filter=['S'])
    assert ok is True


def test_build_run_file_shares_station_coord_cache(tmp_path, monkeypatch):
    phases_df = pd.DataFrame([
        _phases_row('NC', 'AAA', 'P', '2020-01-01T00:00:01Z', dist=1.0),
        _phases_row('NC', 'BBB', 'P', '2020-01-01T00:00:02Z', dist=2.0),
    ])
    monkeypatch.setattr(usgs, 'get_usgs_event', lambda anss_id: {})
    monkeypatch.setattr(usgs, 'get_phases_df', lambda geojson: phases_df)

    lookup_calls = []
    def _tracking_coords(network, station, channel, origin_time_iso):
        lookup_calls.append(station)
        return {'AAA': (37.0, -120.0), 'BBB': (38.0, -121.0)}[station]
    monkeypatch.setattr(usgs, 'get_station_coords', _tracking_coords)

    cache = {}
    out_path = tmp_path / 'evt.run'
    usgs.build_run_file_from_usgs('evt1', '2020-01-01T00:00:00', str(out_path),
                                  station_coord_cache=cache)
    assert len(cache) == 2
    assert sorted(lookup_calls) == ['AAA', 'BBB']

    # A second event reusing the same cache must not re-query already-cached stations.
    out_path2 = tmp_path / 'evt2.run'
    usgs.build_run_file_from_usgs('evt2', '2020-01-01T00:00:00', str(out_path2),
                                  station_coord_cache=cache)
    assert sorted(lookup_calls) == ['AAA', 'BBB']  # unchanged -- no new lookups


# ---------------------------------------------------------------------------
# download_case_study_catalog — cache-hit path (no network)
# ---------------------------------------------------------------------------

def test_download_case_study_catalog_uses_cache_when_present(tmp_path):
    cs = {'name': 'Test Sequence', 'starttime': '2020-01-01', 'endtime': '2020-01-02',
          'bounds': (-121.0, -119.0, 36.0, 38.0), 'min_mag': 3.0}
    cache_path = tmp_path / 'Test_Sequence_catalog.parquet'
    expected = pd.DataFrame({'id': ['ci1'], 'time': pd.to_datetime(['2020-01-01T00:00:00Z']),
                             'latitude': [37.0], 'longitude': [-120.0],
                             'depth': [5.0], 'mag': [4.0]})
    expected.to_parquet(cache_path, index=False)

    df = usgs.download_case_study_catalog(cs, str(tmp_path), REDOWNLOAD=False)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


# ---------------------------------------------------------------------------
# build_run_files_for_case_study — skip_existing + error isolation
# ---------------------------------------------------------------------------

def test_build_run_files_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(usgs.time, 'sleep', lambda s: None)
    (tmp_path / 'ci1.run').write_text('version,order\n0,1\n')  # pre-existing

    calls = []
    def fake_builder(**kwargs):
        calls.append(kwargs['anss_id'])
        return True
    monkeypatch.setattr(usgs, 'build_run_file_from_usgs', fake_builder)

    catalog_df = pd.DataFrame({'id': ['ci1', 'ci2'],
                               'time': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    written = usgs.build_run_files_for_case_study(catalog_df, str(tmp_path), skip_existing=True)

    assert calls == ['ci2']              # ci1 skipped, not rebuilt
    assert written == ['ci1', 'ci2']     # both counted as available


def test_build_run_files_isolates_individual_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(usgs.time, 'sleep', lambda s: None)

    def flaky_builder(**kwargs):
        if kwargs['anss_id'] == 'ci_bad':
            raise RuntimeError('boom')
        return True
    monkeypatch.setattr(usgs, 'build_run_file_from_usgs', flaky_builder)

    catalog_df = pd.DataFrame({'id': ['ci_bad', 'ci_good'],
                               'time': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    written = usgs.build_run_files_for_case_study(catalog_df, str(tmp_path), skip_existing=False)

    assert written == ['ci_good']  # the failing event is skipped, not fatal
