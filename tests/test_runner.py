"""
Unit tests for benchmark/runner.py.

Tests avoid calling bEPIC's locator directly.  BenchmarkRunner internals that
are pure Python (column normalisation, DataFrame assembly) are tested via
lightweight mock objects.  BenchmarkRunner.run_event/run_all control flow
(version skipping, MAX_EVENT_TRIGS breaks, prior scheduling) is tested by
monkeypatching bEPIC's E2Location_locate so no real physics is invoked.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from bEPIC import EPIC_locate_prelim

from benchmark import runner as runner_mod
from benchmark.runner import (
    runner_results_to_df, BenchmarkRunner, make_epic_params,
    get_unique_stations, load_station_availability_cache,
    repair_inversion_json_paths, load_reference_catalog,
    load_reference_catalog_usgs,
)
from benchmark.metrics import COVERAGE_RADII_KM


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class _MockSearchOut:
    """Minimal stand-in for bEPIC's SearchOut."""
    def __init__(self, lat=37.0, lon=-120.0):
        self.posterior_lat = lat
        self.posterior_lon = lon
        self.exp_lat       = lat
        self.exp_lon       = lon
        self.like_lat      = lat
        self.like_lon      = lon
        self.like_exp_lat  = lat
        self.like_exp_lon  = lon
        self.best_misfit   = 0.5
        self.best_like     = 1.2
        self.best_prior    = 0.001
        self.frac_misfit   = 0.1
        self.best_depth    = 8.0


def _make_mock_runner(n_events=2):
    """Return a mock BenchmarkRunner-like object with n_events results."""
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.results = {}
    runner.metrics = {}
    runner.n_trigs = {}
    for i in range(n_events):
        key = (i + 1, 0)
        runner.results[key] = _MockSearchOut(37.0 + i * 0.1, -120.0 - i * 0.1)
        runner.metrics[key] = {
            'map_err_km':                float(i * 5),
            'posterior_confidence_level':       0.3,
            'prior_confidence_level': 0.4,
            **{f'coverage_{r}km': 0.5 for r in COVERAGE_RADII_KM},
        }
        runner.n_trigs[key] = i + 1
    return runner


# ---------------------------------------------------------------------------
# runner_results_to_df
# ---------------------------------------------------------------------------

def test_results_to_df_row_count():
    df = runner_results_to_df(_make_mock_runner(n_events=3))
    assert len(df) == 3


def test_results_to_df_required_columns():
    df = runner_results_to_df(_make_mock_runner())
    required = {
        'event_id', 'version', 'n_trigs',
        'posterior_lat', 'posterior_lon',
        'best_misfit', 'best_like', 'best_prior', 'frac_misfit', 'best_depth',
        'map_err_km', 'posterior_confidence_level', 'prior_confidence_level',
    }
    assert required.issubset(set(df.columns))


def test_results_to_df_coverage_columns_present():
    df = runner_results_to_df(_make_mock_runner())
    for r in COVERAGE_RADII_KM:
        assert f'coverage_{r}km' in df.columns


def test_results_to_df_sorted_by_event_and_version():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.results = {
        (2, 1): _MockSearchOut(),
        (1, 0): _MockSearchOut(),
        (2, 0): _MockSearchOut(),
    }
    runner.metrics = {}
    runner.n_trigs = {(2, 1): 2, (1, 0): 1, (2, 0): 1}
    df = runner_results_to_df(runner)
    assert list(df['event_id']) == [1, 2, 2]
    assert list(df['version'])  == [0, 0, 1]


def test_results_to_df_metric_values_propagated():
    df = runner_results_to_df(_make_mock_runner(n_events=1))
    assert df.iloc[0]['map_err_km'] == pytest.approx(0.0)
    assert df.iloc[0]['posterior_confidence_level'] == pytest.approx(0.3)


def test_results_to_df_missing_metrics_become_none():
    """Events with no entry in metrics should have NaN/None metric columns."""
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.results = {(1, 0): _MockSearchOut()}
    runner.metrics = {}          # no metrics at all
    runner.n_trigs = {(1, 0): 1}
    df = runner_results_to_df(runner)
    assert pd.isna(df.iloc[0]['map_err_km'])


def test_results_to_df_posterior_coords_match():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.results = {(1, 0): _MockSearchOut(lat=36.5, lon=-119.5)}
    runner.metrics = {}
    runner.n_trigs = {(1, 0): 2}
    df = runner_results_to_df(runner)
    assert df.iloc[0]['posterior_lat'] == pytest.approx(36.5)
    assert df.iloc[0]['posterior_lon'] == pytest.approx(-119.5)


# ---------------------------------------------------------------------------
# BenchmarkRunner._normalize_columns
# ---------------------------------------------------------------------------

def test_normalize_columns_replaces_spaces():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    df = pd.DataFrame({'col one': [1], 'col two': [2]})
    out = runner._normalize_columns(df)
    assert 'col_one' in out.columns
    assert 'col_two' in out.columns


def test_normalize_columns_leaves_no_spaces():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    df = pd.DataFrame({'a b c': [1], 'x y': [2], 'nospace': [3]})
    out = runner._normalize_columns(df)
    assert all(' ' not in c for c in out.columns)


def test_normalize_columns_preserves_data():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    df = pd.DataFrame({'trigger time': [1.0, 2.0]})
    out = runner._normalize_columns(df)
    assert list(out['trigger_time']) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# BenchmarkRunner.__init__ with catalog_df
# ---------------------------------------------------------------------------

def test_runner_init_ref_lookup_populated():
    """Providing catalog_df builds a non-empty reference lookup."""
    catalog = pd.DataFrame({
        'event_id': [1, 2],
        'usgs_lat': [37.0, 38.0],
        'usgs_lon': [-120.0, -121.0],
    })
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.__init__(prior=None, params=None, run_dir='/tmp', catalog_df=catalog)
    assert '1' in runner._ref_lookup
    assert '2' in runner._ref_lookup


def test_runner_init_no_catalog_empty_lookup():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.__init__(prior=None, params=None, run_dir='/tmp', catalog_df=None)
    assert runner._ref_lookup == {}


def test_runner_init_results_empty():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.__init__(prior=None, params=None, run_dir='/tmp')
    assert runner.results == {}
    assert runner.metrics == {}
    assert runner.n_trigs == {}


# ---------------------------------------------------------------------------
# make_epic_params
# ---------------------------------------------------------------------------

def test_make_epic_params_required_keys_propagate():
    params = make_epic_params(prior='PRIOR', use_prior=True,
                              benchmark_params={'grid_size': 42, 'grid_km': 300, 'max_trigs': 7})
    assert params.prior == 'PRIOR'
    assert params.use_prior is True
    assert params.GridSize == 42
    assert params.GridKm == 300
    assert params.MAX_EVENT_TRIGS == 7
    assert params.method == 'EPIC C'


def test_make_epic_params_optional_defaults():
    params = make_epic_params('P', False, {'grid_size': 1, 'grid_km': 1, 'max_trigs': 1})
    assert params.migrate_grid is True
    assert params.migrate_grid_min_triggers == 1
    assert params.station_inventory is None
    assert params.activity_threshold == pytest.approx(0.40)
    assert params.resample_distant_events is True
    assert params.edt_sigma_s == pytest.approx(0.2)
    assert params.sigma_s == pytest.approx(1.0)
    assert params.dtt_weight == pytest.approx(0.5)
    assert params.search_depths == [8.0]


def test_make_epic_params_optional_overrides():
    benchmark_params = {
        'grid_size': 1, 'grid_km': 1, 'max_trigs': 1,
        'migrate_grid': False, 'migrate_grid_min_triggers': 3,
        'activity_threshold': 0.9, 'resample_distant_events': False,
        'edt_sigma_s': 0.5, 'sigma_s': 2.0, 'dtt_weight': 0.1,
        'search_depths': [4.0, 8.0, 12.0],
    }
    params = make_epic_params('P', True, benchmark_params)
    assert params.migrate_grid is False
    assert params.migrate_grid_min_triggers == 3
    assert params.activity_threshold == pytest.approx(0.9)
    assert params.resample_distant_events is False
    assert params.edt_sigma_s == pytest.approx(0.5)
    assert params.sigma_s == pytest.approx(2.0)
    assert params.dtt_weight == pytest.approx(0.1)
    assert params.search_depths == [4.0, 8.0, 12.0]


def test_make_epic_params_station_inventory_passthrough():
    df = pd.DataFrame({'station': ['A'], 'network': ['NC'],
                       'longitude': [-120.0], 'latitude': [37.0]})
    params = make_epic_params('P', True, {'grid_size': 1, 'grid_km': 1, 'max_trigs': 1},
                              station_inventory=df)
    assert params.station_inventory is df


# ---------------------------------------------------------------------------
# get_unique_stations
# ---------------------------------------------------------------------------

def _write_run_file(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_get_unique_stations_dedupes_across_files(tmp_path):
    _write_run_file(tmp_path / '1.run', [
        {'station': 'AAA', 'network': 'NC', 'longitude': -120.0, 'latitude': 37.0},
        {'station': 'BBB', 'network': 'NC', 'longitude': -121.0, 'latitude': 38.0},
    ])
    _write_run_file(tmp_path / '2.run', [
        {'station': 'AAA', 'network': 'NC', 'longitude': -120.0, 'latitude': 37.0},
        {'station': 'CCC', 'network': 'NC', 'longitude': -122.0, 'latitude': 39.0},
    ])
    stations = get_unique_stations(str(tmp_path))
    assert len(stations) == 3
    assert set(stations['station']) == {'AAA', 'BBB', 'CCC'}


def test_get_unique_stations_same_station_different_network_kept_separate(tmp_path):
    _write_run_file(tmp_path / '1.run', [
        {'station': 'AAA', 'network': 'NC', 'longitude': -120.0, 'latitude': 37.0},
        {'station': 'AAA', 'network': 'CI', 'longitude': -120.0, 'latitude': 37.0},
    ])
    stations = get_unique_stations(str(tmp_path))
    assert len(stations) == 2


# ---------------------------------------------------------------------------
# load_station_availability_cache
# ---------------------------------------------------------------------------

def test_load_station_availability_cache_groups_by_event(tmp_path):
    df = pd.DataFrame({
        'event_id':  [1, 1, 2],
        'station':   ['AAA', 'BBB', 'CCC'],
        'network':   ['NC', 'NC', 'NC'],
        'longitude': [-120.0, -121.0, -122.0],
        'latitude':  [37.0, 38.0, 39.0],
    })
    cache_path = tmp_path / 'availability.parquet'
    df.to_parquet(cache_path, index=False)

    cache = load_station_availability_cache(str(cache_path))
    assert set(cache.keys()) == {'1', '2'}
    assert len(cache['1']) == 2
    assert len(cache['2']) == 1
    assert 'event_id' not in cache['1'].columns


# ---------------------------------------------------------------------------
# repair_inversion_json_paths
# ---------------------------------------------------------------------------

def test_repair_paths_noop_when_all_exist(tmp_path):
    catalog = tmp_path / 'catalog.csv'
    catalog.write_text('a,b\n1,2\n')
    json_path = tmp_path / 'params.json'
    cfg = {'fn_catalog': str(catalog), 'fn_ip': str(catalog), 'fn_src': str(catalog)}
    json_path.write_text(json.dumps(cfg))

    mtime_before = json_path.stat().st_mtime_ns
    repair_inversion_json_paths(str(json_path))
    assert json_path.stat().st_mtime_ns == mtime_before  # no rewrite happened


def test_repair_paths_falls_back_to_sibling(tmp_path):
    sibling = tmp_path / 'catalog.csv'
    sibling.write_text('a,b\n1,2\n')
    json_path = tmp_path / 'params.json'
    stale_path = '/some/stale/moved/directory/catalog.csv'
    cfg = {'fn_catalog': stale_path, 'fn_ip': stale_path, 'fn_src': stale_path}
    json_path.write_text(json.dumps(cfg))

    with pytest.warns(UserWarning):
        repair_inversion_json_paths(str(json_path))

    repaired = json.loads(json_path.read_text())
    assert repaired['fn_catalog'] == str(sibling)
    assert repaired['fn_ip'] == str(sibling)
    assert repaired['fn_src'] == str(sibling)


def test_repair_paths_leaves_unresolvable_as_is(tmp_path):
    json_path = tmp_path / 'params.json'
    stale_path = '/some/stale/moved/directory/nonexistent_anywhere.csv'
    cfg = {'fn_catalog': stale_path}
    json_path.write_text(json.dumps(cfg))

    with pytest.warns(UserWarning):
        repair_inversion_json_paths(str(json_path))

    # Nothing was resolvable, so the file must be left untouched.
    unchanged = json.loads(json_path.read_text())
    assert unchanged['fn_catalog'] == stale_path


def test_repair_paths_missing_keys_ignored(tmp_path):
    """A JSON with none of fn_catalog/fn_ip/fn_src present should no-op silently."""
    json_path = tmp_path / 'params.json'
    json_path.write_text(json.dumps({'other_field': 1}))
    repair_inversion_json_paths(str(json_path))  # must not raise


# ---------------------------------------------------------------------------
# load_reference_catalog / load_reference_catalog_usgs
# ---------------------------------------------------------------------------

def test_load_reference_catalog_parses_columns(tmp_path):
    path = tmp_path / 'bEPIC_testing_catalog.txt'
    path.write_text(
        "postgres id\tANSS ID\tANSS date\tANSS lat\tANSS lon\tANSS depth\tANSS mag\n"
        "101\tnc123\t2019-07-04-17:33:29.420000-GMT\t35.7\t-117.5\t8.0\t6.4\n"
    )
    df = load_reference_catalog(str(path))
    assert list(df.columns) == ['event_id', 'anss_id', 'usgs_time', 'usgs_lat',
                                'usgs_lon', 'usgs_depth', 'usgs_mag']
    assert df.iloc[0]['event_id'] == 101
    assert df.iloc[0]['anss_id'] == 'nc123'
    assert df.iloc[0]['usgs_lat'] == pytest.approx(35.7)


def test_load_reference_catalog_usgs_csv(tmp_path):
    path = tmp_path / 'catalog.csv'
    pd.DataFrame({
        'id':        ['ci38457511'],
        'time':      ['2019-07-06T03:19:53.040Z'],
        'latitude':  [35.77],
        'longitude': [-117.6],
        'depth':     [8.0],
        'mag':       [7.1],
    }).to_csv(path, index=False)

    df = load_reference_catalog_usgs(str(path))
    assert df.iloc[0]['event_id'] == 'ci38457511'
    assert df.iloc[0]['anss_id'] == 'ci38457511'
    assert df.iloc[0]['usgs_lat'] == pytest.approx(35.77)


def test_load_reference_catalog_usgs_parquet(tmp_path):
    path = tmp_path / 'catalog.parquet'
    pd.DataFrame({
        'id':        ['ci38457511'],
        'time':      pd.to_datetime(['2019-07-06T03:19:53.040Z']),
        'latitude':  [35.77],
        'longitude': [-117.6],
        'mag':       [7.1],
    }).to_parquet(path, index=False)

    df = load_reference_catalog_usgs(str(path))
    assert df.iloc[0]['event_id'] == 'ci38457511'
    assert pd.isna(df.iloc[0]['usgs_depth'])  # 'depth' column absent


def test_load_reference_catalog_usgs_rejects_unsupported_extension(tmp_path):
    path = tmp_path / 'catalog.txt'
    path.write_text('irrelevant')
    with pytest.raises(ValueError):
        load_reference_catalog_usgs(str(path))


# ---------------------------------------------------------------------------
# BenchmarkRunner.run_event / run_all control flow
#
# E2Location_locate is monkeypatched so no real bEPIC physics runs; the mock
# records every call and returns a deterministic SearchOut + empty grid.
# ---------------------------------------------------------------------------

def _write_full_run_file(path, versions_n_trigs, start_time=1000.0):
    """versions_n_trigs: list of trigger counts, one per version (0-indexed)."""
    rows = []
    order = 1
    for version, n in enumerate(versions_n_trigs):
        for i in range(n):
            rows.append({
                'version': version, 'order': order,
                'station': f'S{order}', 'channel': 'HHZ', 'network': 'NC',
                'longitude': -120.0 - i * 0.01, 'latitude': 37.0 + i * 0.01,
                'trigger time': start_time + order,
            })
            order += 1
    pd.DataFrame(rows).to_csv(path, index=False)


class _CallRecordingLocator:
    """Monkeypatch target for EPIC_locate_prelim.E2Location_locate."""
    def __init__(self):
        self.calls = []  # list of (params.prior, event.version, n_trigs)

    def __call__(self, params, event):
        self.calls.append((params.prior, event.version, len(event.trigs)))
        t = EPIC_locate_prelim.SearchOut()
        t.posterior_lat = t.exp_lat = t.like_lat = t.like_exp_lat = 37.0
        t.posterior_lon = t.exp_lon = t.like_lon = t.like_exp_lon = -120.0
        out_df = pd.DataFrame({'lat': [37.0], 'lon': [-120.0],
                               'post': [1.0], 'prior': [1.0], 'like': [1.0]})
        return t, out_df


def _mock_params(max_trigs=100):
    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.MAX_EVENT_TRIGS = max_trigs
    return params


def test_run_event_calls_locator_once_per_distinct_trigger_count(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1, 2, 3])  # 3 distinct versions, all grow
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1)

    assert len(locator.calls) == 3
    assert [c[2] for c in locator.calls] == [1, 2, 3]
    assert set(runner.results.keys()) == {(1, 0), (1, 1), (1, 2)}


def test_run_event_skips_versions_with_unchanged_trigger_count(tmp_path, monkeypatch):
    """Consecutive versions at the same trigger count (bEPIC refresh without
    a new trigger) must be collapsed into a single locator call/result row."""
    run_dir = tmp_path
    # version 0: 1 trigger, version 1: 1 trigger (same count -> skip), version 2: 2 triggers
    _write_full_run_file(run_dir / '1.run', [1, 1, 2])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1)

    assert len(locator.calls) == 2
    assert (1, 1) not in runner.results
    assert set(runner.results.keys()) == {(1, 0), (1, 2)}


def test_run_event_breaks_at_max_event_trigs(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1, 2, 3, 4, 5])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(max_trigs=3), run_dir=str(run_dir))
    runner.run_event(1)

    # Should stop once a version reaches MAX_EVENT_TRIGS=3 triggers.
    assert [c[2] for c in locator.calls] == [1, 2, 3]
    assert set(runner.results.keys()) == {(1, 0), (1, 1), (1, 2)}


def test_run_event_prior_for_n_trigs_overrides_prior_per_version(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1, 2, 3])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='BASE_PRIOR', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1, prior_for_n_trigs=lambda n: f'PRIOR_FOR_{n}')

    assert [c[0] for c in locator.calls] == ['PRIOR_FOR_1', 'PRIOR_FOR_2', 'PRIOR_FOR_3']


def test_run_event_default_uses_fixed_prior(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1, 2])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='FIXED_PRIOR', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1)

    assert all(c[0] == 'FIXED_PRIOR' for c in locator.calls)


def test_run_event_records_n_trigs(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [2, 4])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1)

    assert runner.n_trigs[(1, 0)] == 2
    assert runner.n_trigs[(1, 1)] == 4


def test_run_event_station_availability_sets_inventory_per_event(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    inv_df = pd.DataFrame({'station': ['AAA'], 'network': ['NC'],
                           'longitude': [-120.0], 'latitude': [37.0]})
    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir),
                             station_availability={'1': inv_df})
    runner.run_event(1)

    assert runner.params.station_inventory is inv_df


def test_run_event_computes_metrics_when_catalog_provided(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    catalog = pd.DataFrame({'event_id': [1], 'usgs_lat': [37.0], 'usgs_lon': [-120.0]})
    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir),
                             catalog_df=catalog)
    runner.run_event(1)

    assert (1, 0) in runner.metrics
    assert runner.metrics[(1, 0)]['map_err_km'] == pytest.approx(0.0, abs=1e-3)


def test_run_event_no_metrics_without_catalog(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_event(1)

    assert runner.metrics == {}


# ---------------------------------------------------------------------------
# BenchmarkRunner.run_all — event ordering, ETAS update scheduling,
# after_event_fn, n_trigs_schedule_fn
# ---------------------------------------------------------------------------

def test_run_all_calls_run_event_for_every_event(tmp_path, monkeypatch):
    run_dir = tmp_path
    for eid in (1, 2, 3):
        _write_full_run_file(run_dir / f'{eid}.run', [1])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1, 2, 3])

    assert set(eid for eid, ver in runner.results.keys()) == {1, 2, 3}


def test_run_all_no_etas_update_fn_prior_never_changes(tmp_path, monkeypatch):
    run_dir = tmp_path
    for eid in (1, 2):
        _write_full_run_file(run_dir / f'{eid}.run', [1], start_time=1000.0 * eid)
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    runner = BenchmarkRunner(prior='FIXED', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1, 2])

    assert all(c[0] == 'FIXED' for c in locator.calls)


def test_run_all_etas_update_fn_called_on_first_event_regardless_of_interval(tmp_path, monkeypatch):
    run_dir = tmp_path
    for eid in (1, 2):
        _write_full_run_file(run_dir / f'{eid}.run', [1], start_time=1000.0 * eid)
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    update_calls = []

    def etas_update_fn(event_time):
        update_calls.append(event_time)
        return f'PRIOR_AT_{event_time}'

    runner = BenchmarkRunner(prior='INITIAL', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1, 2], etas_update_fn=etas_update_fn, update_interval_s=1e9)

    # First event always triggers an update (last_update_time starts as None);
    # the second event's gap (1000s) is far below the 1e9s interval, so it's throttled.
    assert len(update_calls) == 1
    assert locator.calls[0][0] == 'PRIOR_AT_1001.0'
    assert locator.calls[1][0] == 'PRIOR_AT_1001.0'  # unchanged, still throttled


def test_run_all_etas_update_fn_respects_update_interval(tmp_path, monkeypatch):
    run_dir = tmp_path
    # Event times 1 second apart at start_time=1 and start_time=100000 -> big gap
    _write_full_run_file(run_dir / '1.run', [1], start_time=0.0)
    _write_full_run_file(run_dir / '2.run', [1], start_time=100000.0)
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    update_calls = []

    def etas_update_fn(event_time):
        update_calls.append(event_time)
        return f'PRIOR_AT_{event_time}'

    runner = BenchmarkRunner(prior='INITIAL', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1, 2], etas_update_fn=etas_update_fn, update_interval_s=3600)

    # Gap between event 1 (t=1) and event 2 (t=100001) exceeds 3600s -> both trigger an update.
    assert len(update_calls) == 2


def test_run_all_update_interval_zero_updates_every_event(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1], start_time=0.0)
    _write_full_run_file(run_dir / '2.run', [1], start_time=1.0)
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    update_calls = []
    runner = BenchmarkRunner(prior='INITIAL', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1, 2], etas_update_fn=lambda t: update_calls.append(t) or 'P',
                  update_interval_s=0)

    assert len(update_calls) == 2


def test_run_all_after_event_fn_called_once_per_event_in_order(tmp_path, monkeypatch):
    run_dir = tmp_path
    for eid in (1, 2, 3):
        _write_full_run_file(run_dir / f'{eid}.run', [1])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    after_calls = []
    runner = BenchmarkRunner(prior='P', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([3, 1, 2], after_event_fn=after_calls.append)

    assert after_calls == [3, 1, 2]


def test_run_all_n_trigs_schedule_fn_receives_current_prior(tmp_path, monkeypatch):
    run_dir = tmp_path
    _write_full_run_file(run_dir / '1.run', [1, 2])
    locator = _CallRecordingLocator()
    monkeypatch.setattr(runner_mod.EPIC_locate_prelim, 'E2Location_locate', locator)

    seen_base_priors = []

    def n_trigs_schedule_fn(base_prior):
        seen_base_priors.append(base_prior)
        return lambda n: f'{base_prior}_n{n}'

    runner = BenchmarkRunner(prior='BASE', params=_mock_params(), run_dir=str(run_dir))
    runner.run_all([1], n_trigs_schedule_fn=n_trigs_schedule_fn)

    assert seen_base_priors == ['BASE']
    assert [c[0] for c in locator.calls] == ['BASE_n1', 'BASE_n2']


# ---------------------------------------------------------------------------
# BenchmarkRunner.update_prior
# ---------------------------------------------------------------------------

def test_update_prior_updates_both_prior_and_params():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.prior = 'OLD'
    runner.params = _mock_params()
    runner.params.prior = 'OLD'
    runner.update_prior('NEW')
    assert runner.prior == 'NEW'
    assert runner.params.prior == 'NEW'
