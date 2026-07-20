"""
Unit tests for benchmark/runner.py.

Tests avoid calling bEPIC's locator directly.  BenchmarkRunner internals that
are pure Python (column normalisation, DataFrame assembly) are tested via
lightweight mock objects.
"""
import numpy as np
import pandas as pd
import pytest

from benchmark.runner import runner_results_to_df, BenchmarkRunner
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
        'best_misfit', 'best_like', 'best_prior', 'frac_misfit',
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
