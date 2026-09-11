"""
Unit tests for benchmark/config_cascadia.py.

Mirrors tests/test_config.py's structure/sanity-check style for the
California config, but for the Cascadia region config. Also checks the
invariants config_cascadia.py's own docstring claims (e.g. that it reuses
config.py's tag helpers and that its polygon reaches further north than
config.py's).
"""
import pandas as pd
import pytest

from benchmark import config
from benchmark import config_cascadia as cc


# ---------------------------------------------------------------------------
# REFERENCE_CATALOG_CONFIG
# ---------------------------------------------------------------------------

def test_reference_catalog_config_has_required_keys():
    required = {'starttime', 'endtime', 'bounds', 'min_mag'}
    assert required.issubset(set(cc.REFERENCE_CATALOG_CONFIG.keys()))


def test_reference_catalog_config_bounds_valid():
    lon_min, lon_max, lat_min, lat_max = cc.REFERENCE_CATALOG_CONFIG['bounds']
    assert lon_min < lon_max
    assert lat_min < lat_max


def test_reference_catalog_config_time_ordering():
    t0 = pd.Timestamp(cc.REFERENCE_CATALOG_CONFIG['starttime'])
    t1 = pd.Timestamp(cc.REFERENCE_CATALOG_CONFIG['endtime'])
    assert t0 < t1


# ---------------------------------------------------------------------------
# ETAS_INVERSION_CONFIG
# ---------------------------------------------------------------------------

def test_etas_inversion_config_has_required_keys():
    required = {'mc', 'auxiliary_start', 'timewindow_start', 'timewindow_end',
                'id', 'shape_coords'}
    assert required.issubset(set(cc.ETAS_INVERSION_CONFIG.keys()))


def test_etas_inversion_config_time_ordering():
    t_aux   = pd.Timestamp(cc.ETAS_INVERSION_CONFIG['auxiliary_start'])
    t_start = pd.Timestamp(cc.ETAS_INVERSION_CONFIG['timewindow_start'])
    assert t_aux <= t_start
    if cc.ETAS_INVERSION_CONFIG['timewindow_end'] is not None:
        t_end = pd.Timestamp(cc.ETAS_INVERSION_CONFIG['timewindow_end'])
        assert t_start < t_end


def test_etas_inversion_config_id_is_cascadia():
    assert cc.ETAS_INVERSION_CONFIG['id'] == 'cascadia'


def test_etas_inversion_config_id_differs_from_california():
    """A shared ETAS_UPDATER cache/output path must never collide between regions."""
    assert cc.ETAS_INVERSION_CONFIG['id'] != config.ETAS_INVERSION_CONFIG['id']


def test_shape_coords_is_closed_polygon():
    """The [lat, lon] polygon must close (first vertex == last vertex)."""
    coords = cc.ETAS_INVERSION_CONFIG['shape_coords']
    assert len(coords) >= 4
    assert coords[0] == coords[-1]


def test_shape_coords_reaches_further_north_than_california():
    """Per the module docstring: config.py's polygon caps at ~44N and doesn't
    reach Washington, so Cascadia's polygon must extend further north."""
    cascadia_max_lat = max(lat for lat, lon in cc.ETAS_INVERSION_CONFIG['shape_coords'])
    california_max_lat = max(lat for lat, lon in config.ETAS_INVERSION_CONFIG['shape_coords'])
    assert cascadia_max_lat > california_max_lat


# ---------------------------------------------------------------------------
# ETAS_UPDATER_CONFIG
# ---------------------------------------------------------------------------

def test_etas_updater_config_has_bounds():
    assert 'bounds' in cc.ETAS_UPDATER_CONFIG


def test_etas_updater_config_bounds_match_reference_catalog():
    assert cc.ETAS_UPDATER_CONFIG['bounds'] == cc.REFERENCE_CATALOG_CONFIG['bounds']


def test_etas_updater_config_grid_spacing_positive():
    assert cc.ETAS_UPDATER_CONFIG['grid_spacing'] > 0


# ---------------------------------------------------------------------------
# BENCHMARK_CATALOG_CONFIG
# ---------------------------------------------------------------------------

def test_benchmark_catalog_config_has_required_keys():
    required = {'name', 'starttime', 'endtime', 'bounds', 'min_mag'}
    assert required.issubset(set(cc.BENCHMARK_CATALOG_CONFIG.keys()))


def test_benchmark_catalog_config_bounds_match_reference_catalog():
    assert cc.BENCHMARK_CATALOG_CONFIG['bounds'] == cc.REFERENCE_CATALOG_CONFIG['bounds']


def test_benchmark_catalog_config_time_ordering():
    t0 = pd.Timestamp(cc.BENCHMARK_CATALOG_CONFIG['starttime'])
    t1 = pd.Timestamp(cc.BENCHMARK_CATALOG_CONFIG['endtime'])
    assert t0 < t1


# ---------------------------------------------------------------------------
# CASE_STUDIES / FOCUS_EVENTS placeholders
# ---------------------------------------------------------------------------

def test_case_studies_is_dict():
    assert isinstance(cc.CASE_STUDIES, dict)


def test_focus_events_are_dicts():
    assert isinstance(cc.FOCUS_EVENTS, dict)
    assert isinstance(cc.FOCUS_EVENTS_MAINSHOCK, dict)


# ---------------------------------------------------------------------------
# Shared helpers imported from config.py (etas_run_tag etc.) work with
# Cascadia's own ETAS_INVERSION_CONFIG shape.
# ---------------------------------------------------------------------------

def test_etas_run_tag_works_with_cascadia_config():
    tag = config.etas_run_tag(cc.ETAS_INVERSION_CONFIG)
    assert isinstance(tag, str) and len(tag) > 0


def test_etas_output_id_same_context_and_flags_collide_across_regions():
    """etas_run_tag() encodes only fb/fp/mc/m_ref/bw_sq -- NOT the cfg 'id'
    field -- so two regions sharing both a context_name and those tunable
    values produce the SAME output id. Callers must pass a region-distinct
    context_name (e.g. 'cascadia_benchmark' vs 'benchmark') to avoid
    silently overwriting one region's inversion output with another's."""
    cascadia_id   = config.etas_output_id('benchmark', cc.ETAS_INVERSION_CONFIG)
    california_id = config.etas_output_id('benchmark', config.ETAS_INVERSION_CONFIG)
    assert cascadia_id == california_id


def test_etas_output_id_differs_with_region_scoped_context_name():
    """The actual collision-avoidance mechanism: distinct context_name."""
    cascadia_id   = config.etas_output_id('cascadia_benchmark', cc.ETAS_INVERSION_CONFIG)
    california_id = config.etas_output_id('benchmark', config.ETAS_INVERSION_CONFIG)
    assert cascadia_id != california_id


def test_etas_catalog_tag_works_with_cascadia_config():
    tag = config.etas_catalog_tag('cascadia_benchmark', cc.ETAS_INVERSION_CONFIG)
    assert isinstance(tag, str) and len(tag) > 0


# ---------------------------------------------------------------------------
# KDE_SEISMICITY_PARAMS
# ---------------------------------------------------------------------------

def test_kde_seismicity_params_has_required_keys():
    required = {'lon_col', 'lat_col', 'grid_size', 'bw_method'}
    assert required.issubset(set(cc.KDE_SEISMICITY_PARAMS.keys()))


def test_kde_seismicity_params_grid_size_positive():
    assert cc.KDE_SEISMICITY_PARAMS['grid_size'] > 0
