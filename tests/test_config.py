"""
Unit tests for benchmark/config.py.

Sanity checks that the configuration dictionaries have the expected structure
and contain valid values.  These tests catch accidental key deletions or
type errors introduced while editing config.py.
"""
import pandas as pd
import pytest

from benchmark import config


# ---------------------------------------------------------------------------
# PRIOR_FILENAMES
# ---------------------------------------------------------------------------

def test_prior_filenames_has_all_expected_keys():
    expected = {'Gear1', 'NSHM', 'Helmstetter', 'KDE_Seismicity', 'Uniform'}
    assert expected.issubset(set(config.PRIOR_FILENAMES.keys()))


def test_prior_filenames_string_values_are_tt3():
    """Any non-None filename must end with .tt3."""
    for name, fname in config.PRIOR_FILENAMES.items():
        if fname is not None:
            assert fname.endswith('.tt3'), f"{name} filename should be a .tt3 file"


# ---------------------------------------------------------------------------
# PRIOR_CONSTRUCTION_PARAMS
# ---------------------------------------------------------------------------

def test_construction_params_has_bounds():
    assert 'bounds' in config.PRIOR_CONSTRUCTION_PARAMS


def test_construction_params_bounds_valid():
    lon_min, lon_max, lat_min, lat_max = config.PRIOR_CONSTRUCTION_PARAMS['bounds']
    assert lon_min < lon_max, "lon_min must be less than lon_max"
    assert lat_min < lat_max, "lat_min must be less than lat_max"


def test_construction_params_bounds_covers_western_us():
    lon_min, lon_max, lat_min, lat_max = config.PRIOR_CONSTRUCTION_PARAMS['bounds']
    assert lon_min < -100, "bounds should be in the western US (negative longitudes)"
    assert lat_min >= 25
    assert lat_max <= 60


def test_construction_params_has_source_paths():
    assert 'source_paths' in config.PRIOR_CONSTRUCTION_PARAMS


def test_construction_params_source_paths_has_gear1_and_nshm():
    src = config.PRIOR_CONSTRUCTION_PARAMS['source_paths']
    assert 'Gear1' in src
    assert 'NSHM' in src


def test_construction_params_out_of_bounds_fill_keys():
    oob = config.PRIOR_CONSTRUCTION_PARAMS.get('out_of_bounds_fill', {})
    expected = {'Gear1', 'NSHM', 'Helmstetter', 'KDE_Seismicity'}
    assert expected.issubset(set(oob.keys()))


def test_construction_params_oob_values_are_positive():
    oob = config.PRIOR_CONSTRUCTION_PARAMS.get('out_of_bounds_fill', {})
    for name, val in oob.items():
        if isinstance(val, (int, float)):
            assert val > 0, f"out_of_bounds_fill for {name} should be positive"


# ---------------------------------------------------------------------------
# BENCHMARK_PARAMS
# ---------------------------------------------------------------------------

def test_benchmark_params_has_required_keys():
    required = {'max_trigs', 'grid_size', 'grid_km'}
    assert required.issubset(set(config.BENCHMARK_PARAMS.keys()))


def test_benchmark_params_positive_values():
    p = config.BENCHMARK_PARAMS
    assert p['max_trigs'] > 0
    assert p['grid_size'] > 0
    assert p['grid_km'] > 0


def test_benchmark_params_grid_km_reasonable():
    """Grid half-width should cover at least a few hundred km."""
    assert config.BENCHMARK_PARAMS['grid_km'] >= 100


# ---------------------------------------------------------------------------
# ETAS_INVERSION_CONFIG
# ---------------------------------------------------------------------------

def test_etas_inversion_config_has_required_keys():
    required = {'mc', 'auxiliary_start', 'timewindow_start', 'timewindow_end', 'id'}
    assert required.issubset(set(config.ETAS_INVERSION_CONFIG.keys()))


def test_etas_inversion_config_time_ordering():
    cfg = config.ETAS_INVERSION_CONFIG
    t_aux   = pd.Timestamp(cfg['auxiliary_start'])
    t_start = pd.Timestamp(cfg['timewindow_start'])
    assert t_aux <= t_start, "auxiliary_start should be <= timewindow_start"
    if cfg['timewindow_end'] is not None:
        t_end = pd.Timestamp(cfg['timewindow_end'])
        assert t_start < t_end, "timewindow_start should be < timewindow_end"


def test_etas_inversion_config_mc_positive():
    assert config.ETAS_INVERSION_CONFIG['mc'] > 0


def test_etas_inversion_config_id_is_string():
    assert isinstance(config.ETAS_INVERSION_CONFIG['id'], str)


# ---------------------------------------------------------------------------
# KDE_START_DATE
# ---------------------------------------------------------------------------

def test_kde_start_date_parseable():
    ts = pd.Timestamp(config.KDE_START_DATE)
    assert ts.year < 2010, "KDE start date should be a historical date"


# ---------------------------------------------------------------------------
# ETAS_UPDATER_CONFIG
# ---------------------------------------------------------------------------

def test_etas_updater_config_has_bounds():
    assert 'bounds' in config.ETAS_UPDATER_CONFIG


def test_etas_updater_config_bounds_match_construction_params():
    assert config.ETAS_UPDATER_CONFIG['bounds'] == config.PRIOR_CONSTRUCTION_PARAMS['bounds']


def test_etas_updater_config_grid_spacing_positive():
    assert config.ETAS_UPDATER_CONFIG['grid_spacing'] > 0


def test_etas_updater_config_spatial_flags_default_off():
    # parameters_benchmark.json was inverted with free_productivity=False and
    # without store_spatial_fields=True, so use_spatial_productivity=True
    # would fail against it; use_spatial_background=True would work today but
    # stays opt-in until deliberately enabled.
    assert config.ETAS_UPDATER_CONFIG['use_spatial_background'] is False
    assert config.ETAS_UPDATER_CONFIG['use_spatial_productivity'] is False
