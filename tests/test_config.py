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


def test_benchmark_params_search_depths_default_preserves_behaviour():
    """A single 8.0 km entry keeps bEPIC's original fixed-depth search."""
    depths = config.BENCHMARK_PARAMS['search_depths']
    assert depths == [8.0]


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


def test_etas_inversion_config_mc_valid():
    """mc is either a rolling-completeness mode ('positive'/'var') or a
    fixed positive magnitude; m_ref (the fixed floor that mode still
    requires) must always be positive."""
    cfg = config.ETAS_INVERSION_CONFIG
    if cfg['mc'] in ('positive', 'var'):
        assert cfg['m_ref'] > 0
    else:
        assert cfg['mc'] > 0


def test_etas_inversion_config_id_is_string():
    assert isinstance(config.ETAS_INVERSION_CONFIG['id'], str)


# ---------------------------------------------------------------------------
# etas_run_tag / etas_output_id / etas_catalog_tag
# ---------------------------------------------------------------------------

def test_etas_run_tag_default_config_stable():
    tag = config.etas_run_tag()
    assert tag == config.etas_run_tag(), "tag must be deterministic for the same config"


def test_etas_run_tag_filename_safe():
    tag = config.etas_run_tag()
    unsafe = set('/\\:*?"<>| ')
    assert not (unsafe & set(tag)), f"tag contains filesystem-unsafe characters: {tag!r}"


def test_etas_run_tag_reflects_free_background_and_productivity():
    cfg_off = dict(config.ETAS_INVERSION_CONFIG, free_background=False, free_productivity=False)
    cfg_on  = dict(config.ETAS_INVERSION_CONFIG, free_background=True,  free_productivity=True)
    assert config.etas_run_tag(cfg_off) != config.etas_run_tag(cfg_on)
    assert 'fb0' in config.etas_run_tag(cfg_off) and 'fp0' in config.etas_run_tag(cfg_off)
    assert 'fb1' in config.etas_run_tag(cfg_on)  and 'fp1' in config.etas_run_tag(cfg_on)


def test_etas_run_tag_positive_vs_var_mc_differ():
    cfg_pos = dict(config.ETAS_INVERSION_CONFIG, mc='positive', m_ref=3.0)
    cfg_var = dict(config.ETAS_INVERSION_CONFIG, mc='var',      m_ref=3.0)
    assert config.etas_run_tag(cfg_pos) != config.etas_run_tag(cfg_var)


def test_etas_run_tag_fixed_mc_uses_mc_not_m_ref():
    """etas_2 ignores m_ref when mc is a fixed float, so the tag must encode
    the mc value actually used, not a stale/irrelevant m_ref."""
    cfg = dict(config.ETAS_INVERSION_CONFIG, mc=3.5, m_ref=99.0)
    tag = config.etas_run_tag(cfg)
    assert 'mc-fixed3.5' in tag
    assert 'mref3.5' in tag
    assert '99' not in tag


def test_etas_run_tag_reflects_bw_sq():
    """Different bw_sq must produce different tags -- otherwise runs with
    different KDE bandwidths silently overwrite each other's output files."""
    cfg_a = dict(config.ETAS_INVERSION_CONFIG, bw_sq=4)
    cfg_b = dict(config.ETAS_INVERSION_CONFIG, bw_sq=6)
    assert config.etas_run_tag(cfg_a) != config.etas_run_tag(cfg_b)
    assert 'bw4' in config.etas_run_tag(cfg_a)
    assert 'bw6' in config.etas_run_tag(cfg_b)


def test_etas_output_id_includes_context_and_tag():
    out_id = config.etas_output_id('benchmark')
    assert out_id.startswith('benchmark__')
    assert out_id == f'benchmark__{config.etas_run_tag()}'


def test_etas_output_id_differs_across_contexts_same_config():
    assert config.etas_output_id('benchmark') != config.etas_output_id('Ridgecrest')


def test_etas_catalog_tag_independent_of_free_background_and_productivity():
    """catalog_{context}.csv content doesn't depend on fb/fp/mc mode, only
    context + m_ref, so its tag must be stable across those flags."""
    cfg_off = dict(config.ETAS_INVERSION_CONFIG, free_background=False, free_productivity=False)
    cfg_on  = dict(config.ETAS_INVERSION_CONFIG, free_background=True,  free_productivity=True)
    assert config.etas_catalog_tag('benchmark', cfg_off) == config.etas_catalog_tag('benchmark', cfg_on)


def test_etas_catalog_tag_differs_by_m_ref():
    # m_ref only affects the tag in 'positive'/'var' mc mode (fixed-mc mode
    # ignores m_ref entirely, per _etas_mc_tag) — pin mc explicitly so this
    # test doesn't depend on ETAS_INVERSION_CONFIG['mc']'s current value.
    cfg_a = dict(config.ETAS_INVERSION_CONFIG, mc='positive', m_ref=3.0)
    cfg_b = dict(config.ETAS_INVERSION_CONFIG, mc='positive', m_ref=3.5)
    assert config.etas_catalog_tag('benchmark', cfg_a) != config.etas_catalog_tag('benchmark', cfg_b)


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
