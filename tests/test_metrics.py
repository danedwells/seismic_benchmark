"""
Unit tests for benchmark/metrics.py.

All functions here are pure numpy/pandas math — no bEPIC calls, no network,
no disk I/O.  These should always pass in any environment where the package
is installed.
"""
import os

import numpy as np
import pandas as pd
import pytest

from benchmark.metrics import (
    hdr_levels,
    location_error_km,
    posterior_confidence_level,
    prior_confidence_level,
    like_confidence_level,
    _haversine_km,
    posterior_coverage,
    COVERAGE_RADII_KM,
    log_score,
    brier_score,
    energy_score,
    likelihood_value_at_location,
    likelihood_value_at_location_unnormalized,
    posterior_value_at_location,
    load_final_values,
    load_final_rows,
    load_per_version_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(n=100, seed=42):
    """Synthetic posterior grid near (37°N, 120°W) for reuse across tests."""
    rng = np.random.default_rng(seed)
    lats  = np.linspace(36.0, 38.0, n)
    lons  = np.linspace(-121.0, -119.0, n)
    post  = rng.random(n)
    prior = rng.random(n)
    return pd.DataFrame({'lat': lats, 'lon': lons, 'post': post, 'prior': prior})


# ---------------------------------------------------------------------------
# _haversine_km
# ---------------------------------------------------------------------------

def test_haversine_same_point_is_zero():
    d = _haversine_km(37.0, -120.0, np.array([37.0]), np.array([-120.0]))
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_one_degree_latitude():
    """1° of latitude ≈ 111.195 km at the equator."""
    d = _haversine_km(0.0, 0.0, np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(111.195, abs=0.5)


def test_haversine_vectorized_shape():
    lats = np.array([37.0, 38.0, 39.0])
    lons = np.full(3, -120.0)
    d = _haversine_km(37.0, -120.0, lats, lons)
    assert d.shape == (3,)


def test_haversine_monotone_northward():
    lats = np.array([37.0, 38.0, 39.0])
    lons = np.full(3, -120.0)
    d = _haversine_km(37.0, -120.0, lats, lons)
    assert d[0] < d[1] < d[2]


def test_haversine_nonnegative():
    rng = np.random.default_rng(0)
    lats = rng.uniform(30, 50, 50)
    lons = rng.uniform(-130, -110, 50)
    d = _haversine_km(37.0, -120.0, lats, lons)
    assert np.all(d >= 0)


# ---------------------------------------------------------------------------
# hdr_levels
# ---------------------------------------------------------------------------

def test_hdr_levels_uniform_distribution():
    """For a uniform distribution every cell has equal density."""
    n = 100
    post = np.ones(n) / n
    thresholds = hdr_levels(post, credible_levels=(0.5, 0.9))
    assert thresholds[0.5] == pytest.approx(1.0 / n, rel=1e-6)
    assert thresholds[0.9] == pytest.approx(1.0 / n, rel=1e-6)


def test_hdr_levels_peaked_distribution():
    """Single dominant cell: the 50% HDR threshold is the peak mass."""
    post = np.zeros(100)
    post[0] = 0.9
    post[1:] = 0.1 / 99
    thresholds = hdr_levels(post, credible_levels=(0.5,))
    assert thresholds[0.5] == pytest.approx(0.9, rel=1e-6)


def test_hdr_levels_returns_all_requested_levels():
    post = np.ones(50) / 50
    levels = (0.10, 0.50, 0.90, 0.95)
    thresholds = hdr_levels(post, credible_levels=levels)
    assert set(thresholds.keys()) == set(levels)


def test_hdr_levels_thresholds_are_nonnegative():
    rng = np.random.default_rng(7)
    post = rng.random(200)
    thresholds = hdr_levels(post)
    assert all(v >= 0 for v in thresholds.values())


# ---------------------------------------------------------------------------
# location_error_km
# ---------------------------------------------------------------------------

def test_location_error_same_point():
    assert location_error_km(37.0, -120.0, 37.0, -120.0) == pytest.approx(0.0, abs=1e-3)


def test_location_error_one_degree_latitude():
    err = location_error_km(37.0, -120.0, 38.0, -120.0)
    assert err == pytest.approx(111.2, abs=1.5)


def test_location_error_positive():
    err = location_error_km(37.0, -120.0, 38.0, -121.0)
    assert err > 0


def test_location_error_symmetric():
    err1 = location_error_km(37.0, -120.0, 38.0, -121.0)
    err2 = location_error_km(38.0, -121.0, 37.0, -120.0)
    assert err1 == pytest.approx(err2, rel=1e-6)


# ---------------------------------------------------------------------------
# posterior_confidence_level
# ---------------------------------------------------------------------------

def test_posterior_confidence_level_at_map_peak_is_minimum():
    """USGS at the MAP peak gives the smallest credible level for this distribution.

    posterior_confidence_level returns the mass of the HDR that just contains the USGS
    location (cells with density >= density at USGS).  The MAP peak has the
    highest density, so its HDR is the smallest — it equals just the cell's own
    mass, which is less than or equal to any other cell's credible level.
    """
    df = _make_grid()
    peak_idx = df['post'].idxmax()
    cred_at_peak = posterior_confidence_level(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    # Spot-check a few non-peak cells: their credible level should be >= the peak's
    for idx in df.nsmallest(5, 'post').index:
        cred_other = posterior_confidence_level(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert cred_at_peak <= cred_other + 1e-9


def test_posterior_confidence_level_in_unit_interval():
    cred = posterior_confidence_level(_make_grid(), 37.0, -120.0)
    assert 0.0 <= cred <= 1.0


def test_posterior_confidence_level_single_cell():
    """Grid with one cell: credible level is always 1.0."""
    df = pd.DataFrame({'lat': [37.0], 'lon': [-120.0], 'post': [1.0], 'prior': [1.0]})
    assert posterior_confidence_level(df, 37.0, -120.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# prior_confidence_level
# ---------------------------------------------------------------------------

def test_prior_confidence_level_in_unit_interval():
    cred = prior_confidence_level(_make_grid(), 37.0, -120.0)
    assert 0.0 <= cred <= 1.0


def test_prior_confidence_level_at_prior_peak_is_minimum():
    """Prior credible level at the prior MAP peak is the minimum for this distribution."""
    df = _make_grid()
    peak_idx = df['prior'].idxmax()
    cred_at_peak = prior_confidence_level(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    for idx in df.nsmallest(5, 'prior').index:
        cred_other = prior_confidence_level(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert cred_at_peak <= cred_other + 1e-9


# ---------------------------------------------------------------------------
# posterior_coverage
# ---------------------------------------------------------------------------

def test_posterior_coverage_exact_location():
    """Single cell exactly at the reference → full coverage at all radii."""
    df = pd.DataFrame({'lat': [37.0], 'lon': [-120.0], 'post': [1.0]})
    cov = posterior_coverage(df, 37.0, -120.0)
    for r in COVERAGE_RADII_KM:
        assert cov[r] == pytest.approx(1.0, abs=1e-9)


def test_posterior_coverage_far_from_reference():
    """All mass far from the reference → zero coverage at small radii."""
    df = pd.DataFrame({'lat': [60.0, 61.0], 'lon': [10.0, 11.0], 'post': [0.5, 0.5]})
    cov = posterior_coverage(df, 0.0, 0.0, radii_km=(1.0,))
    assert cov[1.0] == pytest.approx(0.0, abs=1e-9)


def test_posterior_coverage_scalar_radius_returns_float():
    cov = posterior_coverage(_make_grid(), 37.0, -120.0, radii_km=50.0)
    assert isinstance(cov, float)


def test_posterior_coverage_dict_radius_returns_dict():
    cov = posterior_coverage(_make_grid(), 37.0, -120.0, radii_km=(25.0, 50.0))
    assert isinstance(cov, dict)
    assert set(cov.keys()) == {25.0, 50.0}


def test_posterior_coverage_monotone_with_radius():
    """Coverage is non-decreasing as the radius grows."""
    cov = posterior_coverage(_make_grid(), 37.0, -120.0)
    radii = sorted(COVERAGE_RADII_KM)
    for i in range(len(radii) - 1):
        assert cov[radii[i]] <= cov[radii[i + 1]] + 1e-9


def test_posterior_coverage_in_unit_interval():
    cov = posterior_coverage(_make_grid(), 37.0, -120.0)
    for v in cov.values():
        assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# COVERAGE_RADII_KM constant
# ---------------------------------------------------------------------------

def test_coverage_radii_constant_values():
    assert COVERAGE_RADII_KM == (10, 25, 50, 100)


def test_coverage_radii_ascending():
    assert list(COVERAGE_RADII_KM) == sorted(COVERAGE_RADII_KM)


# ---------------------------------------------------------------------------
# log_score
# ---------------------------------------------------------------------------

def _single_cell_grid(lat=37.0, lon=-120.0, post=1.0):
    return pd.DataFrame({'lat': [lat], 'lon': [lon], 'post': [post], 'prior': [1.0]})


def test_log_score_all_mass_at_truth_is_zero():
    """Posterior entirely at the true cell → log(1) = 0."""
    df = _single_cell_grid()
    assert log_score(df, 37.0, -120.0) == pytest.approx(0.0, abs=1e-9)


def test_log_score_nonpositive():
    """log(P_true) ≤ 0 since P_true ≤ 1."""
    assert log_score(_make_grid(), 37.0, -120.0) <= 0.0


def test_log_score_higher_at_map_peak():
    """Log-score is highest (least negative) when ref is at the MAP peak."""
    df = _make_grid()
    peak_idx = df['post'].idxmax()
    score_at_peak = log_score(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    for idx in df.nsmallest(5, 'post').index:
        score_other = log_score(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert score_at_peak >= score_other - 1e-9


def test_log_score_floor_no_minus_inf():
    """Returns a finite value even when the nearest cell has zero posterior mass."""
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -119.0],
                       'post': [0.0, 1.0], 'prior': [0.5, 0.5]})
    score = log_score(df, 37.0, -120.0)
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

def test_brier_score_all_mass_at_truth_is_zero():
    """All posterior mass at the true cell → BS = 0."""
    df = _single_cell_grid()
    assert brier_score(df, 37.0, -120.0) == pytest.approx(0.0, abs=1e-9)


def test_brier_score_all_mass_wrong_cell():
    """Two cells, all mass on the wrong cell → BS = 2·(1/2)² + (0−1)² wait, let me think...
    P_j = [1, 0], O_j = [0, 1] (ref at second cell).
    BS = (1−0)² + (0−1)² = 2.0
    """
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -120.0],
                       'post': [1.0, 0.0], 'prior': [0.5, 0.5]})
    assert brier_score(df, 38.0, -120.0) == pytest.approx(2.0, abs=1e-9)


def test_brier_score_in_range():
    """BS ∈ [0, 2] for any normalized posterior."""
    bs = brier_score(_make_grid(), 37.0, -120.0)
    assert 0.0 <= bs <= 2.0 + 1e-9


def test_brier_score_lower_at_map_peak():
    """BS is minimized when ref is at the MAP peak."""
    df = _make_grid()
    peak_idx = df['post'].idxmax()
    bs_at_peak = brier_score(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    for idx in df.nsmallest(5, 'post').index:
        bs_other = brier_score(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert bs_at_peak <= bs_other + 1e-9


def test_brier_score_uniform_two_cells():
    """Uniform over two cells, ref at first: P=[0.5,0.5], O=[1,0].
    BS = (0.5−1)² + (0.5−0)² = 0.25 + 0.25 = 0.5
    """
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -120.0],
                       'post': [1.0, 1.0], 'prior': [0.5, 0.5]})
    assert brier_score(df, 37.0, -120.0) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# like_confidence_level
# ---------------------------------------------------------------------------

def _make_grid_with_like(n=100, seed=42):
    df = _make_grid(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df['like'] = rng.random(n)
    return df


def test_like_confidence_level_in_unit_interval():
    cred = like_confidence_level(_make_grid_with_like(), 37.0, -120.0)
    assert 0.0 <= cred <= 1.0


def test_like_confidence_level_at_like_peak_is_minimum():
    df = _make_grid_with_like()
    peak_idx = df['like'].idxmax()
    cred_at_peak = like_confidence_level(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    for idx in df.nsmallest(5, 'like').index:
        cred_other = like_confidence_level(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert cred_at_peak <= cred_other + 1e-9


def test_like_confidence_level_single_cell():
    df = pd.DataFrame({'lat': [37.0], 'lon': [-120.0], 'like': [1.0]})
    assert like_confidence_level(df, 37.0, -120.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# likelihood_value_at_location / likelihood_value_at_location_unnormalized
# ---------------------------------------------------------------------------

def test_likelihood_value_at_location_matches_normalized_cell():
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -119.0],
                       'like': [3.0, 1.0]})
    val = likelihood_value_at_location(df, 37.0, -120.0)
    assert val == pytest.approx(3.0 / 4.0)


def test_likelihood_value_at_location_unnormalized_is_raw():
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -119.0],
                       'like': [3.0, 1.0]})
    val = likelihood_value_at_location_unnormalized(df, 37.0, -120.0)
    assert val == pytest.approx(3.0)


def test_likelihood_value_at_location_in_unit_interval():
    val = likelihood_value_at_location(_make_grid_with_like(), 37.0, -120.0)
    assert 0.0 <= val <= 1.0


def test_likelihood_value_picks_nearest_cell():
    """Reference exactly on the second cell should read that cell's value,
    regardless of how far the first cell is."""
    df = pd.DataFrame({'lat': [0.0, 37.0], 'lon': [0.0, -120.0],
                       'like': [1.0, 9.0]})
    val = likelihood_value_at_location(df, 37.0, -120.0)
    assert val == pytest.approx(9.0 / 10.0)


# ---------------------------------------------------------------------------
# posterior_value_at_location
# ---------------------------------------------------------------------------

def test_posterior_value_at_location_matches_normalized_cell():
    df = pd.DataFrame({'lat': [37.0, 38.0], 'lon': [-120.0, -119.0],
                       'post': [1.0, 3.0]})
    val = posterior_value_at_location(df, 38.0, -119.0)
    assert val == pytest.approx(3.0 / 4.0)


def test_posterior_value_at_location_equals_exp_log_score():
    """posterior_value_at_location is documented as equivalent to
    exp(log_score(...)) — same p_true term, exposed directly."""
    df = _make_grid()
    val = posterior_value_at_location(df, 37.0, -120.0)
    ls  = log_score(df, 37.0, -120.0)
    assert val == pytest.approx(np.exp(ls), rel=1e-9)


# ---------------------------------------------------------------------------
# energy_score
# ---------------------------------------------------------------------------

def test_energy_score_all_mass_at_truth_is_near_zero():
    """A single-cell posterior exactly at the reference: term1=0 and term2=0
    (all samples are the same point), so ES should be ~0."""
    df = _single_cell_grid()
    es = energy_score(df, 37.0, -120.0, n_samples=100, rng=np.random.default_rng(0))
    assert es == pytest.approx(0.0, abs=1e-6)


def test_energy_score_nonnegative_for_typical_grid():
    """ES = term1 - term2 isn't nonnegative in general, but for a diffuse
    posterior far from a concentrated point mass it should be strictly
    positive here since term1 (mean distance to ref) dominates."""
    es = energy_score(_make_grid(), 60.0, 10.0, n_samples=200, rng=np.random.default_rng(1))
    assert es > 0


def test_energy_score_deterministic_with_seeded_rng():
    df = _make_grid()
    es1 = energy_score(df, 37.0, -120.0, n_samples=200, rng=np.random.default_rng(123))
    es2 = energy_score(df, 37.0, -120.0, n_samples=200, rng=np.random.default_rng(123))
    assert es1 == pytest.approx(es2)


def test_energy_score_uses_default_rng_when_none():
    """rng=None must not raise -- a fresh default_rng() is created internally."""
    es = energy_score(_make_grid(), 37.0, -120.0, n_samples=50, rng=None)
    assert np.isfinite(es)


# ---------------------------------------------------------------------------
# load_final_values / load_final_rows / load_per_version_stats
# ---------------------------------------------------------------------------

def _write_benchmark_csv(path):
    """Two events, three trigger-count versions each, one metric column."""
    rows = []
    for eid in (1, 2):
        for n_trigs, err in zip((1, 2, 3), (50.0, 20.0, 5.0 + eid)):
            rows.append({'event_id': eid, 'version': n_trigs - 1,
                        'n_trigs': n_trigs, 'map_err_km': err})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_final_values_missing_file_returns_none(tmp_path):
    assert load_final_values(str(tmp_path / 'nope.csv'), 'map_err_km') is None


def test_load_final_values_missing_column_returns_none(tmp_path):
    csv_path = tmp_path / 'bench.csv'
    pd.DataFrame({'event_id': [1], 'version': [0]}).to_csv(csv_path, index=False)
    assert load_final_values(str(csv_path), 'map_err_km') is None


def test_load_final_values_default_takes_last_version_per_event(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    vals = load_final_values(str(csv_path), 'map_err_km')
    assert sorted(vals) == pytest.approx(sorted([6.0, 7.0]))  # last (n_trigs=3) row per event


def test_load_final_values_specific_n_trigs(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    vals = load_final_values(str(csv_path), 'map_err_km', n_trigs=1)
    assert sorted(vals) == pytest.approx([50.0, 50.0])


def test_load_final_values_n_trigs_exceeds_max_raises(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    with pytest.raises(ValueError):
        load_final_values(str(csv_path), 'map_err_km', n_trigs=99)


def test_load_final_values_warns_below_min_events(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    with pytest.warns(UserWarning):
        load_final_values(str(csv_path), 'map_err_km', n_trigs=1, min_events_warn=5)


def test_load_final_rows_missing_file_returns_none(tmp_path):
    assert load_final_rows(str(tmp_path / 'nope.csv')) is None


def test_load_final_rows_one_row_per_event(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    df = load_final_rows(str(csv_path))
    assert len(df) == 2
    assert set(df['event_id']) == {1, 2}


def test_load_final_rows_specific_n_trigs_filters_rows(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    df = load_final_rows(str(csv_path), n_trigs=2)
    assert len(df) == 2
    assert set(df['n_trigs']) == {2}


def test_load_per_version_stats_missing_file_returns_none(tmp_path):
    assert load_per_version_stats(str(tmp_path / 'nope.csv'), 'map_err_km') is None


def test_load_per_version_stats_missing_column_returns_none(tmp_path):
    csv_path = tmp_path / 'bench.csv'
    pd.DataFrame({'event_id': [1], 'version': [0]}).to_csv(csv_path, index=False)
    assert load_per_version_stats(str(csv_path), 'map_err_km') is None


def test_load_per_version_stats_filters_by_min_events(tmp_path):
    """min_events=5 with only 2 events per n_trigs -> every row filtered out."""
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    stats = load_per_version_stats(str(csv_path), 'map_err_km', min_events=5)
    assert len(stats) == 0


def test_load_per_version_stats_keeps_rows_meeting_min_events(tmp_path):
    csv_path = _write_benchmark_csv(tmp_path / 'bench.csv')
    stats = load_per_version_stats(str(csv_path), 'map_err_km', min_events=2)
    assert set(stats['n_trigs']) == {1, 2, 3}
    assert set(stats['count']) == {2}


