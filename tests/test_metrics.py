"""
Unit tests for benchmark/metrics.py.

All functions here are pure numpy/pandas math — no bEPIC calls, no network,
no disk I/O.  These should always pass in any environment where the package
is installed.
"""
import numpy as np
import pandas as pd
import pytest

from benchmark.metrics import (
    hdr_levels,
    location_error_km,
    usgs_credible_level,
    usgs_prior_credible_level,
    _haversine_km,
    posterior_coverage,
    COVERAGE_RADII_KM,
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
# usgs_credible_level
# ---------------------------------------------------------------------------

def test_usgs_credible_level_at_map_peak_is_minimum():
    """USGS at the MAP peak gives the smallest credible level for this distribution.

    usgs_credible_level returns the mass of the HDR that just contains the USGS
    location (cells with density >= density at USGS).  The MAP peak has the
    highest density, so its HDR is the smallest — it equals just the cell's own
    mass, which is less than or equal to any other cell's credible level.
    """
    df = _make_grid()
    peak_idx = df['post'].idxmax()
    cred_at_peak = usgs_credible_level(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    # Spot-check a few non-peak cells: their credible level should be >= the peak's
    for idx in df.nsmallest(5, 'post').index:
        cred_other = usgs_credible_level(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
        assert cred_at_peak <= cred_other + 1e-9


def test_usgs_credible_level_in_unit_interval():
    cred = usgs_credible_level(_make_grid(), 37.0, -120.0)
    assert 0.0 <= cred <= 1.0


def test_usgs_credible_level_single_cell():
    """Grid with one cell: credible level is always 1.0."""
    df = pd.DataFrame({'lat': [37.0], 'lon': [-120.0], 'post': [1.0], 'prior': [1.0]})
    assert usgs_credible_level(df, 37.0, -120.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# usgs_prior_credible_level
# ---------------------------------------------------------------------------

def test_usgs_prior_credible_level_in_unit_interval():
    cred = usgs_prior_credible_level(_make_grid(), 37.0, -120.0)
    assert 0.0 <= cred <= 1.0


def test_usgs_prior_credible_level_at_prior_peak_is_minimum():
    """Prior credible level at the prior MAP peak is the minimum for this distribution."""
    df = _make_grid()
    peak_idx = df['prior'].idxmax()
    cred_at_peak = usgs_prior_credible_level(df, df.loc[peak_idx, 'lat'], df.loc[peak_idx, 'lon'])
    for idx in df.nsmallest(5, 'prior').index:
        cred_other = usgs_prior_credible_level(df, df.loc[idx, 'lat'], df.loc[idx, 'lon'])
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
