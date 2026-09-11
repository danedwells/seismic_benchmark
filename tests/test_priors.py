"""
Unit tests for benchmark/priors.py.

build_and_cache_priors() wraps external SeismicPrior factory constructors that
each require raw source data files.  These tests verify the function's error
handling and control flow using paths that don't exist on disk, confirming
that failures in individual priors are caught and logged rather than raised.

blend_priors() is pure numpy/scipy math over SeismicPrior-shaped objects, so
it's tested directly against lightweight stand-ins (no real SeismicPrior /
.tt3 files needed).
"""
import copy

import numpy as np
import pytest

from benchmark.priors import build_and_cache_priors, blend_priors


# ---------------------------------------------------------------------------
# build_and_cache_priors — error handling
# ---------------------------------------------------------------------------

def test_missing_all_source_files_does_not_raise():
    """All priors fail (missing source data) — no exception propagates out."""
    cache_paths = {
        'Gear1':             '/nonexistent/gear1.tt3',
        'NSHM':              '/nonexistent/nshm.tt3',
        'Helmstetter':       '/nonexistent/helm.tt3',
        'Smooth_seismicity': '/nonexistent/ss.tt3',
    }
    build_and_cache_priors(cache_paths, '/nonexistent', construction_params={})


def test_none_construction_params_does_not_raise():
    """Calling with construction_params=None uses empty defaults without crashing."""
    build_and_cache_priors({}, '/nonexistent', construction_params=None)


def test_empty_cache_paths_does_not_raise():
    """Empty cache_paths dict: nothing to build, function exits cleanly."""
    build_and_cache_priors({}, '/nonexistent', construction_params={})


def test_kde_skipped_when_not_in_cache_paths():
    """KDE block is only entered when 'KDE_Seismicity' is a key in cache_paths."""
    # Should not raise even though KDE params are absent
    build_and_cache_priors(
        {'Gear1': '/nonexistent/gear1.tt3'},
        '/nonexistent',
        construction_params={},
    )


def test_kde_fails_gracefully_with_missing_catalog():
    """KDE with a non-existent catalog path raises FileNotFoundError internally
    and is caught — no exception propagates."""
    cache_paths = {'KDE_Seismicity': '/nonexistent/kde.tt3'}
    params = {
        'bounds': (-129, -112, 30, 51),
        'kde_seismicity_params': {
            'catalog_path': '/nonexistent/catalog.parquet',
            'grid_size': 10,
            'bw_method': 'scott',
        },
    }
    build_and_cache_priors(cache_paths, '/nonexistent', construction_params=params)


def test_smooth_seismicity_fails_gracefully_with_no_source_path():
    """Smooth_seismicity with no source_paths entry fails gracefully."""
    cache_paths = {'Smooth_seismicity': '/nonexistent/ss.tt3'}
    build_and_cache_priors(
        cache_paths,
        '/nonexistent',
        construction_params={'source_paths': {}},
    )


def test_individual_prior_failure_does_not_block_others(capsys):
    """A failure in one prior should not prevent the others from being attempted."""
    cache_paths = {
        'Gear1': '/nonexistent/gear1.tt3',
        'NSHM':  '/nonexistent/nshm.tt3',
    }
    build_and_cache_priors(cache_paths, '/nonexistent', construction_params={})
    captured = capsys.readouterr()
    # Both priors should have been attempted — both failure lines should appear
    assert 'Gear1' in captured.out
    assert 'NSHM' in captured.out


# ---------------------------------------------------------------------------
# blend_priors
# ---------------------------------------------------------------------------

class _MockPrior:
    """Minimal stand-in for SeismicPrior: just lats/lons/grid, which is all
    blend_priors() touches (via RegularGridInterpolator + copy.deepcopy)."""
    def __init__(self, lats, lons, grid):
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.grid = np.asarray(grid, dtype=float)


def _etas_prior():
    # 3x3 grid, deliberately peaked (not uniform) and pre-normalized.
    lats = [36.0, 37.0, 38.0]
    lons = [-121.0, -120.0, -119.0]
    grid = np.array([[0.05, 0.05, 0.05],
                     [0.05, 0.6,  0.05],
                     [0.05, 0.05, 0.05]])
    grid = grid / grid.sum()
    return _MockPrior(lats, lons, grid)


def _ti_prior_same_grid():
    # Same lat/lon axes as _etas_prior() so interpolation is identity.
    lats = [36.0, 37.0, 38.0]
    lons = [-121.0, -120.0, -119.0]
    grid = np.array([[1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0],
                     [1.0, 9.0, 1.0]])
    return _MockPrior(lats, lons, grid)


def test_blend_priors_result_sums_to_one():
    mixed = blend_priors(_ti_prior_same_grid(), _etas_prior(), alpha=0.5)
    assert mixed.grid.sum() == pytest.approx(1.0)


def test_blend_priors_alpha_one_is_etas_only():
    """alpha=1 should reduce to the (already-normalized) ETAS grid alone."""
    etas = _etas_prior()
    mixed = blend_priors(_ti_prior_same_grid(), etas, alpha=1.0)
    np.testing.assert_allclose(mixed.grid, etas.grid, atol=1e-12)


def test_blend_priors_alpha_zero_is_ti_only():
    """alpha=0 should reduce to the TI grid, resampled onto the ETAS grid
    (identity here since both share the same lat/lon axes) and normalized."""
    ti = _ti_prior_same_grid()
    mixed = blend_priors(ti, _etas_prior(), alpha=0.0)
    expected = ti.grid / ti.grid.sum()
    np.testing.assert_allclose(mixed.grid, expected, atol=1e-12)


def test_blend_priors_none_ti_prior_uses_uniform_base():
    """ti_prior=None should behave as if TI were a uniform grid."""
    etas = _etas_prior()
    mixed_none = blend_priors(None, etas, alpha=0.5)

    # Build an explicit uniform prior on the same grid and compare.
    lats, lons = etas.lats, etas.lons
    uniform_grid = np.ones_like(etas.grid)
    uniform = _MockPrior(lats, lons, uniform_grid)
    mixed_uniform = blend_priors(uniform, _etas_prior(), alpha=0.5)

    np.testing.assert_allclose(mixed_none.grid, mixed_uniform.grid, atol=1e-12)


def test_blend_priors_does_not_mutate_inputs():
    """blend_priors must not modify the caller's prior objects in place."""
    ti = _ti_prior_same_grid()
    etas = _etas_prior()
    ti_grid_before   = ti.grid.copy()
    etas_grid_before = etas.grid.copy()
    blend_priors(ti, etas, alpha=0.5)
    np.testing.assert_array_equal(ti.grid, ti_grid_before)
    np.testing.assert_array_equal(etas.grid, etas_grid_before)


def test_blend_priors_returns_new_object_not_etas_prior():
    etas = _etas_prior()
    mixed = blend_priors(_ti_prior_same_grid(), etas, alpha=0.5)
    assert mixed is not etas


def test_blend_priors_prior_alpha_tempering_changes_result():
    """prior_alpha < 1 compresses the ETAS grid's dynamic range before
    blending, so it must produce a different combined grid than
    prior_alpha=1 (no tempering) for a non-uniform ETAS grid."""
    ti = _ti_prior_same_grid()
    mixed_no_temper   = blend_priors(ti, _etas_prior(), alpha=0.7, prior_alpha=1.0)
    mixed_tempered    = blend_priors(ti, _etas_prior(), alpha=0.7, prior_alpha=0.5)
    assert not np.allclose(mixed_no_temper.grid, mixed_tempered.grid)


def test_blend_priors_result_nonnegative():
    mixed = blend_priors(_ti_prior_same_grid(), _etas_prior(), alpha=0.3, prior_alpha=0.7)
    assert np.all(mixed.grid >= 0)


def test_blend_priors_zero_sum_ti_grid_falls_back_to_uniform():
    """A ti_prior whose grid interpolates to all-zero on the ETAS axes
    (e.g. entirely out-of-bounds) must not raise or divide by zero — it
    falls back to a uniform ti_grid instead."""
    etas = _etas_prior()
    # lat/lon axes far outside etas's bounds -> interpolation fills 0.0 everywhere.
    ti = _MockPrior([10.0, 11.0, 12.0], [10.0, 11.0, 12.0],
                    np.ones((3, 3)))
    mixed = blend_priors(ti, etas, alpha=0.5)
    assert np.all(np.isfinite(mixed.grid))
    assert mixed.grid.sum() == pytest.approx(1.0)
