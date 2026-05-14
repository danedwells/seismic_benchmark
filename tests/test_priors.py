"""
Unit tests for benchmark/priors.py.

build_and_cache_priors() wraps external SeismicPrior factory constructors that
each require raw source data files.  These tests verify the function's error
handling and control flow using paths that don't exist on disk, confirming
that failures in individual priors are caught and logged rather than raised.
"""
import pytest

from benchmark.priors import build_and_cache_priors


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
