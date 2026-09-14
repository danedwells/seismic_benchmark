"""
benchmark/priors.py — construct static spatial priors from source data and
write them to cached .tt3 files, and blend a static prior with the
time-dependent ETAS prior.

build_and_cache_priors() builds each static prior (Gear1, NSHM,
Helmstetter, Smooth_seismicity, optionally KDE_Seismicity) from its
source data and writes it to a .tt3 cache path so the benchmark scripts
can reload it quickly via SeismicPrior.from_tt3(). blend_priors()
combines a static prior with a time-dependent EtasPriorUpdater grid into
a single weighted SeismicPrior for the mixed-prior workflow.
"""
import os
import copy
import numpy as np
import pandas as pd
from priors import SeismicPrior
from scipy.interpolate import RegularGridInterpolator


def blend_priors(ti_prior, etas_prior, alpha=0.5, prior_alpha=1):
    """
    Blend ti_prior onto the ETAS grid and return a new SeismicPrior:

        combined = alpha * etas_tempered + (1 - alpha) * ti_resampled

    The static prior is bilinearly resampled onto the ETAS prior's
    (lat, lon) grid (out-of-bounds cells filled with 0) before blending;
    both grids are renormalized to sum to 1 (the ETAS grid after optional
    tempering, the resampled static grid after resampling), and the final
    blend is renormalized again.

    Parameters
    ----------
    ti_prior : SeismicPrior or None
        Static (time-independent) prior to blend in. None gives a
        uniform base prior (equivalent to a Uniform time-independent
        prior).
    etas_prior : SeismicPrior
        Time-dependent ETAS prior; its (lat, lon) grid defines the
        output grid that ti_prior is resampled onto.
    alpha : float, optional
        Weight on the ETAS component, in [0, 1]. Default 0.5.
    prior_alpha : float, optional
        Power-law exponent (tempering) applied to the ETAS grid before
        blending, then renormalized. Values < 1 compress the ETAS
        grid's dynamic range (reduce cluster dominance); 1 (default)
        leaves it unchanged.

    Returns
    -------
    SeismicPrior
        A deep copy of etas_prior with its grid replaced by the
        normalized blended grid.
    """
    etas_grid = etas_prior.grid.copy()
    if prior_alpha != 1.0:
        etas_grid  = etas_grid ** prior_alpha
        etas_grid /= etas_grid.sum()

    if ti_prior is None:
        ti_grid = np.ones_like(etas_grid)
    else:
        interp = RegularGridInterpolator(
            (ti_prior.lats, ti_prior.lons),
            ti_prior.grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        lat_mesh, lon_mesh = np.meshgrid(etas_prior.lats, etas_prior.lons, indexing='ij')
        ti_grid = interp(
            np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
        ).reshape(len(etas_prior.lats), len(etas_prior.lons))
        ti_grid = np.clip(ti_grid, 0.0, None)
        ti_grid = np.nan_to_num(ti_grid, nan=0.0)

    ti_sum = ti_grid.sum()
    if ti_sum > 0:
        ti_grid /= ti_sum
    else:
        ti_grid = np.ones_like(etas_grid) / etas_grid.size

    combined  = alpha * etas_grid + (1.0 - alpha) * ti_grid
    combined /= combined.sum()

    mixed      = copy.deepcopy(etas_prior)
    mixed.grid = combined
    return mixed


def build_and_cache_priors(cache_paths, data_dir, construction_params=None):
    """
    Build and cache all static prior .tt3 files.

    Attempts to construct each prior from its source data and write it to the
    corresponding .tt3 path in cache_paths. Each prior is tried independently
    (errors are caught and printed) so a failure in one does not prevent the
    others from being built. Handles Gear1, NSHM, Helmstetter,
    Smooth_seismicity, and — if cache_paths has a non-None 'KDE_Seismicity'
    entry — KDE_Seismicity.

    Source file paths (relative to data_dir) and per-prior out_of_bounds_fill
    values are read from construction_params.  Helmstetter is the only prior
    without a source_paths entry — its data comes from pycsep at runtime.
    KDE_Seismicity instead reads a background-seismicity catalog path and
    KDE parameters from construction_params['kde_seismicity_params'].

    ETAS is time-dependent and is handled separately via EtasPriorUpdater
    (see time_dependent_scripts/).

    Parameters
    ----------
    cache_paths : dict
        Mapping of prior name -> .tt3 file path (or None to skip that
        prior). Recognized keys: 'Gear1', 'NSHM', 'Helmstetter',
        'Smooth_seismicity', and optionally 'KDE_Seismicity'.
    data_dir : str
        Path to the priors data directory (SeismicPrior.data_dir).
    construction_params : dict, optional
        Must contain:
          'bounds'               — (lon_min, lon_max, lat_min, lat_max)
          'source_paths'         — {prior_name: path_relative_to_data_dir, ...}
          'out_of_bounds_fill'   — {prior_name: fill_value, ...}
          'target_resolution_deg'— {prior_name: float or None, ...}  (optional)
          'kde_seismicity_params'— {catalog_path, min_mag, grid_size, bw_method,
                                    lon_col, lat_col}, only needed when
                                    building KDE_Seismicity  (optional)
        Any other keys are passed through unchanged as shared_kwargs to
        every SeismicPrior.from_*() constructor call. Defaults to an
        empty dict.

    Returns
    -------
    None
        Each successfully built prior is written to its .tt3 path in
        cache_paths as a side effect; progress and failures are printed
        to stdout.
    """
    if construction_params is None:
        construction_params = {}

    oob_fills   = construction_params.get('out_of_bounds_fill', {})
    rel_sources = construction_params.get('source_paths', {})
    target_res  = construction_params.get('target_resolution_deg', {})
    shared_kwargs = {k: v for k, v in construction_params.items()
                     if k not in ('out_of_bounds_fill', 'source_paths', 'target_resolution_deg')}

    def _abs(name):
        """
        Resolve a prior's source path to an absolute path under data_dir.

        Parameters
        ----------
        name : str
            Prior name key to look up in rel_sources (the enclosing
            construction_params['source_paths']).

        Returns
        -------
        str or None
            os.path.join(data_dir, rel_sources[name]) if name has an
            entry in rel_sources, otherwise None.
        """
        rel = rel_sources.get(name)
        return os.path.join(data_dir, rel) if rel is not None else None

    def _maybe_resample(p, name):
        """
        Resample a SeismicPrior to a target resolution if one is configured.

        Parameters
        ----------
        p : SeismicPrior
            Prior to conditionally resample.
        name : str
            Prior name key to look up in target_res (the enclosing
            construction_params['target_resolution_deg']).

        Returns
        -------
        SeismicPrior
            p.resample(res) if target_res has a non-None entry for
            name, otherwise p unchanged.
        """
        res = target_res.get(name)
        if res is not None:
            p = p.resample(res)
            print(f"  {name}: resampled to {res}° ({len(p.lons)}×{len(p.lats)} cells)")
        return p

    try:
        p = SeismicPrior.from_gear1(_abs('Gear1'),
                                    out_of_bounds_fill=oob_fills.get('Gear1'),
                                    **shared_kwargs)
        p = _maybe_resample(p, 'Gear1')
        p.to_tt3(cache_paths['Gear1'])
        print("Gear1: built and cached.")
    except Exception as e:
        print(f"Gear1: failed — {e}")

    try:
        p = SeismicPrior.from_nshm(_abs('NSHM'),
                                   fault_data_path=_abs('NSHM_fault'),
                                   out_of_bounds_fill=oob_fills.get('NSHM'),
                                   **shared_kwargs)
        p = _maybe_resample(p, 'NSHM')
        p.to_tt3(cache_paths['NSHM'])
        print("NSHM: built and cached.")
    except Exception as e:
        print(f"NSHM: failed — {e}")

    try:
        p = SeismicPrior.from_helmstetter(out_of_bounds_fill=oob_fills.get('Helmstetter'),
                                          **shared_kwargs)
        p = _maybe_resample(p, 'Helmstetter')
        p.to_tt3(cache_paths['Helmstetter'])
        print("Helmstetter: built and cached.")
    except Exception as e:
        print(f"Helmstetter: failed — {e}")

    try:
        src = _abs('Smooth_seismicity')
        if src is None:
            raise FileNotFoundError("No source_paths entry for Smooth_seismicity in construction_params.")
        p = SeismicPrior.from_tt3(src, name='smooth_seismicity')
        p = _maybe_resample(p, 'Smooth_seismicity')
        bounds = shared_kwargs.get('bounds')
        oob = oob_fills.get('Smooth_seismicity')
        if bounds is not None and oob is not None:
            p.lons, p.lats, p.grid = SeismicPrior._expand_to_bounds(
                p.lons, p.lats, p.grid, bounds, oob)
            p.grid = p.grid / np.nansum(p.grid)
        p.to_tt3(cache_paths['Smooth_seismicity'])
        print("Smooth_seismicity: built and cached.")
    except Exception as e:
        print(f"Smooth_seismicity: failed — {e}")

    if 'KDE_Seismicity' in cache_paths and cache_paths.get('KDE_Seismicity') is not None:
        try:
            kde_params = construction_params.get('kde_seismicity_params', {})
            catalog_path = kde_params.get('catalog_path')
            if catalog_path is None or not os.path.exists(catalog_path):
                raise FileNotFoundError(
                    f"KDE catalog not found: {catalog_path!r}\n"
                    "Set construction_params['kde_seismicity_params']['catalog_path'] "
                    "to the background seismicity parquet before building."
                )
            catalog = (pd.read_parquet(catalog_path) if catalog_path.endswith('.parquet')
                       else pd.read_csv(catalog_path))
            min_mag = kde_params.get('min_mag')
            if min_mag is not None:
                catalog = catalog[catalog['mag'] >= min_mag]
            p = SeismicPrior.from_kde_seismicity(
                catalog        = catalog,
                bounds         = shared_kwargs.get('bounds'),
                grid_size      = kde_params.get('grid_size', 100),
                bw_method      = kde_params.get('bw_method', 'scott'),
                lon_col        = kde_params.get('lon_col', 'longitude'),
                lat_col        = kde_params.get('lat_col', 'latitude'),
                out_of_bounds_fill = oob_fills.get('KDE_Seismicity'),
            )
            p = _maybe_resample(p, 'KDE_Seismicity')
            p.to_tt3(cache_paths['KDE_Seismicity'])
            print(f"KDE_Seismicity: built from {len(catalog):,} events and cached.")
        except Exception as e:
            print(f"KDE_Seismicity: failed — {e}")


