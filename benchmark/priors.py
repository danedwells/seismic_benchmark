import os
import numpy as np
import pandas as pd
from priors import SeismicPrior


def build_and_cache_priors(cache_paths, data_dir, construction_params=None):
    """
    Build and cache all prior .tt3 files.

    Attempts to construct each prior from its source data and write it to the
    corresponding .tt3 path in cache_paths. Each prior is tried independently
    so a failure in one does not prevent the others from being built.

    Source file paths (relative to data_dir) and per-prior out_of_bounds_fill
    values are read from construction_params.  Helmstetter is the only prior
    without a source_paths entry — its data comes from pycsep at runtime.

    ETAS is time-dependent and is handled separately via EtasPriorUpdater
    (see time_dependent_scripts/).

    Parameters
    ----------
    cache_paths : dict
        Mapping of prior name -> .tt3 file path (or None for Uniform).
        Expected keys: 'Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'Uniform'.
    data_dir : str
        Path to the priors data directory (SeismicPrior.data_dir).
    construction_params : dict, optional
        Must contain:
          'bounds'               — (lon_min, lon_max, lat_min, lat_max)
          'source_paths'         — {prior_name: path_relative_to_data_dir, ...}
          'out_of_bounds_fill'   — {prior_name: fill_value, ...}
          'target_resolution_deg'— {prior_name: float or None, ...}  (optional)
        Defaults to an empty dict.
    """
    if construction_params is None:
        construction_params = {}

    oob_fills   = construction_params.get('out_of_bounds_fill', {})
    rel_sources = construction_params.get('source_paths', {})
    target_res  = construction_params.get('target_resolution_deg', {})
    shared_kwargs = {k: v for k, v in construction_params.items()
                     if k not in ('out_of_bounds_fill', 'source_paths', 'target_resolution_deg')}

    def _abs(name):
        rel = rel_sources.get(name)
        return os.path.join(data_dir, rel) if rel is not None else None

    def _maybe_resample(p, name):
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


# def build_or_load_kde_prior(context_name, cutoff_date, kde_catalog_path, data_dir,
#                              construction_params=None, kde_start='1990-01-01',
#                              force_rebuild=False, force_redownload=False):
#     """
#     Build (or load from cache) a KDE seismicity prior for a specific context.

#     Downloads a shared base catalog (kde_start → present) once, then filters
#     to events strictly before cutoff_date, fits a KDE, and caches the result
#     as kde_seismicity_{context_name}.tt3 in data_dir.

#     Parameters
#     ----------
#     context_name : str
#         Label for this context (e.g. 'benchmark', 'Ridgecrest').  Used to
#         name the per-context .tt3 cache file.
#     cutoff_date : str or datetime-like
#         Upper bound (exclusive) for catalog events.  Pass the first event time
#         of the benchmark/case-study so the prior only sees past seismicity.
#     kde_catalog_path : str
#         Path to the shared base parquet file.  Downloaded from USGS if absent.
#     data_dir : str
#         Directory where the .tt3 is written.
#     construction_params : dict, optional
#         Merged config dict with keys:
#           'bounds'                — (lon_min, lon_max, lat_min, lat_max)
#           'kde_seismicity_params' — grid_size, bw_method, min_mag, etc.
#           'out_of_bounds_fill'    — {prior_name: fill_value}
#           'target_resolution_deg' — {prior_name: float or None}
#     kde_start : str
#         Start date for base catalog download.
#     force_rebuild : bool
#         Rebuild the .tt3 even if a cached one already exists.
#     force_redownload : bool
#         Re-download the base parquet catalog even if it already exists.
#         Independent of force_rebuild — use this to refresh the catalog
#         without necessarily rebuilding every .tt3.

#     Returns
#     -------
#     str  — path to kde_seismicity_{context_name}.tt3
#     """
#     import pandas as pd
#     from benchmark.background import load_background_seismicity

#     if construction_params is None:
#         construction_params = {}

#     bounds     = construction_params.get('bounds', (-129, -112, 30, 51))
#     kde_params = construction_params.get('kde_seismicity_params', {})
#     oob_fills  = construction_params.get('out_of_bounds_fill', {})
#     target_res = construction_params.get('target_resolution_deg', {})

#     tt3_path = os.path.join(data_dir, f'kde_seismicity_{context_name}.tt3')
#     if not force_rebuild and os.path.exists(tt3_path):
#         print(f"KDE_Seismicity ({context_name}): loaded from cache "
#               f"{os.path.basename(tt3_path)}")
#         return tt3_path

#     os.makedirs(os.path.dirname(kde_catalog_path), exist_ok=True)

#     start_year  = pd.Timestamp(kde_start).year
#     end_year    = pd.Timestamp('today').year
#     kde_min_mag = kde_params.get('min_mag', 2.5)
#     print(f"KDE Min Mag: {kde_min_mag}")

#     catalog = load_background_seismicity(
#         cache_path    = kde_catalog_path,
#         bounds        = bounds,
#         start_year    = start_year,
#         end_year      = end_year,
#         min_mag       = kde_min_mag,
#         force_refresh = force_redownload,
#     )

#     # Filter to strictly before the context cutoff
#     cutoff_ts = pd.Timestamp(cutoff_date)
#     if cutoff_ts.tzinfo is None:
#         cutoff_ts = cutoff_ts.tz_localize('UTC')
#     else:
#         cutoff_ts = cutoff_ts.tz_convert('UTC')
#     catalog = catalog[catalog['time'] < cutoff_ts]

#     # Belt-and-suspenders: re-apply min_mag in case the parquet was previously
#     # cached at a lower threshold (e.g. an old 2.5 run, now min_mag=3).
#     catalog = catalog[catalog['mag'] >= kde_min_mag]

#     if len(catalog) == 0:
#         raise ValueError(
#             f"No events in KDE catalog before {cutoff_ts.date()} "
#             f"(context: '{context_name}')."
#         )

#     p = SeismicPrior.from_kde_seismicity(
#         catalog            = catalog,
#         bounds             = bounds,
#         grid_size          = kde_params.get('grid_size', 100),
#         bw_method          = kde_params.get('bw_method', 'scott'),
#         adaptive           = kde_params.get('adaptive', False),
#         adaptive_alpha     = kde_params.get('adaptive_alpha', 0.5),
#         lon_col            = kde_params.get('lon_col', 'longitude'),
#         lat_col            = kde_params.get('lat_col', 'latitude'),
#         out_of_bounds_fill = oob_fills.get('KDE_Seismicity', 0.0001),
#     )

#     res = target_res.get('KDE_Seismicity')
#     if res is not None:
#         p = p.resample(res)
#         print(f"  KDE_Seismicity ({context_name}): resampled to {res}° "
#               f"({len(p.lons)}×{len(p.lats)} cells)")

#     p.to_tt3(tt3_path)
#     print(f"KDE_Seismicity ({context_name}): {len(catalog):,} events → "
#           f"{os.path.basename(tt3_path)}")
#     return tt3_path

