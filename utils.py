import os
from priors.prior_model import SeismicPrior


def build_and_cache_priors(cache_paths, data_dir):
    """
    Build and cache all prior .tt3 files.

    Attempts to construct each prior from its source data and write it to the
    corresponding .tt3 path in cache_paths. Each prior is tried independently
    so a failure in one does not prevent the others from being built.

    ETAS is always skipped here — it requires external ETAS output and must
    be built manually:
        p = SeismicPrior.from_etas(lats, lons, lambda_grid, forecast_time=t)
        p.to_tt3(cache_paths['ETAS'])

    Parameters
    ----------
    cache_paths : dict
        Mapping of prior name -> .tt3 file path (or None for Uniform).
        Expected keys: 'Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform'.
    data_dir : str
        Path to the priors data directory (SeismicPrior.data_dir).
    """
    try:
        p = SeismicPrior.from_gear1(os.path.join(data_dir, 'GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'))
        p.to_tt3(cache_paths['Gear1'])
        print("Gear1: built and cached.")
    except Exception as e:
        print(f"Gear1: failed — {e}")

    try:
        p = SeismicPrior.from_nshm(os.path.join(data_dir, 'USGS_NSHM_data', 'gridded_moment_rates.xyz'))
        p.to_tt3(cache_paths['NSHM'])
        print("NSHM: built and cached.")
    except Exception as e:
        print(f"NSHM: failed — {e}")

    try:
        p = SeismicPrior.from_helmstetter()
        p.to_tt3(cache_paths['Helmstetter'])
        print("Helmstetter: built and cached.")
    except Exception as e:
        print(f"Helmstetter: failed — {e}")

    try:
        SeismicPrior.from_smooth_seismicity()  # validates the file loads cleanly
        print("Smooth_seismicity: ready (pre-built).")
    except Exception as e:
        print(f"Smooth_seismicity: failed — {e}")

    print("ETAS: skipped (requires external ETAS output — build manually).")
