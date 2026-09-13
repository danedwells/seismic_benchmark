"""
benchmark/config_cascadia.py

Region-scoped config for the Cascadia / Pacific Northwest ETAS work.

This mirrors the shape of benchmark/config.py but is kept as a fully
separate module rather than folded into it: config.py stays the
single-source-of-truth for the California/main-benchmark catalog and its
CASE_STUDIES, and nothing here changes that. Cascadia needs its own ETAS
spatial polygon (config.py's ETAS_INVERSION_CONFIG['shape_coords'] caps at
~44N and doesn't reach Washington) and its own background/reference
catalog, so those live here instead of overloading config.py's globals.

Purely mechanical, geography-agnostic helpers (etas_run_tag, etas_run_label,
etas_output_id, etas_catalog_tag) are imported from config.py and reused
as-is -- they're pure functions of whatever cfg dict is passed in, so they
work identically for this module's ETAS_INVERSION_CONFIG.

Static priors (Gear1/NSHM/Helmstetter/Smooth_seismicity/Uniform) are NOT
duplicated here: config.PRIOR_CONSTRUCTION_PARAMS['bounds'] already spans
to 51N, so those five priors already cover Cascadia and are shared via
SeismicPrior.data_dir -- see preparation_scripts/build_priors.py.
"""


# ---------------------------------------------------------------------------
# Region-scale background/reference catalog
# ---------------------------------------------------------------------------
# The Cascadia equivalent of the main benchmark's data/reference/ +
# data/etas_inversion/input/ catalogs: a long-baseline, wide-area background
# catalog used to build the region's KDE_Seismicity prior and as the ETAS
# inversion's auxiliary catalog. NOT a case study (no .run files/bEPIC
# trigger benchmark) -- that's what CASE_STUDIES below is for.
#
# Matches what's actually on disk at data/cascadia/reference/
# cascadia_reference_catalog.csv, downloaded via
# examples/download_usgs_catalog.py (which mirrors these same values).
REFERENCE_CATALOG_CONFIG = {
    'starttime': '1980-01-01T00:00:00',
    'endtime':   '2026-07-31T00:00:00',
    'bounds':    (-131.0, -115.0, 40.5, 50.5),  # lon_min, lon_max, lat_min, lat_max
    'min_mag':   2.5,
}

# ---------------------------------------------------------------------------
# KDE seismicity prior configuration (Cascadia catalog instead of the main
# benchmark's data/reference/ catalog)
# ---------------------------------------------------------------------------
KDE_SEISMICITY_PARAMS = {
    'catalog_path':   None,   # filled in at build time from data/cascadia/reference/
    'lon_col':        'longitude',
    'lat_col':        'latitude',
    'grid_size':      100,
    'bw_method':      0.4,
    'min_mag':        3.0,
    'adaptive':       True,
    'adaptive_alpha': 0.5,
}
# Cached .tt3 filenames written into SeismicPrior.data_dir.
# KDE_Seismicity filename varies per context; set explicitly in each script.
PRIOR_FILENAMES = {
    'Gear1':          'GEAR1_prior.tt3',
    'NSHM':           'USGS_NSHM_prior.tt3',
    'Helmstetter':    'helmstetter_prior.tt3',
    'KDE_Seismicity': None,   # set per-script: kde_seismicity_{context}.tt3
    'Uniform':        None,
}
# Parameters for the main benchmark run.
BENCHMARK_PARAMS = {
    'prior':                     'KDE_Seismicity',
    'max_trigs':                 10,
    'grid_size':                 100,
    'grid_km':                   200,
    'migrate_grid':              False,  # re-centre grid on posterior MAP between versions
    'migrate_grid_min_triggers': 8,     # suppress migration until this many triggers have reported
    'activity_threshold':        0.40,  # operational EPIC value; pass station_inventory=None to disable
    'station_inventory':         None,
    'resample_distant_events':   False,  # re-run with random trigger subset when nearest station > 200 km
    'sigma_s':                   0.35,     # estimated travel time uncertainty per pick
    'edt_sigma_s':               0.02,   # estimated travel time uncertainty per pick for dff. travle time
    'dtt_weight':                0.0,   # How much to weight the differential travel time (0 = none, 1 = all)
    'search_depths':             [8.0],  # km; candidate source depths for bEPIC's grid search.
                                          # [8.0] preserves the original fixed-depth behaviour.
}

# ---------------------------------------------------------------------------
# ETAS inversion configuration
# ---------------------------------------------------------------------------
# Same shape as config.ETAS_INVERSION_CONFIG, but with a wider spatial
# polygon reaching up through Washington. mc/m_ref are placeholders pending
# a completeness check against the Cascadia catalog -- TODO before treating
# an inversion built from this as final.
ETAS_INVERSION_CONFIG = {
    'auxiliary_start':  '1981-01-01 00:00:00',
    'timewindow_start': '1990-01-01 00:00:00',
    'timewindow_end':   None,   # set per-context in build_initial_prior_cascadia.py

    'mc': 3.0,  # TODO: verify magnitude of completeness for the Cascadia catalog
    'delta_m': 0.1,
    'm_ref': 3.0,

    # -- Spatial region (Pacific Northwest polygon, [lat, lon] pairs) --
    # Wider than config.py's CA polygon -- reaches to ~51N to cover
    # Washington/the full Cascadia subduction zone.
    'shape_coords': [
        [51.0, -129.0], [51.0, -122.0], [46.0, -117.0], [42.0, -117.0],
        [40.5, -122.0], [40.5, -125.5], [43.0, -129.0], [51.0, -129.0],
    ],

    'coppersmith_multiplier': 100,
    'bw_sq':                  4,
    'free_background':        True,
    'free_productivity':      False,

    # -- Initial parameter guess (copied from config.py's CA guess as a
    #    starting point; not refit for this region yet) --
    'theta_0': {
        'log10_mu': -5.8,
        'log10_k0': -2.6,
        'a':         1.8,
        'log10_c':  -2.5,
        'omega':    -0.02,
        'log10_tau': 3.5,
        'log10_d':  -0.85,
        'gamma':     1.3,
        'rho':       0.66,
    },

    'id': 'cascadia',
}

# ---------------------------------------------------------------------------
# EtasPriorUpdater runtime config
# ---------------------------------------------------------------------------
ETAS_UPDATER_CONFIG = {
    'bounds':           REFERENCE_CATALOG_CONFIG['bounds'],
    'grid_spacing':     0.05,
    'out_of_bounds_fill': 1E-9,
    'use_spatial_background':   True,
    'use_spatial_productivity': False,
    'max_lookback_days': 365,
}

# ---------------------------------------------------------------------------
# Benchmark catalog (Cascadia's equivalent of the main CA benchmark's
# ~700-event catalog / data/run_files/, consumed by run_benchmarks.py-style
# scripts) -- NOT a single-sequence case study. Built by
# preparation_scripts/cascadia_preparation.py, which reuses
# download_case_study_catalog()/build_run_files_for_case_study() from
# benchmark/usgs.py -- those functions only need a plain
# name/starttime/endtime/bounds/min_mag dict, so they work here unchanged
# despite the "case_study" naming.
# ---------------------------------------------------------------------------
BENCHMARK_CATALOG_CONFIG = {
    'name':      'Cascadia Benchmark',
    'starttime': '2022-01-01T00:00:00',
    'endtime':   '2026-07-31T00:00:00',
    'bounds':    REFERENCE_CATALOG_CONFIG['bounds'],
    'min_mag':   3.0,
}

# ---------------------------------------------------------------------------
# Case studies within the Cascadia region (single aftershock sequences, in
# the same sense as config.CASE_STUDIES -- none yet)
# ---------------------------------------------------------------------------
CASE_STUDIES = {}

FOCUS_EVENTS = {}

FOCUS_EVENTS_MAINSHOCK = {}
