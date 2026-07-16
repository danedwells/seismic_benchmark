import os

# Parameters passed to SeismicPrior factory constructors when building .tt3 files.
# 'bounds' is (lon_min, lon_max, lat_min, lat_max).

PRIOR_CONSTRUCTION_PARAMS = {
    'bounds': (-129, -112, 30, 51), # Include Washingotn and Oregon
    'out_of_bounds_fill': {
        'Gear1':          'mean',   # global model; offshore cells have low but real rates
        'NSHM':           5000000., # land-only source; offshore needs a background value
        'Helmstetter':    0.00001,  # CSEP testing region; offshore needs a background value
        'KDE_Seismicity': 0.0001,  # KDE tails may not fully cover offshore areas
    },
    # Optional resampling to a common resolution before caching.
    # Set to None to keep each prior's native resolution.
    # All priors upsampled to ~0.02°:
    'target_resolution_deg': {
        'Gear1':          0.02,
        'NSHM':           0.02,
        'Helmstetter':    0.02,
        'KDE_Seismicity': None,  # resolution set by grid_size at build time
    },
    # Paths to source data files, relative to SeismicPrior.data_dir.
    # Helmstetter is omitted — its source data comes from pycsep at runtime.
    'source_paths': {
        'Gear1':     os.path.join('GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'),
        'NSHM':      os.path.join('USGS_NSHM_data', 'gridded_moment_rates.xyz'),
        'NSHM_fault': os.path.join('USGS_NSHM_data', 'fault_moment_rates.xyz'),
    },
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

# ---------------------------------------------------------------------------
# KDE seismicity prior configuration
# ---------------------------------------------------------------------------
# catalog_path  — parquet/CSV of historical seismicity (latitude, longitude cols).
#                 Defaults to the benchmark background seismicity cache; override
#                 to use a different catalog or date range.
# lon_col/lat_col — column names in that file.
# grid_size     — (nx, ny) or scalar; number of grid points per axis.
# bw_method     — bandwidth selector passed to scipy.stats.gaussian_kde.
# min_mag       — optional magnitude filter applied before fitting the KDE.

KDE_SEISMICITY_PARAMS = {
    'catalog_path':   None,   # filled in at build time from the benchmark data dir
    'lon_col':        'longitude',
    'lat_col':        'latitude',
    'grid_size':      100,
    'bw_method':      0.4, #'scott',
    'min_mag':        3.0,
    'adaptive':       True,  # set True to use adaptive (variable-bandwidth) KDE
    'adaptive_alpha': 0.5,    # Silverman sensitivity: 0=fixed, 0.5=standard, 1=max
}

# Start date for the shared KDE base catalog download.
# Events from this date to the day before each context's first event are used.
KDE_START_DATE = '1990-01-01'

# ---------------------------------------------------------------------------
# ETAS inversion configuration
# ---------------------------------------------------------------------------
# Parameters passed to ETASParameterCalculation when building the time-dependent
# prior from scratch.  Edit these before running
# time_dependent_scripts/build_initial_prior.py.
#
# shape_coords: polygon boundary in [lat, lon] pairs (the etas_2 convention).
#   This is the California/Pacific-NW region used for the benchmark catalog.
#   To use a .npy file instead, set shape_coords to its absolute path (str).
#
# theta_0: initial parameter guess — does not affect final values but a
#   reasonable guess speeds convergence.  Values here are from the
#   etas_2 California example inversion.
#
# mc: magnitude of completeness — catalog must be complete above this value.
#   3.6 for the California example catalog; adjust if using a different catalog.
#
# auxiliary_start / timewindow_start / timewindow_end:
#   auxiliary events act as sources only (not targets); primary window events
#   are both sources and targets.  timewindow_end is the forecast origin time
#   for the initial prior.
#
# id: labels the output files (parameters_{id}.json, pij_{id}.csv, …).
#   Change if you want to keep multiple inversion results side-by-side.

ETAS_INVERSION_CONFIG = {
    # -- Catalog time windows --
    # auxiliary_start and timewindow_start are fixed across all contexts.
    # timewindow_end is overridden per context in build_initial_prior.py
    # (set to the cutoff date for each sequence).
    # id is also overridden per context (e.g. 'benchmark', 'Ridgecrest', …).
    'auxiliary_start':  '1971-01-01 00:00:00',
    'timewindow_start': '1981-01-01 00:00:00',
    'timewindow_end':   None,   # set per-context in build_initial_prior.py

    # -- Magnitude completeness --
    'mc':      3.0, # default value
    'delta_m': 0.1,

    # -- Spatial region (California + PNW benchmark polygon, [lat, lon] pairs) --

    'shape_coords': [
    [43.5, -127.7], [43.5, -117.5], [39.7, -117.5], [36.1, -112.6],
    [34.6, -111.6], [34.3, -111.6], [32.7, -112.1], [31.8, -112.2],
    [31.2, -113.5], [31.0, -117.1], [31.1, -117.4], [31.5, -118.3],
    [32.4, -118.8], [33.3, -122.3], [34.0, -124.0], [37.5, -126.3],
    [40.0, -127.9], [40.5, -127.9], [43.0, -127.7], [43.5, -127.7],
    ],

    # -- Model settings --
    'coppersmith_multiplier': 100,
    'free_background':        True,
    'bw_sq':                  4,

    # -- Initial parameter guess --
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

    # -- Output label (output files will be parameters_{id}.json, etc.) --
    'id': 'benchmark',
}

# Parameters for the EtasPriorUpdater built from the inversion output above.
# These are passed to EtasPriorUpdater.from_inversion_json() at runtime.
ETAS_UPDATER_CONFIG = {
    'bounds':           PRIOR_CONSTRUCTION_PARAMS['bounds'],
    'grid_spacing':     0.1,
    'out_of_bounds_fill': 0.0000001,  # fill for cells outside the ETAS polygon
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
    'sigma_s':                   0.1,     # estimated travel time uncertainty per pick
    'edt_sigma_s':               0.02,   # estimated travel time uncertainty per pick for dff. travle time
    'dtt_weight':                0.0,   # How much to weight the differential travel time (0 = none, 1 = all)
}

# ---------------------------------------------------------------------------
# Case study definitions — single authoritative source
# ---------------------------------------------------------------------------
# Shared by preparation_scripts/case_study_preparation.py and all case_studies.py
# workflow scripts.  Add new sequences here; they become available everywhere.
CASE_STUDIES = {
    'Ridgecrest': {
        'name':      'Ridgecrest 2019',
        'starttime': '2019-07-04T17:00:00',
        'endtime':   '2019-08-07T00:00:00',
        'bounds':    (-118.5, -116.5, 35.0, 36.5),
        'min_mag':   3.0,
    },
    'Ferndale': {
        'name':      'Ferndale 2022',
        'starttime': '2022-12-20T10:00:00',
        'endtime':   '2023-01-20T00:00:00',
        'bounds':    (-127.0, -122.5, 39, 41.0),
        'min_mag':   3.0,
    },
    'ElMayor': {
        'name':      'El Mayor-Cucapah 2010',
        'starttime': '2010-04-04T22:00:00',
        'endtime':   '2010-05-04T00:00:00',
        'bounds':    (-117.0, -114.5, 31.5, 33.5),
        'min_mag':   3.0,
    },
}

# Representative aftershocks used for single-event posterior / trajectory figures.
# Set _MS_ = True in any case_studies.py to use FOCUS_EVENTS_MAINSHOCK instead.
FOCUS_EVENTS = {
    'Ridgecrest': 'ci38548295',  # M 4.9 aftershock
    'Ferndale':   'nc73831091',  # M 4.05 aftershock
    'ElMayor':    'ci10148002',  # M 5.2 aftershock
}

FOCUS_EVENTS_MAINSHOCK = {
    'Ridgecrest': 'ci38457511',  # M7.1 mainshock  2019-07-06
    'Ferndale':   'nc73821036',  # M6.4 mainshock  2022-12-20
    'ElMayor':    'ci14607652',  # M7.2 mainshock
}
