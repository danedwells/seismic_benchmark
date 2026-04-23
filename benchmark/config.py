import os

# Parameters passed to SeismicPrior factory constructors when building .tt3 files.
# 'bounds' is (lon_min, lon_max, lat_min, lat_max).
# from_smooth_seismicity is excluded here — it is pre-built and just needs fill/expand.

PRIOR_CONSTRUCTION_PARAMS = {
    'bounds': (-129, -112, 30, 51), # Include Washingotn and Oregon
    'out_of_bounds_fill': {
        'Gear1':             'mean',   # global model; offshore cells have low but real rates
        'NSHM':              5000000.,   # land-only source; offshore needs a background value
        'Helmstetter':       0.00001,   # CSEP testing region; offshore needs a background value
        'Smooth_seismicity': 0.0001,   # US/Canada file; may not extend to all offshore areas
    },
    # Optional resampling to a common resolution before caching.
    # Set to None to keep each prior's native resolution.
    #
    # Experiment A — downsample smooth seismicity to match others (~0.1°):
    #   'Smooth_seismicity': 0.1,  all others: None
    #
    # Smooth_seismicity source is ~0.02°; GEAR1/NSHM/Helmstetter/ETAS are ~0.1°.
    # 'target_resolution_deg': {
    #     'Gear1':             None,
    #     'NSHM':              None,
    #     'Helmstetter':       None,
    #     'Smooth_seismicity': 0.1,
    # },

    #Experiment B — all at ~0.02° (coarse priors upsampled):
    'target_resolution_deg': {
        'Gear1':             0.02,
        'NSHM':              0.02,
        'Helmstetter':       0.02,
        'Smooth_seismicity': None,
    },
    # Paths to source data files, relative to SeismicPrior.data_dir.
    # Helmstetter is omitted — its source data comes from pycsep at runtime.
    'source_paths': {
        'Gear1':             os.path.join('GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'),
        'NSHM':              os.path.join('USGS_NSHM_data', 'gridded_moment_rates.xyz'),
        'NSHM_fault':        os.path.join('USGS_NSHM_data', 'fault_moment_rates.xyz'),
        'Smooth_seismicity': os.path.join('smooth_seismicity_data', 'prior_seis_grid_US_Canada.tt3'),
    },
}

# Cached .tt3 filenames written into SeismicPrior.data_dir.
# Smooth_seismicity is a filled/expanded copy of the source file.
PRIOR_FILENAMES = {
    'Gear1':             'GEAR1_prior.tt3',
    'NSHM':              'USGS_NSHM_prior.tt3',
    'Helmstetter':       'helmstetter_prior.tt3',
    'Smooth_seismicity': 'prior_seis_grid_US_Canada_filled.tt3',
    'Uniform':           None,
}

# Parameters for the reference location run.
# NOTE - deprecated in the sense that for reference locations,
# We are NOT currently running htis algorithm. Instead, we are using ANSS/COMCAT
# Locations.
# This is independent of the main benchmarking workflow — change these
# without affecting benchmark runs, and vice versa.
# 'prior' must match a key in PRIOR_FILENAMES.
REFERENCE_PARAMS = {
    'prior':     'Smooth_seismicity',
    'max_trigs': 100,
    'grid_size': 100,
    'grid_km':   200,
}

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
    'auxiliary_start':  '1971-01-01 00:00:00', # These are default values for the example catalog
    'timewindow_start': '1981-01-01 00:00:00',
    'timewindow_end':   '2007-01-01 00:00:00',

    # -- Magnitude completeness --
    'mc':      3.6, # default value
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
    'prior':                     'Smooth_seismicity',
    'max_trigs':                 15,
    'grid_size':                 100,
    'grid_km':                   200,
    'migrate_grid':              False,  # re-centre grid on posterior MAP between versions
    'migrate_grid_min_triggers': 4,     # suppress migration until this many triggers have reported
}
