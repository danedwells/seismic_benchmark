import os

# Filenames for cached prior .tt3 files, relative to SeismicPrior.data_dir.
# Set to None to disable a prior (uniform weighting will be used instead).
# Update ETAS_FILENAME whenever a new ETAS prior is generated.

# Parameters passed to SeismicPrior factory constructors when building .tt3 files.
# 'bounds' is (lon_min, lon_max, lat_min, lat_max).
# from_smooth_seismicity and from_etas are excluded: the former is pre-built,
# the latter is constructed externally from ETAS output.

PRIOR_CONSTRUCTION_PARAMS = {
    'bounds': (-129, -112, 30, 45),
    'out_of_bounds_fill': {
        'Gear1':             'mean',   # global model; offshore cells have low but real rates
        'NSHM':              5000000.,   # land-only source; offshore needs a background value
        'Helmstetter':       0.00001,   # CSEP testing region; offshore needs a background value
        'Smooth_seismicity': 0.0001,   # US/Canada file; may not extend to all offshore areas
        'ETAS':              0.0001,   # polygon-masked; offshore outside polygon needs fill
    },
    # Optional resampling to a common resolution before caching.
    # Set to None to keep each prior's native resolution.
    #
    # Experiment A — downsample smooth seismicity to match others (~0.1°):
    #   'Smooth_seismicity': 0.1,  all others: None
    #
    # Smooth_seismicity source is ~0.02°; GEAR1/NSHM/Helmstetter/ETAS are ~0.1°.
    'target_resolution_deg': {
        'Gear1':             None,
        'NSHM':              None,
        'Helmstetter':       None,
        'Smooth_seismicity': 0.1,
        'ETAS':              None,
    },

    # Experiment B — all at ~0.02° (coarse priors upsampled):
    # 'target_resolution_deg': {
    #     'Gear1':             0.02,
    #     'NSHM':              0.02,
    #     'Helmstetter':       0.02,
    #     'Smooth_seismicity': None,
    #     'ETAS':              0.02,
    # },
    # Paths to source data files, relative to SeismicPrior.data_dir.
    # Helmstetter is omitted — its source data comes from pycsep at runtime.
    'source_paths': {
        'Gear1':             os.path.join('GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'),
        'NSHM':              os.path.join('USGS_NSHM_data', 'gridded_moment_rates.xyz'),
        'NSHM_fault':        os.path.join('USGS_NSHM_data', 'fault_moment_rates.xyz'),
        'Smooth_seismicity': os.path.join('smooth_seismicity_data', 'prior_seis_grid_US_Canada.tt3'),
        'ETAS':              os.path.join('ETAS_data', 'etas_prior_20080101_000000.tt3'),
    },
}

# Cached .tt3 filenames written into SeismicPrior.data_dir.
# Smooth_seismicity and ETAS are filled/expanded copies of the source files.
PRIOR_FILENAMES = {
    'Gear1':             'GEAR1_prior.tt3',
    'NSHM':              'USGS_NSHM_prior.tt3',
    'Helmstetter':       'helmstetter_prior.tt3',
    'Smooth_seismicity': 'prior_seis_grid_US_Canada_filled.tt3',
    'ETAS':              'etas_prior_20080101_000000_filled.tt3',
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

# Parameters for the main benchmark run.
BENCHMARK_PARAMS = {
    'prior':                     'Smooth_seismicity',
    'max_trigs':                 6,
    'grid_size':                 100,
    'grid_km':                   200,
    'migrate_grid':              False,  # re-centre grid on posterior MAP between versions
    'migrate_grid_min_triggers': 4,     # suppress migration until this many triggers have reported
}
