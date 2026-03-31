# Filenames for cached prior .tt3 files, relative to SeismicPrior.data_dir.
# Set to None to disable a prior (uniform weighting will be used instead).
# Update ETAS_FILENAME whenever a new ETAS prior is generated.

PRIOR_FILENAMES = {
    'Gear1':             'GEAR1_prior.tt3',
    'NSHM':              'USGS_NSHM_prior.tt3',
    'Helmstetter':       'helmstetter_prior.tt3',
    'Smooth_seismicity': 'prior_seis_grid_US_Canada.tt3',
    'ETAS':              'etas_prior_20080101_000000.tt3',
    'Uniform':           None,
}
