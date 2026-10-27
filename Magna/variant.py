# =============================================================================
# variant.py — environment switches shared by the Magna scripts
# =============================================================================
# Default (no env vars): the original setup — seismicity cut at -111.5, config bw_sq, native-resolution NSHM.
#   MAGNA_NOCUT=1     drop the -111.5 deg longitude seismicity cut (KDE + ETAS training catalogs). Everything that
#                     depends on it gets its own name (CONTEXT, results/output_nocut) so existing outputs are untouched.
#   MAGNA_BW_SQ=16    ETAS background-smoothing bw_sq (file/result names carry a _bw16 tag).
#   MAGNA_NSHM_RES=0.02   resample the NSHM prior to this resolution in degrees (default: native 0.1).
# Example: MAGNA_NOCUT=1 MAGNA_BW_SQ=16 MAGNA_NSHM_RES=0.02 python Magna/run_nshm_benchmarks.py
# =============================================================================
import os

BW_SQ    = float(os.environ['MAGNA_BW_SQ']) if os.environ.get('MAGNA_BW_SQ') else None
NOCUT    = os.environ.get('MAGNA_NOCUT') == '1'
NSHM_RES = float(os.environ['MAGNA_NSHM_RES']) if os.environ.get('MAGNA_NSHM_RES') else None

MAX_LON     = None if NOCUT else -111.5                      # drop background seismicity east of this (None = keep all)
CONTEXT     = 'Magna_nocut' if NOCUT else 'Magna'            # names the ETAS inversion files and the KDE prior
OUT_SUFFIX  = '_nocut' if NOCUT else ''                      # results/output{OUT_SUFFIX}
ETAS_SUFFIX = '' if BW_SQ is None else f'_bw{BW_SQ:g}'       # on ETAS result names
NSHM_TAG    = '' if NSHM_RES is None else f'_{NSHM_RES:g}deg'   # on NSHM prior/result names

# Isolated-event test set (download_isolated.py / run_isolated.py / location_hist_isolated.py):
#   MAGNA_ISO_MAG=2.5   minimum magnitude of the test events (default 3.0); non-default values get their own
#                       data/wasatch_isolated_M{mag}, results/output_isolated_M{mag} and figure name.
ISO_MIN_MAG = float(os.environ['MAGNA_ISO_MAG']) if os.environ.get('MAGNA_ISO_MAG') else 3.0
ISO_TAG     = '' if ISO_MIN_MAG == 3.0 else f'_M{ISO_MIN_MAG:g}'

# Prior adjustment used by run_isolated.py / location_hist_isolated.py (tempering exponent and floor as a fraction of the max):
#   MAGNA_PRIOR_ALPHA=1 MAGNA_FLOOR_FRAC=0.005   (defaults: 0.5 and 0.05, the settings of the original Magna runs)
# Non-default values add _a{alpha}_f{floor} to the isolated-event results folder and figure names.
PRIOR_ALPHA = float(os.environ['MAGNA_PRIOR_ALPHA']) if os.environ.get('MAGNA_PRIOR_ALPHA') else 0.5
FLOOR_FRAC  = float(os.environ['MAGNA_FLOOR_FRAC']) if os.environ.get('MAGNA_FLOOR_FRAC') else 0.05
PRIOR_TAG   = '' if (PRIOR_ALPHA, FLOOR_FRAC) == (0.5, 0.05) else f'_a{PRIOR_ALPHA:g}_f{FLOOR_FRAC:g}'

# Fixed floor on the ETAS free-background RATE field (same units as ETAS_UPDATER_CONFIG['out_of_bounds_fill'] = 1e-9), used by
# run_isolated.py instead of the peak-relative floor for the ETAS-family priors:  MAGNA_BG_FLOOR=1e-8
# The floor is applied to mu(x) before the triggered terms are added (lambda = max(mu, floor) + triggered), with no tempering, so it
# does not depend on the recent-event peak. KDE and static NSHM keep MAGNA_PRIOR_ALPHA / MAGNA_FLOOR_FRAC. Adds _bgfloor{value} to names.
BG_FLOOR = float(os.environ['MAGNA_BG_FLOOR']) if os.environ.get('MAGNA_BG_FLOOR') else None
if BG_FLOOR is not None:
    PRIOR_TAG += f'_bgfloor{BG_FLOOR:g}'
