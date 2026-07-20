"""
plot_scripts/error_scatter_calibration_sigma_s.py — scatter of
posterior_confidence_level quantile vs map_err_km, swept over sigma_s.

For a case study run with the sigma_s sweep (`_VARY_SIG = True` in
time_independent_scripts/case_studies.py), plots one point per event: its
posterior_confidence_level (by default, converted to its own empirical
quantile) against its map_err_km (by default, left as a raw km value).
Unlike the Q-Q plots in plot_qq_calibration_sigma_s.py /
plot_qq_calibration_vs_map_err_km.py — which compare the two columns as
independent marginal distributions — this pairs them event-by-event, giving
a direct visual read on how well confidence tracks actual location error.

6-panel grid: one panel per time-independent prior (including Uniform) plus
one panel for the dynamic ETAS prior; one colour per sigma_s value tested.

Requires results/case_studies/{case_study}/output/time_independent/sig_*/
max_trigs_{N}/ directories, produced by that sweep.
"""
#%%
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt

from benchmark import config
from benchmark.plots import plot_scatter_calibration_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
ACTIVE_CASE_STUDY = 'Ferndale'   # 'Ridgecrest', 'Ferndale', 'ElMayor'
MAX_TRIGS         = config.BENCHMARK_PARAMS['max_trigs']
PRIOR_ORDER       = list(config.PRIOR_FILENAMES.keys())  # includes 'Uniform'
N_TRIGS           = 5   # per-event trigger count to plot; None = each event's last (most-triggered) row
SIGMA_S           = None   # single sigma_s value to plot (e.g. 0.3); None = all discovered values

# Metrics to compare — any column in the benchmark results CSVs works, e.g.
# 'best_misfit', 'usgs_credible_level', 'coverage_100km'. Each axis is
# independently switchable between its raw value and its own empirical
# quantile via the *_QUANTILE flags below.
Y_COLUMN   = 'posterior_confidence_level'
X_COLUMN   = 'map_err_km'
Y_QUANTILE = True    # plot the quantile of Y_COLUMN (default: True)
X_QUANTILE = False   # plot the raw value of X_COLUMN (default: False)
LOG_X      = True
LOG_Y      = True

CS_TI_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                            ACTIVE_CASE_STUDY, 'output', 'time_independent')
CS_TD_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                            ACTIVE_CASE_STUDY, 'output', 'time_dependent')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                            ACTIVE_CASE_STUDY, 'figures', 'time_independent')
os.makedirs(FIGURES_DIR, exist_ok=True)

sig_dir_re = re.compile(r'^sig_(\d+(?:\.\d+)?)$')


def _discover_sigma_s_dirs(base_dir):
    """Map sigma_s value -> {base_dir}/sig_{value}/max_trigs_{MAX_TRIGS}/."""
    found = {}
    for entry in sorted(os.listdir(base_dir)):
        m = sig_dir_re.match(entry)
        if not m:
            continue
        trig_dir = os.path.join(base_dir, entry, f'max_trigs_{MAX_TRIGS}')
        if os.path.isdir(trig_dir):
            found[float(m.group(1))] = trig_dir
    return dict(sorted(found.items()))


def _filter_sigma_s(dirs, sigma_s, strict=True):
    """Restrict a {sigma_s: dir} map to a single value, or pass it through if None."""
    if sigma_s is None:
        return dirs
    if sigma_s in dirs:
        return {sigma_s: dirs[sigma_s]}
    if strict:
        raise ValueError(f'sigma_s={sigma_s} not found; available: {list(dirs.keys())}')
    print(f'  sigma_s={sigma_s} not found among ETAS dirs (available: {list(dirs.keys())}); skipping ETAS panel.')
    return {}


# ---------------------------------------------------------------------------
# Discover sigma_s sweep directories: sig_{value}/max_trigs_{MAX_TRIGS}/
# ---------------------------------------------------------------------------
output_dirs = _filter_sigma_s(_discover_sigma_s_dirs(CS_TI_DIR), SIGMA_S)
print(f'Found sigma_s values (time-independent priors): {list(output_dirs.keys())}')

etas_output_dirs = _filter_sigma_s(_discover_sigma_s_dirs(CS_TD_DIR), SIGMA_S, strict=False)
print(f'Found sigma_s values (dynamic ETAS): {list(etas_output_dirs.keys())}')

#%%
# ---------------------------------------------------------------------------
# Figure: Y_COLUMN vs X_COLUMN scatter — one panel per prior, one colour per
# sigma_s, plus a 6th panel for the dynamic ETAS prior.
# ---------------------------------------------------------------------------
extra_panel = None
if etas_output_dirs:
    extra_panel = {
        'name':         'ETAS (dynamic)',
        'output_dirs':  etas_output_dirs,
        'csv_filename': 'etas_dynamic_benchmark_results.csv',
    }

_trigs_label = f'{N_TRIGS} triggers' if N_TRIGS is not None else 'final version'
_trigs_tag   = f'{N_TRIGS}trigs' if N_TRIGS is not None else 'final'
_sig_tag     = f'_sig{SIGMA_S}' if SIGMA_S is not None else ''

fig = plot_scatter_calibration_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    y_column    = Y_COLUMN,
    x_column    = X_COLUMN,
    y_quantile  = Y_QUANTILE,
    x_quantile  = X_QUANTILE,
    title       = f'{{y_axis}} vs {{x_axis}} — {ACTIVE_CASE_STUDY}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'error_scatter_calibration_vs_sigma_s_{_trigs_tag}{_sig_tag}.png'),
    extra_panel = extra_panel,
    log_x       = LOG_X,
    log_y       = False,
    n_trigs     = N_TRIGS,
)
plt.show()

# %%
