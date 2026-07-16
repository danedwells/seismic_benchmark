"""
plot_scripts/plot_qq_calibration_vs_map_err_km.py — Q-Q calibration vs
map_err_km, swept over sigma_s.

Functionally identical to plot_qq_calibration_sigma_s.py, except the x-axis
is not the theoretical Uniform(0,1) quantile — it's the empirical quantile
function of map_err_km. This compares the *shape* of the
posterior_confidence_level distribution against the shape of the location
error distribution (marginally, not paired event-by-event) for each prior
and sigma_s value.

Requires results/case_studies/{case_study}/output/time_independent/sig_*/
max_trigs_{N}/ directories, produced by the sigma_s sweep, and (optionally)
the equivalent time_dependent/sig_*/ directories for the dynamic ETAS panel.
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
from benchmark.plots import plot_qq_calibration_by_param

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
ACTIVE_CASE_STUDY = 'Ridgecrest'   # 'Ridgecrest', 'Ferndale', 'ElMayor'
MAX_TRIGS         = config.BENCHMARK_PARAMS['max_trigs']
PRIOR_ORDER       = list(config.PRIOR_FILENAMES.keys())  # includes 'Uniform'

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

#%%
# ---------------------------------------------------------------------------
# Discover sigma_s sweep directories: sig_{value}/max_trigs_{MAX_TRIGS}/
# ---------------------------------------------------------------------------
output_dirs = _discover_sigma_s_dirs(CS_TI_DIR)
print(f'Found sigma_s values (time-independent priors): {list(output_dirs.keys())}')

etas_output_dirs = _discover_sigma_s_dirs(CS_TD_DIR)
print(f'Found sigma_s values (dynamic ETAS): {list(etas_output_dirs.keys())}')

#%%
# ---------------------------------------------------------------------------
# Figure: posterior_confidence_level quantiles (y) vs map_err_km quantiles
# (x) — one panel per prior, one line per sigma_s, plus a 6th panel for the
# dynamic ETAS prior.
# ---------------------------------------------------------------------------
extra_panel = None
if etas_output_dirs:
    extra_panel = {
        'name':         'ETAS (dynamic)',
        'output_dirs':  etas_output_dirs,
        'csv_filename': 'etas_dynamic_benchmark_results.csv',
    }

fig = plot_qq_calibration_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'Posterior calibration vs map_err_km — {ACTIVE_CASE_STUDY}  ({MAX_TRIGS} triggers)',
    save_path   = os.path.join(FIGURES_DIR, f'qq_calibration_vs_map_err_km_{MAX_TRIGS}trigs.png'),
    extra_panel = extra_panel,
    x_column    = 'map_err_km',
    x_label     = 'map_err_km quantile',
    log_x       = True,
)
plt.show()

# %%
