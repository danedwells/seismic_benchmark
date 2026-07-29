"""
plot_scripts/plot_qq_calibration_sigma_s.py — Q-Q calibration vs sigma_s.

For a case study run with the sigma_s sweep (`_VARY_SIG = True` in
time_independent_scripts/case_studies.py), plots posterior_confidence_level
Q-Q calibration curves in a grid with one panel per prior (including
Uniform) and one line per sigma_s value tested.

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
from benchmark.plots import plot_qq_calibration_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
ACTIVE_CASE_STUDY = 'Ridgecrest'   # 'Ridgecrest', 'Ferndale', 'ElMayor'
MAX_TRIGS         = config.BENCHMARK_PARAMS['max_trigs']
PRIOR_ORDER       = list(config.PRIOR_FILENAMES.keys())  # includes 'Uniform'
N_TRIGS           = 5   # per-event trigger count to plot; None = each event's last (most-triggered) row

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
# Figure: Q-Q calibration grid — one panel per prior, one line per sigma_s,
# plus a 6th panel for the dynamic ETAS prior.
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

fig = plot_qq_calibration_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'Posterior calibration vs sigma_s — {ACTIVE_CASE_STUDY}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'qq_calibration_vs_sigma_s_{_trigs_tag}.png'),
    extra_panel = extra_panel,
    n_trigs     = N_TRIGS,
)
plt.show()

# %%

COLUMN_NAME = 'post_val_at_usgs'
fig = plot_qq_calibration_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'{COLUMN_NAME} calibration vs sigma_s — {ACTIVE_CASE_STUDY}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'{COLUMN_NAME}_qq_calibration_vs_sigma_s_{_trigs_tag}.png'),
    extra_panel = extra_panel,
    y_column    = COLUMN_NAME,
    n_trigs     = N_TRIGS,
)
plt.show()

#%%