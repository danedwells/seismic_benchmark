"""
plot_scripts/log_likelihood_vs_sigma_s.py — total log-likelihood vs sigma_s.

For a run with the sigma_s sweep (`_VARY_SIG = True` in
time_independent_scripts/case_studies.py, time_independent_scripts/run_benchmarks.py,
or run_run_benchmarks.sh --sigma-s "..."), computes, for each sigma_s value
tested, the sum of log(like_val_at_usgs) across all events at a fixed trigger
count — the total log-likelihood of the true locations under that sigma_s.
Maximizing this sum picks the sigma_s that best explains where events
actually occurred; it's the numerically stable stand-in for maximizing the
raw product of per-event likelihood values (which underflows to 0 almost
immediately).

6-panel grid: one panel per time-independent prior (including Uniform) plus
one panel for the dynamic ETAS prior. Each panel has one point per sigma_s
value tested (a handful of points, not a distribution), with the best value
starred. Panels/points with no data at the requested trigger count are
skipped gracefully.

Set ACTIVE_CASE_STUDY to None to use the main benchmark's sweep output
(results/output/time_independent/sig_*/max_trigs_{N}/), or to one of
'Ridgecrest', 'Ferndale', 'ElMayor' to use that case study's sweep output
(results/case_studies/{case_study}/output/time_independent/sig_*/max_trigs_{N}/).
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
from benchmark.plots import plot_log_likelihood_sum_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
ACTIVE_CASE_STUDY = None   # None (main benchmark), 'Ridgecrest', 'Ferndale', 'ElMayor'
MAX_TRIGS         = config.BENCHMARK_PARAMS['max_trigs']
PRIOR_ORDER       = list(config.PRIOR_FILENAMES.keys())  # includes 'Uniform'
N_TRIGS           = 5   # per-event trigger count to use; None = each event's last (most-triggered) row

# Column treated as a per-event probability; sum of its log is the total
# log-likelihood plotted against sigma_s. 'like_val_at_usgs' (raw normalized
# likelihood-surface value at the true location) is the natural default.
#COLUMN = 'like_val_at_usgs'
COLUMN = 'post_val_at_usgs'
#COLUMN = 'like_val_raw_at_usgs'

if ACTIVE_CASE_STUDY is None:
    CS_TI_DIR   = os.path.join(PROJECT_ROOT, 'results', 'output', 'time_independent')
    CS_TD_DIR   = os.path.join(PROJECT_ROOT, 'results', 'output', 'time_dependent')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'time_independent')
else:
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
# Figure: Σ log(COLUMN) vs sigma_s — one panel per prior, plus a 6th panel
# for the dynamic ETAS prior.
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
_run_label   = ACTIVE_CASE_STUDY if ACTIVE_CASE_STUDY is not None else 'main benchmark'

fig = plot_log_likelihood_sum_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    column      = COLUMN,
    title       = f'Total log sum of column \'{COLUMN}\' vs sigma_s — {_run_label}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'log_likelihood_vs_sigma_s_{_trigs_tag}_col_{COLUMN}.png'),
    extra_panel = extra_panel,
    n_trigs     = N_TRIGS,
    log_floor   = 1E-30,
)
plt.show()

# %%
