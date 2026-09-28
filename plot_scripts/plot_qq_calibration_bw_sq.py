"""
plot_scripts/plot_qq_calibration_bw_sq.py — Q-Q calibration vs bw_sq.

For a bw_sq sweep produced by data_examination_scripts/bw_sq_calibration.py
(results/california/output/time_dependent/max_trigs_{N}/
etas_dynamic_benchmark_results_bw{value}.csv), plots
posterior_confidence_level Q-Q calibration curves — one line per bw_sq
value tested.

Single-panel plot: bw_sq is baked into the ETAS inversion itself, so only
the dynamic ETAS prior is relevant here (there's no Uniform/other-prior
variant to compare against, unlike the sigma_s sweep).

California 'benchmark' context only — matches bw_sq_calibration.py's own
scope.
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

from benchmark import config_california as config
from benchmark.plots import plot_qq_calibration_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
MAX_TRIGS = config.BENCHMARK_PARAMS['max_trigs']
N_TRIGS   = 5   # per-event trigger count to plot; None = each event's last (most-triggered) row

TD_DIR      = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent', f'max_trigs_{MAX_TRIGS}')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}')
os.makedirs(FIGURES_DIR, exist_ok=True)

bw_sq_file_re = re.compile(r'^etas_dynamic_benchmark_results_bw(\d+(?:\.\d+)?)\.csv$')


def _discover_bw_sq_files(base_dir):
    """Map bw_sq value -> base_dir (same directory for every value; the filename carries bw_sq)."""
    found = {}
    for entry in sorted(os.listdir(base_dir)):
        m = bw_sq_file_re.match(entry)
        if m:
            found[float(m.group(1))] = base_dir
    return dict(sorted(found.items()))

#%%
# ---------------------------------------------------------------------------
# Discover bw_sq sweep files: etas_dynamic_benchmark_results_bw{value}.csv
# ---------------------------------------------------------------------------
output_dirs = _discover_bw_sq_files(TD_DIR)
print(f'Found bw_sq values: {list(output_dirs.keys())}')

#%%
# ---------------------------------------------------------------------------
# Figure: Q-Q calibration — single panel (dynamic ETAS), one line per bw_sq
# ---------------------------------------------------------------------------
_trigs_label = f'{N_TRIGS} triggers' if N_TRIGS is not None else 'final version'
_trigs_tag   = f'{N_TRIGS}trigs' if N_TRIGS is not None else 'final'

extra_panel = {
    'name':         'ETAS',
    'output_dirs':  output_dirs,
    'csv_filename': 'etas_dynamic_benchmark_results_bw{param_value:g}.csv',
}

fig = plot_qq_calibration_by_param(
    prior_names = [],
    output_dirs = output_dirs,
    param_label = 'bw_sq',
    title       = f'Posterior calibration vs bw_sq — california  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'qq_calibration_vs_bw_sq_{_trigs_tag}.png'),
    ncols       = 1,
    extra_panel = extra_panel,
    n_trigs     = N_TRIGS,
)
plt.show()

# %%

COLUMN_NAME = 'post_val_at_usgs'
fig = plot_qq_calibration_by_param(
    prior_names = [],
    output_dirs = output_dirs,
    param_label = 'bw_sq',
    title       = f'{COLUMN_NAME} calibration vs bw_sq — california  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'{COLUMN_NAME}_qq_calibration_vs_bw_sq_{_trigs_tag}.png'),
    ncols       = 1,
    extra_panel = extra_panel,
    y_column    = COLUMN_NAME,
    n_trigs     = N_TRIGS,
)
plt.show()

#%%
