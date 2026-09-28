"""
plot_scripts/log_likelihood_vs_bw_sq.py — total log-likelihood vs bw_sq.

For a bw_sq sweep produced by data_examination_scripts/bw_sq_calibration.py
(results/california/output/time_dependent/max_trigs_{N}/
etas_dynamic_benchmark_results_bw{value}.csv), computes the sum of
log(post_val_at_usgs) across all events at a fixed trigger count — the
total log-likelihood of the true locations under that bw_sq. Maximizing
this sum picks the bw_sq that best explains where events actually
occurred.

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
from benchmark.plots import plot_log_likelihood_sum_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
MAX_TRIGS = config.BENCHMARK_PARAMS['max_trigs']
N_TRIGS   = 5   # per-event trigger count to use; None = each event's last (most-triggered) row

# Column treated as a per-event probability; sum of its log is the total
# log-likelihood plotted against bw_sq.
#COLUMN = 'like_val_at_usgs'
COLUMN = 'post_val_at_usgs'
#COLUMN = 'like_val_raw_at_usgs'

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
# Figure: Σ log(COLUMN) vs bw_sq — single panel, dynamic ETAS prior only
# ---------------------------------------------------------------------------
_trigs_label = f'{N_TRIGS} triggers' if N_TRIGS is not None else 'final version'
_trigs_tag   = f'{N_TRIGS}trigs' if N_TRIGS is not None else 'final'

extra_panel = {
    'name':         'ETAS',
    'output_dirs':  output_dirs,
    'csv_filename': 'etas_dynamic_benchmark_results_bw{param_value:g}.csv',
}

fig = plot_log_likelihood_sum_by_param(
    prior_names = [],
    output_dirs = output_dirs,
    param_label = 'bw_sq',
    column      = COLUMN,
    title       = f"Total log sum of '{COLUMN}' vs bw_sq — california  ({_trigs_label})",
    save_path   = os.path.join(FIGURES_DIR, f'log_likelihood_vs_bw_sq_{_trigs_tag}_col_{COLUMN}.png'),
    ncols       = 1,
    extra_panel = extra_panel,
    n_trigs     = N_TRIGS,
    log_floor   = 1E-30,
)
plt.show()

# %%
