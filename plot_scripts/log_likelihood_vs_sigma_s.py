"""
plot_scripts/log_likelihood_vs_sigma_s.py — total log-likelihood vs sigma_s.

For a sigma_s sweep produced by data_examination_scripts/sigma_calibration.py
(sig_{value}/max_trigs_{N}/uniform_benchmark_results.csv), computes the sum
of log(like_val_at_usgs) across all events at a fixed trigger count — the
total log-likelihood of the true locations under that sigma_s. Maximizing
this sum picks the sigma_s that best explains where events actually
occurred; it's the numerically stable stand-in for maximizing the raw
product of per-event likelihood values (which underflows to 0 almost
immediately).

Single-panel plot: only the Uniform prior is relevant here (calibrating
sigma_s is a likelihood-only question — posterior ∝ likelihood when the
prior is Uniform, so no other prior's calibration differs).

Works for any region or case study — set TARGET_KIND / TARGET below.
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
from benchmark.plots import plot_log_likelihood_sum_by_param

#%%
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
TARGET_KIND = 'region'        # 'region' | 'case_study'
TARGET      = 'cascadia'      # 'california'|'cascadia' (region), or a
                               # benchmark.config.CASE_STUDIES key (case_study)

if TARGET == 'cascadia':
    from benchmark import config_cascadia as config
elif TARGET == 'california':
    from benchmark import config_california as config


MAX_TRIGS = config.BENCHMARK_PARAMS['max_trigs']
N_TRIGS   = 5   # per-event trigger count to use; None = each event's last (most-triggered) row

# Column treated as a per-event probability; sum of its log is the total
# log-likelihood plotted against sigma_s. 'like_val_at_usgs' (raw normalized
# likelihood-surface value at the true location) is the natural default.
#COLUMN = 'like_val_at_usgs'
COLUMN = 'post_val_at_usgs'
#COLUMN = 'like_val_raw_at_usgs'

if TARGET_KIND == 'region':
    CS_TI_DIR   = os.path.join(PROJECT_ROOT, 'results', TARGET, 'output', 'time_independent')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', TARGET, 'figures', 'time_independent')
elif TARGET_KIND == 'case_study':
    CS_TI_DIR   = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', TARGET, 'output', 'time_independent')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', TARGET, 'figures', 'time_independent')
else:
    raise ValueError(f"Unknown TARGET_KIND '{TARGET_KIND}' — expected 'region' or 'case_study'")
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
print(f'Found sigma_s values: {list(output_dirs.keys())}')

#%%
# ---------------------------------------------------------------------------
# Figure: Σ log(COLUMN) vs sigma_s — single panel, Uniform prior only
# ---------------------------------------------------------------------------
_trigs_label = f'{N_TRIGS} triggers' if N_TRIGS is not None else 'final version'
_trigs_tag   = f'{N_TRIGS}trigs' if N_TRIGS is not None else 'final'

fig = plot_log_likelihood_sum_by_param(
    prior_names = ['Uniform'],
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    column      = COLUMN,
    title       = f"Total log sum of '{COLUMN}' vs sigma_s — {TARGET}  ({_trigs_label})",
    save_path   = os.path.join(FIGURES_DIR, f'log_likelihood_vs_sigma_s_{_trigs_tag}_col_{COLUMN}.png'),
    ncols       = 1,
    n_trigs     = N_TRIGS,
    log_floor   = 1E-30,
)
plt.show()

# %%
