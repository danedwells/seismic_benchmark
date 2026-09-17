"""
plot_scripts/plot_qq_calibration_sigma_s.py — Q-Q calibration vs sigma_s.

For a sigma_s sweep produced by data_examination_scripts/sigma_calibration.py
(sig_{value}/max_trigs_{N}/uniform_benchmark_results.csv), plots
posterior_confidence_level Q-Q calibration curves — one line per sigma_s
value tested.

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



from benchmark.plots import plot_qq_calibration_by_param

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
N_TRIGS   = 5   # per-event trigger count to plot; None = each event's last (most-triggered) row

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
# Figure: Q-Q calibration — single panel (Uniform), one line per sigma_s
# ---------------------------------------------------------------------------
_trigs_label = f'{N_TRIGS} triggers' if N_TRIGS is not None else 'final version'
_trigs_tag   = f'{N_TRIGS}trigs' if N_TRIGS is not None else 'final'

fig = plot_qq_calibration_by_param(
    prior_names = ['Uniform'],
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'Posterior calibration vs sigma_s — {TARGET}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'qq_calibration_vs_sigma_s_{_trigs_tag}.png'),
    ncols       = 1,
    n_trigs     = N_TRIGS,
)
plt.show()

# %%

COLUMN_NAME = 'post_val_at_usgs'
fig = plot_qq_calibration_by_param(
    prior_names = ['Uniform'],
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'{COLUMN_NAME} calibration vs sigma_s — {TARGET}  ({_trigs_label})',
    save_path   = os.path.join(FIGURES_DIR, f'{COLUMN_NAME}_qq_calibration_vs_sigma_s_{_trigs_tag}.png'),
    ncols       = 1,
    y_column    = COLUMN_NAME,
    n_trigs     = N_TRIGS,
)
plt.show()

#%%
