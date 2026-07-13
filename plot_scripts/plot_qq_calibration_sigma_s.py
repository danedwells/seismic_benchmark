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

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
ACTIVE_CASE_STUDY = 'Ridgecrest'   # 'Ridgecrest', 'Ferndale', 'ElMayor'
MAX_TRIGS         = config.BENCHMARK_PARAMS['max_trigs']
PRIOR_ORDER       = list(config.PRIOR_FILENAMES.keys())  # includes 'Uniform'

CS_TI_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                            ACTIVE_CASE_STUDY, 'output', 'time_independent')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                            ACTIVE_CASE_STUDY, 'figures', 'time_independent')
os.makedirs(FIGURES_DIR, exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# Discover sigma_s sweep directories: sig_{value}/max_trigs_{MAX_TRIGS}/
# ---------------------------------------------------------------------------
sig_dir_re = re.compile(r'^edt_(\d+(?:\.\d+)?)$')

output_dirs = {}
for entry in sorted(os.listdir(CS_TI_DIR)):
    m = sig_dir_re.match(entry)
    if not m:
        continue
    trig_dir = os.path.join(CS_TI_DIR, entry, f'max_trigs_{MAX_TRIGS}')
    if os.path.isdir(trig_dir):
        output_dirs[float(m.group(1))] = trig_dir

output_dirs = dict(sorted(output_dirs.items()))
print(f'Found sigma_s values: {list(output_dirs.keys())}')

#%%
# ---------------------------------------------------------------------------
# Figure: Q-Q calibration grid — one panel per prior, one line per sigma_s
# ---------------------------------------------------------------------------
fig = plot_qq_calibration_by_param(
    prior_names = PRIOR_ORDER,
    output_dirs = output_dirs,
    param_label = 'sigma_s',
    title       = f'Posterior calibration vs sigma_s — {ACTIVE_CASE_STUDY}  ({MAX_TRIGS} triggers)',
    save_path   = os.path.join(FIGURES_DIR, f'qq_calibration_vs_edt_sigma_s_{MAX_TRIGS}trigs.png'),
)
plt.show()

# %%
