"""
plot_scripts/plot_comparison_mixed.py — cross-workflow comparison for mixed priors.

Plots median metric vs trigger count for the five blended (TI + ETAS) priors,
with ETAS_dynamic always shown as a dashed reference line.  Set
INCLUDE_BASELINES = True to also overlay the pure time-independent priors as
dotted lines, giving a three-way comparison:

    solid  — mixed prior   (ALPHA * ETAS + (1-ALPHA) * TI)
    dashed — ETAS dynamic  (reference upper bound for time-dependence)
    dotted — TI prior      (reference for time-independence) [optional]

Set ACTIVE_CASE_STUDY to None for the main benchmark, or to one of the keys
in CASE_STUDIES for a case-study sequence.

Set ALPHA to the value used when running mixed_prior_scripts/ so it appears
correctly in figure titles.
"""
#%%
import os
import sys

import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import config
from benchmark.metrics import COVERAGE_RADII_KM, load_per_version_stats
from benchmark.plots import plot_median_vs_triggers, plot_mean_vs_triggers
from benchmark.plots import plot_mean_posterior_coverage, plot_median_posterior_coverage
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

ACTIVE_CASE_STUDY  = "Ridgecrest" # "ElMayor"  # None = main benchmark; 'Ridgecrest' / 'Ferndale' / 'ElMayor'
INCLUDE_BASELINES  = True   # overlay pure TI (dotted) and ETAS (dashed) for context
ALPHA              = 0.1    # blend weight used when running mixed_prior_scripts/
ALPHA_TAG          = f'alpha_{ALPHA:.2f}'

CASE_STUDIES = {
    'Ridgecrest': {'name': 'Ridgecrest 2019'},
    'Ferndale':   {'name': 'Ferndale 2022'},
    'ElMayor':    {'name': 'El Mayor-Cucapah 2010'},
}

# ---------------------------------------------------------------------------
# Resolve output paths
# ---------------------------------------------------------------------------

MAX_TRIGS = config.BENCHMARK_PARAMS['max_trigs']

if ACTIVE_CASE_STUDY is None:
    OUTPUT_DIR_MIXED   = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    OUTPUT_DIR_DYNAMIC = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_dependent', f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_independent', f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(PROJECT_ROOT, 'results', 'figures',
                                      'comparison_mixed', ALPHA_TAG)
    PLOT_TITLE_SUFFIX  = 'main benchmark'
else:
    cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
    _cs_base           = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                                      ACTIVE_CASE_STUDY)
    OUTPUT_DIR_MIXED   = os.path.join(_cs_base, 'output', 'mixed',
                                      f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    OUTPUT_DIR_DYNAMIC = os.path.join(_cs_base, 'output', 'time_dependent',
                                      f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(_cs_base, 'output', 'time_independent',
                                      f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(_cs_base, 'figures', 'comparison_mixed', ALPHA_TAG)
    PLOT_TITLE_SUFFIX  = cs['name']

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prior specs
# Each entry: display name, csv path, linestyle, linewidth, alpha
# ---------------------------------------------------------------------------

TI_NAMES = list(config.PRIOR_FILENAMES.keys())   # Gear1, NSHM, …, Uniform

PRIOR_SPECS = [
    {
        'name':      f'{name}+ETAS',
        'csv':       os.path.join(OUTPUT_DIR_MIXED,
                                  f'{name.lower()}_etas_mixed_benchmark_results.csv'),
        'ls':        '-',
        'lw':        2.5,
        'group':     'mixed',
    }
    for name in TI_NAMES
]

PRIOR_SPECS += [
    {
        'name':  'ETAS (dynamic)',
        'csv':   os.path.join(OUTPUT_DIR_DYNAMIC, 'etas_dynamic_benchmark_results.csv'),
        'ls':    '-',
        'lw':    2.5,
        'group': 'dynamic',
    },
]

if INCLUDE_BASELINES:
    PRIOR_SPECS += [
        {
            'name':  name,
            'csv':   os.path.join(OUTPUT_DIR_STATIC,
                                  f'{name.lower()}_benchmark_results.csv'),
            'ls':    '--',
            'lw':    1.2,
            'group': 'static',
        }
        for name in TI_NAMES
    ]



#%%
# ---------------------------------------------------------------------------
# Figure 1: posterior_confidence_level vs trigger count
# ---------------------------------------------------------------------------
fig_cred = plot_median_vs_triggers(
    metric    = 'posterior_confidence_level',
    ylabel    = 'Median posterior_confidence_level  (↓ better)',
    title     = f'Posterior calibration vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    ylim      = (0, 1),
    ref_line  = 0.5,
    ref_label = 'calibrated median  (0.5)',
    PRIOR_SPECS = PRIOR_SPECS,
    save_path = os.path.join(FIGURES_DIR, f'posterior_confidence_median_vs_triggers_{ALPHA_TAG}.png'),
)
plt.show()

# ---------------------------------------------------------------------------
# Figure 1: posterior_confidence_level vs trigger count
# ---------------------------------------------------------------------------
fig_cred = plot_mean_vs_triggers(
    metric        = 'posterior_confidence_level',
    ylabel        = 'Mean posterior_confidence_level  (↓ better)',
    title         = f'Posterior calibration vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    ylim          = (0, 1),
    ref_line      = 0.5,
    ref_label     = 'calibrated median  (0.5)',
    PRIOR_SPECS   = PRIOR_SPECS,
    save_path     = os.path.join(FIGURES_DIR, f'posterior_confidence_mean_vs_triggers_{ALPHA_TAG}.png'),
    shade_groups  = ('mixed', 'dynamic'),
)
plt.show()

#%%
# ---------------------------------------------------------------------------
# Figure 2: posterior coverage at fixed radii vs trigger count (2×2 panel)
# ---------------------------------------------------------------------------
fig_cov = plot_median_posterior_coverage(
    PRIOR_SPECS = PRIOR_SPECS,
    title       = f'Posterior coverage vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, f'coverage_vs_triggers_median_{ALPHA_TAG}.png'),
    legend_ncol = 2 if INCLUDE_BASELINES else 1,
)
plt.show()

fig_cov = plot_mean_posterior_coverage(
    PRIOR_SPECS = PRIOR_SPECS,
    title       = f'Posterior coverage vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, f'coverage_vs_triggers_mean_{ALPHA_TAG}.png'),
    legend_ncol = 2 if INCLUDE_BASELINES else 1,
)
plt.show()

#%%
# ---------------------------------------------------------------------------
# Figure 3: location error (km) vs trigger count
# ---------------------------------------------------------------------------
fig_err = plot_median_vs_triggers(
    metric    = 'map_err_km',
    ylabel    = 'Median location error  km  (↓ better)',
    title     = f'Location error vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    log_y     = True,
    PRIOR_SPECS = PRIOR_SPECS,
    save_path = os.path.join(FIGURES_DIR, f'location_error_median_vs_triggers_{ALPHA_TAG}.png'),
)
plt.show()

fig_err = plot_mean_vs_triggers(
    metric       = 'map_err_km',
    ylabel       = 'Mean location error  km  (↓ better)',
    title        = f'Location error vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    log_y        = True,
    PRIOR_SPECS  = PRIOR_SPECS,
    save_path    = os.path.join(FIGURES_DIR, f'location_error_mean_vs_triggers_{ALPHA_TAG}.png'),
    shade_groups = ('mixed', 'dynamic'),
)
plt.show()

# %%

import numpy as np
import pandas as pd

def load_final_values(csv_path, metric, n_trigs=None, min_events_warn=5):
    import warnings
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if metric not in df.columns or df[metric].isna().all():
        return None

    if 'n_trigs' not in df.columns:
        df['n_trigs'] = (df.groupby('event_id')['version']
                            .rank(method='dense')
                            .astype(int))

    if n_trigs is None:
        vals = df.groupby('event_id').last()[metric].dropna().values
    else:
        max_available = int(df['n_trigs'].max())
        if n_trigs > max_available:
            raise ValueError(
                f"Requested n_trigs={n_trigs} exceeds the maximum available "
                f"({max_available}) in {os.path.basename(csv_path)}."
            )
        subset = df[df['n_trigs'] == n_trigs][metric].dropna()
        if len(subset) < min_events_warn:
            warnings.warn(
                f"Only {len(subset)} events have data at n_trigs={n_trigs} "
                f"in {os.path.basename(csv_path)} (min_events_warn={min_events_warn}). "
                "Results may be unreliable.",
                UserWarning, stacklevel=2,
            )
        vals = subset.values

    return vals if len(vals) > 0 else None

# Assign consistent colors: mixed and their TI counterparts share a color.
trigger_number = 5
colors = plt.cm.tab10.colors
# Build color index: mixed priors get indices 0..4; TI baselines reuse same indices.
mixed_specs  = [s for s in PRIOR_SPECS if s['group'] == 'mixed']
color_lookup = {s['name']: colors[i % len(colors)]
                for i, s in enumerate(mixed_specs)}

fig = plt.figure(figsize=(14,8),dpi=300)
ax1 = fig.add_axes([0.02, 0.5, 0.28, 0.43])  
ax2 = fig.add_axes([0.36, 0.5, 0.28, 0.43])   
ax3 = fig.add_axes([0.7, 0.5, 0.28, 0.43])   
ax4 = fig.add_axes([0.02, 0.02, 0.28, 0.43])  
ax5 = fig.add_axes([0.36, 0.02, 0.28, 0.43])   
ax6 = fig.add_axes([0.7, 0.02, 0.28, 0.43])   

bins = np.logspace(-1,3,20)

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

for i,ax in enumerate(axes):
    spec = PRIOR_SPECS[i]
    stats = load_final_values(spec['csv'], 'map_err_km', n_trigs=trigger_number)
    ax.hist(stats, bins = bins, rwidth=0.9, label=spec['name'], alpha=0.8)
    ax.set_xscale('log')
    ax.set_title(spec["name"])
    ax.grid()

# Enforce consistent y-axis across all panels
y_max = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, y_max)

for ax in [ax1,ax2,ax3]:
    ax.set_xticklabels([])

for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])

fig.suptitle(f"Sequence: {ACTIVE_CASE_STUDY}  Number of triggers: {trigger_number}   Alpha: {ALPHA}",fontsize=16)

plt.show()
fig.savefig(os.path.join(FIGURES_DIR,"hist_location_error_{trigger_number}_{ALPHA}.png"))
# %%
