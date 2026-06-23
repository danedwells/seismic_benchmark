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
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import config
from benchmark.metrics import COVERAGE_RADII_KM, load_per_version_stats, load_final_values
from benchmark.plots import plot_median_vs_triggers, plot_mean_vs_triggers
from benchmark.plots import plot_mean_posterior_coverage, plot_median_posterior_coverage
from benchmark.plots import plot_score_scatter
# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

ACTIVE_CASE_STUDY  = "Ferndale" # "ElMayor"  # None = main benchmark; 'Ridgecrest' / 'Ferndale' / 'ElMayor'
INCLUDE_BASELINES  = True   # overlay pure TI (dotted) and ETAS (dashed) for context
ALPHA              = 0.5    # blend weight used when running mixed_prior_scripts/
ALPHA_TAG          = f'alpha_{ALPHA:.2f}'

CASE_STUDIES = {
    'Ridgecrest': {'name': 'Ridgecrest 2019'},
    'Ferndale':   {'name': 'Ferndale 2022'},
    'ElMayor':    {'name': 'El Mayor-Cucapah 2010'},
}

# ---------------------------------------------------------------------------
# Resolve output paths
# ---------------------------------------------------------------------------

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
EDT_SIGMA_S = config.BENCHMARK_PARAMS['edt_sigma_s']
EDT_TAG     = f'edt_{EDT_SIGMA_S}'

if ACTIVE_CASE_STUDY is None:
    OUTPUT_DIR_MIXED   = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'mixed', EDT_TAG, f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    OUTPUT_DIR_DYNAMIC = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_dependent', f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_independent', EDT_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(PROJECT_ROOT, 'results', 'figures',
                                      'comparison_mixed', EDT_TAG, ALPHA_TAG)
    PLOT_TITLE_SUFFIX  = 'main benchmark'
else:
    cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
    _cs_base           = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                                      ACTIVE_CASE_STUDY)
    OUTPUT_DIR_MIXED   = os.path.join(_cs_base, 'output', 'mixed',
                                      EDT_TAG, f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    OUTPUT_DIR_DYNAMIC = os.path.join(_cs_base, 'output', 'time_dependent',
                                      f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(_cs_base, 'output', 'time_independent',
                                      EDT_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(_cs_base, 'figures', 'comparison_mixed', EDT_TAG, ALPHA_TAG)
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
# ---------------------------------------------------------------------------
# Figure 4: log-score vs trigger count
# ---------------------------------------------------------------------------
fig_ls = plot_median_vs_triggers(
    metric       = 'log_score',
    ylabel       = 'Median log-score  (↑ better)',
    title        = f'Log-score vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    PRIOR_SPECS  = PRIOR_SPECS,
    save_path    = os.path.join(FIGURES_DIR, f'log_score_median_vs_triggers_{ALPHA_TAG}.png'),
    shade_groups = ('mixed', 'dynamic'),
)
plt.show()

# ---------------------------------------------------------------------------
# Figure 5: Brier score vs trigger count
# ---------------------------------------------------------------------------
fig_bs = plot_median_vs_triggers(
    metric       = 'brier_score',
    ylabel       = 'Median Brier score  (↓ better)',
    title        = f'Brier score vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    PRIOR_SPECS  = PRIOR_SPECS,
    save_path    = os.path.join(FIGURES_DIR, f'brier_score_median_vs_triggers_{ALPHA_TAG}.png'),
    shade_groups = ('mixed', 'dynamic'),
)
plt.show()

# ---------------------------------------------------------------------------
# Figure 6: scoring metrics vs location error scatter (1×2 panel)
# ---------------------------------------------------------------------------
fig_sc = plot_score_scatter(
    PRIOR_SPECS = PRIOR_SPECS,
    title       = f'Scoring metrics vs location error — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, f'score_vs_location_error_{ALPHA_TAG}.png'),
)
plt.show()

# %%

# Assign consistent colors: mixed and their TI counterparts share a color.
trigger_number = 10
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
spec = PRIOR_SPECS[5] # Uniform as reference
ref_stats = load_final_values(spec['csv'], 'map_err_km', n_trigs = trigger_number)

for i,ax in enumerate(axes):
    spec = PRIOR_SPECS[i]

    # Skip uniform as a dedicated plot - go to dynamic etas
    if spec['name'] == 'Uniform+ETAS':
        i = i+1
        spec = PRIOR_SPECS[i]
    stats = load_final_values(spec['csv'], 'map_err_km', n_trigs=trigger_number)
    ax.hist(ref_stats, bins = bins, rwidth=0.9, color='b', label=['Uniform + ETAS (ref)'], alpha=0.4)
    ax.hist(stats, bins = bins, rwidth=0.9, color='r', alpha=0.6)
    ax.set_xscale('log')
    ax.set_title(spec["name"])
    ax.grid()
    if i == 0:
        ax.legend()

# Enforce consistent y-axis across all panels
y_max = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, y_max)

for ax in [ax1,ax2,ax3]:
    ax.set_xticklabels([])

for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])

for ax in [ax4, ax5, ax6]:
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

fig.suptitle(f"Location error (km):  Sequence: {ACTIVE_CASE_STUDY}  Number of triggers: {trigger_number}   Alpha: {ALPHA}",fontsize=16)

plt.show()
fig.savefig(os.path.join(FIGURES_DIR,"hist_location_error_{trigger_number}_{ALPHA}.png"))

# %%
# ---------------------------------------------------------------------------
# Figure 7: count of events with location error >= threshold vs trigger count
# ---------------------------------------------------------------------------
# 2×2 panel, one per radius in COVERAGE_RADII_KM (10, 25, 50, 100 km).
# Each panel tallies how many events still have MAP error ≥ that threshold at
# each trigger count.  Lower = better.  Mirrors the posterior_coverage() radii.
# ---------------------------------------------------------------------------
colors = plt.cm.tab10.colors
mixed_specs  = [s for s in PRIOR_SPECS if s['group'] == 'mixed']
color_lookup = {s['name']: colors[i % len(colors)]
                for i, s in enumerate(mixed_specs)}
if mixed_specs:
    for s in PRIOR_SPECS:
        if s['group'] == 'static':
            mixed_counterpart = f"{s['name']}+ETAS"
            if mixed_counterpart in color_lookup:
                color_lookup[s['name']] = color_lookup[mixed_counterpart]
else:
    for i, s in enumerate(s for s in PRIOR_SPECS if s['group'] == 'static'):
        color_lookup[s['name']] = colors[i % len(colors)]
for s in PRIOR_SPECS:
    if s['group'] == 'dynamic':
        color_lookup[s['name']] = 'black'

# Pre-load CSVs once; skip priors without usable data
loaded = {}
for spec in PRIOR_SPECS:
    if not os.path.exists(spec['csv']):
        print(f"  [{spec['name']}] CSV not found — skipping")
        continue
    df = pd.read_csv(spec['csv'])
    if 'map_err_km' not in df.columns or df['map_err_km'].isna().all():
        print(f"  [{spec['name']}] no map_err_km data — skipping")
        continue
    df = df.dropna(subset=['map_err_km']).copy()
    if 'n_trigs' not in df.columns:
        df['n_trigs'] = (df.groupby('event_id')['version']
                           .rank(method='dense')
                           .astype(int))
    loaded[spec['name']] = (spec, df)

fig_large, axes_large = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes_large = axes_large.flatten()

for ax, threshold in zip(axes_large, COVERAGE_RADII_KM):
    for name, (spec, df) in loaded.items():
        counts = (df.groupby('n_trigs')['map_err_km']
                    .apply(lambda x: (x >= threshold).sum())
                    .reset_index(name='n_large_errors'))
        color    = color_lookup.get(name, 'gray')
        n_events = df['event_id'].nunique()
        ax.plot(counts['n_trigs'], counts['n_large_errors'],
                color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                label=f"{name}  (n={n_events})")

    ax.set_title(f'Error ≥ {threshold} km', fontsize=11)
    ax.set_xlim(left=1)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

for ax in axes_large[2:]:
    ax.set_xlabel('Number of triggers', fontsize=11)
for ax in axes_large[::2]:
    ax.set_ylabel('Event count  (↓ better)', fontsize=11)

handles, labels = axes_large[0].get_legend_handles_labels()
fig_large.legend(handles, labels, fontsize=8, loc='lower center',
                 ncol=min(len(loaded), 4), bbox_to_anchor=(0.5, -0.02))
fig_large.suptitle(
    f'Events exceeding error threshold vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    fontsize=13)
plt.tight_layout(rect=[0, 0.06, 1, 1])

_save = os.path.join(FIGURES_DIR, f'large_error_count_vs_triggers_{ALPHA_TAG}.png')
fig_large.savefig(_save, dpi=150, bbox_inches='tight')
print(f'Saved: {_save}')
plt.show()

# %%
