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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import config
from benchmark.metrics import COVERAGE_RADII_KM

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

ACTIVE_CASE_STUDY  = None   # None = main benchmark; 'Ridgecrest' / 'Ferndale' / 'ElMayor'
INCLUDE_BASELINES  = True   # overlay pure TI (dotted) and ETAS (dashed) for context
ALPHA              = 0.5    # blend weight used when running mixed_prior_scripts/

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
                                      'mixed', f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_DYNAMIC = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_dependent', f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(PROJECT_ROOT, 'results', 'output',
                                      'time_independent', f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(PROJECT_ROOT, 'results', 'figures',
                                      'comparison_mixed')
    PLOT_TITLE_SUFFIX  = 'main benchmark'
else:
    cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
    _cs_base           = os.path.join(PROJECT_ROOT, 'results', 'case_studies',
                                      ACTIVE_CASE_STUDY)
    OUTPUT_DIR_MIXED   = os.path.join(_cs_base, 'output', 'mixed',
                                      f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_DYNAMIC = os.path.join(_cs_base, 'output', 'time_dependent',
                                      f'max_trigs_{MAX_TRIGS}')
    OUTPUT_DIR_STATIC  = os.path.join(_cs_base, 'output', 'time_independent',
                                      f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR        = os.path.join(_cs_base, 'figures', 'comparison_mixed')
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
        'lw':        1.8,
        'group':     'mixed',
    }
    for name in TI_NAMES
]

PRIOR_SPECS += [
    {
        'name':  'ETAS (dynamic)',
        'csv':   os.path.join(OUTPUT_DIR_DYNAMIC, 'etas_dynamic_benchmark_results.csv'),
        'ls':    '--',
        'lw':    2.0,
        'group': 'dynamic',
    },
]

if INCLUDE_BASELINES:
    PRIOR_SPECS += [
        {
            'name':  name,
            'csv':   os.path.join(OUTPUT_DIR_STATIC,
                                  f'{name.lower()}_benchmark_results.csv'),
            'ls':    ':',
            'lw':    1.2,
            'group': 'static',
        }
        for name in TI_NAMES
    ]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_per_version_stats(csv_path, metric, min_events=5):
    """
    Load a benchmark CSV and return per-trigger-count aggregate statistics.

    Returns a DataFrame with columns [n_trigs, median, q5, q95, count],
    or None if the file is missing or the metric column is absent / all-NaN.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if metric not in df.columns or df[metric].isna().all():
        return None

    df = df.dropna(subset=[metric]).copy()
    if 'n_trigs' not in df.columns:
        df['n_trigs'] = (df.groupby('event_id')['version']
                           .rank(method='dense')
                           .astype(int))

    stats = (df.groupby('n_trigs')[metric]
               .agg(median='median',
                    q5=lambda x: x.quantile(0.05),
                    q95=lambda x: x.quantile(0.95),
                    count='count')
               .reset_index())
    return stats[stats['count'] >= min_events]


def _group_color(group, i, colors):
    """Return a color for a spec, consistent within each group."""
    if group == 'dynamic':
        return 'black'
    return colors[i % len(colors)]


def plot_metric_vs_triggers(metric, ylabel, title, save_path=None,
                            ylim=None, ref_line=None, ref_label=None,
                            log_y=False):
    """
    Plot median metric vs trigger count for all priors, with 5–95 % band.

    Mixed priors: solid lines.
    ETAS dynamic: dashed black reference.
    TI baselines (if INCLUDE_BASELINES): dotted lines, same color palette.
    """
    # Assign consistent colors: mixed and their TI counterparts share a color.
    colors = plt.cm.tab10.colors
    # Build color index: mixed priors get indices 0..4; TI baselines reuse same indices.
    mixed_names  = [s['name'] for s in PRIOR_SPECS if s['group'] == 'mixed']
    color_lookup = {s['name']: colors[i % len(colors)]
                    for i, s in enumerate(s for s in PRIOR_SPECS if s['group'] == 'mixed')}
    for s in PRIOR_SPECS:
        if s['group'] == 'static':
            ti_name = s['name']
            mixed_counterpart = f'{ti_name}+ETAS'
            if mixed_counterpart in color_lookup:
                color_lookup[ti_name] = color_lookup[mixed_counterpart]
        elif s['group'] == 'dynamic':
            color_lookup[s['name']] = 'black'

    fig, ax = plt.subplots(figsize=(11, 6))

    for spec in PRIOR_SPECS:
        stats = load_per_version_stats(spec['csv'], metric)
        if stats is None:
            print(f"  [{spec['name']}] no data for '{metric}' — skipping")
            continue

        color   = color_lookup.get(spec['name'], 'gray')
        n_max   = int(stats['count'].max())
        label   = f"{spec['name']}  (n≈{n_max})"

        ax.plot(stats['n_trigs'], stats['median'],
                color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                label=label)

        # Only shade IQR for mixed and dynamic (not baselines, to reduce clutter)
        if spec['group'] in ('mixed', 'dynamic'):
            ax.fill_between(stats['n_trigs'], stats['q5'], stats['q95'],
                            color=color, alpha=0.10)
            ax.plot(stats['n_trigs'], stats['q5'],
                    color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)
            ax.plot(stats['n_trigs'], stats['q95'],
                    color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)

    if ref_line is not None:
        ax.axhline(ref_line, color='gray', linestyle=':', linewidth=1,
                   label=ref_label or str(ref_line))

    ax.set_xlabel('Number of triggers', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(left=1)
    if log_y:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Split legend: mixed priors on left, references on right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc='upper right',
              ncol=2 if len(handles) > 6 else 1)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


#%%
# ---------------------------------------------------------------------------
# Figure 1: usgs_credible_level vs trigger count
# ---------------------------------------------------------------------------
fig_cred = plot_metric_vs_triggers(
    metric    = 'usgs_credible_level',
    ylabel    = 'Median usgs_credible_level  (↓ better)',
    title     = f'Posterior calibration vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    ylim      = (0, 1),
    ref_line  = 0.5,
    ref_label = 'calibrated median  (0.5)',
    save_path = os.path.join(FIGURES_DIR, 'credible_level_vs_triggers.png'),
)
plt.show()

#%%
# ---------------------------------------------------------------------------
# Figure 2: posterior coverage at fixed radii vs trigger count (2×2 panel)
# ---------------------------------------------------------------------------
colors = plt.cm.tab10.colors

mixed_names  = [s['name'] for s in PRIOR_SPECS if s['group'] == 'mixed']
color_lookup = {s['name']: colors[i % len(colors)]
                for i, s in enumerate(s for s in PRIOR_SPECS if s['group'] == 'mixed')}
for s in PRIOR_SPECS:
    if s['group'] == 'static':
        color_lookup[s['name']] = color_lookup.get(f"{s['name']}+ETAS", 'gray')
color_lookup['ETAS (dynamic)'] = 'black'

fig_cov, axes_cov = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

for ax, radius_km in zip(axes_cov.flatten(), COVERAGE_RADII_KM):
    col = f'coverage_{radius_km}km'
    for spec in PRIOR_SPECS:
        stats = load_per_version_stats(spec['csv'], col)
        if stats is None:
            continue
        color = color_lookup.get(spec['name'], 'gray')
        n_max = int(stats['count'].max())
        ax.plot(stats['n_trigs'], stats['median'],
                color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                label=f"{spec['name']}  (n≈{n_max})")
        if spec['group'] in ('mixed', 'dynamic'):
            ax.fill_between(stats['n_trigs'], stats['q5'], stats['q95'],
                            color=color, alpha=0.10)
    ax.set_xlabel('Number of triggers', fontsize=10)
    ax.set_ylabel('Median coverage  (↑ better)', fontsize=10)
    ax.set_title(f'Within {radius_km} km', fontsize=11)
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='lower right', ncol=2 if INCLUDE_BASELINES else 1)

fig_cov.suptitle(
    f'Posterior coverage vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    fontsize=13)
plt.tight_layout()
_cov_path = os.path.join(FIGURES_DIR, 'coverage_vs_triggers.png')
plt.savefig(_cov_path, dpi=150, bbox_inches='tight')
print(f'Saved: {_cov_path}')
plt.show()

#%%
# ---------------------------------------------------------------------------
# Figure 3: location error (km) vs trigger count
# ---------------------------------------------------------------------------
fig_err = plot_metric_vs_triggers(
    metric    = 'map_err_km',
    ylabel    = 'Median location error  km  (↓ better)',
    title     = f'Location error vs trigger count — mixed priors (α={ALPHA}) — {PLOT_TITLE_SUFFIX}',
    log_y     = True,
    save_path = os.path.join(FIGURES_DIR, 'location_error_vs_triggers.png'),
)
plt.show()

# %%
