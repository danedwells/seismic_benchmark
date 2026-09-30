#%%
# =============================================================================
# coverage_vs_triggers.py — paper figure: posterior coverage vs trigger count
# =============================================================================
# Mean fraction of posterior probability mass within 10 / 25 / 50 / 100 km of
# the USGS location, as a function of number of triggers, for the California
# benchmark. Only KDE_Seismicity, Uniform and ETAS are shown.
#
# Adapted from plot_scripts/plot_comparison.py (plot_mean_posterior_coverage).
# Panel positions are set explicitly via AX_POS so the layout can be hand-tuned.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import matplotlib.pyplot as plt

from benchmark.metrics import COVERAGE_RADII_KM, load_per_version_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config — must match the california benchmark run these CSVs came from.
# ---------------------------------------------------------------------------
MAX_TRIGS          = 10
OUTPUT_DIR_STATIC  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}')
OUTPUT_DIR_DYNAMIC = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent',   f'max_trigs_{MAX_TRIGS}')

# Same prior order / colors as location_hist_case_study.py
PRIOR_SPECS = [
    {'name': name, 'csv': os.path.join(OUTPUT_DIR_STATIC, f'{name.lower()}_benchmark_results.csv')}
    for name in ('KDE_Seismicity', 'Uniform')
] + [
    {'name': 'ETAS', 'csv': os.path.join(OUTPUT_DIR_DYNAMIC, 'etas_dynamic_benchmark_results.csv')}
]
_colors = plt.cm.tab10.colors
COLOR_LOOKUP = {spec['name']: _colors[i % len(_colors)] for i, spec in enumerate(PRIOR_SPECS)}

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    10:  [0.07, 0.54, 0.42, 0.38],
    25:  [0.52, 0.54, 0.42, 0.38],
    50:  [0.07, 0.08, 0.42, 0.38],
    100: [0.52, 0.08, 0.42, 0.38],
}

fig = plt.figure(figsize=(11, 8))

for i, radius_km in enumerate(COVERAGE_RADII_KM):
    ax = fig.add_axes(AX_POS[radius_km])
    for spec in PRIOR_SPECS:
        stats = load_per_version_stats(spec['csv'], f'coverage_{radius_km}km')
        if stats is None:
            print(f"  [{spec['name']}] no data for {radius_km} km — skipping")
            continue
        ax.plot(stats['n_trigs'], stats['mean'], '-o', color=COLOR_LOOKUP[spec['name']],
                lw=2.5, ms=5, label=spec['name'])
    # Shared axes: x label/ticks only on the bottom row, y label/ticks only on the left column
    if i >= 2:
        ax.set_xlabel('Number of triggers')
    else:
        ax.set_xticklabels([])
    if i % 2 == 0:
        ax.set_ylabel('Mean posterior coverage')
    else:
        ax.set_yticklabels([])
    ax.set_title(f'Within {radius_km} km')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.03, f'({"abcd"[i]})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=13, fontweight='bold')
    if i == 0:
        ax.legend(loc='lower right', fontsize=10)

fig.savefig(os.path.join(OUTPUT_DIR, 'coverage_vs_triggers.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'coverage_vs_triggers.pdf'), bbox_inches='tight')
plt.show()

# %%
