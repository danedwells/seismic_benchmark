#%%
# =============================================================================
# confidence_vs_triggers.py — paper figure: posterior calibration vs trigger count
# =============================================================================
# posterior_confidence_level is the smallest highest-density credible level whose
# region contains the USGS location. For a calibrated posterior it is U(0, 1), so
# the fraction of events with level <= p (USGS inside the p credible region)
# should equal p. Panels: p = 25 / 50 / 68 / 95 %, vs number of triggers, for the
# California benchmark; the dashed line is perfect calibration. Same layout and
# priors as confidence_vs_triggers.py. Above the line = underconfident posterior,
# below = overconfident.
# Panel positions are set explicitly via AX_POS so the layout can be hand-tuned.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import matplotlib.pyplot as plt

import pandas as pd

CREDIBLE_LEVELS = [0.25, 0.50, 0.68, 0.95]
MIN_EVENTS = 5

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
COLOR_LOOKUP = {'KDE_Seismicity': 'gray', 'Uniform': 'darkblue', 'ETAS': 'red'}

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
AX_POS = [[0.07, 0.54, 0.42, 0.38], [0.52, 0.54, 0.42, 0.38],
          [0.07, 0.08, 0.42, 0.38], [0.52, 0.08, 0.42, 0.38]]

fig = plt.figure(figsize=(11, 8))

data = {}
for spec in PRIOR_SPECS:
    df = pd.read_csv(spec['csv']).dropna(subset=['posterior_confidence_level'])
    data[spec['name']] = df[df.groupby('n_trigs')['event_id'].transform('count') >= MIN_EVENTS]

for i, p in enumerate(CREDIBLE_LEVELS):
    ax = fig.add_axes(AX_POS[i])
    ax.axhline(p, color='k', ls='--', lw=1.2, zorder=1)
    for spec in PRIOR_SPECS:
        df = data[spec['name']]
        frac = (df['posterior_confidence_level'] <= p).groupby(df['n_trigs']).mean()
        ax.plot(frac.index, frac.values, '-o', color=COLOR_LOOKUP[spec['name']],
                lw=2.5, ms=5, label=spec['name'])
    # Shared axes: x label/ticks only on the bottom row, y label/ticks only on the left column
    if i >= 2:
        ax.set_xlabel('Number of triggers')
    else:
        ax.set_xticklabels([])
    if i % 2 == 0:
        ax.set_ylabel('Fraction of events inside region')
    else:
        ax.set_yticklabels([])
    ax.set_title(f'{p:.0%} credible region')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.03, f'({"abcd"[i]})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=13, fontweight='bold')
    if i == 0:
        ax.legend(loc='upper right', fontsize=10)
    if i == 3:
        ax.text(0.98, 0.04, 'dashed: perfect calibration', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'confidence_vs_triggers.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'confidence_vs_triggers.pdf'), bbox_inches='tight')
plt.show()

# %%
