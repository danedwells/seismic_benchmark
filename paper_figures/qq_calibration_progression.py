#%%
# =============================================================================
# qq_calibration_progression.py — paper figure: cumulative calibration story
# =============================================================================
# Q-Q calibration plot (posterior_confidence_level vs U(0,1)) showing the
# cumulative effect of the work done in this repo, as a sequence of stages
# rather than just the current end-state:
#
#   1. Naive                 — Uniform prior, bEPIC's default sigma_s=1.0
#   2. + sigma_s calibration — Uniform prior, tuned sigma_s
#   3. + spatial prior       — KDE_Seismicity prior, tuned sigma_s
#   4. + dynamic ETAS prior  — ETAS (dynamic), tuned sigma_s
#
# Two panels, same story on two different event sets:
#   (a) California — statewide benchmark catalog, tuned sigma_s=0.35
#   (b) Ridgecrest — case study, tuned sigma_s=0.22 (case studies were
#       calibrated separately from the statewide catalog; see
#       time_dependent_scripts/case_studies.py's hardcoded SIGMA_S=0.22).
#       Just one representative case study here — the full per-case-study
#       breakdown belongs in a table, not this figure.
#
# For a perfectly calibrated posterior, posterior_confidence_level (the
# smallest HDR fraction needed to contain the USGS true location) is
# Uniform(0,1) across events, so points falling on the diagonal indicate good
# calibration. Below the diagonal = overconfident (too narrow/peaked);
# above = underconfident (too diffuse). See benchmark/plots.py's
# plot_qq_calibration() for the same diagnostic applied to a single run.
#
# sigma_s=1.0 is bEPIC's own package default (EPIC_locate_prelim.py:
# getattr(params, 'sigma_s', 1.)). Calibrating sigma_s is a likelihood-only
# question (posterior ∝ likelihood under the Uniform prior), which is why
# data_examination_scripts/sigma_calibration.py only ever sweeps the Uniform
# prior — stages 3-4 use the other priors' main (non-swept) results at the
# already-tuned sigma_s.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import matplotlib.pyplot as plt

from benchmark.metrics import load_final_values

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

#%%
# ── Stages, one list per panel ───────────────────────────────────────────
CALI_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output')
RIDGECREST_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies',
                               'Ridgecrest', 'output')


def _stages(base_dir, sigma_calibrated):
    return [
        {'label': 'Naive (Uniform, σₛ=1.0)',
         'csv': os.path.join(base_dir, 'time_independent', 'sig_1.0', 'max_trigs_10',
                              'uniform_benchmark_results.csv')},
        {'label': f'+ σₛ calibration (Uniform, σₛ={sigma_calibrated:g})',
         'csv': os.path.join(base_dir, 'time_independent', f'sig_{sigma_calibrated:g}', 'max_trigs_10',
                              'uniform_benchmark_results.csv')},
        {'label': '+ spatial prior (KDE Seismicity)',
         'csv': os.path.join(base_dir, 'time_independent', f'sig_{sigma_calibrated:g}', 'max_trigs_10',
                              'kde_seismicity_benchmark_results.csv')},
        {'label': '+ dynamic ETAS prior',
         'csv': os.path.join(base_dir, 'time_dependent', f'sig_{sigma_calibrated:g}', 'max_trigs_10',
                              'etas_dynamic_benchmark_results.csv')},
    ]


PANELS = [
    {'title': 'California', 'stages': _stages(CALI_DIR, 0.35)},
    {'title': 'Ridgecrest case study', 'stages': _stages(RIDGECREST_DIR, 0.22)},
]

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact panel placement [left, bottom, width, height] (figure fraction).
# Edit directly to hand-tune the layout.
AX_POS = {
    'panel_a': [0.08, 0.10, 0.40, 0.78],
    'panel_b': [0.56, 0.10, 0.40, 0.78],
}

fig = plt.figure(figsize=(14, 7))


def _plot_calibration_panel(ax, stages, title, letter):
    # Sequential (not categorical) colormap — these are cumulative steps,
    # not independent categories, so color communicates progression toward
    # the final (brightest) stage.
    colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
    for stage, color in zip(stages, colors):
        vals = load_final_values(stage['csv'], 'posterior_confidence_level')
        if vals is None:
            print(f"  [{title} / {stage['label']}] no data — skipping")
            continue
        vals = np.sort(vals)
        n = len(vals)
        theoretical = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(theoretical, vals, color=color, linewidth=2.5, label=f"{stage['label']}  (n={n})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='ideal calibration')
    ax.set_xlabel('Theoretical quantile  U(0,1)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(title)
    ax.text(0.02, 0.98, f'({letter})', transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold')


ax_a = fig.add_axes(AX_POS['panel_a'])
ax_b = fig.add_axes(AX_POS['panel_b'])
_plot_calibration_panel(ax_a, PANELS[0]['stages'], PANELS[0]['title'], 'a')
_plot_calibration_panel(ax_b, PANELS[1]['stages'], PANELS[1]['title'], 'b')
ax_a.set_ylabel('Empirical posterior confidence level')
ax_b.set_yticklabels([])

fig.savefig(os.path.join(OUTPUT_DIR, 'qq_calibration_progression.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'qq_calibration_progression.pdf'), bbox_inches='tight')
plt.show()

# %%
