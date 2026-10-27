#%%
# =============================================================================
# prior_error_hist.py — location-error histograms for the California benchmark.
# =============================================================================
# prior_error_hist:            every static prior (everything except ETAS), for dismissing
#                              them early in the results section. Labeled, with medians.
# prior_error_hist_conclusion: the same five plus ETAS as a 6th, bottom panel, with no text or
#                              x ticks (pure visual for the conclusion slide); median lines kept.
# One row per prior, stacked with a shared log x-axis (same style as
# location_hist_case_study.py). Errors are MAP location errors at TRIGGER_NUMBER triggers.
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from benchmark.metrics import load_final_values

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

MAX_TRIGS      = 10
TRIGGER_NUMBER = 4
OUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output')

# name -> (color, results csv). KDE_Seismicity, Uniform and ETAS use the same colors as the other figures.
STATIC = {'Gear1': 'tab:green', 'NSHM': 'tab:orange', 'Helmstetter': 'tab:purple',
          'KDE_Seismicity': 'darkblue', 'Uniform': 'gray'}
PRIORS = {name: (color, os.path.join(OUT_DIR, 'time_independent', f'max_trigs_{MAX_TRIGS}',
                                     f'{name.lower()}_benchmark_results.csv'))
          for name, color in STATIC.items()}
PRIORS['ETAS'] = ('red', os.path.join(OUT_DIR, 'time_dependent', f'max_trigs_{MAX_TRIGS}',
                                      'etas_dynamic_benchmark_results.csv'))

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.labelsize': 12, 'figure.dpi': 150})
bins = np.logspace(-1, 3, 20)


def plot_error_hists(names, out_name, labeled=True):
    fig, axes = plt.subplots(len(names), 1, figsize=(6.5, 1.6 * len(names)), sharex=True,
                             gridspec_kw={'hspace': 0})
    for ax, name in zip(axes, names):
        color, csv = PRIORS[name]
        stats = load_final_values(csv, 'map_err_km', n_trigs=TRIGGER_NUMBER)
        ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7)
        median = np.median(stats)
        ax.axvline(median, color=color, ls='--', lw=1.5)
        if labeled:
            ax.text(0.03, 0.92, name, transform=ax.transAxes, ha='left', va='top',
                    fontsize=13, color=color, fontweight='bold')
            ax.text(0.03, 0.66, f'median: {median:.1f} km', transform=ax.transAxes, ha='left', va='top',
                    fontsize=11, color=color, fontweight='bold')
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        print(f'{name}: {len(stats)} events, median {median:.1f} km')

    # Same vertical scale in every row so bar heights are comparable
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)
    axes[-1].set_xscale('log')
    if labeled:
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False, bottom=False)
        axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        axes[-1].set_xlabel('Location error (km)')
    else:
        for ax in axes:
            ax.tick_params(which='both', bottom=False, labelbottom=False)

    fig.savefig(os.path.join(OUTPUT_DIR, f'{out_name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{out_name}.pdf'), bbox_inches='tight')
    plt.show()


plot_error_hists(list(STATIC), 'prior_error_hist')
plot_error_hists(list(PRIORS), 'prior_error_hist_conclusion', labeled=False)
