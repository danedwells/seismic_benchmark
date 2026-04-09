"""
benchmark/plots.py — reusable figure helpers for prior comparison plots.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_prior_histograms(
    prior_names,
    output_dir,
    column,
    bins,
    title,
    xlabel,
    save_path,
    filter_fn=None,
    catalog_df=None,
    color='crimson',
    n_cols=3,
):
    """
    Plot a grid of per-prior histograms and save to disk.

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior names; one panel per entry.  The grid is
        ``ceil(len / n_cols)`` rows × ``n_cols`` columns.
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv``.
    column : str
        Column to histogram.  If ``'location_error_km'`` and that column is
        absent from the CSV, it is computed on the fly from ``catalog_df``
        via ``compute_location_error()``.
    bins : array-like
        Bin edges passed directly to ``ax.hist``.
    title : str
        Figure suptitle.
    xlabel : str
        x-axis label for every panel.
    save_path : str
        Full path for the saved PNG (written at 150 dpi).
    filter_fn : callable(df) -> df, optional
        Applied to the per-event final-trigger DataFrame before extracting
        the column.  Useful for spatial subsets (e.g. ``in_extent``).
    catalog_df : DataFrame, optional
        Reference catalog with ``event_id``, ``usgs_lat``, ``usgs_lon``
        columns.  Required when ``column='location_error_km'``.
    color : str
        Bar fill color (default ``'crimson'``).
    n_cols : int
        Number of columns in the subplot grid (default 3).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle for further customisation before display or
        additional saving.
    """
    from .runner import compute_location_error  # local import avoids top-level circularity

    n_priors = len(prior_names)
    n_rows   = math.ceil(n_priors / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 16 / 3, n_rows * 4),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    for ax, prior_name in zip(axes_flat, prior_names):
        csv_path = os.path.join(output_dir,
                                f'{prior_name.lower()}_benchmark_results.csv')

        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(prior_name, fontsize=11)
            ax.set_xlabel(xlabel, fontsize=9)
            continue

        df    = pd.read_csv(csv_path)
        final = df.groupby('event_id').last().reset_index()

        if filter_fn is not None:
            final = filter_fn(final)

        if column == 'location_error_km' and column not in final.columns:
            if catalog_df is None:
                ax.text(0.5, 0.5, 'no catalog', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                ax.set_title(prior_name, fontsize=11)
                ax.set_xlabel(xlabel, fontsize=9)
                continue
            final = compute_location_error(final, catalog_df)

        values = final[column].dropna()
        if values.empty:
            ax.text(0.5, 0.5, 'no data\nafter filter', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        else:
            ax.hist(values, bins=bins, color=color, alpha=0.6)
            ax.axvline(values.median(), color='black', linestyle='--',
                       linewidth=1, label=f'median {values.median():.2g}')
            ax.legend(fontsize=7)

        ax.set_title(prior_name, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)

    # Hide any unused panels (when n_priors < n_rows * n_cols)
    for ax in axes_flat[n_priors:]:
        ax.set_visible(False)

    # y-axis labels on leftmost column only
    for row_i in range(n_rows):
        axes_flat[row_i * n_cols].set_ylabel('count', fontsize=9)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

    return fig
