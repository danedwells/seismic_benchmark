"""
benchmark/plots.py — reusable figure helpers for prior comparison plots.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import obspy

from .metrics import load_per_version_stats
from .metrics import load_final_values
from .metrics import load_final_rows
from .metrics import COVERAGE_RADII_KM
# Plots

def plot_median_vs_triggers(metric, ylabel, title, save_path=None,
                            ylim=None, ref_line=None, ref_label=None,
                            PRIOR_SPECS=None, log_y=False,
                            shade_groups=('mixed', 'dynamic')):
    """
    Plot median metric vs trigger count for all priors, with 5–95 % band.

    Mixed priors: solid lines.
    ETAS dynamic: dashed black reference.
    TI baselines (if INCLUDE_BASELINES): dotted lines, same color palette.
    """
    # Assign consistent colors: mixed and their TI counterparts share a color.
    colors = plt.cm.tab10.colors
    # Build color index: mixed priors get indices 0..4; TI baselines reuse same indices.
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

        if spec['group'] in shade_groups:
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

def plot_mean_vs_triggers(metric, ylabel, title, save_path=None,
                            ylim=None, ref_line=None, ref_label=None,
                            PRIOR_SPECS=None, log_y=False,
                            shade_groups=None):
    """
    Plot median metric vs trigger count for all priors, with IQR shading.

    Parameters
    ----------
    metric : str
        Column name in the benchmark CSVs.
    ylabel : str
        Y-axis label (include direction hint, e.g. '↓ better').
    title : str
        Figure title.
    save_path : str or None
    ylim : tuple or None
        (ymin, ymax) passed to ax.set_ylim.  None = matplotlib auto.
    ref_line : float or None
        If given, draws a horizontal dashed reference line at this value.
    ref_label : str or None
        Legend label for the reference line.
    """
    # Assign consistent colors: mixed and their TI counterparts share a color.
    colors = plt.cm.tab10.colors
    # Build color index: mixed priors get indices 0..4; TI baselines reuse same indices.
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
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, spec in enumerate(PRIOR_SPECS):
        stats = load_per_version_stats(spec['csv'], metric)
        if stats is None:
            print(f"  [{spec['name']}] no data for '{metric}' — skipping")
            continue

        color = color_lookup.get(spec['name'], colors[i % len(colors)])
        n_max = int(stats['count'].max())
        ax.plot(stats['n_trigs'], stats['mean'],
                color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                label=f"{spec['name']}  (n≈{n_max})")
        if shade_groups is None or spec['group'] in shade_groups:
            ax.fill_between(stats['n_trigs'], stats['q5'], stats['q95'],
                            color=color, alpha=0.06)
            ax.plot(stats['n_trigs'], stats['q5'],
                    color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.5)
            ax.plot(stats['n_trigs'], stats['q95'],
                    color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.5)

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
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig

def plot_median_posterior_coverage(
    PRIOR_SPECS,
    title,
    save_path=None,
    legend_ncol=1,
    shade_groups=('mixed', 'dynamic'),
):
    """
    2×2 panel: median posterior coverage vs trigger count for each of COVERAGE_RADII_KM.

    Parameters
    ----------
    PRIOR_SPECS : list[dict]
        Each dict must have: name, csv, ls, lw, group.
        group in {'mixed', 'static', 'dynamic'}.
    title : str
        Figure suptitle.
    save_path : str or None
    legend_ncol : int
        Number of columns in each subplot legend (default 1).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = plt.cm.tab10.colors
    color_lookup = {}

    mixed_specs  = [s for s in PRIOR_SPECS if s['group'] == 'mixed']
    static_specs = [s for s in PRIOR_SPECS if s['group'] == 'static']

    if mixed_specs:
        for i, s in enumerate(mixed_specs):
            color_lookup[s['name']] = colors[i % len(colors)]
        for s in static_specs:
            mixed_name = f"{s['name']}+ETAS"
            color_lookup[s['name']] = color_lookup.get(mixed_name, 'gray')
    else:
        for i, s in enumerate(static_specs):
            color_lookup[s['name']] = colors[i % len(colors)]

    for s in PRIOR_SPECS:
        if s['group'] == 'dynamic':
            color_lookup[s['name']] = 'black'

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for ax, radius_km in zip(axes.flatten(), COVERAGE_RADII_KM):
        col = f'coverage_{radius_km}km'
        for spec in PRIOR_SPECS:
            stats = load_per_version_stats(spec['csv'], col)
            if stats is None:
                print(f"  [{spec['name']}] no data for '{col}' — skipping")
                continue
            color = color_lookup.get(spec['name'], 'gray')
            n_max = int(stats['count'].max())
            ax.plot(stats['n_trigs'], stats['median'],
                    color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                    label=f"{spec['name']}  (n≈{n_max})")
            if spec['group'] in shade_groups:
                ax.fill_between(stats['n_trigs'], stats['q5'], stats['q95'],
                                color=color, alpha=0.10)
                ax.plot(stats['n_trigs'], stats['q5'],
                        color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)
                ax.plot(stats['n_trigs'], stats['q95'],
                        color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)
        ax.set_xlabel('Number of triggers', fontsize=10)
        ax.set_ylabel('Median coverage  (↑ better)', fontsize=10)
        ax.set_title(f'Within {radius_km} km', fontsize=11)
        ax.set_xlim(left=1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='lower right', ncol=legend_ncol)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig

def plot_mean_posterior_coverage(
    PRIOR_SPECS,
    title,
    save_path=None,
    legend_ncol=1,
    shade_groups=('mixed', 'dynamic'),
):
    """
    2×2 panel: median posterior coverage vs trigger count for each of COVERAGE_RADII_KM.

    Parameters
    ----------
    PRIOR_SPECS : list[dict]
        Each dict must have: name, csv, ls, lw, group.
        group in {'mixed', 'static', 'dynamic'}.
    title : str
        Figure suptitle.
    save_path : str or None
    legend_ncol : int
        Number of columns in each subplot legend (default 1).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = plt.cm.tab10.colors
    color_lookup = {}

    mixed_specs  = [s for s in PRIOR_SPECS if s['group'] == 'mixed']
    static_specs = [s for s in PRIOR_SPECS if s['group'] == 'static']

    if mixed_specs:
        for i, s in enumerate(mixed_specs):
            color_lookup[s['name']] = colors[i % len(colors)]
        for s in static_specs:
            mixed_name = f"{s['name']}+ETAS"
            color_lookup[s['name']] = color_lookup.get(mixed_name, 'gray')
    else:
        for i, s in enumerate(static_specs):
            color_lookup[s['name']] = colors[i % len(colors)]

    for s in PRIOR_SPECS:
        if s['group'] == 'dynamic':
            color_lookup[s['name']] = 'black'

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for ax, radius_km in zip(axes.flatten(), COVERAGE_RADII_KM):
        col = f'coverage_{radius_km}km'
        for spec in PRIOR_SPECS:
            stats = load_per_version_stats(spec['csv'], col)
            if stats is None:
                print(f"  [{spec['name']}] no data for '{col}' — skipping")
                continue
            color = color_lookup.get(spec['name'], 'gray')
            n_max = int(stats['count'].max())
            ax.plot(stats['n_trigs'], stats['mean'],
                    color=color, linestyle=spec['ls'], linewidth=spec['lw'],
                    label=f"{spec['name']}  (n≈{n_max})")
            if spec['group'] in shade_groups:
                ax.fill_between(stats['n_trigs'], stats['q5'], stats['q95'],
                                color=color, alpha=0.10)
                ax.plot(stats['n_trigs'], stats['q5'],
                        color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)
                ax.plot(stats['n_trigs'], stats['q95'],
                        color=color, linestyle=spec['ls'], linewidth=0.6, alpha=0.6)
        ax.set_xlabel('Number of triggers', fontsize=10)
        ax.set_ylabel('Median coverage  (↑ better)', fontsize=10)
        ax.set_title(f'Within {radius_km} km', fontsize=11)
        ax.set_xlim(left=1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='lower right', ncol=legend_ncol)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig

def plot_prior_histograms(
    prior_names,
    output_dir,
    column,
    bins,
    title,
    xlabel,
    save_path,
    filter_fn=None,
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
        Column to histogram (e.g. ``'map_err_km'``).
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

        values = final[column].dropna() if column in final.columns else pd.Series(dtype=float)
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


def plot_overview_map(
    output_dir,
    prior_order,
    extent,
    events_df=None,
    stations_df=None,
    bg=None,
    title='bEPIC final locations — prior comparison',
    save_path=None,
):
    """
    Single-panel overview map of bEPIC final posterior locations for all priors.

    Parameters
    ----------
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv`` files.
    prior_order : list[str]
        Prior names; each gets a distinct tab10 color in the legend.
    extent : list
        [lon_min, lon_max, lat_min, lat_max] for the map.
    events_df : DataFrame or None
        Events shown as black dots; must have ``longitude`` and ``latitude`` columns.
    stations_df : DataFrame or None
        Stations shown as orange triangles; must have ``longitude`` and ``latitude``.
    bg : DataFrame or None
        Background seismicity shown as gray dots; must have ``longitude`` and ``latitude``.
    title : str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    proj   = ccrs.PlateCarree()
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray',   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightyellow', zorder=0)

    if bg is not None:
        ax.scatter(bg['longitude'], bg['latitude'],
                   s=8, c='gray', alpha=0.1, transform=proj, zorder=0, linewidths=0)

    if events_df is not None:
        ax.scatter(events_df['longitude'], events_df['latitude'],
                   s=10, color='black', alpha=0.4, transform=proj,
                   label='ANSS catalog', zorder=1)

    if stations_df is not None:
        ax.scatter(stations_df['longitude'], stations_df['latitude'],
                   s=20, color='orange', edgecolor='k', alpha=0.7, marker='v',
                   transform=proj, label='Stations', zorder=2)

    for i, prior_name in enumerate(prior_order):
        csv_path = os.path.join(output_dir, f'{prior_name.lower()}_benchmark_results.csv')
        if not os.path.exists(csv_path):
            continue
        df    = pd.read_csv(csv_path)
        final = df.groupby('event_id').last().reset_index()
        ax.scatter(final['posterior_lon'], final['posterior_lat'],
                   s=8, color=colors[i % len(colors)], alpha=0.2,
                   transform=proj, label=prior_name, zorder=3)

    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_location_grid(
    output_dir,
    prior_order,
    extent,
    ref_catalog=None,
    events_df=None,
    stations_df=None,
    bg=None,
    cache_paths=None,
    filter_fn=None,
    show_scale_bar=False,
    title='bEPIC locations — prior comparison',
    save_path=None,
):
    """
    2×3 panel map of bEPIC posterior locations, one panel per prior.

    Parameters
    ----------
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv`` files.
    prior_order : list[str]
        Prior names in panel order (≤6 entries; unused 2×3 panels are hidden).
    extent : list
        [lon_min, lon_max, lat_min, lat_max] for every panel.
    ref_catalog : DataFrame or None
        Reference catalog used to draw USGS→posterior error lines.
        Must have columns ``event_id``, ``usgs_lon``, ``usgs_lat``.
    events_df : DataFrame or None
        Events shown as black dots in every panel; needs ``longitude``, ``latitude``.
    stations_df : DataFrame or None
        Stations shown as orange triangles; needs ``longitude``, ``latitude``.
    bg : DataFrame or None
        Background seismicity gray dots; needs ``longitude``, ``latitude``.
    cache_paths : dict or None
        Mapping of prior name → .tt3 path.  When provided, a log₁₀ prior density
        pcolormesh is drawn behind each panel's results.
    filter_fn : callable(df) -> df, optional
        Applied to the per-prior final-trigger DataFrame before plotting posteriors
        and drawing error lines (e.g. a spatial subset like ``in_extent``).
    show_scale_bar : bool
        If True, draws a 100 km scale bar on the bottom-left panel.
    title : str
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    proj  = ccrs.PlateCarree()
    n     = len(prior_order)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 16 / 3, nrows * 5.0),
                             subplot_kw={'projection': proj},
                             squeeze=False)

    for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), prior_order)):
        row_idx, col_idx = divmod(idx, ncols)
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.STATES,    linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax.add_feature(cfeature.LAND,      facecolor='lightgray',   zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor='lightyellow', zorder=0)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = (col_idx == 0)
        gl.bottom_labels = (row_idx == nrows - 1)

        # Optional prior density pcolormesh background
        if cache_paths is not None:
            from priors import SeismicPrior
            pcache = cache_paths.get(prior_name)
            if pcache is not None and os.path.exists(pcache):
                sp = SeismicPrior.from_tt3(pcache)
                ax.pcolormesh(sp.lons, sp.lats, np.log10(sp.grid + 1e-12),
                              transform=proj, cmap='viridis', alpha=0.4,
                              shading='auto', zorder=1)

        if bg is not None:
            ax.scatter(bg['longitude'], bg['latitude'],
                       s=8, c='gray', alpha=0.1, transform=proj, zorder=2, linewidths=0)

        if events_df is not None:
            ax.scatter(events_df['longitude'], events_df['latitude'],
                       s=14, color='black', alpha=0.4, transform=proj,
                       label='ANSS catalog', zorder=3)

        if stations_df is not None:
            ax.scatter(stations_df['longitude'], stations_df['latitude'],
                       s=40, color='orange', edgecolor='k', alpha=0.85, marker='v',
                       transform=proj, label='Stations', zorder=4)

        csv_path = os.path.join(output_dir, f'{prior_name.lower()}_benchmark_results.csv')
        if os.path.exists(csv_path):
            df    = pd.read_csv(csv_path)
            final = df.groupby('event_id').last().reset_index()
            if filter_fn is not None:
                final = filter_fn(final)

            # Error lines: USGS → posterior
            if ref_catalog is not None and not final.empty:
                matched = final.merge(
                    ref_catalog[['event_id', 'usgs_lon', 'usgs_lat']],
                    on='event_id', how='inner',
                )
                if not matched.empty:
                    n = len(matched)
                    seg_lons = np.empty(n * 3)
                    seg_lats = np.empty(n * 3)
                    seg_lons[0::3] = matched['usgs_lon'].values
                    seg_lons[1::3] = matched['posterior_lon'].values
                    seg_lons[2::3] = np.nan
                    seg_lats[0::3] = matched['usgs_lat'].values
                    seg_lats[1::3] = matched['posterior_lat'].values
                    seg_lats[2::3] = np.nan
                    ax.plot(seg_lons, seg_lats, color='black', linewidth=0.5,
                            alpha=0.35, transform=proj, zorder=5)

            ax.scatter(final['posterior_lon'], final['posterior_lat'],
                       s=16, color='crimson', alpha=0.5, transform=proj,
                       label=prior_name, zorder=6)
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')

        ax.set_title(prior_name, fontsize=11)
        ax.legend(loc='upper right', fontsize=7)

    # Optional 100 km scale bar on bottom-left panel
    if show_scale_bar:
        lon_min, lon_max, lat_min, lat_max = extent
        ax_sb    = axes[nrows - 1, 0]
        lat_mid  = (lat_min + lat_max) / 2
        scale_km = 100
        scale_deg = scale_km / (111.32 * np.cos(np.radians(lat_mid)))
        x0 = lon_min + 0.25
        y0 = lat_min + 0.25
        x1 = x0 + scale_deg
        tick_h = 0.06
        for xs in ([x0, x1], [x0, x0], [x1, x1]):
            ys = [y0, y0] if xs == [x0, x1] else [y0 - tick_h, y0 + tick_h]
            ax_sb.plot(xs, ys, color='black', linewidth=2,
                       transform=proj, zorder=10, solid_capstyle='butt')
        ax_sb.text((x0 + x1) / 2, y0 + tick_h + 0.04, f'{scale_km} km',
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   transform=proj, zorder=10)

    for ax in axes.flatten()[len(prior_order):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_posterior_grid(
    focus_run_path,
    cache_paths,
    prior_order,
    params_kw,
    ref_lat=None,
    ref_lon=None,
    focus_version=None,
    extent_pad_deg=0.4,
    extent=None,
    title='bEPIC posterior grid',
    save_path=None,
    prior_results=None,
):
    """
    2×3 panel figure showing the prior density background and bEPIC posterior
    contours for a single event, one panel per prior.

    Parameters
    ----------
    focus_run_path : str
        Path to the .run trigger file for the target event.
    cache_paths : dict
        Mapping of prior name → .tt3 file path (or None for Uniform).
    prior_order : list[str]
        Prior names in panel order (≤6 entries; unused 2×3 panels are hidden).
    params_kw : dict
        Keys: grid_size, grid_km, max_trigs — passed to run_single_event_get_grid.
        Ignored when prior_results is provided.
    ref_lat, ref_lon : float or None
        USGS reference location plotted as a gold star.  Omitted if None.
    focus_version : int or None
        Trigger version to plot (0-based).  None = last available version.
        Ignored when prior_results is provided.
    extent_pad_deg : float
        Degrees of padding added around the posterior grid extent (ignored when
        `extent` is provided).
    extent : list[float] or None
        ``[min_lon, max_lon, min_lat, max_lat]`` — overrides the auto-derived
        extent.  Useful for zooming in on a specific region.
    title : str
        Figure suptitle.
    save_path : str or None
        If given, the figure is saved as a PNG at 150 dpi.
    prior_results : dict or None
        Pre-computed results keyed by prior name:
        ``{pname: (t, odf, actual_v, sp, pcache)}``.
        When provided, skips the internal run_single_event_get_grid calls so
        the caller can reuse a grid already computed for coverage metrics.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from priors import SeismicPrior
    from .runner import run_single_event_get_grid

    grid_width = 2 * params_kw['grid_size'] + 1
    proj = ccrs.PlateCarree()

    # -- Run all priors for this event (skip if pre-computed) ------------------
    if prior_results is None:
        prior_results = {}
        for pname in prior_order:
            pcache = cache_paths.get(pname)
            if pcache is not None and os.path.exists(pcache):
                sp       = SeismicPrior.from_tt3(pcache)
                use_p    = True
            else:
                fallback = next((v for v in cache_paths.values() if v is not None), None)
                if fallback is None:
                    print(f'  [{pname}] no prior file available — skipping panel.')
                    continue
                sp    = SeismicPrior.from_tt3(fallback)
                use_p = False
            t, odf, actual_v = run_single_event_get_grid(
                focus_run_path, sp, use_p, params_kw, focus_version=focus_version
            )
            prior_results[pname] = (t, odf, actual_v, sp, pcache)

    # -- Derive map extent from first valid posterior grid ---------------------
    first_odf = next((v[1] for v in prior_results.values() if v[1] is not None), None)
    if first_odf is None:
        print('[plot_posterior_grid] No valid posterior found — figure skipped.')
        return None
    ext = extent if extent is not None else [
        float(first_odf['lon'].min()) - extent_pad_deg,
        float(first_odf['lon'].max()) + extent_pad_deg,
        float(first_odf['lat'].min()) - extent_pad_deg,
        float(first_odf['lat'].max()) + extent_pad_deg,
    ]

    # -- Build figure ----------------------------------------------------------
    n = len(prior_order)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 16 / 3, nrows * 5.0),
                             subplot_kw={'projection': proj}, squeeze=False)

    for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), prior_order)):
        row_idx, col_idx = divmod(idx, ncols)
        t, odf, actual_v, sp, pcache = prior_results[prior_name]

        ax.set_extent(ext, crs=proj)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = (col_idx == 0)
        gl.bottom_labels = (row_idx == nrows - 1)

        # Prior density background (viridis pcolormesh).
        # vmin is anchored to the mean log10 value so the colormap reflects
        # absolute probability density — a flat (tempered) prior will look
        # visually uniform rather than being stretched to fill the palette.
        if pcache is not None and os.path.exists(pcache):
            log_grid = np.log10(sp.grid + 1e-12)
            ax.pcolormesh(sp.lons, sp.lats, log_grid,
                          transform=proj, cmap='viridis', alpha=0.5,
                          shading='auto', zorder=1,
                          vmin=np.nanmean(log_grid), vmax=np.nanmax(log_grid))

        # Posterior contours (Reds)
        if odf is not None:
            post_2d = odf['post'].values.reshape(grid_width, grid_width)
            lats_2d = odf['lat'].values.reshape(grid_width, grid_width)
            lons_2d = odf['lon'].values.reshape(grid_width, grid_width)
            pmax = post_2d.max()
            if pmax > 0:
                ax.contour(lons_2d, lats_2d, post_2d / pmax,
                           levels=np.linspace(0.1, 1.0, 10),
                           cmap='Reds', transform=proj, zorder=4, linewidths=1.0)

        # Posterior MAP location (red star)
        if t is not None:
            ax.scatter(t.posterior_lon, t.posterior_lat,
                       s=100, color='red', marker='*', edgecolors='darkred',
                       linewidths=0.5, transform=proj, zorder=5, label='bEPIC MAP')

        # USGS reference location (gold star)
        if ref_lat is not None and ref_lon is not None:
            ax.scatter(ref_lon, ref_lat,
                       s=120, color='gold', marker='*', edgecolors='black',
                       linewidths=0.6, transform=proj, zorder=6, label='USGS')

        v_label = f'v{actual_v}' if t is not None else 'no data'
        ax.set_title(f'{prior_name}  ({v_label})', fontsize=11)
        ax.legend(loc='upper right', fontsize=7)

    for ax in axes.flatten()[len(prior_order):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_location_trajectory(
    event_id,
    output_dir,
    prior_order,
    run_dir,
    min_triggers=4,
    ref_lat=None,
    ref_lon=None,
    cache_paths=None,
    extent = None,
    extent_pad_deg=0.5,
    title='bEPIC location trajectory',
    save_path=None,
):
    """
    2×3 panel figure showing how bEPIC's MAP location evolves with trigger count
    for a single event, one panel per prior.

    Reads trigger counts from the event's .run file and location estimates from
    the per-prior benchmark CSVs already written to output_dir — no re-running
    of bEPIC required.

    Parameters
    ----------
    event_id : int or str
        Event ID matching the .run filename stem and the event_id column in CSVs.
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv`` files.
    prior_order : list[str]
        Prior names in panel order (≤6 entries; unused 2×3 panels are hidden).
    run_dir : str
        Directory containing ``{event_id}.run`` files; used to map version → trigger count.
    min_triggers : int
        First trigger count to plot (default 4).
    ref_lat, ref_lon : float or None
        USGS reference location plotted as a gold star.
    cache_paths : dict or None
        Mapping of prior name → .tt3 path.  When provided, a log₁₀ prior density
        pcolormesh (viridis, alpha 0.4) is drawn behind each panel's trajectory.
    extent_pad_deg : float
        Degrees of padding around the trajectory bounding box.
    title : str
        Figure suptitle.
    save_path : str or None
        If given, the figure is saved as a PNG at 150 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    proj = ccrs.PlateCarree()

    # -- Map version → trigger count from the .run file -----------------------
    run_path = os.path.join(run_dir, f'{event_id}.run')
    if os.path.exists(run_path):
        df_run = pd.read_csv(run_path)
        df_run.columns = [c.replace(' ', '_') for c in df_run.columns]
        version_to_ntrigs = df_run.groupby('version').size().to_dict()
    else:
        print(f'[plot_location_trajectory] .run file not found: {run_path}')
        print('  → falling back to version + 1 as trigger count estimate')
        version_to_ntrigs = None

    # -- Load per-prior trajectories ------------------------------------------
    trajectories = {}
    all_lons, all_lats = [], []

    for prior_name in prior_order:
        csv_path = os.path.join(output_dir, f'{prior_name.lower()}_benchmark_results.csv')
        if not os.path.exists(csv_path):
            trajectories[prior_name] = None
            continue

        df = pd.read_csv(csv_path)
        df_event = df[df['event_id'] == event_id].copy()

        if df_event.empty:
            trajectories[prior_name] = None
            continue

        if version_to_ntrigs is not None:
            df_event['n_trigs'] = df_event['version'].map(version_to_ntrigs)
        else:
            df_event['n_trigs'] = df_event['version'] + 1

        df_event = (df_event[df_event['n_trigs'] >= min_triggers]
                    .sort_values('n_trigs')
                    .reset_index(drop=True))

        if df_event.empty:
            trajectories[prior_name] = None
            continue

        trajectories[prior_name] = df_event
        all_lons.extend(df_event['posterior_lon'].tolist())
        all_lats.extend(df_event['posterior_lat'].tolist())

    if ref_lat is not None:
        all_lats.append(ref_lat)
    if ref_lon is not None:
        all_lons.append(ref_lon)

    if not all_lons:
        print('[plot_location_trajectory] No trajectory data found — skipping.')
        return None

    # Enforce a minimum map span of 0.5° so nearby points aren't over-zoomed
    lon_span = max(max(all_lons) - min(all_lons), 0.5)
    lat_span = max(max(all_lats) - min(all_lats), 0.5)
    lon_mid  = (max(all_lons) + min(all_lons)) / 2
    lat_mid  = (max(all_lats) + min(all_lats)) / 2
    
    ext = extent if extent is not None else [
        lon_mid - lon_span / 2 - extent_pad_deg,
        lon_mid + lon_span / 2 + extent_pad_deg,
        lat_mid - lat_span / 2 - extent_pad_deg,
        lat_mid + lat_span / 2 + extent_pad_deg,
    ]

    # Colormap: sequential by trigger count, shared across all panels
    all_n = np.concatenate([df['n_trigs'].values for df in trajectories.values()
                            if df is not None])
    vmin, vmax = int(all_n.min()), int(all_n.max())
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    n = len(prior_order)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 16 / 3 + 1.5, nrows * 5.0),
                             subplot_kw={'projection': proj}, squeeze=False)

    for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), prior_order)):
        row_idx, col_idx = divmod(idx, ncols)
        df_traj = trajectories.get(prior_name)

        ax.set_extent(ext, crs=proj)
        ax.add_feature(cfeature.STATES,    linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = (col_idx == 0)
        gl.bottom_labels = (row_idx == nrows - 1)

        # Optional prior density background
        if cache_paths is not None:
            from priors import SeismicPrior
            pcache = cache_paths.get(prior_name)
            if pcache is not None and os.path.exists(pcache):
                sp = SeismicPrior.from_tt3(pcache)
                ax.pcolormesh(sp.lons, sp.lats, np.log10(sp.grid + 1e-12),
                              transform=proj, cmap='viridis', alpha=0.4,
                              shading='auto', zorder=1)

        if df_traj is not None:
            lons   = df_traj['posterior_lon'].values
            lats   = df_traj['posterior_lat'].values
            ntrigs = df_traj['n_trigs'].values

            # Connecting path
            ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.6,
                    transform=proj, zorder=2)

            # Intermediate locations colored by trigger count
            if len(lons) > 1:
                ax.scatter(lons[:-1], lats[:-1],
                           c=ntrigs[:-1], cmap=cmap, norm=norm,
                           s=50, zorder=3, transform=proj,
                           edgecolors='white', linewidths=0.4)

            # Trigger count labels on intermediate points
            outline = [pe.withStroke(linewidth=2, foreground='white')]
            for lon, lat, n in zip(lons[:-1], lats[:-1], ntrigs[:-1]):
                ax.text(lon, lat, str(int(n)), fontsize=6.5, ha='center', va='bottom',
                        transform=proj, zorder=4,
                        path_effects=outline)

            # Final location — red star
            ax.scatter(lons[-1], lats[-1],
                       s=150, color='red', marker='*', edgecolors='darkred',
                       linewidths=0.5, transform=proj, zorder=5,
                       label=f'Final (n={int(ntrigs[-1])})')
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')

        # USGS reference location — gold star
        if ref_lat is not None and ref_lon is not None:
            ax.scatter(ref_lon, ref_lat,
                       s=150, color='gold', marker='*', edgecolors='black',
                       linewidths=0.6, transform=proj, zorder=6, label='USGS')

        ax.set_title(prior_name, fontsize=11)
        ax.legend(loc='upper right', fontsize=7)

    for ax in axes.flatten()[len(prior_order):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05)

    # Shared colorbar anchored to a manually placed axis outside the panel grid
    cbar_ax = fig.add_axes([0.905, 0.15, 0.015, 0.68])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Number of triggers', fontsize=9)
    cbar.set_ticks(range(vmin, vmax + 1))
    cbar.ax.tick_params(labelsize=7)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_score_scatter(
    PRIOR_SPECS,
    title=None,
    save_path=None,
    loc_err_clip_km=200.0,
):
    """
    Scatter of log_score and brier_score vs location error (1×2 panel).

    Each point is one event at its final trigger version.  Priors are colored
    using the same tab10 scheme as the other PRIOR_SPECS-based plots.  Gray
    dashed/dotted lines mark the pooled medians, forming four quadrants.

    The key diagnostic quadrant is large error + good score (top-right for
    log_score, bottom-right for brier_score): the posterior is confident but
    spatially wrong, meaning the prior is misleading bEPIC.

    Uses ``map_err_km`` from the CSV as the x-axis (per-version location error
    already stored by the runner).  Priors whose CSVs are missing or whose
    score columns have not yet been populated (requires a re-run) are skipped
    with a printed notice.

    Parameters
    ----------
    PRIOR_SPECS : list[dict]
        Each dict must have ``name``, ``csv``, ``group`` keys, following the
        same convention as ``plot_median_vs_triggers``.
    title : str or None
    save_path : str or None
    loc_err_clip_km : float
        X-axis upper limit; events beyond this are dropped as outliers.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = plt.cm.tab10.colors

    # Shared color logic (mirrors plot_median_vs_triggers)
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

    _panels = [
        ('log_score',   'Log-score  (↑ better)'),
        ('brier_score', 'Brier score  (↓ better)'),
    ]

    if title is None:
        title = 'Scoring metrics vs location error — final trigger version'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (score_col, score_ylabel) in zip(axes, _panels):
        all_x, all_y = [], []

        for spec in PRIOR_SPECS:
            if not os.path.exists(spec['csv']):
                continue
            df    = pd.read_csv(spec['csv'])
            final = df.groupby('event_id').last().reset_index()

            err_col = 'map_err_km' if ('map_err_km' in final.columns and not final['map_err_km'].isna().all()) else None
            if err_col is None:
                print(f"  [{spec['name']}] no location error column — skipping")
                continue

            if score_col not in final.columns or final[score_col].isna().all():
                print(f"  [{spec['name']}] '{score_col}' absent — "
                      "re-run benchmark to populate scoring columns")
                continue

            sub = final[[err_col, score_col]].dropna()
            sub = sub[sub[err_col] <= loc_err_clip_km]
            if sub.empty:
                continue

            x = sub[err_col].values
            y = sub[score_col].values
            color = color_lookup.get(spec['name'], 'gray')
            ax.scatter(x, y, s=18, color=color, alpha=0.45,
                       label=spec['name'], edgecolors='none',
                       linestyle=spec.get('ls', '-'))
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())

        if all_x and all_y:
            med_x = float(np.median(all_x))
            med_y = float(np.median(all_y))
            ax.axvline(med_x, color='gray', linewidth=0.9, linestyle='--', alpha=0.55,
                       label=f'median error ({med_x:.0f} km)')
            ax.axhline(med_y, color='gray', linewidth=0.9, linestyle=':',  alpha=0.55,
                       label=f'median score ({med_y:.3g})')

        ax.set_xlabel('Location error (km)  (↓ better)', fontsize=11)
        ax.set_ylabel(score_ylabel, fontsize=11)
        ax.set_xlim(0, loc_err_clip_km)
        ax.legend(fontsize=8, loc='best', markerscale=1.4)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


def plot_coverage_panel(
    prior_names,
    output_dir,
    title,
    save_path=None,
    filter_fn=None,
    bins=None,
):
    """
    2×2 panel of posterior coverage histograms, one subplot per fixed radius.

    Each subplot overlays step histograms for every prior so distributions can
    be compared directly.  Panels correspond to COVERAGE_RADII_KM in reading
    order (top-left → top-right → bottom-left → bottom-right).

    Parameters
    ----------
    prior_names : list[str]
    output_dir : str
    title : str
    save_path : str or None
    filter_fn : callable(df) -> df, optional
    bins : array-like or None
        Defaults to np.linspace(0, 1, 41).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, radius_km in zip(axes.flatten(), COVERAGE_RADII_KM):
        col = f'coverage_{radius_km}km'
        for i, prior_name in enumerate(prior_names):
            csv_path = os.path.join(output_dir,
                                    f'{prior_name.lower()}_benchmark_results.csv')
            if not os.path.exists(csv_path):
                continue
            df    = pd.read_csv(csv_path)
            final = df.groupby('event_id').last().reset_index()
            if filter_fn is not None:
                final = filter_fn(final)
            if col not in final.columns:
                continue
            values = final[col].dropna()
            if values.empty:
                continue
            # Scale bin count to sample size so small case studies don't produce
            # sparse isolated bars.  Cap at 40 bins for large benchmarks.
            if bins is not None:
                plot_bins = bins
            else:
                n_bins = min(max(int(np.sqrt(len(values)) * 2), 5), 40)
                plot_bins = np.linspace(0, 1, n_bins + 1)
            ax.hist(values, bins=plot_bins,
                    color=colors[i % len(colors)], alpha=0.6,
                    histtype='step', linewidth=1.8,
                    label=f'{prior_name}  (med={values.median():.2f})')

        ax.set_title(f'Within {radius_km} km', fontsize=11)
        ax.set_xlabel('Posterior coverage fraction', fontsize=9)
        ax.set_ylabel('count', fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_qq_calibration(
    prior_names,
    output_dir,
    title='bEPIC posterior calibration — posterior_confidence_level vs U(0,1)',
    save_path=None,
    filter_fn=None,
    n_trigs=None,
    csv_paths=None,
):
    """
    Single-panel Q-Q calibration plot: posterior_confidence_level vs Uniform(0,1).

    Each prior is plotted as a separate colored line.  For a perfectly
    calibrated posterior, posterior_confidence_level is Uniform(0,1) across events
    and all lines fall on the diagonal.

    Interpretation of deviations (assuming roughly unimodal posteriors):
    - Below diagonal: posteriors are overconfident (too narrow/peaked) — USGS
      tends to fall in high-probability regions, requiring only a small HDR to
      contain it; the posterior concentrates mass near the true location more
      than a calibrated system would.
    - Above diagonal: posteriors are underconfident (too wide/diffuse) — USGS
      tends to fall in lower-probability regions; more mass is needed to
      include it in an HDR.

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior names; one line per entry.
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv``.
        Ignored for any prior_name present in ``csv_paths``.
    title : str
        Figure title.
    save_path : str or None
        Full path for the saved PNG (written at 150 dpi).
    filter_fn : callable(df) -> df, optional
        Applied to the per-event DataFrame before extracting
        posterior_confidence_level (e.g. a spatial subset like ``in_extent``).
    n_trigs : int or None
        posterior_confidence_level is computed for every trigger version, not
        just the final one — without filtering, events with more trigger
        versions would contribute more (correlated) points than events with
        fewer, and different trigger counts would be blended together in the
        same curve. None (default) takes each event's last available
        version; pass an int to instead use each event's row at that
        specific trigger count (events that never reached it are excluded).
    csv_paths : dict[str, str] or None
        Optional {prior_name: csv_path} overrides for priors whose results
        don't follow the ``output_dir/{name.lower()}_benchmark_results.csv``
        convention (e.g. dynamic ETAS runs living in a per-config subfolder).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(7, 6))

    for color, prior_name in zip(colors, prior_names):
        if csv_paths is not None and prior_name in csv_paths:
            csv_path = csv_paths[prior_name]
        else:
            csv_path = os.path.join(output_dir,
                                    f'{prior_name.lower()}_benchmark_results.csv')

        final = load_final_rows(csv_path, n_trigs=n_trigs)
        if final is None:
            continue
        final = final.dropna(subset=['posterior_confidence_level'])
        if filter_fn is not None:
            final = filter_fn(final)

        vals = np.sort(final['posterior_confidence_level'].values)
        n = len(vals)
        if n == 0:
            continue

        theoretical = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(theoretical, vals, color=color, linewidth=1.5,
                label=f'{prior_name}  (n={n})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='ideal')
    ax.set_xlabel('Theoretical quantile  U(0,1)', fontsize=11)
    ax.set_ylabel('Empirical posterior_confidence_level', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_qq_calibration_prior(
    prior_names,
    output_dir,
    title='bEPIC prior calibration — prior_confidence_level vs U(0,1)',
    save_path=None,
    filter_fn=None,
    n_trigs=None,
    csv_paths=None,
):
    """
    Single-panel Q-Q calibration plot: prior_confidence_level vs Uniform(0,1).

    Analogous to plot_qq_calibration but evaluates the spatial *prior* rather
    than the posterior.  Comparing the two plots reveals how much of the
    calibration signal comes from the prior alone vs. the seismic data.

    Interpretation is identical to plot_qq_calibration:
    - Below diagonal: prior concentrates mass near the true location (prior is
      well-placed / overconfident).
    - Above diagonal: prior spreads mass away from the true location.

    Note: for the Uniform prior, prior_confidence_level is ~1.0 for every
    event (a flat grid has no high-density region), so the Uniform line will
    sit at the top of the plot and is usually uninformative.

    Parameters
    ----------
    prior_names : list[str]
    output_dir : str
        Ignored for any prior_name present in ``csv_paths``.
    title : str
    save_path : str or None
    filter_fn : callable(df) -> df, optional
    n_trigs : int or None
        Which trigger-count version to use per event (see plot_qq_calibration).
        None (default) takes each event's last available version.
    csv_paths : dict[str, str] or None
        Optional {prior_name: csv_path} overrides — see plot_qq_calibration.
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(7, 6))

    for color, prior_name in zip(colors, prior_names):
        if csv_paths is not None and prior_name in csv_paths:
            csv_path = csv_paths[prior_name]
        else:
            csv_path = os.path.join(output_dir,
                                    f'{prior_name.lower()}_benchmark_results.csv')

        # Deduplicate by event: the prior is evaluated once per event, so all
        # trigger versions share the same prior_confidence_level.  Using all
        # rows inflates n with identical values, creating rectangular plateaus.
        final = load_final_rows(csv_path, n_trigs=n_trigs)
        if final is None:
            continue
        final = final.dropna(subset=['prior_confidence_level'])
        if filter_fn is not None:
            final = filter_fn(final)

        vals = np.sort(final['prior_confidence_level'].values)
        n = len(vals)
        if n == 0:
            continue

        theoretical = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(theoretical, vals, color=color, linewidth=1.5,
                label=f'{prior_name}  (n={n})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='ideal')
    ax.set_xlabel('Theoretical quantile  U(0,1)', fontsize=11)
    ax.set_ylabel('Empirical prior_confidence_level', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_qq_prior_comparison(
    prior_names,
    output_dir,
    column='map_err_km',
    title='Q-Q prior comparison — map_err_km',
    save_path=None,
    n_trigs=None,
):
    """
    5×5 Q-Q comparison of location errors across all prior pairs.

    Panel [i, j] plots sorted errors of prior_i (y-axis) against sorted
    errors of prior_j (x-axis) at 200 common quantile levels.  Points below
    the diagonal mean prior_i (row) has smaller errors at that quantile than
    prior_j (column).  Diagonal panels show the prior name.

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior names (determines grid size and ordering).
    output_dir : str
        Directory containing ``{prior_name.lower()}_benchmark_results.csv``.
    column : str
        Column to compare across priors.  Defaults to ``'map_err_km'``.
    title : str
        Figure suptitle.
    save_path : str or None
        Full path for the saved PNG (written at 150 dpi).
    n_trigs : int or None
        Which trigger-count version to use per event (see plot_qq_calibration).
        None (default) takes each event's last available version.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(prior_names)
    if n < 2:
        print(f'[plot_qq_prior_comparison] skipped — need ≥2 priors, got {n}.')
        return None

    fig, axes = plt.subplots(n, n, figsize=(n * 3.2, n * 3.2))

    # Load sorted error values for each prior
    data = {}
    for prior_name in prior_names:
        csv_path = os.path.join(output_dir,
                                f'{prior_name.lower()}_benchmark_results.csv')
        final = load_final_rows(csv_path, n_trigs=n_trigs)
        if final is None or column not in final.columns or final[column].isna().all():
            data[prior_name] = None
            continue

        data[prior_name] = np.sort(final[column].dropna().values)

    # Shared axis limit clipped at the 99th percentile across all priors
    all_vals = [v for v in data.values() if v is not None]
    global_max = float(np.percentile(np.concatenate(all_vals), 99)) if all_vals else 100.0

    q_levels = np.linspace(0, 1, 200)

    for i, prior_i in enumerate(prior_names):
        for j, prior_j in enumerate(prior_names):
            ax = axes[i, j]

            if i == j:
                ax.text(0.5, 0.5, prior_i, transform=ax.transAxes,
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='0.25')
                ax.set_facecolor('0.91')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            vals_i = data[prior_i]
            vals_j = data[prior_j]

            if vals_i is None or vals_j is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            q_i = np.quantile(vals_i, q_levels)
            q_j = np.quantile(vals_j, q_levels)

            ax.plot(q_j, q_i, color='steelblue', linewidth=1.2)
            ax.plot([0, global_max], [0, global_max],
                    'k--', linewidth=0.8, alpha=0.55)
            ax.set_xlim(0, global_max)
            ax.set_ylim(0, global_max)

            # Suppress inner tick labels; keep outer edges readable
            ax.tick_params(labelsize=7)
            if j != 0:
                ax.tick_params(labelleft=False)
            if i != n - 1:
                ax.tick_params(labelbottom=False)

            # Outer axis labels carry the prior name
            if j == 0:
                ax.set_ylabel(prior_i, fontsize=9)
            if i == n - 1:
                ax.set_xlabel(prior_j, fontsize=9)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_qq_calibration_by_param(
    prior_names,
    output_dirs,
    param_label='sigma_s',
    title='bEPIC posterior calibration vs {param_label}',
    save_path=None,
    ncols=3,
    extra_panel=None,
    y_column = 'posterior_confidence_level',
    x_column=None,
    x_label=None,
    log_x=False,
    n_trigs=None,
):
    """
    Grid of Q-Q calibration plots: one panel per prior, one line per output
    directory (e.g. a sweep over sigma_s or edt_sigma_s).

    Mirrors plot_qq_calibration, but instead of one panel with one line per
    prior (all sharing a single output_dir), this produces one panel per
    prior with one line per entry in `output_dirs` — letting calibration be
    compared across a swept parameter, per prior.

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior display names; one panel per entry.
    output_dirs : dict[str or float, str]
        Maps a parameter value (used as the line label and for colour
        ordering) to a directory containing
        ``{prior_name.lower()}_benchmark_results.csv``. Iterated in the
        order given — sort the dict before calling if a specific line-colour
        order is wanted. Also sets the colour scale shared with `extra_panel`.
    param_label : str
        Name of the swept parameter, used in the title and legend.
    title : str
        Figure suptitle; ``{param_label}`` is substituted if present.
    save_path : str or None
        Full path for the saved PNG (written at 150 dpi).
    ncols : int
        Number of panel columns.
    extra_panel : dict, optional
        Adds one more panel for a series that lives in its own directory
        tree with a fixed CSV filename (e.g. the dynamic ETAS prior, whose
        results aren't named ``{prior}_benchmark_results.csv``). Keys:
          'name'         — panel title / legend label prefix
          'output_dirs'  — dict[param_value, dir], analogous to `output_dirs`
          'csv_filename' — filename read from each directory in 'output_dirs'
        Param values shared with `output_dirs` reuse the same colour so
        sigma_s is comparable across all panels; values unique to
        `extra_panel` fall back to a separate colour cycle.
    x_column : str or None
        If None (default), the x-axis is the theoretical Uniform(0,1)
        quantile — the standard calibration Q-Q plot. If set to another
        column name (e.g. ``'map_err_km'``), the x-axis instead becomes
        that column's own empirical quantile function, evaluated at the
        same quantile levels as the posterior_confidence_level values on
        the y-axis — a two-distribution Q-Q plot comparing the shape of
        posterior_confidence_level against the shape of `x_column`, rather
        than against a uniform reference. The two columns are not paired
        event-by-event; each is ranked independently, since only their
        marginal distributions are being compared. The diagonal reference
        line and fixed [0,1] x-limits are skipped in this mode.
    x_label : str or None
        X-axis label when `x_column` is set. Defaults to `x_column`.
    log_x : bool
        Log-scale the x-axis. Only meaningful when `x_column` is set.
    n_trigs : int or None
        Which trigger-count version to use per event. Defaults to None,
        which takes each event's last (most-triggered) row — matches
        `load_final_values`'s default. Pass an int to instead use each
        event's row at that specific trigger count (events that never
        reached it are excluded).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    panel_specs = [
        {'name': name, 'output_dirs': output_dirs,
         'csv_filename': f'{name.lower()}_benchmark_results.csv'}
        for name in prior_names
    ]
    if extra_panel is not None:
        panel_specs.append(extra_panel)

    n = len(panel_specs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.0),
                              squeeze=False)
    axes_flat = axes.flatten()

    # Shared colour scale, keyed by param_value, so the same sigma_s draws
    # the same colour in every panel including the extra one.
    param_values = list(output_dirs.keys())
    color_map = dict(zip(param_values, plt.cm.viridis(np.linspace(0, 1, len(param_values)))))
    fallback_colors = plt.cm.plasma(np.linspace(0, 1, max(len(param_values), 1)))

    for ax, spec in zip(axes_flat, panel_specs):
        panel_param_values = list(spec['output_dirs'].keys())
        for i, param_value in enumerate(panel_param_values):
            color = color_map.get(param_value, fallback_colors[i % len(fallback_colors)])
            csv_path = os.path.join(spec['output_dirs'][param_value], spec['csv_filename'])

            try:
                y_src = load_final_values(csv_path, y_column, n_trigs=n_trigs)
            except ValueError as e:
                print(f"  [{spec['name']}, {param_label}={param_value}] {e}")
                continue
            if y_src is None or len(y_src) == 0:
                continue

            y_vals = np.sort(y_src)
            n_vals = len(y_vals)
            q_levels = (np.arange(1, n_vals + 1) - 0.5) / n_vals

            if x_column is None:
                x_vals = q_levels
            else:
                x_src = load_final_values(csv_path, x_column, n_trigs=n_trigs)
                if x_src is None or len(x_src) == 0:
                    continue
                # Independent quantile function of x_column at the same
                # quantile levels — a marginal-distribution comparison, not
                # an event-by-event pairing.
                x_vals = np.quantile(x_src, q_levels)

            ax.plot(x_vals, y_vals, color=color, linewidth=1.5,
                    label=f'{param_label}={param_value}')

        if x_column is None:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='ideal')
            ax.set_xlim(0, 1)
        if log_x:
            ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_title(spec['name'], fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    x_axis_label = ('Theoretical quantile  U(0,1)' if x_column is None
                     else (x_label or x_column))
    for ax in axes_flat[:n]:
        ax.set_xlabel(x_axis_label, fontsize=9)
    for i, ax in enumerate(axes_flat[:n]):
        if i % ncols == 0:
            ax.set_ylabel(f'Empirical {y_column}', fontsize=9)

    fig.suptitle(title.format(param_label=param_label), fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def _load_paired_values(csv_path, columns, n_trigs=None):
    """
    Load several metric columns from a benchmark CSV, paired per event.

    Unlike load_final_values (one column, NaNs dropped independently), this
    keeps `columns` aligned row-by-row: a row is dropped only if any of the
    requested columns is NaN for that event. Returns a DataFrame indexed by
    event_id, or None if the file/columns are missing or nothing survives.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if any(c not in df.columns for c in columns):
        return None

    if 'n_trigs' not in df.columns:
        df['n_trigs'] = (df.groupby('event_id')['version']
                            .rank(method='dense').astype(int))

    if n_trigs is None:
        sub = df.groupby('event_id').last()
    else:
        sub = df[df['n_trigs'] == n_trigs].set_index('event_id')

    sub = sub[list(columns)].dropna()
    return sub if len(sub) > 0 else None


def _to_quantile(values):
    """Map each value to its own empirical quantile (plotting-position rank)."""
    ranks = np.argsort(np.argsort(values))
    return (ranks + 0.5) / len(values)


def plot_scatter_calibration_by_param(
    prior_names,
    output_dirs,
    param_label='sigma_s',
    y_column='posterior_confidence_level',
    x_column='map_err_km',
    y_quantile=True,
    x_quantile=False,
    title='{y_axis} vs {x_axis}, by {param_label}',
    save_path=None,
    ncols=3,
    extra_panel=None,
    x_label=None,
    y_label=None,
    log_x=False,
    log_y=False,
    n_trigs=None,
    alpha=0.5,
    point_size=14,
):
    """
    Grid of scatter plots: one panel per prior, one colour per swept
    parameter value (e.g. sigma_s), one point per event.

    Unlike plot_qq_calibration_by_param — which compares `x_column` and
    `y_column` as independent marginal distributions — this pairs the two
    columns event-by-event, so the scatter shows the actual per-event
    relationship between them (e.g. does a higher posterior confidence level
    correlate with a smaller location error?).

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior display names; one panel per entry.
    output_dirs : dict[str or float, str]
        Maps a swept parameter value (used as the point colour/label) to a
        directory containing ``{prior_name.lower()}_benchmark_results.csv``.
        Also sets the colour scale shared with `extra_panel`.
    param_label : str
        Name of the swept parameter, used in the legend.
    y_column, x_column : str
        Benchmark CSV columns to plot on the y- and x-axes. Any column in
        the results CSV works (e.g. 'map_err_km', 'best_misfit',
        'usgs_credible_level', 'coverage_100km').
    y_quantile, x_quantile : bool
        If True, that column's per-event values are replaced by their own
        empirical quantile (plotting-position rank in [0, 1]) before
        plotting, rather than the raw value. Defaults match the standard
        use case: quantile-transformed posterior_confidence_level (y) against
        raw map_err_km in km (x).
    title : str
        Figure suptitle; ``{param_label}``, ``{x_axis}``, ``{y_axis}`` are
        substituted if present.
    save_path : str or None
        Full path for the saved PNG (written at 150 dpi).
    ncols : int
        Number of panel columns.
    extra_panel : dict, optional
        Adds one more panel for a series that lives in its own directory
        tree with a fixed CSV filename (e.g. the dynamic ETAS prior). Keys:
          'name'         — panel title / legend label prefix
          'output_dirs'  — dict[param_value, dir], analogous to `output_dirs`
          'csv_filename' — filename read from each directory in 'output_dirs'
    x_label, y_label : str or None
        Axis labels; default to the column name (with a "quantile" suffix
        when that axis is quantile-transformed).
    log_x, log_y : bool
        Log-scale the respective axis.
    n_trigs : int or None
        Which trigger-count version to use per event. Defaults to None,
        which takes each event's last (most-triggered) row. Pass an int to
        instead use each event's row at that specific trigger count (events
        that never reached it are excluded).
    alpha : float
        Marker transparency, useful when many events overlap.
    point_size : float
        Marker size passed to ax.scatter.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    panel_specs = [
        {'name': name, 'output_dirs': output_dirs,
         'csv_filename': f'{name.lower()}_benchmark_results.csv'}
        for name in prior_names
    ]
    if extra_panel is not None:
        panel_specs.append(extra_panel)

    n = len(panel_specs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.0),
                              squeeze=False)
    axes_flat = axes.flatten()

    # Shared colour scale, keyed by param_value, so the same sigma_s draws
    # the same colour in every panel including the extra one.
    param_values = list(output_dirs.keys())
    color_map = dict(zip(param_values, plt.cm.viridis(np.linspace(0, 1, len(param_values)))))
    fallback_colors = plt.cm.plasma(np.linspace(0, 1, max(len(param_values), 1)))

    for ax, spec in zip(axes_flat, panel_specs):
        panel_param_values = list(spec['output_dirs'].keys())
        for i, param_value in enumerate(panel_param_values):
            color = color_map.get(param_value, fallback_colors[i % len(fallback_colors)])
            csv_path = os.path.join(spec['output_dirs'][param_value], spec['csv_filename'])

            paired = _load_paired_values(csv_path, [y_column, x_column], n_trigs=n_trigs)
            if paired is None:
                print(f"  [{spec['name']}, {param_label}={param_value}] no paired data for "
                      f"{y_column}/{x_column}")
                continue

            y_vals = _to_quantile(paired[y_column].values) if y_quantile else paired[y_column].values
            x_vals = _to_quantile(paired[x_column].values) if x_quantile else paired[x_column].values

            ax.scatter(x_vals, y_vals, color=color, s=point_size, alpha=alpha,
                       edgecolors='none', label=f'{param_label}={param_value}')

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        if y_quantile:
            ax.set_ylim(0, 1)
        if x_quantile:
            ax.set_xlim(0, 1)
        ax.set_title(spec['name'], fontsize=11)
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    x_axis_label = x_label or (f'{x_column} quantile' if x_quantile else x_column)
    y_axis_label = y_label or (f'{y_column} quantile' if y_quantile else y_column)
    for ax in axes_flat[:n]:
        ax.set_xlabel(x_axis_label, fontsize=9)
    for i, ax in enumerate(axes_flat[:n]):
        if i % ncols == 0:
            ax.set_ylabel(y_axis_label, fontsize=9)

    fig.suptitle(title.format(param_label=param_label, x_axis=x_axis_label, y_axis=y_axis_label),
                 fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_log_likelihood_sum_by_param(
    prior_names,
    output_dirs,
    param_label='sigma_s',
    column='like_val_at_usgs',
    title='Total log-likelihood vs {param_label}',
    save_path=None,
    ncols=3,
    extra_panel=None,
    n_trigs=None,
    log_floor=1e-200,
    highlight_best=True,
):
    """
    Scatter of Σ log(column) — the total log-likelihood across events —
    against a swept parameter (e.g. sigma_s), one panel per prior (plus an
    optional extra_panel, e.g. dynamic ETAS), one point per swept value.

    Maximizing Σ log(p_i) across events is equivalent to maximizing the
    joint likelihood Π p_i (the parameter value best supported by where
    events actually occurred), but is numerically stable — raw products of
    many small per-event probabilities underflow to 0 long before the sum of
    their logs does. This is the natural way to pick, e.g., the sigma_s that
    best explains the benchmark catalog as a whole, rather than relying on a
    raw mean (dominated by a few large outliers) or a median (ignores
    magnitude, only reflects the middle event).

    Parameters
    ----------
    prior_names : list[str]
        Ordered list of prior display names; one panel per entry.
    output_dirs : dict[str or float, str]
        Maps a swept parameter value to a directory containing
        ``{prior_name.lower()}_benchmark_results.csv``.
    param_label : str
        Name of the swept parameter, used in the axis label.
    column : str
        Benchmark CSV column to treat as a per-event probability and sum the
        log of. Defaults to 'like_val_at_usgs' (the raw normalized
        likelihood-surface value at the true location) — the natural choice
        for picking the sigma_s that maximizes the travel-time likelihood.
    title : str
        Figure suptitle; ``{param_label}`` and ``{column}`` are substituted.
    save_path : str or None
        Full path for the saved PNG (written at 150 dpi).
    ncols : int
        Number of panel columns.
    extra_panel : dict, optional
        Adds one more panel for a series that lives in its own directory
        tree with a fixed CSV filename (e.g. the dynamic ETAS prior). Keys:
          'name'         — panel title
          'output_dirs'  — dict[param_value, dir], analogous to `output_dirs`
          'csv_filename' — filename read from each directory in 'output_dirs'
    n_trigs : int or None
        Which trigger-count version to use per event. Defaults to None,
        which takes each event's last (most-triggered) row.
    log_floor : float
        Values are clipped to this floor before taking the log, so an exact
        zero for one event doesn't send the whole panel's sum to -inf.
    highlight_best : bool
        If True, marks the argmax point (best parameter value) in each panel
        with a red star.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    panel_specs = [
        {'name': name, 'output_dirs': output_dirs,
         'csv_filename': f'{name.lower()}_benchmark_results.csv'}
        for name in prior_names
    ]
    if extra_panel is not None:
        panel_specs.append(extra_panel)

    n = len(panel_specs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.0),
                              squeeze=False)
    axes_flat = axes.flatten()

    for ax, spec in zip(axes_flat, panel_specs):
        panel_param_values = sorted(spec['output_dirs'].keys())
        xs, ys = [], []
        for param_value in panel_param_values:
            csv_path = os.path.join(spec['output_dirs'][param_value], spec['csv_filename'])
            vals = load_final_values(csv_path, column, n_trigs=n_trigs)
            if vals is None or len(vals) == 0:
                print(f"  [{spec['name']}, {param_label}={param_value}] no data for {column}")
                continue
            n_floored = int((vals <= log_floor).sum())
            if n_floored > 0:
                print(f"  [{spec['name']}, {param_label}={param_value}] "
                      f"{n_floored}/{len(vals)} events hit log_floor={log_floor:.0e}")
            log_sum = float(np.log(np.clip(vals, log_floor, None)).sum())
            xs.append(param_value)
            ys.append(log_sum)

        if not xs:
            ax.set_title(f"{spec['name']} (no data)", fontsize=11)
            ax.grid(True, alpha=0.2)
            continue

        ax.scatter(xs, ys, color='tab:blue', s=40, zorder=3)

        if highlight_best:
            best_i = int(np.argmax(ys))
            ax.scatter([xs[best_i]], [ys[best_i]], color='tab:red', s=90,
                       zorder=4, marker='*',
                       label=f'best {param_label}={xs[best_i]}')
            ax.legend(fontsize=7)

        ax.set_title(spec['name'], fontsize=11)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax in axes_flat[:n]:
        ax.set_xlabel(param_label, fontsize=9)
    for i, ax in enumerate(axes_flat[:n]):
        if i % ncols == 0:
            ax.set_ylabel(f'Σ log({column})', fontsize=9)

    fig.suptitle(title.format(param_label=param_label, column=column), fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    return fig
