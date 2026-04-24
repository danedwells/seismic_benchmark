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

from .metrics import hdr_levels, usgs_credible_level, posterior_coverage  # re-exported for scripts

# Plots

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
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={'projection': proj})

    for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), prior_order)):
        row_idx, col_idx = divmod(idx, 3)
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
        gl.bottom_labels = (row_idx == 1)

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
        ax_sb    = axes[1, 0]
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
        plt.savefig(save_path, dpi=150)

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
    ext = [
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
    plt.tight_layout()
    # Shrink panels left to make room for the colorbar on the far right
    plt.subplots_adjust(right=0.88)

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
