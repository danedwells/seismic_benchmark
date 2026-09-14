"""
benchmark/metrics.py — location accuracy and posterior probability metrics,
plus helpers for loading aggregated results from benchmark CSVs.

Most functions here operate on the out_df grid returned by
E2Location_locate (columns: lat, lon, like, prior, post) and on SearchOut
posterior coordinates, computing per-event accuracy/calibration/scoring
metrics (location error, HDR credible levels, coverage, log/Brier/energy
scores). The load_final_values/load_final_rows/load_per_version_stats
functions instead read the `{prior}_benchmark_results.csv` files these
metrics are written to (via runner.py), for downstream plotting/analysis.
"""
import numpy as np
from obspy.geodetics import gps2dist_azimuth
import pandas as pd
import os
from scipy.stats import kstest

def load_final_values(csv_path, metric, n_trigs=None, min_events_warn=5):
    """
    Load one metric value per event from a benchmark results CSV.

    Reads csv_path and, per event, takes either the metric value at the
    last available trigger version (n_trigs=None) or at a specific
    trigger count. Column-level counterpart to load_final_rows, which
    returns the full row instead of a single column's values.

    Parameters
    ----------
    csv_path : str
        Path to a `{prior}_benchmark_results.csv` file.
    metric : str
        Name of the column to extract.
    n_trigs : int or None, optional
        If given, take each event's value at this specific trigger count
        instead of its last available version. None (default) uses each
        event's last row.
    min_events_warn : int, optional
        If fewer than this many events have data at the requested
        n_trigs, emit a UserWarning that results may be unreliable.
        Default 5. Ignored when n_trigs is None.

    Returns
    -------
    numpy.ndarray or None
        Array of metric values (one per event, NaNs dropped), or None if
        csv_path doesn't exist, metric is not a column, the column is
        entirely NaN, or no values remain after filtering.
    """
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


def load_final_rows(csv_path, n_trigs=None, min_events_warn=5):
    """
    Load one row per event from a benchmark CSV: either the last available
    trigger version (n_trigs=None) or the row at a specific trigger count.

    Row-level counterpart to load_final_values — returns a DataFrame (so
    callers can apply their own column selection and filter_fn) instead of
    a single column's values.

    Parameters
    ----------
    csv_path : str
        Path to a `{prior}_benchmark_results.csv` file.
    n_trigs : int or None, optional
        If given, keep only each event's row at this specific trigger
        count instead of its last available version. None (default)
        uses each event's last row.
    min_events_warn : int, optional
        If fewer than this many events have data at the requested
        n_trigs, emit a UserWarning that results may be unreliable.
        Default 5. Ignored when n_trigs is None.

    Returns
    -------
    pandas.DataFrame or None
        One row per event (with an n_trigs column added if not already
        present), or None if csv_path doesn't exist.
    """
    import warnings
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    if 'n_trigs' not in df.columns:
        df['n_trigs'] = (df.groupby('event_id')['version']
                            .rank(method='dense')
                            .astype(int))

    if n_trigs is None:
        return df.groupby('event_id').last().reset_index()

    max_available = int(df['n_trigs'].max())
    if n_trigs > max_available:
        raise ValueError(
            f"Requested n_trigs={n_trigs} exceeds the maximum available "
            f"({max_available}) in {os.path.basename(csv_path)}."
        )
    subset = df[df['n_trigs'] == n_trigs].reset_index(drop=True)
    if len(subset) < min_events_warn:
        warnings.warn(
            f"Only {len(subset)} events have data at n_trigs={n_trigs} "
            f"in {os.path.basename(csv_path)} (min_events_warn={min_events_warn}). "
            "Results may be unreliable.",
            UserWarning, stacklevel=2,
        )
    return subset

def load_per_version_stats(csv_path, metric, min_events=5):
    """
    Load a benchmark CSV and return per-trigger-count aggregate statistics.

    Parameters
    ----------
    csv_path : str
        Path to a `{prior}_benchmark_results.csv` file.
    metric : str
        Name of the column to aggregate.
    min_events : int, optional
        Minimum number of non-NaN observations required at a given
        n_trigs for that row to be kept in the output. Default 5.

    Returns
    -------
    pandas.DataFrame or None
        One row per n_trigs value with columns n_trigs, median, mean,
        q1, q5, q95, q99, min, max, count, restricted to rows with
        count >= min_events. None if csv_path doesn't exist or metric
        is absent/all-NaN.
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
                    mean = 'mean',
                    q1=lambda x: x.quantile(0.01),
                    q5=lambda x: x.quantile(0.05),
                    q95=lambda x: x.quantile(0.95),
                    q99=lambda x: x.quantile(0.99),
                    min = 'min',
                    max = 'max',
                    count='count')
               .reset_index())
    return stats[stats['count'] >= min_events]

def hdr_levels(post_flat, credible_levels=(0.1, 0.50, 0.67, 0.90, 0.95)):
    """
    Compute per-grid-cell probability thresholds for the smallest highest
    density regions (HDRs) containing each requested credible mass.

    For each credible level, grid cells are ranked by probability from
    highest to lowest and accumulated until the cumulative sum reaches
    that level; the cell probability at that point is the threshold such
    that {cells with density >= threshold} is the smallest-area region
    containing at least that fraction of the total probability mass.
    Used to draw HDR contours on prior/posterior grids.

    Parameters
    ----------
    post_flat : numpy.ndarray
        Flattened (1-D) array of (unnormalized) probability values for
        every grid cell, e.g. a prior or posterior grid raveled to 1-D.
    credible_levels : sequence of float, optional
        Credible mass levels in (0, 1] to compute thresholds for.
        Default (0.1, 0.50, 0.67, 0.90, 0.95).

    Returns
    -------
    dict
        Maps each requested credible level to the probability-density
        threshold (float) of the smallest HDR containing at least that
        much of the normalized probability mass.
    """
    p = post_flat / post_flat.sum()
    idx = np.argsort(p)[::-1]
    cumsum = np.cumsum(p[idx])
    thresholds = {}
    for cl in credible_levels:
        i = np.searchsorted(cumsum, cl)
        thresholds[cl] = float(p[idx[min(i, len(idx) - 1)]])
    return thresholds


def location_error_km(posterior_lat, posterior_lon, ref_lat, ref_lon):
    """
    Great-circle (geodetic) distance in km between two lat/lon points.

    Despite the parameter names, this is a general-purpose distance
    helper — runner.py uses it for the MAP posterior estimate as well
    as the expectation, likelihood, and likelihood-expectation location
    estimates.

    Parameters
    ----------
    posterior_lat, posterior_lon : float
        Latitude and longitude (degrees) of the estimated location.
    ref_lat, ref_lon : float
        Latitude and longitude (degrees) of the reference (e.g. USGS
        catalog) location.

    Returns
    -------
    float
        Geodetic distance between the two points, in kilometers.
    """
    m, _, _ = gps2dist_azimuth(ref_lat, ref_lon, posterior_lat, posterior_lon)
    return m / 1000.0


def posterior_confidence_level(out_df, usgs_lat, usgs_lon):
    """
    Credible level of the smallest HDR that contains the USGS location.
    Returns a value in [0, 1]: lower is better (USGS is in a high-density region).

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        post.
    usgs_lat, usgs_lon : float
        Reference location (e.g. USGS catalog) to evaluate.

    Returns
    -------
    float
        Credible level in [0, 1] of the smallest HDR of the posterior
        containing the reference location; lower is better (means the
        reference location sits in a higher-density region).
    """
    p = out_df['post'].values
    p_norm = p / p.sum()

    dlat = out_df['lat'].values - usgs_lat
    # Correct for longitude compression at non-equatorial latitudes.
    dlon = (out_df['lon'].values - usgs_lon) * np.cos(np.radians(usgs_lat))
    p_usgs = p_norm[np.argmin(np.hypot(dlat, dlon))]

    return float(p_norm[p_norm >= p_usgs].sum())


def prior_confidence_level(out_df, usgs_lat, usgs_lon):
    """
    Credible level of the smallest HDR of the *prior* that contains the USGS location.
    Returns a value in [0, 1]: lower is better (USGS is in a high-density prior region).

    Analogous to posterior_confidence_level but uses the prior column instead of post.
    Comparing the two reveals how much bEPIC's posterior improves on the raw prior.
    When use_prior=False the prior grid is uniform, so this returns ~1.0 for most
    events and the column should be excluded from analysis for Uniform runs.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        prior.
    usgs_lat, usgs_lon : float
        Reference location (e.g. USGS catalog) to evaluate.

    Returns
    -------
    float
        Credible level in [0, 1] of the smallest HDR of the prior
        containing the reference location; lower is better (means the
        reference location sits in a higher-density prior region).
    """
    p = out_df['prior'].values
    p_norm = p / p.sum()

    dlat = out_df['lat'].values - usgs_lat
    # Correct for longitude compression at non-equatorial latitudes.
    dlon = (out_df['lon'].values - usgs_lon) * np.cos(np.radians(usgs_lat))
    p_usgs = p_norm[np.argmin(np.hypot(dlat, dlon))]

    return float(p_norm[p_norm >= p_usgs].sum())


def like_confidence_level(out_df, usgs_lat, usgs_lon):
    """
    Credible level of the smallest HDR of the *likelihood surface* that
    contains the USGS location. Returns a value in [0, 1]: lower is better
    (USGS is in a high-density region of the likelihood alone).

    Analogous to posterior_confidence_level/prior_confidence_level but uses
    the 'like' column instead of 'post'/'prior' — shows how well the
    travel-time misfit alone (before any prior is applied) constrains the
    true location.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        like.
    usgs_lat, usgs_lon : float
        Reference location (e.g. USGS catalog) to evaluate.

    Returns
    -------
    float
        Credible level in [0, 1] of the smallest HDR of the likelihood
        surface containing the reference location; lower is better
        (means the reference location sits in a higher-density
        likelihood region).
    """
    p = out_df['like'].values
    p_norm = p / p.sum()

    dlat = out_df['lat'].values - usgs_lat
    # Correct for longitude compression at non-equatorial latitudes.
    dlon = (out_df['lon'].values - usgs_lon) * np.cos(np.radians(usgs_lat))
    p_usgs = p_norm[np.argmin(np.hypot(dlat, dlon))]

    return float(p_norm[p_norm >= p_usgs].sum())


def _haversine_km(ref_lat, ref_lon, lats, lons):
    """
    Vectorized haversine distance (km) from one point to an array of points.

    Accurate to < 0.5 km within the ~400 km search boxes used here; replaces
    per-row gps2dist_azimuth calls that would otherwise loop over the full grid.

    Parameters
    ----------
    ref_lat, ref_lon : float
        Latitude and longitude (degrees) of the reference point.
    lats, lons : numpy.ndarray
        Arrays of latitude and longitude (degrees) of the target points.

    Returns
    -------
    numpy.ndarray
        Haversine distance (km) from the reference point to each target
        point; same shape as lats/lons.
    """
    R = 6371.0
    dlat = np.radians(lats - ref_lat)
    dlon = np.radians(lons - ref_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(ref_lat)) * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# Default radii (km) at which posterior_coverage() reports cumulative
# posterior probability mass around a reference location.
COVERAGE_RADII_KM = (10, 25, 50, 100)


def posterior_coverage(out_df, ref_lat, ref_lon, radii_km=COVERAGE_RADII_KM):
    """
    Fraction of posterior probability mass within each radius of ref_lat/ref_lon.

    Parameters
    ----------
    out_df : pd.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon, post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).
    radii_km : float or sequence of float
        Radius or radii (km) at which to evaluate cumulative posterior mass.

    Returns
    -------
    dict or float
        Dict mapping each radius to its coverage fraction in [0, 1], or a
        single float if a scalar radii_km was supplied.
    """
    dists_km = _haversine_km(ref_lat, ref_lon,
                             out_df['lat'].values, out_df['lon'].values)
    post = out_df['post'].values
    total = post.sum()
    if total > 0:
        post = post / total

    scalar = np.isscalar(radii_km)
    radii_km = (radii_km,) if scalar else radii_km
    result = {r: float(post[dists_km <= r].sum()) for r in radii_km}
    return result[radii_km[0]] if scalar else result


def _nearest_cell_index(out_df, ref_lat, ref_lon):
    """
    Index of the grid cell nearest to ref_lat/ref_lon (cosine-corrected).

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon.
    ref_lat, ref_lon : float
        Reference location to find the nearest grid cell to.

    Returns
    -------
    int
        Row index into out_df of the grid cell nearest ref_lat/ref_lon,
        with longitude distance cosine-corrected for latitude.
    """
    dlat = out_df['lat'].values - ref_lat
    dlon = (out_df['lon'].values - ref_lon) * np.cos(np.radians(ref_lat))
    return int(np.argmin(np.hypot(dlat, dlon)))


def log_score(out_df, ref_lat, ref_lon):
    """
    Log-score: log of the normalized posterior probability at the reference location.

    Finds the grid cell nearest ref_lat/ref_lon and returns log(P_true), where
    P_true is that cell's share of the total posterior mass.

    Higher (less negative) is better. A posterior with all mass at the true cell
    returns 0.0. Values are bounded below by log(1/G) for a G-cell uniform grid.

    Note: comparisons are only meaningful across grids of the same resolution.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).

    Returns
    -------
    float
        log(P_true), the natural log of the normalized posterior
        probability at the grid cell nearest the reference location.
        Higher (less negative) is better; 0.0 is the best possible
        value.
    """
    p = out_df['post'].values
    p_norm = p / p.sum()
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    p_true = float(p_norm[idx])
    return float(np.log(max(p_true, 1e-300)))


def brier_score(out_df, ref_lat, ref_lon):
    """
    Spatial Brier score: MSE of the normalized posterior against a point-mass at ref.

    Treats the grid cell nearest ref_lat/ref_lon as the single true outcome
    (O_j = 1) and all other cells as negative outcomes (O_j = 0).

        BS = Σ_j (P_j − O_j)²  =  Σ_j P_j²  −  2·P_true  +  1

    Lower is better. A posterior with all mass at the true cell gives 0.0;
    a posterior with all mass on the wrong cell gives 2.0.

    Note: like the log-score, this is grid-resolution dependent — only compare
    across events or priors evaluated on the same grid.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).

    Returns
    -------
    float
        Spatial Brier score (Σ_j P_j² − 2·P_true + 1). Lower is better;
        0.0 for a posterior with all mass at the true cell, 2.0 for a
        posterior with all mass on the wrong cell.
    """
    p = out_df['post'].values
    p_norm = p / p.sum()
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    p_true = float(p_norm[idx])
    return float(np.sum(p_norm ** 2) - 2.0 * p_true + 1.0)


def likelihood_value_at_location(out_df, ref_lat, ref_lon):
    """
    Normalized likelihood-surface mass at the grid cell nearest ref_lat/ref_lon.

    Unlike posterior_confidence_level/prior_confidence_level (the smallest HDR
    credible level containing ref), this is the raw value of the likelihood
    surface itself at that cell — how much weight the travel-time misfit
    alone assigns to the true location, independent of any prior. Same
    quantity as the p_true term computed internally by log_score/brier_score,
    but evaluated on the 'like' column instead of 'post'.

    Higher is better: a value near the grid's max means the likelihood alone
    (before any prior is applied) already favors the true location.

    Note: like log_score/brier_score, this is grid-resolution dependent —
    only compare across events or priors evaluated on the same grid.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        like.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).

    Returns
    -------
    float
        Normalized likelihood value in [0, 1] at the grid cell nearest
        the reference location.
    """
    p = out_df['like'].values
    p_norm = p / p.sum()
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    return float(p_norm[idx])


# TEMP: sigma_s sweep diagnostic — compares raw vs. normalized likelihood at
# the USGS location to check whether sigma_s selection is normalization-
# sensitive. Delete alongside likelihood_value_at_location_unnormalized once
# the experiment is done.
def likelihood_value_at_location_unnormalized(out_df, ref_lat, ref_lon):
    """
    Raw (un-normalized) likelihood-surface value at the grid cell nearest
    ref_lat/ref_lon — same lookup as likelihood_value_at_location, but skips
    the p / p.sum() normalization step.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        like.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).

    Returns
    -------
    float
        Raw (un-normalized) likelihood value at the grid cell nearest the
        reference location.
    """
    p = out_df['like'].values
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    return float(p[idx])


def posterior_value_at_location(out_df, ref_lat, ref_lon):
    """
    Normalized posterior mass at the grid cell nearest ref_lat/ref_lon.

    Unlike posterior_confidence_level (the smallest HDR credible level
    containing ref), this is the raw posterior value itself at that cell —
    the same p_true term computed internally by log_score/brier_score,
    exposed directly (equivalently, exp(log_score(out_df, ref_lat, ref_lon))).

    Higher is better: a value near the grid's max means the posterior
    concentrates its mass at the true location.

    Note: like log_score/brier_score, this is grid-resolution dependent —
    only compare across events or priors evaluated on the same grid.

    Parameters
    ----------
    out_df : pandas.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon,
        post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).

    Returns
    -------
    float
        Normalized posterior value in [0, 1] at the grid cell nearest
        the reference location.
    """
    p = out_df['post'].values
    p_norm = p / p.sum()
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    return float(p_norm[idx])


def energy_score(out_df, ref_lat, ref_lon, n_samples=1000, rng=None):
    """
    Energy Score for a 2-D gridded posterior vs a point observation.

    ES(F, y) = E[‖X − y‖] − ½ E[‖X − X'‖]

    where X, X' are independent draws from the posterior and y is the reference
    location.  The multivariate generalisation of CRPS to 2-D; reduces to CRPS
    when the forecast is 1-D.  Lower is better; a posterior with all mass at the
    true location gives 0.

    Term 1 is computed exactly over the full grid (O(G)).  Term 2 uses a
    split-sample Monte Carlo estimator: n_samples indices are drawn once, then
    pairwise distances between the two halves estimate ½ E[‖X − X'‖] (O(n_samples)).

    Parameters
    ----------
    out_df : pd.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon, post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).
    n_samples : int
        Number of posterior samples for the MC term (default 1000).
    rng : np.random.Generator or None
        Random generator; a fresh default_rng() is used if None.

    Returns
    -------
    float
        Energy Score in km.  Lower is better.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = out_df['post'].values
    p = p / p.sum()
    lats = out_df['lat'].values
    lons = out_df['lon'].values

    # Term 1: exact posterior-weighted mean distance to reference, O(G)
    term1 = float(np.dot(p, _haversine_km(ref_lat, ref_lon, lats, lons)))

    # Term 2: ½ E[‖X − X'‖] via split-sample MC, O(n_samples)
    idx = rng.choice(len(p), size=n_samples, p=p, replace=True)
    mid = n_samples // 2
    term2 = 0.5 * float(np.mean(
        _haversine_km(lats[idx[:mid]], lons[idx[:mid]],
                      lats[idx[mid:]], lons[idx[mid:]])
    ))

    return term1 - term2


