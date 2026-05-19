"""
benchmark/metrics.py — location accuracy and posterior probability metrics.

These functions operate on the out_df grid returned by E2Location_locate
(columns: lat, lon, like, prior, post) and on SearchOut posterior coordinates.
"""
import numpy as np
from obspy.geodetics import gps2dist_azimuth
import pandas as pd
import os
from scipy.stats import kstest

def load_final_values(csv_path, metric, n_trigs=None, min_events_warn=5):
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
    p = post_flat / post_flat.sum()
    idx = np.argsort(p)[::-1]
    cumsum = np.cumsum(p[idx])
    thresholds = {}
    for cl in credible_levels:
        i = np.searchsorted(cumsum, cl)
        thresholds[cl] = float(p[idx[min(i, len(idx) - 1)]])
    return thresholds


def location_error_km(posterior_lat, posterior_lon, ref_lat, ref_lon):
    """Geodetic distance in km between a posterior MAP estimate and a reference location."""
    m, _, _ = gps2dist_azimuth(ref_lat, ref_lon, posterior_lat, posterior_lon)
    return m / 1000.0


def posterior_confidence_level(out_df, usgs_lat, usgs_lon):
    """
    Credible level of the smallest HDR that contains the USGS location.
    Returns a value in [0, 1]: lower is better (USGS is in a high-density region).
    """
    p = out_df['post'].values
    p_norm = p / p.sum()

    dlat = out_df['lat'].values - usgs_lat
    # Correct for longitude compression at non-equatorial latitudes.
    dlon = (out_df['lon'].values - usgs_lon) * np.cos(np.radians(usgs_lat))
    p_usgs = p_norm[np.argmin(np.hypot(dlat, dlon))]

    return float(p_norm[p_norm >= p_usgs].sum())


def usgs_prior_credible_level(out_df, usgs_lat, usgs_lon):
    """
    Credible level of the smallest HDR of the *prior* that contains the USGS location.
    Returns a value in [0, 1]: lower is better (USGS is in a high-density prior region).

    Analogous to posterior_confidence_level but uses the prior column instead of post.
    Comparing the two reveals how much bEPIC's posterior improves on the raw prior.
    When use_prior=False the prior grid is uniform, so this returns ~1.0 for most
    events and the column should be excluded from analysis for Uniform runs.
    """
    p = out_df['prior'].values
    p_norm = p / p.sum()

    dlat = out_df['lat'].values - usgs_lat
    # Correct for longitude compression at non-equatorial latitudes.
    dlon = (out_df['lon'].values - usgs_lon) * np.cos(np.radians(usgs_lat))
    p_usgs = p_norm[np.argmin(np.hypot(dlat, dlon))]

    return float(p_norm[p_norm >= p_usgs].sum())


def _haversine_km(ref_lat, ref_lon, lats, lons):
    """Vectorized haversine distance (km) from one point to an array of points.

    Accurate to < 0.5 km within the ~400 km search boxes used here; replaces
    per-row gps2dist_azimuth calls that would otherwise loop over the full grid.
    """
    R = 6371.0
    dlat = np.radians(lats - ref_lat)
    dlon = np.radians(lons - ref_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(ref_lat)) * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


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
    dict mapping each radius to its coverage fraction in [0, 1], or a single
    float if a scalar radii_km was supplied.
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
    """Index of the grid cell nearest to ref_lat/ref_lon (cosine-corrected)."""
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
    """
    p = out_df['post'].values
    p_norm = p / p.sum()
    idx = _nearest_cell_index(out_df, ref_lat, ref_lon)
    p_true = float(p_norm[idx])
    return float(np.sum(p_norm ** 2) - 2.0 * p_true + 1.0)


def ks_calibration(credible_levels):
    """
    KS statistic testing whether a vector of credible levels is Uniform[0, 1].

    For a perfectly calibrated posterior, posterior_confidence_level values are
    i.i.d. Uniform[0, 1] across events. This function quantifies the departure
    from that ideal using the two-sided Kolmogorov-Smirnov test.

    Parameters
    ----------
    credible_levels : array-like
        Per-event posterior_confidence_level values, each in [0, 1].

    Returns
    -------
    statistic : float
        KS distance from the empirical CDF to Uniform[0, 1]. Lower = better
        calibration; 0 is perfect.
    p_value : float
        Two-sided p-value. Small values indicate significant miscalibration.
    """
    vals = np.asarray(credible_levels, dtype=float)
    stat, pval = kstest(vals, 'uniform')
    return float(stat), float(pval)
