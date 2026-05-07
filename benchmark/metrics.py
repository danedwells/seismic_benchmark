"""
benchmark/metrics.py — location accuracy and posterior probability metrics.

These functions operate on the out_df grid returned by E2Location_locate
(columns: lat, lon, like, prior, post) and on SearchOut posterior coordinates.
"""
import numpy as np
from obspy.geodetics import gps2dist_azimuth


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


def usgs_credible_level(out_df, usgs_lat, usgs_lon):
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

    Analogous to usgs_credible_level but uses the prior column instead of post.
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
