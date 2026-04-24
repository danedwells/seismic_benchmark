"""
benchmark/metrics.py — location accuracy and posterior probability metrics.

These functions operate on the out_df grid returned by E2Location_locate
(columns: lat, lon, post) and on SearchOut posterior coordinates.
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


def posterior_coverage(out_df, ref_lat, ref_lon, radii_km=50):
    """
    Fraction of posterior probability mass within radii_km of ref_lat/ref_lon.

    Parameters
    ----------
    out_df : pd.DataFrame
        Grid output from E2Location_locate — must have columns lat, lon, post.
    ref_lat, ref_lon : float
        Reference location (e.g. USGS catalog).
    radii_km : float
        Radius at which to evaluate cumulative posterior mass.  Typically set to
        the MAP location error so the metric answers "what fraction of the posterior
        was within the same distance as the final estimate?"

    Returns
    -------
    float : coverage fraction in [0, 1]
    """
    dists_km = np.array([
        gps2dist_azimuth(ref_lat, ref_lon, row.lat, row.lon)[0] / 1000.0
        for row in out_df.itertuples(index=False)
    ])

    post = out_df['post'].values
    total = post.sum()
    if total > 0:
        post = post / total

    return float(post[dists_km <= radii_km].sum())
