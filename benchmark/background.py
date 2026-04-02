"""
seismicity_background.py

Download and cache background seismicity from USGS ComCat for use as a
spatial reference layer in benchmark plots.

Main entry point
----------------
    df = load_background_seismicity(cache_path, bounds, start_year, end_year,
                                    min_mag=2.5, force_refresh=False)

The function returns a DataFrame with columns:
    time (UTC datetime), latitude, longitude, depth (km), mag

On first call it downloads from USGS and writes a parquet cache file.
Subsequent calls load directly from the cache.  Pass force_refresh=True
to re-download (e.g. after extending the date range).

USGS ComCat limits each request to 20,000 events.  The downloader
automatically splits the time range into chunks small enough to stay
under this limit, using a recursive bisection strategy.

Plotting helper
---------------
    add_background_seismicity(ax, df, transform, ...)
"""

import time
import requests
from io import StringIO
from datetime import date

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
MAX_EVENTS_PER_REQUEST = 20_000
REQUEST_DELAY = 0.5   # seconds between requests — be polite to the API


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_events(bounds, starttime, endtime, min_mag):
    """Return the number of events USGS would return for this query."""
    lon_min, lon_max, lat_min, lat_max = bounds
    params = {
        "format":       "geojson",
        "minlatitude":  lat_min,
        "maxlatitude":  lat_max,
        "minlongitude": lon_min,
        "maxlongitude": lon_max,
        "starttime":    starttime,
        "endtime":      endtime,
        "minmagnitude": min_mag,
    }
    r = requests.get(USGS_COUNT_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["count"]


def _fetch_chunk(bounds, starttime, endtime, min_mag):
    """
    Download one chunk as CSV and return a DataFrame.
    Caller must ensure the chunk has <= MAX_EVENTS_PER_REQUEST events.
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    params = {
        "format":       "csv",
        "minlatitude":  lat_min,
        "maxlatitude":  lat_max,
        "minlongitude": lon_min,
        "maxlongitude": lon_max,
        "starttime":    starttime,
        "endtime":      endtime,
        "minmagnitude": min_mag,
        "orderby":      "time-asc",
    }
    r = requests.get(USGS_QUERY_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    return df[["time", "latitude", "longitude", "depth", "mag"]].copy()


def _download_range(bounds, starttime, endtime, min_mag, chunks):
    """
    Recursively split [starttime, endtime] until each piece is under the
    event limit, then download and collect all chunks into `chunks` list.
    """
    n = _count_events(bounds, starttime, endtime, min_mag)
    if n == 0:
        return
    if n <= MAX_EVENTS_PER_REQUEST:
        print(f"  Fetching {n:>6,} events  {starttime} → {endtime}")
        time.sleep(REQUEST_DELAY)
        df = _fetch_chunk(bounds, starttime, endtime, min_mag)
        chunks.append(df)
    else:
        # Bisect the time range
        t0 = pd.Timestamp(starttime)
        t1 = pd.Timestamp(endtime)
        mid = t0 + (t1 - t0) / 2
        mid_str = mid.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"  Splitting [{starttime} → {endtime}] ({n:,} events > limit)")
        _download_range(bounds, starttime, mid_str,  min_mag, chunks)
        _download_range(bounds, mid_str,   endtime,  min_mag, chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_background_seismicity(bounds, start_year, end_year,
                                   min_mag=2.5, cache_path=None):
    """
    Download background seismicity from USGS ComCat and return a DataFrame.

    Automatically splits the time range into chunks that respect the
    20,000-event-per-request limit using recursive bisection.

    Parameters
    ----------
    bounds : tuple
        (lon_min, lon_max, lat_min, lat_max)
    start_year : int
    end_year   : int   (inclusive)
    min_mag    : float
    cache_path : str or None
        If provided, save the result as a parquet file at this path.

    Returns
    -------
    pd.DataFrame  columns: time (UTC), latitude, longitude, depth, mag
    """
    starttime = f"{start_year}-01-01"
    endtime   = f"{end_year}-12-31"

    print(f"Downloading seismicity: M≥{min_mag}  "
          f"{bounds}  {starttime} → {endtime}")

    chunks = []
    _download_range(bounds, starttime, endtime, min_mag, chunks)

    if not chunks:
        print("No events returned.")
        return pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag"])

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["mag"]   = pd.to_numeric(df["mag"],   errors="coerce")

    print(f"Downloaded {len(df):,} events total.")

    if cache_path is not None:
        df.to_parquet(cache_path, index=False)
        print(f"Cached to {cache_path}")

    return df


def load_background_seismicity(cache_path, bounds, start_year, end_year,
                               min_mag=2.5, force_refresh=False):
    """
    Load background seismicity, downloading and caching if needed.

    On first call (or if force_refresh=True) this queries USGS and writes
    a parquet file to cache_path.  Subsequent calls load from the cache.

    Parameters
    ----------
    cache_path    : str   Path to parquet cache file.
    bounds        : tuple (lon_min, lon_max, lat_min, lat_max)
    start_year    : int
    end_year      : int   (inclusive)
    min_mag       : float  Default 2.5
    force_refresh : bool   Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame  columns: time (UTC), latitude, longitude, depth, mag
    """
    import os
    if not force_refresh and os.path.exists(cache_path):
        print(f"Loading background seismicity from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  {len(df):,} events  M≥{df['mag'].min():.1f}  "
              f"{df['time'].min().date()} → {df['time'].max().date()}")
        return df

    return download_background_seismicity(
        bounds, start_year, end_year, min_mag, cache_path=cache_path
    )


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def add_background_seismicity(ax, df, transform,
                               color="0.3", alpha=0.15, size=2,
                               mag_scale=False, zorder=1):
    """
    Plot background seismicity as low-alpha scatter dots on a cartopy axes.

    Parameters
    ----------
    ax        : cartopy GeoAxes
    df        : DataFrame from load_background_seismicity
    transform : cartopy CRS (typically ccrs.PlateCarree())
    color     : marker color (ignored if mag_scale=True)
    alpha     : marker transparency
    size      : base marker size in points²
    mag_scale : bool  If True, scale dot area by magnitude (2^mag).
    zorder    : drawing order (keep low so it sits behind other layers)
    """
    s = 2.0 ** df["mag"].clip(lower=0) if mag_scale else size
    ax.scatter(
        df["longitude"], df["latitude"],
        s=s, c=color, alpha=alpha,
        transform=transform, zorder=zorder,
        linewidths=0,
    )
