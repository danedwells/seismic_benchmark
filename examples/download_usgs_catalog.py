"""
download_usgs_catalog.py

Standalone example: query the USGS ComCat earthquake catalog over a region
and time range, and save every matching event into a single CSV file.

USGS caps each request at 20,000 events. This recursively splits up the chunks
for queries over 20,000, until each chunk < 20,000

Usage
-----
Edit the parameters in the `if __name__ == "__main__":` block at the
bottom (region bounds, date range, minimum magnitude, output path) and run:

    python download_usgs_catalog.py

Or import and call `download_usgs_catalog(...)` directly from your own code.

Requires: requests, pandas, matplotlib, cartopy
"""

import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import StringIO

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
MAX_EVENTS_PER_REQUEST = 20_000
REQUEST_DELAY = 0.5  # seconds between requests -- be polite to the API


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
    return pd.read_csv(StringIO(r.text), low_memory=False)


def _download_range(bounds, starttime, endtime, min_mag, chunks):
    """
    Recursively split [starttime, endtime] until each piece is under the
    event limit, then download and collect all chunks into `chunks` list.
    """
    n = _count_events(bounds, starttime, endtime, min_mag)
    if n == 0:
        return
    if n <= MAX_EVENTS_PER_REQUEST:
        print(f"  Fetching {n:>6,} events  {starttime} -> {endtime}")
        time.sleep(REQUEST_DELAY)
        df = _fetch_chunk(bounds, starttime, endtime, min_mag)
        chunks.append(df)
    else:
        # Bisect the time range and recurse on each half.
        t0 = pd.Timestamp(starttime)
        t1 = pd.Timestamp(endtime)
        mid = t0 + (t1 - t0) / 2
        mid_str = mid.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"  Splitting [{starttime} -> {endtime}] ({n:,} events > limit)")
        _download_range(bounds, starttime, mid_str, min_mag, chunks)
        _download_range(bounds, mid_str, endtime, min_mag, chunks)


def download_usgs_catalog(bounds, starttime, endtime, min_mag, out_csv):
    """
    Download every USGS event matching the query and write it to one CSV.

    Parameters
    ----------
    bounds : tuple
        (lon_min, lon_max, lat_min, lat_max)
    starttime, endtime : str
        ISO-8601 dates/datetimes, e.g. "2019-01-01" or "2019-01-01T00:00:00".
    min_mag : float
        Minimum magnitude to include.
    out_csv : str
        Path to write the combined CSV to.

    Returns
    -------
    pd.DataFrame with all downloaded events.
    """
    print(f"Downloading catalog: M>={min_mag}  {bounds}  {starttime} -> {endtime}")

    chunks = []
    _download_range(bounds, starttime, endtime, min_mag, chunks)

    if not chunks:
        print("No events found for this query.")
        df = pd.DataFrame()
    else:
        df = pd.concat(chunks, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

    print(f"Downloaded {len(df):,} events total.")
    df.to_csv(out_csv, index=False)
    print(f"Wrote catalog to {out_csv}")
    return df


def plot_catalog_map(df, bounds, out_png=None):
    """
    Quick sanity-check map: event locations (sized by magnitude) over
    coastlines/borders, with the requested query box overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'latitude', 'longitude', 'mag' columns.
    bounds : tuple
        (lon_min, lon_max, lat_min, lat_max) -- the query extent to draw.
    out_png : str or None
        If given, save the figure to this path.
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    pad = 1.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad],
                  crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")

    # query box
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color="red", linewidth=1.5, transform=ccrs.PlateCarree())

    if not df.empty:
        ax.scatter(df["longitude"], df["latitude"],
                   s=5 * 2 ** df["mag"], c=df["mag"], cmap="viridis",
                   alpha=0.6, edgecolor="k", linewidth=0.2,
                   transform=ccrs.PlateCarree())

    ax.set_title(f"USGS catalog: {len(df):,} events")
    ax.gridlines(draw_labels=True, linewidth=0.3)

    if out_png is not None:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Wrote map to {out_png}")
    plt.show()


if __name__ == "__main__":
    # --- Example: Southern California, 2019, M >= 3.0 ---
    # (this window includes the Ridgecrest sequence, so it exercises the
    # recursive bisection path)
    # Matches benchmark/config_cascadia.py's REFERENCE_CATALOG_CONFIG.
    REF_DIR   = "/home/a01738353/2024_NEHRP/seismic_benchmark/"
    BOUNDS    = (-131.0, -115.0, 40.5, 50.5)  # lon_min, lon_max, lat_min, lat_max
    STARTTIME = "2022-01-01"
    ENDTIME   = "2026-07-31"
    MIN_MAG   = 3.0

    OUT_CSV   = f"{REF_DIR}data/cascadia/reference/cascadia_test_catalog.csv"

    catalog = download_usgs_catalog(BOUNDS, STARTTIME, ENDTIME, MIN_MAG, OUT_CSV)

    # Sanity-check plot
    plot_catalog_map(catalog, BOUNDS, out_png=f"{REF_DIR}data/cascadia/reference/usgs_catalog_map.png")
    
