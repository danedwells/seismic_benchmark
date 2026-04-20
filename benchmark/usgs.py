import os
import time
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from pathlib import Path

from benchmark import runner as benchmark_runner
from bEPIC import EPIC_locate_prelim

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def get_usgs_event(anss_id):
    """
    Return the USGS ComCat GeoJSON for a given ANSS event ID.

    Parameters
    ----------
    anss_id : str
        USGS/ANSS event ID (e.g. 'nc73093981').
    """
    url = USGS_QUERY_URL
    r = requests.get(url, params={"eventid": anss_id, "format": "geojson"}, timeout=30)
    r.raise_for_status()
    print("get_usgs_event: ",r.json())
    return r.json()


def get_phases_df(geojson):
    """
    Download phase arrivals from a USGS event GeoJSON.

    Tries phases.csv first; falls back to parsing quakeml.xml when phases.csv
    is absent.

    Returns a DataFrame with columns:
        Channel, Distance, Phase, Arrival Time, Residual
    or None if no phase product is available.
    """
    # The GeoJSON 'products' dict maps product type → list of versions.
    # 'phase-data' is the USGS product that contains arrival picks.
    products       = geojson.get("properties", {}).get("products", {})
    phase_products = products.get("phase-data", [])
    if not phase_products:
        return None

    # Each product version has a 'contents' dict mapping filename → metadata
    # (contentType, url, sha256, etc.).  We always take the first (preferred) version.
    contents = phase_products[0].get("contents", {})

    # phases.csv is a flat tabular format — straightforward to read directly.
    if "phases.csv" in contents:
        url = contents["phases.csv"]["url"]
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))

    # quakeml.xml is the standard seismological XML exchange format.
    # Parse it manually to extract the same columns phases.csv would have.
    if "quakeml.xml" in contents:
        return _parse_phases_from_quakeml(contents["quakeml.xml"]["url"])

    return None


def _parse_phases_from_quakeml(url):
    """
    Extract phase arrivals from a USGS QuakeML XML file.

    QuakeML separates the *when* (picks) from the *what* (arrivals):
      - <pick>    records the observed arrival time at a specific channel.
      - <arrival> (inside <origin>) links a pick to a phase label (P, S, ...)
                  and adds derived quantities like distance and time residual.

    The two are linked by publicID/pickID references, so we build a pick
    lookup first, then join it with the arrival table.

    Returns a DataFrame with columns matching phases.csv
    (Channel, Distance, Phase, Arrival Time, Residual), or None if
    no usable arrivals are found.
    """
    import xml.etree.ElementTree as ET

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # ET.fromstring parses the raw XML bytes into an element tree.
    root = ET.fromstring(r.content)

    # QuakeML files embed a namespace URI in every tag, e.g.
    # '{http://quakeml.org/xmlns/bed/1.2}pick'.  _loc() strips that prefix
    # so we can match by plain local name regardless of which namespace
    # variant USGS used.
    def _loc(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    # Return the first direct child of el whose local tag name matches.
    def _child(el, local_name):
        return next((c for c in el if _loc(c.tag) == local_name), None)

    # --- Pass 1: collect all <pick> elements into a lookup keyed by publicID ---
    # Each <pick> looks like:
    #   <pick publicID="quakeml:.../pick/12345">
    #     <time><value>2019-07-04T17:33:29.42Z</value></time>
    #     <waveformID networkCode="CI" stationCode="WMF" channelCode="HHZ" locationCode="--"/>
    #   </pick>
    picks = {}
    for el in root.iter():
        if _loc(el.tag) != 'pick':
            continue
        pid     = el.get('publicID')
        time_el = _child(el, 'time')
        # Arrival time is nested: <time><value>ISO-string</value></time>
        val_el  = _child(time_el, 'value') if time_el is not None else None
        # waveformID attributes identify the channel (net/sta/cha/loc)
        wfid    = _child(el, 'waveformID')
        if not pid or val_el is None or wfid is None:
            continue
        picks[pid] = {
            'time': val_el.text,
            'net':  wfid.get('networkCode', ''),
            'sta':  wfid.get('stationCode', ''),
            'cha':  wfid.get('channelCode', ''),
            # locationCode can be absent or empty; normalise to '--'
            'loc':  wfid.get('locationCode', '--') or '--',
        }

    # --- Pass 2: collect all <arrival> elements and join with picks ---
    # Each <arrival> looks like:
    #   <arrival>
    #     <pickID>quakeml:.../pick/12345</pickID>
    #     <phase>P</phase>
    #     <distance>0.47</distance>          <!-- epicentral distance, degrees -->
    #     <timeResidual>0.12</timeResidual>  <!-- observed minus predicted travel time -->
    #   </arrival>
    rows = []
    for el in root.iter():
        if _loc(el.tag) != 'arrival':
            continue
        pick_id_el = _child(el, 'pickID')
        phase_el   = _child(el, 'phase')
        dist_el    = _child(el, 'distance')
        resid_el   = _child(el, 'timeResidual')
        if pick_id_el is None or phase_el is None:
            continue
        # Look up the corresponding pick by its ID
        p = picks.get(pick_id_el.text)
        if p is None:
            continue
        # Format Channel to match the phases.csv convention: 'CI WMF HHZ --'
        rows.append({
            'Channel':      f"{p['net']} {p['sta']} {p['cha']} {p['loc']}",
            'Distance':     float(dist_el.text) if dist_el is not None else None,
            'Phase':        phase_el.text,
            'Arrival Time': p['time'],
            'Residual':     float(resid_el.text) if resid_el is not None else None,
        })

    return pd.DataFrame(rows) if rows else None

def download_case_study_catalog(cs, cache_dir):
    """
    Download a USGS earthquake catalog for a case study.

    Uses the FDSN event CSV endpoint.  Results are cached as a parquet file
    so subsequent calls are instant.

    Parameters
    ----------
    cs : dict
        Case study dict with keys: name, starttime, endtime, bounds, min_mag.
        bounds = (min_lon, max_lon, min_lat, max_lat).
    cache_dir : str
        Directory for the parquet cache file.

    Returns
    -------
    DataFrame with columns: id, time, latitude, longitude, depth, mag
    """
    name       = cs['name'].replace(' ', '_')
    cache_path = os.path.join(cache_dir, f"{name}_catalog.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached catalog: {cache_path}")
        return pd.read_parquet(cache_path)

    min_lon, max_lon, min_lat, max_lat = cs['bounds']
    params = {
        'format':        'csv',
        'starttime':     cs['starttime'],
        'endtime':       cs['endtime'],
        'minlongitude':  min_lon,
        'maxlongitude':  max_lon,
        'minlatitude':   min_lat,
        'maxlatitude':   max_lat,
        'minmagnitude':  cs['min_mag'],
        'orderby':       'time',
    }

    print(f"Downloading catalog for {cs['name']}...")
    r = requests.get(USGS_QUERY_URL, params=params, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))

    # The USGS FDSN CSV uses 'id' as the event ID column name.
    # Normalise defensively in case the header includes a BOM prefix.
    df.columns = [c.lstrip('#').strip() for c in df.columns]
    df = df.rename(columns={'EventID': 'id', 'eventid': 'id'})
    df = df[['id', 'time', 'latitude', 'longitude', 'depth', 'mag']].copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)

    print(f"  {len(df)} events downloaded.")
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


#%%
# ---------------------------------------------------------------------------
# Step 2: Build .run files from USGS phase data
# ---------------------------------------------------------------------------

def _parse_channel(ch):
    """Parse a USGS phases.csv Channel string into (net, sta, cha, loc).

    Example input: 'NC SAO HHZ --'
    """
    parts = ch.strip().split()
    loc   = parts[3] if len(parts) > 3 else '--'
    return parts[0], parts[1], parts[2], loc


def build_run_file_from_usgs(
    anss_id,
    origin_time_iso,
    out_path,
    max_dist_deg=5.0,
    phases_filter=None,
    station_coord_cache=None,
):
    """
    Fetch USGS phase data for a single event and write a .run trigger file.

    The output format mirrors data/run_files/*.run:
        version, order, station, channel, network, location,
        longitude, latitude, trigger time, tterr, logPd

    Versioning: triggers are sorted by arrival time; a new version is started
    each time a new unique station (net+sta) arrives.  Version k contains all
    triggers accumulated up to and including the k-th distinct station.

    logPd is set to 0.0 — peak-displacement values are not available from
    USGS phase data and are not used by the bEPIC location algorithm.

    Parameters
    ----------
    anss_id : str
        USGS event ID (e.g. 'ci38457511').
    origin_time_iso : str
        ISO-8601 origin time, used to query IRIS FDSNWS for station coords.
    out_path : str
        Destination path for the .run CSV file.
    max_dist_deg : float
        Drop phases with epicentral distance > this value (degrees).
    phases_filter : list[str] or None
        Phase types to retain. Defaults to ['P', 'Pn', 'Pg', 'Pb'].
    station_coord_cache : dict or None
        Mutable {(net, sta): (lat, lon)} dict; shared across events to avoid
        redundant IRIS queries.

    Returns
    -------
    bool : True if the file was written, False if skipped (insufficient data).
    """
    if phases_filter is None:
        phases_filter = ['P', 'Pn', 'Pg', 'Pb']
    if station_coord_cache is None:
        station_coord_cache = {}

    # --- fetch event and phases from USGS ---
    geojson   = get_usgs_event(anss_id)
    #print(geojson)
    phases_df = get_phases_df(geojson)

    if phases_df is None or phases_df.empty:
        print(f"  {anss_id}: no phases.csv — skipped")
        #raise Exception
        return False

    phases_df = phases_df[phases_df['Phase'].isin(phases_filter)].copy()
    phases_df = phases_df[phases_df['Distance'] <= max_dist_deg].copy()
    if phases_df.empty:
        print(f"  {anss_id}: no phases after filter — skipped")
        return False

    # --- parse 'Channel' column: 'NC SAO HHZ --' → (net, sta, cha, loc) ---
    parsed = phases_df['Channel'].apply(_parse_channel)
    phases_df['network']  = [p[0] for p in parsed]
    phases_df['station']  = [p[1] for p in parsed]
    phases_df['channel']  = [p[2] for p in parsed]
    phases_df['location'] = [p[3] for p in parsed]

    # --- arrival time → Unix timestamp ---
    phases_df['trigger_time'] = phases_df['Arrival Time'].apply(
        lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    )
    phases_df['tterr'] = phases_df['Residual'].fillna(0.0)

    # --- sort by arrival time; deduplicate same channel (keep first arrival) ---
    phases_df = (phases_df
                 .sort_values('trigger_time')
                 .drop_duplicates(subset=['network', 'station', 'channel'], keep='first')
                 .reset_index(drop=True))

    # Assign a permanent global order (1-based, in arrival-time sequence).
    phases_df['order'] = range(1, len(phases_df) + 1)

    # --- resolve station coordinates via IRIS FDSNWS ---
    rows = []
    for _, row in phases_df.iterrows():
        key = (row['network'], row['station'])
        if key not in station_coord_cache:
            lat, lon = benchmark_runner.get_station_coords(
                row['network'], row['station'], row['channel'], origin_time_iso
            )
            station_coord_cache[key] = (lat, lon)

        sta_lat, sta_lon = station_coord_cache[key]
        if sta_lat is None:
            continue   # IRIS returned nothing for this station

        rows.append({
            'network':      row['network'],
            'station':      row['station'],
            'channel':      row['channel'],
            'location':     row['location'],
            'latitude':     sta_lat,
            'longitude':    sta_lon,
            'trigger_time': row['trigger_time'],
            'tterr':        row['tterr'],
            'order':        int(row['order']),
        })

    if len(rows) < 2:
        print(f"  {anss_id}: fewer than 2 triggers with coordinates — skipped")
        return False

    trig_df = (pd.DataFrame(rows)
               .sort_values('trigger_time')
               .reset_index(drop=True))

    # --- assign version numbers ---
    # A new version is created each time a new (net, sta) is encountered.
    # version_first[i] = the version index at which trigger i first appears.
    seen_stations = []
    version_first = []
    for _, row in trig_df.iterrows():
        sta_key = (row['network'], row['station'])
        if sta_key not in seen_stations:
            seen_stations.append(sta_key)
        version_first.append(len(seen_stations) - 1)   # 0-indexed

    trig_df['version_first'] = version_first
    n_versions = len(seen_stations)

    # --- build cumulative version blocks ---
    # Version v contains ALL triggers whose version_first <= v.
    final_rows = []
    for v in range(n_versions):
        for _, row in trig_df[trig_df['version_first'] <= v].iterrows():
            final_rows.append({
                'version':      v,
                'order':        row['order'],
                'station':      row['station'],
                'channel':      row['channel'],
                'network':      row['network'],
                'location':     row['location'],
                'longitude':    row['longitude'],
                'latitude':     row['latitude'],
                'trigger time': row['trigger_time'],
                'tterr':        round(float(row['tterr']), 4),
                'logPd':        0.0,
            })

    out_df = (pd.DataFrame(final_rows)
              [['version', 'order', 'station', 'channel', 'network', 'location',
                'longitude', 'latitude', 'trigger time', 'tterr', 'logPd']]
              .sort_values(['version', 'order']))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  {anss_id}: {n_versions} versions, {len(trig_df)} triggers → written")
    return True


def build_run_files_for_case_study(
    catalog_df,
    run_dir,
    max_dist_deg=5.0,
    phases_filter=None,
    request_delay=1.5,
    skip_existing=True,
):
    """
    Build .run files for every event in a case-study catalog.

    Iterates over rows, calls build_run_file_from_usgs for each, and shares
    a station-coordinate cache across events to reduce IRIS API calls.

    Parameters
    ----------
    catalog_df : DataFrame
        Must have columns: id, time.
    run_dir : str
        Output directory for .run files.
    max_dist_deg : float
        Passed to build_run_file_from_usgs.
    phases_filter : list[str] or None
        Passed to build_run_file_from_usgs.
    request_delay : float
        Seconds to pause between events (avoids USGS/IRIS rate-limiting).
    skip_existing : bool
        Skip events whose .run file already exists in run_dir.

    Returns
    -------
    list[str] : ANSS IDs for which a .run file was successfully written.
    """
    os.makedirs(run_dir, exist_ok=True)
    station_coord_cache = {}
    written = []
    n = len(catalog_df)

    for i, row in enumerate(catalog_df.itertuples(index=False), start=1):
        anss_id    = row.id
        origin_iso = (row.time.isoformat()
                      if hasattr(row.time, 'isoformat') else str(row.time))
        out_path   = os.path.join(run_dir, f"{anss_id}.run")

        if skip_existing and os.path.exists(out_path):
            written.append(anss_id)
            continue

        print(f"[{i}/{n}] {anss_id}")
        try:
            ok = build_run_file_from_usgs(
                anss_id             = anss_id,
                origin_time_iso     = origin_iso,
                out_path            = out_path,
                max_dist_deg        = max_dist_deg,
                phases_filter       = phases_filter,
                station_coord_cache = station_coord_cache,
            )
            if ok:
                written.append(anss_id)
        except Exception as exc:
            print(f"  {anss_id}: ERROR — {exc}")

        time.sleep(request_delay)

    print(f"\n{len(written)}/{n} run files written to {run_dir}")
    return written

def get_station_coords(network, station, channel, origin_time_iso):
    """
    Query IRIS FDSNWS for the lat/lon of a station at the time of the event.

    Returns (lat, lon) or (None, None) if not found.
    """
    url = "https://service.iris.edu/fdsnws/station/1/query"
    params = {
        "net":      network,
        "sta":      station,
        "cha":      channel,
        "level":    "station",
        "format":   "text",
        "endafter": origin_time_iso[:10],
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        lines = [ln for ln in r.text.strip().split("\n") if not ln.startswith("#")]
        if lines:
            parts = lines[0].split("|")
            return float(parts[2]), float(parts[3])
    except Exception:
        pass
    return None, None


def build_event_and_triggers(anss_id, max_dist_deg=5.0, phases_filter=None):
    """
    Download USGS phase data for a given ANSS event ID and return a populated
    EPIC_locate_prelim.Event object ready for E2Location_locate().

    To look up the ANSS ID from a postgres ID, use load_reference_catalog()
    first and index into the returned DataFrame:
        catalog_df = load_reference_catalog('bEPIC_testing_catalog.txt')
        anss_id = catalog_df.set_index('event_id').loc[postgres_id, 'anss_id']

    Parameters
    ----------
    anss_id : str
        USGS/ANSS event ID (e.g. 'nc73093981').
    max_dist_deg : float
        Keep only phases with epicentral distance <= this many degrees.
    phases_filter : list of str or None
        Phase types to include. Defaults to ['P', 'Pn', 'Pg', 'Pb'].

    Returns
    -------
    EPIC_locate_prelim.Event or None on failure.
    """
    if phases_filter is None:
        phases_filter = ["P", "Pn", "Pg", "Pb"]

    print(f"\nFetching event {anss_id} from USGS ComCat...")
    geojson = get_usgs_event(anss_id)

    props  = geojson["properties"]
    coords = geojson["geometry"]["coordinates"]  # [lon, lat, depth_km]
    evlon   = coords[0]
    evlat   = coords[1]
    evmag   = props["mag"]
    evtime  = props["time"] / 1000.0   # ms → seconds since epoch
    origin_iso = datetime.fromtimestamp(evtime, tz=timezone.utc).isoformat()

    print(f"  Title : {props.get('title', anss_id)}")
    print(f"  Origin: lat={evlat:.4f}  lon={evlon:.4f}  M={evmag}")

    print("\nDownloading phases.csv from USGS...")
    phases_df = get_phases_df(geojson)
    if phases_df is None:
        print("  ERROR: No phases.csv product found.")
        return None

    phases_df = phases_df[phases_df["Phase"].isin(phases_filter)].copy()
    phases_df = phases_df[phases_df["Distance"] <= max_dist_deg].copy()
    print(f"  Phases after filter: {len(phases_df)}")
    if phases_df.empty:
        print("  No phases remain after filtering.")
        return None

    def parse_channel(ch):
        parts = ch.strip().split()
        return parts[0], parts[1], parts[2], (parts[3] if len(parts) > 3 else "--")

    parsed = phases_df["Channel"].apply(parse_channel)
    phases_df["net"] = [p[0] for p in parsed]
    phases_df["sta"] = [p[1] for p in parsed]
    phases_df["cha"] = [p[2] for p in parsed]

    event = EPIC_locate_prelim.Event(
        lat=evlat, lon=evlon, time=evtime,
        misfit_rms=0, misfit_ave=0, eventid=anss_id, version=0,
    )

    print("\nFetching station coordinates from IRIS FDSNWS...")
    coord_cache = {}
    seen = set()
    for _, row in phases_df.iterrows():
        key = (row["net"], row["sta"], row["cha"])
        if key in seen:
            continue
        seen.add(key)

        cache_key = (row["net"], row["sta"])
        if cache_key in coord_cache:
            sta_lat, sta_lon = coord_cache[cache_key]
        else:
            sta_lat, sta_lon = get_station_coords(row["net"], row["sta"], row["cha"], origin_iso)
            coord_cache[cache_key] = (sta_lat, sta_lon)

        if sta_lat is None:
            print(f"  SKIP {row['net']}.{row['sta']}.{row['cha']} — coordinates not found")
            continue

        arrival_dt = datetime.fromisoformat(row["Arrival Time"].replace("Z", "+00:00"))
        t = EPIC_locate_prelim.TriggerManager(
            lon=sta_lon, lat=sta_lat,
            sta=row["sta"], net=row["net"], chan=row["cha"],
            trigger_time=arrival_dt.timestamp(),
        )
        event.trigs.append(t)
        print(f"  ADD {row['net']}.{row['sta']}.{row['cha']:5s}  "
              f"lat={sta_lat:.4f}  lon={sta_lon:.4f}  dist={row['Distance']:.2f}°")

    print(f"\nTotal triggers added: {len(event.trigs)}")
    return event