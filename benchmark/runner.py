import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from bEPIC import EPIC_locate_prelim

def run_single_event_get_grid(run_path, prior, use_prior, params_kw, focus_version=None):
    """
    Run bEPIC for one event up through *focus_version* and return
    (SearchOut, output_df, actual_version).

    Parameters
    ----------
    run_path : str
        Path to the .run trigger file.
    prior : SeismicPrior
        Spatial prior to use.
    use_prior : bool
        Whether to weight by the prior (False = uniform/likelihood-only).
    params_kw : dict
        Keys: grid_size, grid_km, max_trigs.
    focus_version : int or None
        Version to stop at (inclusive). None = last.
    """
    df_run = pd.read_csv(run_path)
    df_run.columns = [c.replace(' ', '_') for c in df_run.columns]

    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                    = prior
    params.use_prior                = use_prior
    params.GridSize                 = params_kw['grid_size']
    params.GridKm                   = params_kw['grid_km']
    params.method                   = 'EPIC C'
    params.MAX_EVENT_TRIGS          = params_kw['max_trigs']
    params.migrate_grid             = params_kw.get('migrate_grid', True)
    params.migrate_grid_min_triggers = params_kw.get('migrate_grid_min_triggers', 1)

    first = df_run.sort_values('order').iloc[0]
    event = EPIC_locate_prelim.Event(
        lat        = first['latitude'],
        lon        = first['longitude'],
        time       = first['trigger_time'],
        misfit_rms = 0,
        misfit_ave = 0,
        eventid    = Path(run_path).stem,
        version    = 0,
    )

    versions = sorted(df_run['version'].unique())
    target_v = versions[-1] if focus_version is None else int(focus_version)

    t_out = out_df = None
    for version in versions:
        df_v = (df_run[df_run['version'] == version]
                .sort_values('order')
                .head(params.MAX_EVENT_TRIGS))

        event.trigs   = []
        event.version = int(version)
        for row in df_v.itertuples(index=False):
            trig = EPIC_locate_prelim.TriggerManager(
                lon          = row.longitude,
                lat          = row.latitude,
                trigger_time = row.trigger_time,
                sta          = row.station,
                net          = row.network,
                chan          = row.channel,
            )
            event.trigs.append(trig)

        t_out, out_df = EPIC_locate_prelim.E2Location_locate(params, event)

        if version >= target_v or len(df_v) >= params.MAX_EVENT_TRIGS:
            target_v = version   # record the version that was actually used
            break

    return t_out, out_df, target_v


class BenchmarkRunner:
    """
    Runs bEPIC location over a collection of events from .run files.

    For each event all versions are processed in order, mirroring how
    bEPIC receives triggers sequentially in real time.  The prior is
    shared across all events and can be updated between events — either
    on a fixed time schedule via an ETAS callback, or manually via
    update_prior().

    Parameters
    ----------
    prior : SeismicPrior
        Initial spatial prior.  For fixed priors (NSHM, GEAR1, etc.)
        this never changes.  For ETAS, pass etas_update_fn to run_all().
    params : EPIC_locate_prelim.EPIC_PARAMS
        Configured parameter object (GridSize, GridKm, method, etc.).
        params.prior is overwritten on each event with the current prior.
    run_dir : str
        Directory containing <event_id>.run files.
    """

    def __init__(self, prior, params, run_dir):
        self.prior   = prior
        self.params  = params
        self.run_dir = run_dir
        self.results = {}   # keyed by (event_id, version) -> (SearchOut, DataFrame)

    def _normalize_columns(self, df):
        df.columns = [c.replace(' ', '_') for c in df.columns]
        return df

    def update_prior(self, new_prior):
        # TODO - add in condion of some sort to automatically retreive a new prior for ETAS only (AND other time dependent later)
        self.prior = new_prior

    def run_event(self, event_id):
        """
        Run all versions for a single event.

        Parameters
        ----------
        event_id : int
        """
        run_path = os.path.join(self.run_dir, f'{event_id}.run')
        df_run   = self._normalize_columns(pd.read_csv(run_path))
        self.params.prior = self.prior

        first = df_run.sort_values('order').iloc[0]

        # Create initial event
        event = EPIC_locate_prelim.Event(
            lat        = first['latitude'],
            lon        = first['longitude'],
            time       = first['trigger_time'],
            misfit_rms = 0,
            misfit_ave = 0,
            eventid    = event_id,
            version    = 0,
        )

        # Iterate over the versions (new version every time new trigger)
        for version in sorted(df_run['version'].unique()):
            df_v = (df_run[df_run['version'] == version]
                    .sort_values('order')
                    .head(self.params.MAX_EVENT_TRIGS))

            event.trigs = []
            event.version = int(version)
            for row in df_v.itertuples(index=False):
                trig = EPIC_locate_prelim.TriggerManager(
                    lon          = row.longitude,
                    lat          = row.latitude,
                    trigger_time = row.trigger_time,
                    sta          = row.station,
                    net          = row.network,
                    chan         = row.channel,
                )
                event.trigs.append(trig)

            t, _ = EPIC_locate_prelim.E2Location_locate(self.params, event)
            self.results[(event_id, version)] = t

            if len(df_v) >= self.params.MAX_EVENT_TRIGS:
                break

    def _get_event_time(self, event_id):
        """Return the first trigger time in the run file as a proxy for event time."""
        run_path = os.path.join(self.run_dir, f'{event_id}.run')
        df = pd.read_csv(run_path, nrows=1)
        return float(df['trigger time'].iloc[0])

    def run_all(self, event_ids, etas_update_fn=None, update_interval_s=3600,
                after_event_fn=None):
        """
        Loop over events in order, optionally updating the prior on a
        fixed time schedule.

        Parameters
        ----------
        event_ids : list[int or str]
            Event IDs to process, in the order they should run.
        etas_update_fn : callable, optional
            Called as etas_update_fn(event_time: float) -> SeismicPrior
            whenever the elapsed event time since the last update exceeds
            update_interval_s.  If None, the prior is never updated.
        update_interval_s : float
            How often (in event seconds) to invoke etas_update_fn.
            Set to 0 to update before every event.  Default 3600 (1 hour).
        after_event_fn : callable, optional
            Called as after_event_fn(event_id) immediately after each event
            is located.  Intended for appending the just-located event to a
            time-dependent model (e.g. EtasPriorUpdater) so that the next
            prior update sees it.
        """
        last_update_time = None

        for event_id in event_ids:
            if etas_update_fn is not None:
                event_time = self._get_event_time(event_id)
                if (last_update_time is None or
                        (event_time - last_update_time) >= update_interval_s):
                    new_prior = etas_update_fn(event_time)
                    self.update_prior(new_prior)
                    last_update_time = event_time

            self.run_event(event_id)

            if after_event_fn is not None:
                after_event_fn(event_id)


# ---------------------------------------------------------------------------
# Catalog lookup
# ---------------------------------------------------------------------------

def load_reference_catalog(catalog_path):
    """
    Read a bEPIC testing catalog (tab-separated) and return a DataFrame that
    maps postgres IDs to ANSS reference locations.

    The catalog must contain at minimum the columns:
        postgres id, ANSS ID, ANSS lat, ANSS lon, ANSS depth, ANSS mag

    Parameters
    ----------
    catalog_path : str
        Path to the catalog file (e.g. bEPIC_testing_catalog.txt).

    Returns
    -------
    DataFrame with columns:
        event_id   — postgres id (int); matches run file stems
        anss_id    — USGS/ANSS event ID string (e.g. 'nc73093981')
        usgs_lat   — ANSS catalog latitude
        usgs_lon   — ANSS catalog longitude
        usgs_depth — ANSS catalog depth (km)
        usgs_mag   — ANSS catalog magnitude

    Suitable for passing directly to compute_location_error().
    """
    raw = pd.read_csv(catalog_path, sep='\t')
    return pd.DataFrame({
        'event_id':   raw['postgres id'].astype(int),
        'anss_id':    raw['ANSS ID'],
        'usgs_time':  pd.to_datetime(raw['ANSS date'],
                                     format='%Y-%m-%d-%H:%M:%S.%f-GMT'),
        'usgs_lat':   raw['ANSS lat'],
        'usgs_lon':   raw['ANSS lon'],
        'usgs_depth': raw['ANSS depth'],
        'usgs_mag':   raw['ANSS mag'],
    })


def compute_location_error(results_df, catalog_df=None):
    """
    Adds a location_error_km column to results_df by comparing posterior
    estimates against USGS catalog locations.

    Parameters
    ----------
    results_df : DataFrame
        Output from run_all / run_prior; must have columns
        event_id, posterior_lat, posterior_lon.
    catalog_df : DataFrame or None
        Must have columns: event_id, usgs_lat, usgs_lon.
        If None, results_df is returned unchanged.

    Returns
    -------
    DataFrame with location_error_km column appended (NaN for events
    not present in catalog_df).
    """
    from obspy.geodetics import gps2dist_azimuth

    if catalog_df is None:
        return results_df

    merged = results_df.merge(
        catalog_df[['event_id', 'usgs_lat', 'usgs_lon']],
        on='event_id', how='left'
    )

    def _dist_km(row):
        if pd.isna(row['usgs_lat']) or pd.isna(row['usgs_lon']):
            return np.nan
        m, _, _ = gps2dist_azimuth(
            row['usgs_lat'], row['usgs_lon'],
            row['posterior_lat'], row['posterior_lon']
        )
        return m / 1000.0

    merged['location_error_km'] = merged.apply(_dist_km, axis=1)
    return merged.drop(columns=['usgs_lat', 'usgs_lon'])


def create_reference_locations(run_dir, output_dir, cache_paths, ref_params):
    """
    Run the reference location benchmark and write results to a CSV.

    Uses its own prior and grid parameters (from ref_params), independent of
    whatever the main benchmarking workflow is configured with.

    Parameters
    ----------
    run_dir : str
        Directory containing <event_id>.run files.
    output_dir : str
        Directory where the reference CSV will be written.
    cache_paths : dict
        Mapping of prior name -> .tt3 file path (or None for Uniform).
        Must contain the key named by ref_params['prior'].
    ref_params : dict
        Reference run configuration with keys:
            prior     — prior name, must be a key in cache_paths
            max_trigs — maximum station triggers per version
            grid_size — number of grid points along one axis
            grid_km   — half-width of the search grid in km

    Returns
    -------
    str : path to the written CSV file.
    """
    from priors import SeismicPrior

    prior_name = ref_params['prior']
    cache_key  = next(k for k in cache_paths if k.lower() == prior_name.lower())
    tt3_path   = cache_paths[cache_key]

    if tt3_path is None:
        fallback = next(v for v in cache_paths.values() if v is not None)
        prior = SeismicPrior.from_tt3(fallback)
        use_prior = False
    else:
        prior = SeismicPrior.from_tt3(tt3_path)
        use_prior = True

    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                    = prior
    params.use_prior                = use_prior
    params.GridSize                 = ref_params['grid_size']
    params.GridKm                   = ref_params['grid_km']
    params.method                   = 'EPIC C'
    params.MAX_EVENT_TRIGS          = ref_params['max_trigs']
    params.migrate_grid             = ref_params.get('migrate_grid', True)
    params.migrate_grid_min_triggers = ref_params.get('migrate_grid_min_triggers', 1)

    event_ids = sorted(int(f.stem) for f in Path(run_dir).glob('*.run'))
    runner = BenchmarkRunner(prior=prior, params=params, run_dir=run_dir)
    runner.run_all(event_ids)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"REFERENCE_{ref_params['max_trigs']}.csv")
    _cols = ['event_id', 'version', 'posterior_lat', 'posterior_lon',
             'best_misfit', 'best_like', 'best_prior', 'frac_misfit']
    pd.DataFrame(
        [{'event_id': eid, 'version': ver, 'posterior_lat': t.posterior_lat,
          'posterior_lon': t.posterior_lon, 'best_misfit': t.best_misfit,
          'best_like': t.best_like, 'best_prior': t.best_prior,
          'frac_misfit': t.frac_misfit}
         for (eid, ver), t in runner.results.items()],
        columns=_cols,
    ).sort_values(['event_id', 'version']).to_csv(out_path, index=False)

    return out_path


def run_prior(args):
    """
    Module-level worker for ProcessPoolExecutor.

    Loads a prior from a .tt3 path (or uses uniform weighting if cache_path
    is None), runs all events in run_dir, and saves results to a CSV in
    output_dir named <prior_name>_benchmark_results.csv.

    Parameters (passed as a single dict so the executor only needs one arg)
    ----------
    prior_name : str
    cache_path : str or None   — .tt3 file; None means uniform (no prior)
    nshm_path  : str           — .tt3 used purely for grid geometry when uniform
    run_dir    : str
    output_dir : str
    grid_size  : int
    grid_km    : int
    max_trigs  : int

    Returns
    -------
    prior_name : str
    """
    from priors import SeismicPrior

    prior_name = args['prior_name']
    cache_path = args['cache_path']

    if cache_path is None:
        p = SeismicPrior.from_tt3(args['nshm_path'])
        use_prior = False
    else:
        p = SeismicPrior.from_tt3(cache_path)
        use_prior = True

    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                    = p
    params.use_prior                = use_prior
    params.GridSize                 = args['grid_size']
    params.GridKm                   = args['grid_km']
    params.method                   = 'EPIC C'
    params.MAX_EVENT_TRIGS          = args['max_trigs']
    params.migrate_grid             = args.get('migrate_grid', True)
    params.migrate_grid_min_triggers = args.get('migrate_grid_min_triggers', 1)

    runner = BenchmarkRunner(prior=p, params=params, run_dir=args['run_dir'])
    stems = [f.stem for f in Path(args['run_dir']).glob('*.run')]
    try:
        event_ids = sorted(int(s) for s in stems)
    except ValueError:
        event_ids = sorted(stems)
    runner.run_all(event_ids)

    rows = [
        {'event_id': eid, 'version': ver, 'posterior_lat': t.posterior_lat,
         'posterior_lon': t.posterior_lon, 'best_misfit': t.best_misfit,
         'best_like': t.best_like, 'best_prior': t.best_prior,
         'frac_misfit': t.frac_misfit}
        for (eid, ver), t in runner.results.items()
    ]
    os.makedirs(args['output_dir'], exist_ok=True)
    out_path = os.path.join(args['output_dir'], f"{prior_name.lower()}_benchmark_results.csv")
    _cols = ['event_id', 'version', 'posterior_lat', 'posterior_lon',
             'best_misfit', 'best_like', 'best_prior', 'frac_misfit']
    pd.DataFrame(rows, columns=_cols).sort_values(['event_id', 'version']).to_csv(out_path, index=False)
    return prior_name


def run_all_priors_parallel(worker_fn, job_args):
    """
    Dispatch a list of per-prior job dicts to a ProcessPoolExecutor and
    print pass/fail for each prior as it completes.

    Parameters
    ----------
    worker_fn : callable
        Module-level worker (e.g. run_prior).  Must accept a single dict
        with at least a 'prior_name' key.
    job_args : list of dict
        One dict per prior, each with the keys expected by worker_fn.
    """
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_fn, a): a['prior_name'] for a in job_args}
        for f in as_completed(futures):
            name = futures[f]
            exc  = f.exception()
            if exc:
                print(f"{name} FAILED: {exc}")
            else:
                print(f"{name} done")
