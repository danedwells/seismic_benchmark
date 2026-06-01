import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from bEPIC import EPIC_locate_prelim
from .metrics import (posterior_confidence_level, prior_confidence_level,
                      posterior_coverage, location_error_km, COVERAGE_RADII_KM,
                      log_score, brier_score, energy_score)

# When the nearest triggered station is farther than this after the first
# location, the trigger set is resampled to simulate real-system uncertainty.
_DISTANT_EVENT_THRESHOLD_KM = 200.0
_DISTANT_EVENT_SEED_TRIGS   = 3


def get_unique_stations(run_dir):
    """Return a DataFrame of unique stations (by station+network) across all run files."""
    frames = [pd.read_csv(f, usecols=['station', 'network', 'longitude', 'latitude'])
              for f in Path(run_dir).glob('*.run')]
    return pd.concat(frames).drop_duplicates(subset=['station', 'network']).reset_index(drop=True)


def load_station_availability_cache(path):
    """
    Load the pre-built station availability cache produced by
    preparation_scripts/build_station_availability.py.

    Returns
    -------
    dict[int, pd.DataFrame]
        Maps event_id → DataFrame(station, network, longitude, latitude)
        containing the stations that had ??Z waveform data at event time.
        Pass the per-event DataFrame to make_epic_params(station_inventory=...)
        or set it directly on BenchmarkRunner via station_availability=.
    """
    df = pd.read_parquet(path)
    return {
        str(eid): grp.drop(columns='event_id').reset_index(drop=True)
        for eid, grp in df.groupby('event_id')
    }


def make_epic_params(prior, use_prior, benchmark_params, station_inventory=None):
    """Return a configured EPIC_PARAMS object from a benchmark params dict.

    benchmark_params must contain: grid_size, grid_km, max_trigs.
    Optional keys (with defaults): migrate_grid (True), migrate_grid_min_triggers (1),
    activity_mask_threshold (0.30).

    station_inventory : pandas.DataFrame or None
        DataFrame with columns station, network, longitude, latitude.
        None (default) disables the activity mask.
        Tier 1 (benchmark): pass get_unique_stations(run_dir).
        Tier 2 (case study): pass get_fdsn_station_inventory(origin_time).
    """
    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                     = prior
    params.use_prior                 = use_prior
    params.GridSize                  = benchmark_params['grid_size']
    params.GridKm                    = benchmark_params['grid_km']
    params.method                    = 'EPIC C'
    params.MAX_EVENT_TRIGS           = benchmark_params['max_trigs']
    params.migrate_grid              = benchmark_params.get('migrate_grid', True)
    params.migrate_grid_min_triggers = benchmark_params.get('migrate_grid_min_triggers', 1)
    params.station_inventory         = station_inventory
    params.activity_threshold        = benchmark_params.get('activity_threshold', 0.40)
    params.resample_distant_events   = benchmark_params.get('resample_distant_events', True)
    return params


def runner_results_to_df(runner):
    """Convert a BenchmarkRunner's results and metrics into a tidy DataFrame."""
    cov_cols = [f'coverage_{r}km' for r in COVERAGE_RADII_KM]
    rows = []
    for (eid, ver), t in runner.results.items():
        m = runner.metrics.get((eid, ver), {})
        row = {
            'event_id':                  eid,
            'version':                   ver,
            'n_trigs':                   runner.n_trigs.get((eid, ver)),
            'posterior_lat':             t.posterior_lat,
            'posterior_lon':             t.posterior_lon,
            'exp_lat':                   t.exp_lat,
            'exp_lon':                   t.exp_lon,
            'like_lon':                  t.like_lon,
            'like_lat':                  t.like_lat,
            'like_exp_lon':              t.like_exp_lon,
            'like_exp_lat':              t.like_exp_lat,
            'best_misfit':               t.best_misfit,
            'best_like':                 t.best_like,
            'best_prior':                t.best_prior,
            'frac_misfit':               t.frac_misfit,
            'map_err_km':                    m.get('map_err_km'),
            'exp_err_km':                    m.get('exp_err_km'),
            'like_err_km':                   m.get('like_err_km'),
            'like_exp_err_km':               m.get('like_exp_err_km'),
            'log_score':                     m.get('log_score'),
            'brier_score':                   m.get('brier_score'),
            'energy_score':                  m.get('energy_score'),
            'posterior_confidence_level':    m.get('posterior_confidence_level'),
            'prior_confidence_level':     m.get('prior_confidence_level'),
        }
        for col in cov_cols:
            row[col] = m.get(col)
        rows.append(row)
    _cols = (['event_id', 'version', 'n_trigs', 'posterior_lat', 'posterior_lon',
              'exp_lat', 'exp_lon', 'like_lon', 'like_lat','like_exp_lon','like_exp_lat',
              'like_err_km','like_exp_err_km',
              'best_misfit', 'best_like', 'best_prior', 'frac_misfit',
              'map_err_km', 'exp_err_km', 'log_score', 'brier_score', 'energy_score'] + cov_cols
             + ['posterior_confidence_level', 'prior_confidence_level'])
    return pd.DataFrame(rows, columns=_cols).sort_values(['event_id', 'version'])


def run_single_event_get_grid(run_path, prior, use_prior, params_kw,
                               focus_version=None, station_inventory=None):
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
    station_inventory : pandas.DataFrame or None
        Passed to make_epic_params; enables the activity mask when provided.
    """
    df_run = pd.read_csv(run_path)
    df_run.columns = [c.replace(' ', '_') for c in df_run.columns]

    params = make_epic_params(prior, use_prior, params_kw,
                              station_inventory=station_inventory)

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

    def __init__(self, prior, params, run_dir, catalog_df=None,
                 station_availability=None, rng=None,
                 resample_distant_events=None):
        self.prior   = prior
        self.params  = params
        self.run_dir = run_dir
        self.results = {}   # {(event_id, version): SearchOut}
        self.metrics = {}   # {(event_id, version): {map_err_km, coverage, posterior_confidence_level}}
        self.n_trigs = {}   # {(event_id, version): int trigger count fed to bEPIC}

        # Debug hook: set _debug_event_id to an event_id string/int before
        # calling run_event() to capture out_df for every version of that event.
        self._debug_event_id = None
        self.debug_out_df    = {}   # {(event_id, version): out_df}

        self._rng = np.random.default_rng(rng)
        if resample_distant_events is None:
            self.resample_distant_events = getattr(params, 'resample_distant_events', True)
        else:
            self.resample_distant_events = resample_distant_events

        # Optional per-event station inventory from build_station_availability.py.
        # If provided, params.station_inventory is set per event before locating.
        # If None, params.station_inventory is left as-is (set by the caller or disabled).
        self._station_availability = station_availability  # dict[int, DataFrame] or None

        if catalog_df is not None:
            self._ref_lookup = {
                str(row.event_id): (float(row.usgs_lat), float(row.usgs_lon))
                for row in catalog_df[['event_id', 'usgs_lat', 'usgs_lon']].itertuples(index=False)
            }
        else:
            self._ref_lookup = {}

    def _build_resample_sequence(self, df_run):
        """
        Build a synthetic trigger sequence for a distant event.

        Takes the full trigger set from the final run-file version, randomly
        picks _DISTANT_EVENT_SEED_TRIGS as the starting subset, then appends
        the remaining triggers in their original arrival order (by 'order' column).

        Returns a list of DataFrames, one per synthetic version, each containing
        the cumulative trigger set for that version.
        """
        all_trigs = (df_run[df_run['version'] == df_run['version'].max()]
                     .sort_values('order')
                     .head(self.params.MAX_EVENT_TRIGS)
                     .reset_index(drop=True))
        n = len(all_trigs)
        k = min(_DISTANT_EVENT_SEED_TRIGS, n)

        seed_idx = sorted(self._rng.choice(n, size=k, replace=False).tolist())
        rest_idx = [i for i in range(n) if i not in set(seed_idx)]
        # Seed triggers first (in arrival order), then remaining in arrival order.
        ordered = seed_idx + rest_idx

        return [all_trigs.iloc[ordered[:end]] for end in range(k, n + 1)]

    def _run_resample_event(self, event_id, event, df_run):
        """
        Run the synthetic trigger sequence for a distant event.

        Iterates the versions produced by _build_resample_sequence and stores
        results under version keys offset by 10000 to avoid collision with any
        normal-mode versions already stored for this event.
        """
        for syn_idx, syn_df_v in enumerate(self._build_resample_sequence(df_run)):
            syn_version   = 10000 + syn_idx
            event.trigs   = []
            event.version = syn_version
            for row in syn_df_v.itertuples(index=False):
                event.trigs.append(EPIC_locate_prelim.TriggerManager(
                    lon          = row.longitude,
                    lat          = row.latitude,
                    trigger_time = row.trigger_time,
                    sta          = row.station,
                    net          = row.network,
                    chan         = row.channel,
                ))
            t, out_df = EPIC_locate_prelim.E2Location_locate(self.params, event)
            self.results[(event_id, syn_version)] = t
            self.n_trigs[(event_id, syn_version)] = len(syn_df_v)
            if self._ref_lookup:
                self._compute_event_metrics(event_id, syn_version, t, out_df)
            if len(syn_df_v) >= self.params.MAX_EVENT_TRIGS:
                break

    def _normalize_columns(self, df):
        df.columns = [c.replace(' ', '_') for c in df.columns]
        return df

    def _compute_event_metrics(self, event_id, version, t, out_df):
        """Compute and store posterior accuracy metrics for one (event, version)."""
        ref = self._ref_lookup.get(str(event_id))
        if ref is None or t is None or out_df is None:
            return
        usgs_lat, usgs_lon = ref
        err_km     = location_error_km(t.posterior_lat, t.posterior_lon, usgs_lat, usgs_lon)
        exp_err_km = location_error_km(t.exp_lat,       t.exp_lon,       usgs_lat, usgs_lon)
        like_err_km =  location_error_km(t.like_lat,       t.like_lon,   usgs_lat, usgs_lon)
        like_exp_err_km =  location_error_km(t.like_exp_lat,t.like_exp_lon, usgs_lat, usgs_lon)
        cov        = posterior_coverage(out_df, usgs_lat, usgs_lon)
        cred       = posterior_confidence_level(out_df, usgs_lat, usgs_lon)
        prior_cred = prior_confidence_level(out_df, usgs_lat, usgs_lon)
        ls         = log_score(out_df, usgs_lat, usgs_lon)
        bs         = brier_score(out_df, usgs_lat, usgs_lon)
        es         = energy_score(out_df, usgs_lat, usgs_lon, rng=self._rng)
        m = {'map_err_km': err_km, 'exp_err_km': exp_err_km, 
             'like_err_km': like_err_km,
             'like_exp_err_km': like_exp_err_km,
             'posterior_confidence_level': cred,
             'prior_confidence_level': prior_cred,
             'log_score': ls, 'brier_score': bs, 'energy_score': es}
        for r in COVERAGE_RADII_KM:
            m[f'coverage_{r}km'] = cov[r]
        self.metrics[(event_id, version)] = m

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

        # Station availability inventory is passed to BenchmarkRunner at __init__
        if self._station_availability is not None:
            self.params.station_inventory = self._station_availability.get(str(event_id))

        # Reset per-event state so previous event's posterior doesn't leak in
        self.params.prev_posterior_lat = None
        self.params.prev_posterior_lon = None

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
        prev_n_trigs = -1
        for version in sorted(df_run['version'].unique()):
            df_v = (df_run[df_run['version'] == version]
                    .sort_values('order')
                    .head(self.params.MAX_EVENT_TRIGS))

            # Skip versions where the trigger count hasn't changed — the run
            # file can contain many consecutive versions at the same count
            # (bEPIC refreshes its solution without a new trigger arriving),
            # which would otherwise produce duplicate rows in the output CSV.
            if len(df_v) == prev_n_trigs:
                continue
            prev_n_trigs = len(df_v)

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

            t, out_df = EPIC_locate_prelim.E2Location_locate(self.params, event)
            self.params.prev_posterior_lat = t.posterior_lat
            self.params.prev_posterior_lon = t.posterior_lon
            self.results[(event_id, version)] = t
            self.n_trigs[(event_id, version)] = len(df_v)
            if self._ref_lookup:
                self._compute_event_metrics(event_id, version, t, out_df)
            if self._debug_event_id is not None and str(event_id) == str(self._debug_event_id):
                self.debug_out_df[(event_id, version)] = out_df

            # If every triggered station is farther than the threshold, the
            # initial estimate is unreliable.  Resample the trigger set:
            # randomly pick _DISTANT_EVENT_SEED_TRIGS as the new starting
            # subset, then add the rest one by one in original arrival order.
            if self.resample_distant_events:
                min_dist_km = min(trig.dist for trig in event.trigs)
                if min_dist_km > _DISTANT_EVENT_THRESHOLD_KM:
                    self._run_resample_event(event_id, event, df_run)
                    return

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

    def update_prior(self, new_prior):
        """Swap in a new prior; also updates params.prior so the next run_event uses it."""
        self.prior = new_prior
        self.params.prior = new_prior


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
    if catalog_df is None:
        return results_df

    merged = results_df.merge(
        catalog_df[['event_id', 'usgs_lat', 'usgs_lon']],
        on='event_id', how='left'
    )

    def _dist_km(row):
        if pd.isna(row['usgs_lat']) or pd.isna(row['usgs_lon']):
            return np.nan
        return location_error_km(row['posterior_lat'], row['posterior_lon'],
                                 row['usgs_lat'], row['usgs_lon'])

    merged['location_error_km'] = merged.apply(_dist_km, axis=1)
    return merged.drop(columns=['usgs_lat', 'usgs_lon'])


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

    params = make_epic_params(p, use_prior, args,
                              station_inventory=args.get('station_inventory'))

    catalog_df = args.get('catalog_df')
    if catalog_df is None:
        catalog_path = args.get('catalog_path')
        if catalog_path and os.path.exists(catalog_path):
            catalog_df = load_reference_catalog(catalog_path)

    # Initiate the runner
    runner = BenchmarkRunner(prior=p, params=params, run_dir=args['run_dir'],
                             catalog_df=catalog_df,
                             station_availability=args.get('station_availability'))
    

    stems = [f.stem for f in Path(args['run_dir']).glob('*.run')]
    try:
        event_ids = sorted(int(s) for s in stems)
    except ValueError:
        event_ids = sorted(stems)
    runner.run_all(event_ids)

    os.makedirs(args['output_dir'], exist_ok=True)
    out_path = os.path.join(args['output_dir'], f"{prior_name.lower()}_benchmark_results.csv")
    runner_results_to_df(runner).to_csv(out_path, index=False)
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
