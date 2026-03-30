import os
import pandas as pd
from pathlib import Path
from bEPIC import EPIC_locate_prelim


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

            t, output_df = EPIC_locate_prelim.E2Location_locate(self.params, event)
            self.results[(event_id, version)] = (t, output_df)

            if len(df_v) >= self.params.MAX_EVENT_TRIGS:
                break

    def _get_event_time(self, event_id):
        """Return the first trigger time in the run file as a proxy for event time."""
        run_path = os.path.join(self.run_dir, f'{event_id}.run')
        df = pd.read_csv(run_path, nrows=1)
        return float(df['trigger time'].iloc[0])

    def run_all(self, event_ids, etas_update_fn=None, update_interval_s=3600):
        """
        Loop over events in order, optionally updating the prior on a
        fixed time schedule.

        Parameters
        ----------
        event_ids : list[int]
            Event IDs to process, in the order they should run.
        etas_update_fn : callable, optional
            Called as etas_update_fn(event_time: float) -> SeismicPrior
            whenever the elapsed event time since the last update exceeds
            update_interval_s.  If None, the prior is never updated.
        update_interval_s : float
            How often (in event seconds) to invoke etas_update_fn.
            Default 3600 (1 hour).
        """
        last_update_time = None

        for event_id in event_ids:
            if etas_update_fn is not None:
                event_time = self._get_event_time(event_id)
                if (last_update_time is None or
                        (event_time - last_update_time) >= update_interval_s):
                    self.update_prior(etas_update_fn(event_time))
                    last_update_time = event_time

            self.run_event(event_id)


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
    params.prior          = p
    params.use_prior      = use_prior
    params.GridSize       = args['grid_size']
    params.GridKm         = args['grid_km']
    params.method         = 'EPIC C'
    params.MAX_EVENT_TRIGS = args['max_trigs']

    runner = BenchmarkRunner(prior=p, params=params, run_dir=args['run_dir'])
    event_ids = sorted(int(f.stem) for f in Path(args['run_dir']).glob('*.run'))
    runner.run_all(event_ids)

    rows = [
        {'event_id': eid, 'version': ver, 'posterior_lat': t.posterior_lat,
         'posterior_lon': t.posterior_lon, 'best_misfit': t.best_misfit,
         'best_like': t.best_like, 'best_prior': t.best_prior}
        for (eid, ver), (t, _) in runner.results.items()
    ]
    os.makedirs(args['output_dir'], exist_ok=True)
    out_path = os.path.join(args['output_dir'], f"{prior_name.lower()}_benchmark_results.csv")
    pd.DataFrame(rows).sort_values(['event_id', 'version']).to_csv(out_path, index=False)
    return prior_name
