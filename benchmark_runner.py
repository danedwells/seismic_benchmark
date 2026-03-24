import os
import pandas as pd
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
        df_run   = pd.read_csv(run_path)
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
