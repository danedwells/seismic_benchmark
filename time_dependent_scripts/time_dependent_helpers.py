import os
import pandas as pd
from priors import SeismicPrior


# -- Define callbacks ----------------------------------------------------
def run_trigger_time(event_id, RUN_DIR):
    path = os.path.join(RUN_DIR, f'{event_id}.run')
    try:
        df  = pd.read_csv(path, nrows=1)
        col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
        return float(df[col].iloc[0])
    except Exception:
        return 0.0

def etas_update_fn(event_time_unix: float, updater, PRIOR_ALPHA) -> SeismicPrior:
    t     = pd.Timestamp(event_time_unix, unit='s')
    prior = updater.update(t)
    if PRIOR_ALPHA != 1.0:
        prior.grid  = prior.grid ** PRIOR_ALPHA
        prior.grid /= prior.grid.sum()
    print(f"  [ETAS] prior updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
            f"— catalog size: {updater.n_catalog_events}")
    return prior