#%%
# =============================================================================
# saf_quiet_zones.py — where is the San Andreas quiet in small earthquakes?
# Counts M>=MIN_MAG events (USGS ComCat, 2000-2025) within CORRIDOR_KM of an approximate fault trace, per BIN_KM of
# along-strike distance. The trace is a hand-digitized polyline (a few km accuracy), NOT a QFaults trace.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark.background import load_background_seismicity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'saf_seismicity_M2.parquet')
MIN_MAG, CORRIDOR_KM, BIN_KM = 2.0, 15.0, 20.0

# approximate SAF trace, north to south: (name, lon, lat)
TRACE = [('Point Reyes', -122.80, 38.00), ('San Francisco Pen.', -122.45, 37.65), ('Santa Cruz Mtns', -122.00, 37.20),
         ('San Juan Bautista', -121.55, 36.85), ('Creeping', -121.20, 36.50), ('Creeping S', -120.70, 36.10),
         ('Parkfield', -120.45, 35.90), ('Cholame', -120.30, 35.72), ('Carrizo', -119.83, 35.27),
         ('Frazier Mtn', -119.00, 34.82), ('Palmdale', -118.10, 34.58), ('Cajon Pass', -117.50, 34.32),
         ('San Bernardino', -117.00, 34.15), ('Banning', -116.50, 33.90), ('Indio', -116.00, 33.60),
         ('Bombay Beach', -115.70, 33.35)]

#%%
df = load_background_seismicity(CACHE, (-123.2, -115.0, 32.8, 38.3), 2000, 2025, min_mag=MIN_MAG)
print(len(df), 'events')

kx = 111.32 * np.cos(np.radians(36.0))
tr = np.array([[(lo + 123.2) * kx, (la - 32.8) * 110.57] for _, lo, la in TRACE])
seg_len = np.hypot(*np.diff(tr, axis=0).T)
cum = np.concatenate([[0], np.cumsum(seg_len)])          # along-strike distance of each vertex (km)


def project(x, y):
    """Distance to the polyline and along-strike position of each point."""
    best_d, best_s = np.full(len(x), np.inf), np.zeros(len(x))
    for i in range(len(tr) - 1):
        a, b = tr[i], tr[i + 1]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        d = np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))
        better = d < best_d
        best_d[better], best_s[better] = d[better], cum[i] + t[better] * seg_len[i]
    return best_d, best_s


d, s = project((df['longitude'].values + 123.2) * kx, (df['latitude'].values - 32.8) * 110.57)
near = d <= CORRIDOR_KM
edges = np.arange(0, cum[-1] + BIN_KM, BIN_KM)
counts = np.histogram(s[near], bins=edges)[0]
mid = edges[:-1] + BIN_KM / 2

table = pd.DataFrame({'start_km': edges[:-1], 'count': counts})
# lat/lon of each bin centre
table['lon'] = np.interp(mid, cum, [p[1] for p in TRACE])
table['lat'] = np.interp(mid, cum, [p[2] for p in TRACE])
table['nearest'] = [TRACE[int(np.argmin(np.abs(cum - m)))][0] for m in mid]
print(f'\nM>={MIN_MAG:g} events within {CORRIDOR_KM:g} km of trace, 2000-2025, per {BIN_KM:g} km bin (quietest 12):')
print(table.sort_values('count').head(12).to_string(index=False))

#%%
# ---------------------------------------------------------------------------
# What the priors say along the fault: density per km^2 at each bin centre, relative to the Carrizo bin.
# KDE_Seismicity = static .tt3; ETAS = EtasPriorUpdater at ETAS_DATE (spatial background + triggering from the
# previous year of the historical catalog).
# ---------------------------------------------------------------------------
import benchmark.config_california as config
from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner

ETAS_DATE = pd.Timestamp('2008-01-01')
INVERSION_JSON = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion',
                              f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"])}.json')
HIST_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input',
                            f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"])}.csv')
hist = pd.read_csv(HIST_CATALOG, index_col=0, dtype={'url': str, 'alert': str})
hist['time'] = pd.to_datetime(hist['time'], format='ISO8601', utc=True).dt.tz_convert(None)
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, catalog_df=hist, **config.ETAS_UPDATER_CONFIG)
priors = {'KDE': SeismicPrior.from_tt3(os.path.join(SeismicPrior.data_dir, 'kde_seismicity_benchmark.tt3')),
          'ETAS': updater.update(ETAS_DATE)}


def density_at(prior, lat, lon):
    """Prior probability per km^2 of the grid cell containing (lat, lon)."""
    i, j = np.argmin(np.abs(prior.lats - lat)), np.argmin(np.abs(prior.lons - lon))
    dlat, dlon = np.median(np.diff(prior.lats)), np.median(np.diff(prior.lons))
    return prior.grid[i, j] / (dlat * 110.57 * dlon * 111.32 * np.cos(np.radians(lat)))


for name, pr in priors.items():
    table[name] = [density_at(pr, la, lo) for la, lo in zip(table['lat'], table['lon'])]
ref = table.loc[table['count'].idxmin()]
for name in priors:
    table[name + '_rel'] = table[name] / ref[name]
print(f'\nAlong-strike profile (prior density relative to the quietest bin, which is at {ref.lat:.2f}N {ref.lon:.2f}W):')
print(table[table['start_km'].between(ref.start_km - 220, ref.start_km + 220)]
      [['start_km', 'lat', 'lon', 'nearest', 'count', 'KDE_rel', 'ETAS_rel']].round(2).to_string(index=False))

#%%
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(mid, counts, width=BIN_KM * 0.9, color='steelblue')
for name, _, _ in TRACE:
    ax.axvline(cum[[p[0] for p in TRACE].index(name)], color='gray', lw=0.5)
    ax.text(cum[[p[0] for p in TRACE].index(name)], ax.get_ylim()[1] * 0.98, name, rotation=90, va='top', fontsize=7)
ax.set_xlabel('Distance along approximate San Andreas trace, NW to SE (km)')
ax.set_ylabel(f'M≥{MIN_MAG:g} events per {BIN_KM:g} km, 2000–2025')
ax.set_yscale('log')
fig.savefig(os.path.join(PROJECT_ROOT, 'data_examination_scripts', 'saf_quiet_zones.png'), dpi=200, bbox_inches='tight')
