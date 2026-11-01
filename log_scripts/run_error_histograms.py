"""
Location and magnitude error histograms for a whole EPIC replay run.

Reads every <anss_id>.log in a run folder, takes the first EPIC version whose
station count (nS, the E:I: column) reaches N_TRIGS, and compares its location and
magnitude to the USGS/ANSS solution.  Logs with no event reaching N_TRIGS are skipped.
USGS solutions are cached to a CSV so re-runs don't re-query ComCat.
"""
#%%
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.usgs import get_usgs_event

SEQUENCE = 'Ridgecrest' # 'MTJ'
WHICH = "sigma_and_prior" # 'sigma_and_prior' 'sigma_only'


RUN_DIR = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{WHICH}' / 'epic'
N_TRIGS = 5
MAX_ORIGIN_DIFF_S = 30          # match log event to USGS origin time within this
USGS_CACHE = Path(__file__).resolve().parent / f'usgs_cache_{SEQUENCE}.csv'

#%% Parse logs: one row (event, version) per E:I: line with eventid > 0
def parse_log(path):
    rows = []
    for line in path.open(errors='replace'):
        _, _, msg = line.partition('| INFO | E:I:')
        if msg.startswith('F:'):       # 'E:I:F:' = the event's first alert version (same columns as E:I:)
            msg = msg[2:]
        f = msg.split()
        if len(f) < 17 or not f[0].lstrip('-').isdigit() or int(f[0]) <= 0:
            continue
        rows.append((int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[5]),
                     pd.Timestamp(f[6]), int(f[15]), int(f[16])))
    return pd.DataFrame(rows, columns=['eventid', 'ver', 'lat', 'lon', 'mag', 'time', 'nT', 'nS'])

logs = {p.stem: parse_log(p) for p in sorted(RUN_DIR.glob('*.log'))}
print(f'{len(logs)} logs; {sum(len(d) > 0 for d in logs.values())} contain an alerted event')

#%% USGS solutions (cached)
def fetch_usgs(anss_id):
    try:
        g = get_usgs_event(anss_id)
        lon, lat, dep = g['geometry']['coordinates']
        return dict(anss_id=anss_id, lat=lat, lon=lon, mag=g['properties']['mag'],
                    time=pd.to_datetime(g['properties']['time'], unit='ms'))
    except Exception as e:
        print(f'USGS lookup failed for {anss_id}: {e}')

usgs = pd.read_csv(USGS_CACHE, parse_dates=['time']) if USGS_CACHE.exists() else pd.DataFrame(
    columns=['anss_id', 'lat', 'lon', 'mag', 'time'])
todo = [i for i, d in logs.items() if len(d) and i not in set(usgs.anss_id)]
if todo:
    with ThreadPoolExecutor(8) as ex:
        new = [r for r in ex.map(fetch_usgs, todo) if r]
    usgs = pd.concat([usgs, pd.DataFrame(new)], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.set_index('anss_id')

#%% Errors at the first version with nS >= N_TRIGS
out = []
for anss_id, d in logs.items():
    if not len(d) or anss_id not in usgs.index:
        continue
    u = usgs.loc[anss_id]
    d = d[d.nS >= N_TRIGS]
    if d.empty:
        continue
    # Several events can be alerted in one log; take the one whose origin matches USGS
    dt = (d.time - u.time).abs().dt.total_seconds()
    d = d[dt == dt.min()]
    if dt.min() > MAX_ORIGIN_DIFF_S:
        continue
    r = d.sort_values('ver').iloc[0]
    out.append(dict(anss_id=anss_id, ver=r.ver, nS=r.nS, usgs_mag=u.mag,
                    loc_err_km=gps2dist_azimuth(u.lat, u.lon, r.lat, r.lon)[0] / 1e3,
                    mag_err=r.mag - u.mag))
res = pd.DataFrame(out)
print(f'{len(res)} events with a matched version at nS >= {N_TRIGS}')
print(res[['loc_err_km', 'mag_err']].describe())

#%% Plot
fig, (ax_loc, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_loc.hist(res.loc_err_km, bins=np.logspace(np.log10(res.loc_err_km.min()), np.log10(res.loc_err_km.max()), 41),
            color='k', rwidth=0.9)
ax_loc.set_xscale('log')
ax_loc.set_xlabel('Location error vs USGS (km)'); ax_loc.set_ylabel('Events')
ax_loc.set_title(f'Location error (nS >= {N_TRIGS}, n={len(res)})')
ax_mag.hist(res.mag_err, bins=40, color='k', rwidth=0.9)
ax_mag.axvline(0, color='r', ls='--')
ax_mag.set_xlabel('EPIC mag - USGS mag'); ax_mag.set_ylabel('Events')
ax_mag.set_title(f'Magnitude error (nS >= {N_TRIGS}, n={len(res)})')
fig.tight_layout()
out_path = Path(__file__).resolve().parent / 'figures' / f'{RUN_DIR.parent.name}_error_hist_nS{N_TRIGS}.png'
fig.savefig(out_path, dpi=150)
print('Saved', out_path)

# %%
