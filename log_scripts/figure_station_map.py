"""
Map of which stations report in the benchmark's USGS-phase .run file vs the EPIC replay log
for the 2024 M7.0 MTJ mainshock (nc75095651): only in the benchmark run, only in EPIC, in both,
and in neither (stations either source knows about that never reported).  Two panels: the
first 10 stations to report (what the benchmark locator uses at max_trigs=10) and the first 70
(the size of the full EPIC log).  Stations are matched on (network, station code).
"""
#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RUN  = 'MTJ_sigma_and_prior'      # EPIC run whose log is compared
EVENT_ID = 'nc75095651'
LOG  = ROOT / 'log_files' / 'MTJ-selected' / RUN / 'epic' / f'{EVENT_ID}.log'
CS   = ROOT / 'data' / 'california' / 'case_studies' / 'MTJ_2024_M7'
OUT  = Path(__file__).resolve().parent / 'figures' / 'station_map.png'
USGS = (40.3740, -125.0217)
N_PANELS = (10, 70)

#%% EPIC log: triggers per version (E:I:T:) and stations EPIC considered (A:C:)
trig_hdr, trig_rows, ac_rows = None, [], []
for line in LOG.open(errors='replace'):
    _, _, msg = line.partition('| INFO | E:I:T:')
    if msg:
        f = msg.split()
        if f and f[0] == 'H:':
            trig_hdr = trig_hdr or f[1:]
        elif trig_hdr and len(f) >= len(trig_hdr) and f[0].lstrip('-').isdigit():
            trig_rows.append(f[:len(trig_hdr)])
        continue
    _, _, msg = line.partition('| INFO | A:C:')
    f = msg.split()
    if len(f) >= 6 and f[0].isdigit():
        ac_rows.append((f[3], f[2], float(f[4]), float(f[5])))
t = pd.DataFrame(trig_rows, columns=trig_hdr)
for c in ('eventid', 'ver', 'order', 'lat', 'lon'):
    t[c] = pd.to_numeric(t[c])
t = t[t.eventid > 0]
epic_order = t.sort_values(['ver', 'order']).drop_duplicates(['net', 'sta'])      # first-report order
ac = pd.DataFrame(ac_rows, columns=['net', 'sta', 'lat', 'lon']).drop_duplicates(['net', 'sta'])

#%% Benchmark .run (USGS phases) and per-event availability cache
run = pd.read_csv(CS / 'run_files' / f'{EVENT_ID}.run')
cache = pd.read_parquet(CS / 'station_availability_cache.parquet')
cache = cache[cache.event_id == EVENT_ID].rename(columns={'network': 'net', 'station': 'sta',
                                                          'longitude': 'lon', 'latitude': 'lat'})
run_xy = run.rename(columns={'network': 'net', 'station': 'sta', 'longitude': 'lon', 'latitude': 'lat'})

# Every station either source knows about, with coordinates ("neither" = known but never reported)
known = (pd.concat([cache[['net', 'sta', 'lat', 'lon']], ac, epic_order[['net', 'sta', 'lat', 'lon']],
                    run_xy[['net', 'sta', 'lat', 'lon']]])
         .drop_duplicates(['net', 'sta']).set_index(['net', 'sta']))

def bench_first(N):
    """First N stations to report in the benchmark run (= the stations in its N-trigger version)."""
    v = run[run.version == N - 1].sort_values('order')
    return list(dict.fromkeys(zip(v.network, v.station)))[:N]

def epic_first(N):
    return list(zip(epic_order.net, epic_order.sta))[:N]

#%% Map
STYLE = {'neither': dict(marker='o', s=14, facecolor='none', edgecolor='gray', lw=.6, zorder=3),
         'both':    dict(marker='^', s=70, facecolor='k', edgecolor='k', zorder=5),
         'mine':    dict(marker='^', s=110, facecolor='darkblue', edgecolor='w', lw=.8, zorder=6),
         'epic':    dict(marker='^', s=110, facecolor='red', edgecolor='w', lw=.8, zorder=6)}
PC = ccrs.PlateCarree()
fig, axs = plt.subplots(1, 2, figsize=(15, 8), subplot_kw=dict(projection=PC))
for ax, N in zip(axs, N_PANELS):
    mine, epic = set(bench_first(N)), set(epic_first(N))
    cats = {'both': mine & epic, 'mine': mine - epic, 'epic': epic - mine}
    cats['neither'] = set(known.index) - mine - epic
    ax.set_extent([-126.6, -121.9, 38.4, 42.4])
    ax.add_feature(cfeature.LAND, facecolor='#f2f2f2'); ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')
    ax.add_feature(cfeature.COASTLINE, lw=.7); ax.add_feature(cfeature.STATES, lw=.4, edgecolor='gray')
    for k in ('neither', 'both', 'mine', 'epic'):
        ks = known.loc[sorted(cats[k])]
        ax.scatter(ks.lon, ks.lat, transform=PC, **STYLE[k])
        print(f'first {N}: {k:8s} {len(ks):4d}', sorted(f'{n}.{s}' for n, s in cats[k]) if k != 'neither' and N == 10 else '')
    ax.plot(USGS[1], USGS[0], '*', c='gold', mec='k', ms=20, zorder=9, transform=PC)
    gl = ax.gridlines(draw_labels=True, lw=.3, alpha=.5, ls='--'); gl.top_labels = gl.right_labels = False
    ax.set_title(f'First {N} stations to report'
                 + (' (what the benchmark locator uses)' if N == 10 else ' (size of the full EPIC log)'),
                 fontweight='bold')
    ax.legend(handles=[
        Line2D([], [], marker='^', ls='', c='darkblue', ms=9, label=f'only in benchmark .run ({len(cats["mine"])})'),
        Line2D([], [], marker='^', ls='', c='red', ms=9, label=f'only in EPIC log ({len(cats["epic"])})'),
        Line2D([], [], marker='^', ls='', c='k', ms=8, label=f'in both ({len(cats["both"])})'),
        Line2D([], [], marker='o', ls='', mfc='none', mec='gray', ms=5,
               label=f'in neither ({len(cats["neither"])}; known stations that never reported)'),
        Line2D([], [], marker='*', ls='', c='gold', mec='k', ms=14, label='USGS epicenter')],
        loc='lower left', fontsize=9, framealpha=.95)
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print('Saved', OUT)
