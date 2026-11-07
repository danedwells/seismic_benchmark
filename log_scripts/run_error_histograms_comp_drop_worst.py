"""
Comparison of location and magnitude error histograms across the control, sigma_only and
sigma_and_prior EPIC replay runs of one sequence, overlaid (see run_error_histograms.py
for the single-run version).

Reads every <anss_id>.log in a run folder (parsing_tools.py), finds the EPIC event that matches the
USGS/ANSS event (within MAX_ORIGIN_DIFF_S and 100 km), takes that event's first alerted version whose
station count (nS, the E:I: column) reaches N_TRIGS, and compares its location and magnitude to the
USGS/ANSS solution.  Events without such a version are skipped.
USGS solutions are cached to a CSV so re-runs don't re-query ComCat.

Copy of run_error_histograms_comp.py that also RE-LOCATES each selected EPIC version's triggers with the
benchmark's bEPIC locator and drops the worst-fitting station (the largest travel-time residual at the
all-station solution, as in epic_log_param_search.py's drop test), then re-locates (only if that residual is an outlier: > ABS_RESID_S s or > N_SIGMA x the std of the other residuals).  control uses the
KDE_Seismicity prior and sigma_and_prior the dynamic ETAS prior (built per event in time order); the
relocated rows are 'control_drop' / 'sigma_and_prior_drop' in the figure.  The drop needs >= MIN_STATIONS stations.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark.usgs import get_usgs_event
import parsing_tools

SEQUENCE = 'MTJ' #'Ridgecrest' # 'MTJ'
MAINSHOCK_ID = {'Ridgecrest': 'ci38457511', 'MTJ': 'nc75095651'}[SEQUENCE]   # starred on the map
RUNS = {'control': 'gray', 'sigma_only': 'blue', 'sigma_and_prior': 'red'}

N_TRIGS = 5
MAX_ORIGIN_DIFF_S = 30          # match log event to USGS origin time within this
ALPHA = 0.3
MIN_USGS_MAG = 3.5            # only compare events with USGS magnitude >= this (None = all)
ALERTS_ONLY = True              # use only versions with Ast == 1 (alert actually sent)
USGS_CACHE = Path(__file__).resolve().parent / f'usgs_cache_{SEQUENCE}.csv'

#%% Parse logs: one row (event, version) per E:I: line with eventid > 0
all_logs = {}
for which in RUNS:
    run_dir = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{which}' / 'epic'
    all_logs[which] = {p.stem: parsing_tools.read_ei(p) for p in sorted(run_dir.glob('*.log'))}
    logs = all_logs[which]
    print(f'{which}: {len(logs)} logs; {sum((d.Ast == 1).any() for d in logs.values())} contain an alerted event')

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
todo = sorted({i for logs in all_logs.values() for i, d in logs.items() if (d.Ast == 1).any()} - set(usgs.anss_id))
if todo:
    with ThreadPoolExecutor(8) as ex:
        new = [r for r in ex.map(fetch_usgs, todo) if r]
    usgs = pd.concat([usgs, pd.DataFrame(new)], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.set_index('anss_id')

#%% Errors at the first version with nS >= N_TRIGS
results = {}
for which, logs in all_logs.items():
    out = []
    for anss_id, d in logs.items():
        if d.empty or anss_id not in usgs.index:
            continue
        u = usgs.loc[anss_id]
        if MIN_USGS_MAG is not None and not u.mag >= MIN_USGS_MAG:
            continue
        # Several events can be in one log; take the EPIC event that matches USGS, then its first version at N_TRIGS
        eventid = parsing_tools.match_event(d, u.time, u.lat, u.lon, max_dt_s=MAX_ORIGIN_DIFF_S)
        r = None if eventid is None else parsing_tools.first_at_stations(d, eventid, N_TRIGS, alerted_only=ALERTS_ONLY)
        if r is None:
            continue
        out.append(dict(anss_id=anss_id, eventid=r.eventid, ver=r.ver, nS=r.nS, usgs_mag=u.mag,
                        loc_err_km=gps2dist_azimuth(u.lat, u.lon, r.lat, r.lon)[0] / 1e3,
                        mag_err=r.mag - u.mag, lat=r.lat, lon=r.lon, usgs_lat=u.lat, usgs_lon=u.lon))
    results[which] = pd.DataFrame(out)
    print(f'{which}: {len(results[which])} events with a matched version at nS >= {N_TRIGS}'
          + (f' and USGS M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''))
    print(results[which][['loc_err_km', 'mag_err']].describe())

#%% Re-locate each event's triggers (benchmark locator) with all stations and with the worst station dropped
import io
import contextlib
import os
from priors import EtasPriorUpdater, SeismicPrior
import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import make_epic_params
from benchmark.usgs import download_case_study_catalog
from benchmark.metrics import _haversine_km
from bEPIC import EPIC_locate_prelim
from bEPIC.EPIC_locate_prelim import _get_travel_time_function

CASE = {'Ridgecrest': 'Ridgecrest', 'MTJ': 'MTJ_2024_M7'}[SEQUENCE]
CS_DATA = ROOT / 'data' / 'california' / 'case_studies' / CASE
DROP_RUNS = {'control': 'KDE', 'sigma_and_prior': 'ETAS'}      # EPIC run -> prior used to re-locate it
SIGMA_S, EDT_SIGMA_S, MIN_STATIONS = 0.22, 0.02, 5             # case-study settings; drop only if >= MIN_STATIONS
ABS_RESID_S, N_SIGMA = 2.0, 2.0     # drop the worst station only if its |residual| > ABS_RESID_S s OR > N_SIGMA x the std of the other stations' residuals
                                    # (None disables a test; both None = always drop, as before)
DROP_WORST = False                   # False: keep the all-station relocation (no station dropped), to test what the drop adds
USE_STATION_MASK = False             # True: bEPIC activity mask built from the collaborators' <SEQUENCE>_channels.dat (all stations available)
TRIGGERS = os.environ.get('TRIGGERS', 'all')   # which E:I:T: rows go to the locator (most stations have an HH and an HN row, picks differ by 0.1-0.4 s):
                                                    # 'all' every trigger, as EPIC does (default; each trigger counts equally, A. Williamson) | 'earliest' one per station, earliest pick | 'hh' one per station, HH channel if it has one
TAG = (('drop_worst' if DROP_WORST else 'relocated') + ('_mask' if USE_STATION_MASK else '')
       + ('' if TRIGGERS == 'earliest' else f'_{TRIGGERS}'))   # in output file names; also the suffix of the relocated runs
SUFFIX = 'drop' if DROP_WORST else 'relocated'
STATION_FILE = ROOT / f'{SEQUENCE}_channels.dat'
COLS = 'network station location channel latitude longitude elevation samplerate gain units'.split()
# One row per net.sta (the file has one per channel).  keep_default_na: a station or network named 'NA' must stay a string
INVENTORY = (pd.read_csv(STATION_FILE, sep=r'\s+', comment='#', names=COLS, dtype=str, keep_default_na=False)
             .drop_duplicates(['network', 'station'])[['station', 'network', 'longitude', 'latitude']]
             .astype({'longitude': float, 'latitude': float}).reset_index(drop=True)) if USE_STATION_MASK else None
print(f'DROP_WORST={DROP_WORST}  USE_STATION_MASK={USE_STATION_MASK}'
      + (f'  ({len(INVENTORY)} stations from {STATION_FILE.name})' if USE_STATION_MASK else ''))
PD_A, PD_B, PD_C, PD_WINDOW_S, W_FLOOR = 1.23, 1.39, 5.39, 4.0, 0.3   # EPIC Pd magnitude scaling (see epic_log_param_search.py)
TT8 = _get_travel_time_function(8.0)

def parse_trig_log(path):
    """E:I:T: rows (one trigger per station per version) of a log."""
    hdr, rows = None, []
    for line in path.open(errors='replace'):
        _, _, msg = line.partition('| INFO | E:I:T:')
        f = msg.split()
        if not f:
            continue
        if f[0] == 'H:':
            hdr = hdr or f[1:]
        elif hdr and len(f) >= len(hdr) and f[0].lstrip('-').isdigit():
            rows.append(f[:len(hdr)])
    t = pd.DataFrame(rows, columns=hdr)
    for c in ('eventid', 'ver', 'order', 'lat', 'lon', 'tterr', 'log_pd', 'updm', 'distkm', 'buftime'):
        t[c] = pd.to_numeric(t[c])
    t['t'] = pd.to_datetime(t['trigger_time']).dt.tz_localize('UTC').map(lambda x: x.timestamp())
    return t[t.eventid > 0]

def pick_triggers(tv):
    """Rows of one EPIC version handed to the locator, per TRIGGERS."""
    if TRIGGERS == 'hh':                                                                # HH rows, plus the stations that have no HH channel
        hh = tv[tv.chan.str.startswith('HH')]
        tv = pd.concat([hh, tv[~tv.sta.isin(hh.sta)]])
    g = tv if TRIGGERS == 'all' else tv.sort_values('t').drop_duplicates('sta')
    return g.sort_values('order').reset_index(drop=True)

def map_loc(g, prior):
    pk = dict(config.BENCHMARK_PARAMS)
    pk.update(sigma_s=SIGMA_S, edt_sigma_s=EDT_SIGMA_S, dtt_weight=0.0, search_depths=[8.0], max_trigs=15)
    first = g.iloc[0]
    event = EPIC_locate_prelim.Event(lat=first.lat, lon=first.lon, time=first.t, misfit_rms=0, misfit_ave=0,
                                     eventid='x', version=0)
    event.trigs = [EPIC_locate_prelim.TriggerManager(lon=r.lon, lat=r.lat, trigger_time=r.t, sta=r.sta,
                                                     net=r.net, chan=r.chan) for r in g.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        _, odf = EPIC_locate_prelim.E2Location_locate(make_epic_params(prior, True, pk, station_inventory=INVENTORY), event)
    i = int(np.argmax(odf['post'].values))
    return odf['lat'].values[i], odf['lon'].values[i]

def resid(g, lat, lon):
    """Per-station travel-time residuals (s) at a location, origin time = mean (as the locator)."""
    ot = g.t.values - TT8(np.sqrt(_haversine_km(lat, lon, g.lat.values, g.lon.values) ** 2 + 8.0 ** 2))
    return ot - ot.mean()

def err_km(u, lat, lon):
    return float(_haversine_km(u.lat, u.lon, np.array([lat]), np.array([lon]))[0])

# Priors: KDE static; ETAS from the case-study inversion, fed every catalog event before the trigger time
kde = SeismicPrior.from_tt3(os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{CASE}.tt3'))
config.ETAS_INVERSION_CONFIG['bw_sq'] = 4
inv  = CS_DATA / 'etas_inversion' / f'parameters_{config.etas_output_id(CASE)}.json'
hist = CS_DATA / 'etas_inversion' / 'input' / f'catalog_{config.etas_catalog_tag(CASE)}.csv'
benchmark_runner.repair_inversion_json_paths(str(inv))
updater = EtasPriorUpdater.from_inversion_json(
    json_path=str(inv), catalog_df=pd.read_csv(hist, index_col=0, parse_dates=['time'], dtype={'url': str, 'alert': str}),
    **config.ETAS_UPDATER_CONFIG, spatial_factor=None)
cat = download_case_study_catalog(config.CASE_STUDIES[CASE], cache_dir=str(CS_DATA), REDOWNLOAD=False)
cs = (cat[['id', 'time', 'latitude', 'longitude', 'mag']].rename(columns={'mag': 'magnitude'})
      .assign(time=lambda d: pd.to_datetime(d['time']).dt.tz_localize(None))
      .query(f"magnitude >= {config.ETAS_INVERSION_CONFIG['m_ref']}").sort_values('time'))
added = set()

def etas_prior(t, anss_id):
    t_event = pd.Timestamp(t, unit='s')
    new = cs[(cs.time < t_event) & (cs.id != anss_id) & ~cs.id.isin(added)]
    if len(new):
        updater.append_events(new[['time', 'latitude', 'longitude', 'magnitude']])
        added.update(new.id)
    return updater.update(t_event)

relocated = {}
for which, prior_name in DROP_RUNS.items():          # control (KDE prior) and sigma_and_prior (ETAS prior)
    log_dir = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{which}' / 'epic'

    # -- 1. Collect the triggers EPIC used for each matched event (the version chosen in the Errors cell) --
    jobs = []
    for row in results[which].itertuples():
        trig = parse_trig_log(log_dir / f'{row.anss_id}.log')                           # all E:I:T: rows in this log
        tv = trig[(trig.eventid == row.eventid) & (trig.ver == row.ver)]                # triggers of the selected EPIC event + version
        g = pick_triggers(tv)                                                           # locator input, in EPIC's order (see TRIGGERS)
        jobs.append((g.t.min(), row, tv, g))                                            # g.t.min() = first trigger time, used to order events

    # -- 2. Re-locate each event, oldest first (the ETAS prior must only see earlier events) --
    out = []
    for t, row, tv, g in sorted(jobs, key=lambda j: j[0]):
        u = usgs.loc[row.anss_id]                                                       # USGS location/magnitude (the truth)
        prior = kde if prior_name == 'KDE' else etas_prior(t, row.anss_id)             # static KDE, or ETAS updated to time t

        # All-station solution, and the travel-time residual of every station at that location
        lat0, lon0 = map_loc(g, prior)
        r0 = resid(g, lat0, lon0)
        w = int(np.argmax(np.abs(r0)))                                                  # index of the worst-fitting station

        # Is the worst station an outlier?  (> ABS_RESID_S seconds, or > N_SIGMA x the spread of the OTHER stations' residuals)
        others = np.delete(r0, w)
        outlier = ((ABS_RESID_S is not None and abs(r0[w]) > ABS_RESID_S)
                   or (N_SIGMA is not None and abs(r0[w]) > N_SIGMA * others.std(ddof=1))
                   or (ABS_RESID_S is None and N_SIGMA is None))                        # both tests off = always drop
        dropped = DROP_WORST and len(g) >= MIN_STATIONS and outlier                     # never drop below MIN_STATIONS stations

        # Per-event record: where EPIC put it, where all stations put it, and which station was worst
        d = dict(anss_id=row.anss_id, ver=row.ver, nS=len(g), usgs_lat=u.lat, usgs_lon=u.lon, usgs_mag=u.mag,
                 epic_err_km=row.loc_err_km, all_err_km=err_km(u, lat0, lon0), worst_station=g.sta[w], worst_resid_s=r0[w],
                 dropped=dropped)

        # Final solution: re-locate without the worst station if it was dropped, otherwise keep the all-station solution
        lat1, lon1 = map_loc(g.drop(index=w), prior) if dropped else (lat0, lon0)

        # Magnitude at the final location: EPIC's Pd scaling over the triggers it used for magnitude (updm == 1),
        # without the dropped station, weighted by waveform length (buftime)
        tw = tv[(tv.updm == 1) & (tv.sta != g.sta[w])] if dropped else tv[tv.updm == 1]
        r_new = np.maximum(_haversine_km(lat1, lon1, tw.lat.values, tw.lon.values), 1.0)   # 1 km floor: a station AT the location would give log10(0)
        mag1 = (float(np.average(PD_A * tw.log_pd.values + PD_B * np.log10(r_new) + PD_C,
                                 weights=np.clip(tw.buftime.values / PD_WINDOW_S, W_FLOOR, 1)))
                if len(tw) else np.nan)                                     # no trigger used for the magnitude (updm == 1): no Pd magnitude
        d.update(lat=lat1, lon=lon1, loc_err_km=err_km(u, lat1, lon1), mag_err=mag1 - u.mag,
                 reloc_vs_epic_km=float(_haversine_km(row.lat, row.lon, np.array([lat1]), np.array([lon1]))[0]))   # relocated vs EPIC log location
        out.append(d)

    # -- 3. Save and expose the results --
    relocated[which] = pd.DataFrame(out)                                                # one row per event, all columns -> CSV
    relocated[which].to_csv(Path(__file__).resolve().parent / f'{SEQUENCE}_{which}_{TAG}_nS{N_TRIGS}.csv', index=False)
    rd = relocated[which].dropna(subset=['loc_err_km'])
    results[f'{which}_{SUFFIX}'] =rd[['anss_id', 'lat', 'lon', 'usgs_lat', 'usgs_lon', 'loc_err_km', 'mag_err']].reset_index(drop=True)   # used by the figures
    print(f'{which} ({prior_name}): median location error (km)  EPIC log {rd.epic_err_km.median():.2f}  '
          f'relocated all stations {rd.all_err_km.median():.2f}  final ({SUFFIX}) {rd.loc_err_km.median():.2f}  (n={len(rd)}, dropped in {int(rd.dropped.sum())})\n'
          f'    relocated vs EPIC log location (km): median {rd.reloc_vs_epic_km.median():.2f}  '
          f'90th pct {rd.reloc_vs_epic_km.quantile(0.9):.2f}  max {rd.reloc_vs_epic_km.max():.2f}  [TRIGGERS={TRIGGERS}]')

#%% Plot
# Shared bins so the three runs are directly comparable
loc_all = pd.concat([r.loc_err_km for r in results.values()])
mag_all = pd.concat([r.mag_err for r in results.values()])
loc_bins = np.logspace(np.log10(loc_all.min()), np.log10(loc_all.max()), 41)
mag_bins = np.linspace(mag_all.min(), mag_all.max(), 41)

BIN_OFFSET = 0.15   # per-run shift, as a fraction of bin width, so overlapping bars stay distinguishable
BAR_WIDTH = 0.7     # bar width as a fraction of bin width

def offset_hist(ax, values, bins, k, color, label, log=False):
    """Histogram drawn as bars shifted by k*BIN_OFFSET bin widths (in log10 space if log)."""
    counts, edges = np.histogram(values, bins=bins)
    u = np.log10(edges) if log else edges
    w = np.diff(u)
    left = u[:-1] + (0.5 - BAR_WIDTH / 2 + k * BIN_OFFSET) * w
    right = left + BAR_WIDTH * w
    if log:
        left, right = 10 ** left, 10 ** right
    ax.bar(left, counts, width=right - left, align='edge', color=color, alpha=ALPHA, label=label)

fig, (ax_loc, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.5))
for i, (which, color) in enumerate(RUNS.items()):
    res = results[which]
    k = i - (len(RUNS) - 1) / 2
    label = f'{which} (n={len(res)})'
    offset_hist(ax_loc, res.loc_err_km, loc_bins, k, color, label, log=True)
    offset_hist(ax_mag, res.mag_err, mag_bins, k, color, label)
    for ax, med in ((ax_loc, res.loc_err_km.abs().median()), (ax_mag, res.mag_err.abs().median())):
        unit = ' km' if ax is ax_loc else ''
        ax.text(0.02, 0.97 - 0.07 * i, f'{which}: median |err| = {med:.2f}{unit}', transform=ax.transAxes,
                color=color, ha='left', va='top', fontsize=9)
for ax in (ax_loc, ax_mag):
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)   # headroom for annotations/legend
ax_loc.set_xscale('log')
ax_loc.set_xlabel('Location error vs USGS (km)'); ax_loc.set_ylabel('Events')
mag_note = f', M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''
ax_loc.set_title(f'Location error (nS >= {N_TRIGS}{mag_note})')
ax_loc.legend(loc='upper right')
ax_mag.axvline(0, color='k', ls='--')
ax_mag.set_xlabel('EPIC mag - USGS mag'); ax_mag.set_ylabel('Events')
ax_mag.set_title(f'Magnitude error (nS >= {N_TRIGS}{mag_note})')
ax_mag.legend(loc='upper right')
fig.tight_layout()
out_path = Path(__file__).resolve().parent / 'figures' / f'{SEQUENCE}_error_hist_{N_TRIGS}_comp_{TAG}.png'
fig.savefig(out_path, dpi=150)
print('Saved', out_path)

#%% Paper-style figure (same layout/style as paper_figures/location_hist_case_study.py):
# (a) map of EPIC locations per run with arrows to the USGS location, (b) stacked location-error histograms
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()
RUNS = {**RUNS, f'control_{SUFFIX}': 'black', f'sigma_and_prior_{SUFFIX}': 'darkred'}   # EPIC log runs + relocated (worst station dropped if DROP_WORST)
SIZE_SCALE = dict(zip(RUNS, (1.8, 1.6, 1.4, 1.2, 1.0)))   # decreasing with draw order so overlapping dots read as concentric
BUF = 0.3                                     # degrees added around the posterior locations
MIN_MAP_ASPECT = 6.3 / 7.3                      # map box width/height; tall regions are widened east-west
AX_POS = {'map': [0.05, 0.08, 0.45, 0.86], 'hists': [0.58, 0.183, 0.17, 0.654], 'mags': [0.82, 0.183, 0.17, 0.654]}

fig = plt.figure(figsize=(14, 8.5))

# -- Right: stacked location-error histograms, one per run, shared x-axis --
bins = np.logspace(-1, 3, 20)
hl, hb, hw, hh = AX_POS['hists']
h_each = hh / len(RUNS)
hist_axes = []
for i, (which, color) in enumerate(RUNS.items()):
    ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each], sharex=hist_axes[0] if hist_axes else None)
    hist_axes.append(ax)
    stats = results[which].loc_err_km.values
    if len(stats):
        ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7)
        median_err = np.median(stats)
        ax.axvline(median_err, ymax=0.55, color=color, ls='--', lw=1.5)   # stops above the bars, below the labels
        ax.text(0.03, 0.70, f'median: {median_err:.1f} km', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.text(0.03, 0.92, which, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, color=color, fontweight='bold')
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(RUNS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
hist_axes[-1].set_xscale('log')
hist_axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
hist_axes[-1].set_xlabel('Location error (km)')
_y_max = max(a.get_ylim()[1] for a in hist_axes)
for a in hist_axes:
    a.set_ylim(0, 2.0 * _y_max)   # headroom: bars top out at half height so the name/median labels sit clear above them
hist_axes[0].text(-0.2, 0.87, '(b)', transform=hist_axes[0].transAxes,
                  ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- Middle-right: stacked magnitude-error histograms (same style, shared bins) --
ml, mb, mw, mh = AX_POS['mags']
m_each = mh / len(RUNS)
mag_axes = []
for i, (which, color) in enumerate(RUNS.items()):
    ax = fig.add_axes([ml, mb + mh - (i + 1) * m_each, mw, m_each], sharex=mag_axes[0] if mag_axes else None)
    mag_axes.append(ax)
    stats = results[which].mag_err.dropna().values
    if len(stats):
        ax.hist(stats, bins=mag_bins, rwidth=0.9, color=color, alpha=0.7)
        median_err = np.median(stats)
        ax.axvline(median_err, ymax=0.55, color=color, ls='--', lw=1.5)
        ax.text(0.03, 0.70, f'median: {median_err:+.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.text(0.03, 0.92, which, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, color=color, fontweight='bold')
    ax.axvline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(RUNS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
mag_axes[-1].set_xlabel('EPIC mag − USGS mag')
_y_max = max(a.get_ylim()[1] for a in mag_axes)
for a in mag_axes:
    a.set_ylim(0, 2.0 * _y_max)
mag_axes[0].text(-0.2, 0.87, '(c)', transform=mag_axes[0].transAxes,
                 ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- Left: combined map of EPIC locations, colored by run --
ax_map = fig.add_axes(AX_POS['map'], projection=proj)
all_lats = pd.concat([r.lat for r in results.values()])
all_lons = pd.concat([r.lon for r in results.values()])
extent = [all_lons.min() - BUF, all_lons.max() + BUF, all_lats.min() - BUF, all_lats.max() + BUF]
if (extent[1] - extent[0]) < MIN_MAP_ASPECT * (extent[3] - extent[2]):
    _half = 0.5 * MIN_MAP_ASPECT * (extent[3] - extent[2])
    _lon_c = 0.5 * (extent[0] + extent[1])
    extent[0], extent[1] = _lon_c - _half, _lon_c + _half
ax_map.set_extent(extent, crs=proj)
ax_map.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

# Arrows from each run's location to the USGS location, colored to match that run's points
for which, color in RUNS.items():
    m = results[which]
    ax_map.quiver(m.lon.values, m.lat.values, (m.usgs_lon - m.lon).values, (m.usgs_lat - m.lat).values,
                  angles='xy', scale_units='xy', scale=1, color=color, alpha=0.3, width=0.0025,
                  headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
for which, color in RUNS.items():
    m = results[which]
    ax_map.scatter(m.lon, m.lat, color=color, s=30 * SIZE_SCALE[which], alpha=0.5,
                   edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=which)
# Mainshock: each run's location as a large star in the run's color, drawn above everything else.
# Runs often land on the same spot, so the stars shrink in draw order and nest instead of hiding each other.
for i, (which, color) in enumerate(RUNS.items()):
    ms = results[which][results[which].anss_id == MAINSHOCK_ID]
    ax_map.scatter(ms.lon, ms.lat, color=color, s=300 + 150 * (len(RUNS) - 1 - i), edgecolors='black', linewidths=1.0,
                   marker='*', transform=proj, zorder=8 + i)
# True (USGS) mainshock location: yellow star on top of the run stars (no legend entry)
true_ms = pd.concat([r[r.anss_id == MAINSHOCK_ID] for r in results.values()]).head(1)
ax_map.scatter(true_ms.usgs_lon, true_ms.usgs_lat, marker='*', c='yellow', s=400, edgecolors='black', linewidths=0.8,
               transform=proj, zorder=20)
usgs_pts = pd.concat([r[['anss_id', 'usgs_lat', 'usgs_lon']] for r in results.values()]).drop_duplicates('anss_id')
ax_map.scatter(usgs_pts.usgs_lon, usgs_pts.usgs_lat, marker='x', c='black', s=35, linewidths=1.1,
               transform=proj, zorder=6, alpha=0.5, label='USGS (true)')

ax_map.set_xticks(np.linspace(extent[0], extent[1], 5), crs=proj)
ax_map.set_yticks(np.linspace(extent[2], extent[3], 5), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
ax_map.text(0.98, 0.02, f'EPIC locations at {N_TRIGS} stations', transform=ax_map.transAxes,
            ha='right', va='bottom', fontsize=13, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

out_stem = Path(__file__).resolve().parent / 'figures' / f'{SEQUENCE}_location_hist_{N_TRIGS}_comp_{TAG}'
fig.savefig(out_stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
print('Saved', out_stem)

# %%
