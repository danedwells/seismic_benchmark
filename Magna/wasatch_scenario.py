#%%
# =============================================================================
# wasatch_scenario.py — VERY crude, illustrative "big one" on the Brigham City segment
# =============================================================================
# Presentation sketch only. NOT a physical simulation:
#   * The Brigham City segment trace is approximate and hypothetical; the hypocenter (41.45 N, 112.1 W, 15 km)
#     is the Brigham City scenario source from the paper.
#   * Synthetic P picks: origin time 0 + hypocentral distance / VP (uniform half-space) + Gaussian noise,
#     at the real UU/US/... station locations found in Magna/data/run_files (the "surrounding stations").
#   * The picks are located with the benchmark's bEPIC locator (its ak135 travel-time table differs from the
#     synthetic uniform velocity, so some location error is expected), with and without the Magna KDE prior.
#   * Alert time = (N_ALERT-th station's P pick) + latency, as in eew_alert_map.py.
#   * Warning time at a site = (earliest S arrival from the rupturing fault) - alert time, with the rupture
#     modelled as a line of points that slip at VR from the hypocenter (S travels at VS from each point).
#     Negative = inside the "blind zone" (S-wave arrives before the alert).
#
# Usage: python Magna/wasatch_scenario.py     (needs Magna/data/run_files and data/priors from the Magna scripts)
# =============================================================================
import os
import io
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

from priors import SeismicPrior
import benchmark.config_california as config
from benchmark.runner import make_epic_params
from bEPIC import EPIC_locate_prelim

MAGNA_DIR = Path(__file__).resolve().parent
RUN_DIR   = MAGNA_DIR / 'data' / 'run_files'
KDE_TT3   = MAGNA_DIR / 'data' / 'priors' / 'kde_seismicity_Magna.tt3'
FIG_DIR   = MAGNA_DIR / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenario (all approximate / hypothetical)
# ---------------------------------------------------------------------------
MAG_LABEL   = 7.0
FAULT_S     = (41.38, -112.02)    # (lat, lon) southern end of the Brigham City segment trace (approx.)
FAULT_N     = (41.72, -112.09)    # northern end (approx.)
HYPO         = (41.45, -112.1)   # (lat, lon) Brigham City scenario source (paper value)
HYPO_DEPTH  = 15.0                # km (paper value)
RUP_DEPTH   = 8.0                 # km, depth of the points representing the slipping fault
VP, VS, VR  = 6.0, 3.5, 2.8       # km/s: synthetic P speed, S speed, rupture speed
PICK_SIGMA  = 0.15                # s, synthetic pick noise
SEED        = 0
N_ALERT     = 4                   # alert when this many stations have a P pick
LATENCY_S   = 4.0                 # s from the last needed pick to the alert reaching users
LOCATE_N    = (4, 6, 10)          # station counts at which the picks are located
SIGMA_S     = 0.22                # bEPIC travel-time sigma, as in the case-study scripts

EXTENT = [-112.65, -111.35, 40.55, 41.95]
CITIES = {                         # name: (lat, lon)
    'Brigham City': (41.510, -112.015), 'Logan': (41.737, -111.834), 'Tremonton': (41.712, -112.185),
    'Ogden': (41.223, -111.974), 'Salt Lake City': (40.761, -111.891)}
LABEL_OFFSET = {'Brigham City': (8, -14), 'Logan': (8, 4), 'Tremonton': (-8, 6), 'Ogden': (8, 4), 'Salt Lake City': (8, 4)}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.dpi': 150, 'savefig.transparent': True})
GOOD, BAD = plt.get_cmap('RdBu')(0.85), plt.get_cmap('RdBu')(0.12)   # warning (blue) / blind zone (red), as on the map

KM_LAT = 111.2
def km_lon(lat): return 111.32 * np.cos(np.radians(lat))

def dist_km(lat1, lon1, lat2, lon2, dz=0.0):
    """Flat-earth distance (km), with an optional depth difference."""
    dx = (np.asarray(lon2) - lon1) * km_lon(0.5 * (lat1 + np.asarray(lat2)))
    dy = (np.asarray(lat2) - lat1) * KM_LAT
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

#%% Fault, hypocenter, stations, synthetic picks
frac = np.linspace(0, 1, 41)
fault = np.c_[FAULT_S[0] + frac * (FAULT_N[0] - FAULT_S[0]), FAULT_S[1] + frac * (FAULT_N[1] - FAULT_S[1])]   # (lat, lon)
hypo = HYPO
fault_len = float(np.sum(dist_km(fault[:-1, 0], fault[:-1, 1], fault[1:, 0], fault[1:, 1])))

stations = (pd.concat([pd.read_csv(f)[['station', 'network', 'latitude', 'longitude']] for f in RUN_DIR.glob('*.run')])
            .drop_duplicates(['network', 'station']).reset_index(drop=True))
stations['dist_hypo'] = dist_km(hypo[0], hypo[1], stations.latitude, stations.longitude, HYPO_DEPTH)
rng = np.random.default_rng(SEED)
stations['t_p'] = stations.dist_hypo / VP + rng.normal(0, PICK_SIGMA, len(stations))   # origin time = 0
stations = stations[stations.dist_hypo < 150].sort_values('t_p').reset_index(drop=True)
t_alert = stations.t_p.iloc[N_ALERT - 1] + LATENCY_S
print(f'Fault {fault_len:.0f} km, hypocenter {hypo[0]:.3f}, {hypo[1]:.3f}; {len(stations)} stations within 150 km')
print('First stations:', ', '.join(f'{r.station}({r.t_p:.1f}s)' for r in stations.head(N_ALERT + 2).itertuples()))
print(f'Alert {t_alert:.1f} s after origin ({N_ALERT} picks + {LATENCY_S:g} s latency)')

#%% Locate the synthetic picks with the benchmark's bEPIC locator
kde = SeismicPrior.from_tt3(str(KDE_TT3))

def locate(n, use_prior):
    g = stations.head(n)
    pk = dict(config.BENCHMARK_PARAMS); pk.update(sigma_s=SIGMA_S, max_trigs=max(n, 10))
    params = make_epic_params(kde, use_prior, pk)
    first = g.iloc[0]
    event = EPIC_locate_prelim.Event(lat=first.latitude, lon=first.longitude, time=first.t_p, misfit_rms=0,
                                     misfit_ave=0, eventid='scenario', version=0)
    event.trigs = [EPIC_locate_prelim.TriggerManager(lon=r.longitude, lat=r.latitude, trigger_time=r.t_p,
                                                     sta=r.station, net=r.network, chan='HHZ') for r in g.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        _, odf = EPIC_locate_prelim.E2Location_locate(params, event)
    i = int(np.argmax(odf['post'].values))
    la, lo = float(odf['lat'].values[i]), float(odf['lon'].values[i])
    return la, lo, float(dist_km(hypo[0], hypo[1], la, lo))

locs = {}
print('\nlocation error vs true hypocenter (km):')
for n in LOCATE_N:
    for name, use in (('Uniform', False), ('KDE_Seismicity', True)):
        locs[(name, n)] = locate(n, use)
    print(f'  {n:2d} stations: Uniform {locs[("Uniform", n)][2]:5.1f}   KDE {locs[("KDE_Seismicity", n)][2]:5.1f}')

#%% Warning time from the rupturing fault
t_rup = dist_km(hypo[0], hypo[1], fault[:, 0], fault[:, 1], HYPO_DEPTH - RUP_DEPTH) / VR   # s, slip reaches each point

def warning_s(lat, lon):
    """Seconds between the alert and the earliest S arrival (from any slipping point) at (lat, lon)."""
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    t_s = np.min([t_rup[i] + dist_km(fault[i, 0], fault[i, 1], lat, lon, RUP_DEPTH) / VS for i in range(len(fault))], axis=0)
    return t_s - t_alert

city_w = {c: float(warning_s(la, lo)) for c, (la, lo) in CITIES.items()}
print('\nwarning time at cities (s; negative = no warning):', {c: round(w, 1) for c, w in city_w.items()})

#%% Figure
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(14, 8.5))
ax = fig.add_axes([0.04, 0.07, 0.56, 0.86], projection=proj)
ax.set_extent(EXTENT, crs=proj)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='lightcyan', edgecolor='gray',
               linewidth=0.4, zorder=3)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=3)

lon_g, lat_g = np.linspace(EXTENT[0], EXTENT[1], 260), np.linspace(EXTENT[2], EXTENT[3], 260)
LON, LAT = np.meshgrid(lon_g, lat_g)
W = warning_s(LAT, LON)
levels = np.arange(-10, 31, 2)
cf = ax.contourf(LON, LAT, W, levels=levels, cmap='RdBu', norm=mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=30),
                 extend='both', transform=proj, alpha=0.85, zorder=1)
cs0 = ax.contour(LON, LAT, W, levels=[0], colors='k', linewidths=1.8, transform=proj, zorder=2)
ax.clabel(cs0, fmt=lambda v: 'blind zone', fontsize=10)
cb = fig.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.04, pad=0.06)
cb.set_label('Warning time before S-wave arrival (s)')

ax.plot(fault[:, 1], fault[:, 0], color='crimson', lw=4, transform=proj, zorder=4, solid_capstyle='round')
ax.scatter(stations.longitude, stations.latitude, marker='^', s=30, facecolor='w', edgecolor='k', linewidths=0.7,
           transform=proj, zorder=5)
first = stations.head(N_ALERT)
ax.scatter(first.longitude, first.latitude, marker='^', s=90, facecolor='k', edgecolor='w', linewidths=0.8,
           transform=proj, zorder=6)
for k, r in enumerate(first.itertuples(), 1):
    ax.annotate(str(k), (r.longitude, r.latitude), xycoords=proj._as_mpl_transform(ax), xytext=(5, 5),
                textcoords='offset points', fontsize=9, fontweight='bold', zorder=8)
ax.plot(hypo[1], hypo[0], '*', ms=24, c='gold', mec='k', mew=1, transform=proj, zorder=9)
for name, color, marker in (('Uniform', 'gray', 'o'), ('KDE_Seismicity', 'darkblue', 's')):
    la, lo, e = locs[(name, N_ALERT)]
    ax.plot(lo, la, marker, ms=9, c=color, mec='k', transform=proj, zorder=8)
for c, (la, lo) in CITIES.items():
    ax.scatter(lo, la, s=26, c='k', marker='s', transform=proj, zorder=7)
    ax.annotate(f'{c}\n{city_w[c]:+.0f} s', (lo, la), xycoords=proj._as_mpl_transform(ax), xytext=LABEL_OFFSET[c],
                textcoords='offset points', ha='left' if LABEL_OFFSET[c][0] > 0 else 'right', fontsize=10, fontweight='bold',
                zorder=10, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))
# Lat/lon ticks commented out for the presentation version (restore for the paper):
# ax.set_xticks(np.linspace(EXTENT[0], EXTENT[1], 4), crs=proj); ax.set_yticks(np.linspace(EXTENT[2], EXTENT[3], 5), crs=proj)
# ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.2f}°W'))
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}°N'))
LEGEND = [
    Line2D([], [], color='crimson', lw=4, label=f'Brigham City segment (~{fault_len:.0f} km, approx.)'),
    Line2D([], [], marker='*', ls='', ms=16, c='gold', mec='k', label='hypocenter'),
    Line2D([], [], marker='^', ls='', ms=8, c='k', mec='w', label=f'first {N_ALERT} stations to report'),
    Line2D([], [], marker='^', ls='', ms=6, mfc='w', mec='k', label='other stations'),
    Line2D([], [], marker='o', ls='', ms=8, c='gray', mec='k', label=f'bEPIC location, {N_ALERT} stations, no prior ({locs[("Uniform", N_ALERT)][2]:.0f} km off)'),
    Line2D([], [], marker='s', ls='', ms=8, c='darkblue', mec='k', label=f'bEPIC location, {N_ALERT} stations, KDE prior ({locs[("KDE_Seismicity", N_ALERT)][2]:.0f} km off)')]

# right: timeline (P picks -> alert -> S arrival at each city), then legend + assumptions
ax2 = fig.add_axes([0.69, 0.47, 0.29, 0.46])
t_s = {c: city_w[c] + t_alert for c in CITIES}
order = sorted(CITIES, key=lambda c: t_s[c])[::-1]          # earliest S arrival at the top
ypos = {c: i for i, c in enumerate(order)}
ax2.axvspan(0, t_alert, color='lightgray', alpha=0.45, zorder=0)
ax2.axvline(t_alert, color='k', ls='--', lw=1.5)
ax2.text(t_alert + 0.6, len(order) + 0.6, f'alert {t_alert:.1f} s', ha='left', va='bottom', fontsize=10, fontweight='bold')
ax2.plot(first.t_p, [len(order)] * N_ALERT, 'k^', ms=8)
ax2.text(t_alert + 0.6, len(order), f'P picks of the first {N_ALERT} stations', ha='left', va='center', fontsize=9)
for c in order:
    w, y = city_w[c], ypos[c]
    col = BAD if w < 0 else GOOD
    ax2.hlines(y, t_alert, t_s[c], color=col, lw=3)
    ax2.plot(t_s[c], y, 'o', c=col, mec='k', ms=9, zorder=5)
    ax2.text(max(t_alert, t_s[c]) + 0.6, y, f'{w:+.0f} s', va='center', fontsize=10, fontweight='bold', color=col)
ax2.set_yticks(range(len(order))); ax2.set_yticklabels(order)
ax2.set_ylim(-0.6, len(order) + 1.4); ax2.set_xlim(0, max(t_s.values()) + 7)
ax2.set_xlabel('Time after origin (s)  (dot = earliest S arrival)')
ax2.spines[['top', 'right']].set_visible(False)

ax3 = fig.add_axes([0.69, 0.06, 0.29, 0.34]); ax3.axis('off')
ax3.legend(handles=LEGEND, loc='upper left', fontsize=9, frameon=False)
ax3.text(0.0, 0.2, f'Hypothetical M{MAG_LABEL:g} on the Brigham City segment\n({N_ALERT} stations + {LATENCY_S:g} s latency); illustration only',
         transform=ax3.transAxes, fontsize=11, fontweight='bold', va='bottom')
ax3.text(0.0, 0.0, f'Synthetic picks: Vp {VP:g} km/s, noise {PICK_SIGMA:g} s.  S {VS:g} km/s, rupture {VR:g} km/s.\n'
         f'Alert = {N_ALERT}th pick + {LATENCY_S:g} s.  bEPIC location error: '
         f'{locs[("Uniform", 4)][2]:.0f} km (4 stations), {locs[("Uniform", 10)][2]:.0f} km (10).',
         transform=ax3.transAxes, fontsize=8.5, va='bottom', color='dimgray')

fig.savefig(FIG_DIR / 'wasatch_brigham_scenario.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'wasatch_brigham_scenario.pdf', bbox_inches='tight')
plt.show()
