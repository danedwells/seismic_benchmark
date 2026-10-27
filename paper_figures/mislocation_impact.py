#%%
# =============================================================================
# mislocation_impact.py — conceptual figure: what a bad location does to the
# predicted shaking
# =============================================================================
# Story: the 2024 M7.0 Mendocino Triple Junction mainshock (nc75095651) located
# from 4 triggers is badly misplaced without a good prior —
#     Uniform:         ~107 km too far SW (further offshore) -> under-predicts shaking onshore
#     KDE_Seismicity:  ~ 42 km too far E  (closer to shore)  -> over-predicts shaking onshore
# A real USGS ShakeMap (nc75095651 v9, downloaded once and cached under
# data/) gives the "truth" at each town.
#
#   left:  map: USGS ShakeMap MMI, USGS epicenter (labeled as the 2024 M7 mainshock), each prior's
#          first estimate with >= N_TRIGGERS stations (distance error and magnitude)
#   right: town-by-town shaking residual, USGS ShakeMap MMI minus the MMI predicted if the system
#          believed each estimate (grouped bars, plus the mean over towns on top)
#
# Hypothetical shaking model: the real ShakeMap is the anchor, and a ground-motion
# model supplies only the *change* in MMI from moving the source / changing the
# magnitude:
#
#     MMI_scenario(town) = MMI_ShakeMap(town) + f(R_assumed, M_assumed) - f(R_true, M_true)
#
# f(R, M) = mean of four NGA-West2 GMPEs (Abrahamson+2014, Boore+2014,
# Campbell & Bozorgnia 2014, Chiou & Youngs 2014 — the ones the ShakeMap itself
# used), evaluated for PGV at a point source (strike-slip, 10 km deep, Vs30 = 400 m/s) and converted
# to MMI with the Worden et al. (2012) relation (WGRW12, as implemented in USGS shakelib, which is what this
# ShakeMap used): three bilinear-in-log10 segments plus a magnitude term. As in ShakeMap's VirtualIPE, PGV
# is used (not averaged with PGA). The predicted town MMI is clipped to 1-10.
# Needs openquake.hazardlib (pip install --no-deps openquake.engine). The
# correct-location, correct-magnitude scenario reproduces the ShakeMap exactly.
# Limitation: the M7 is a point source here (R = epicentral distance), not a
# finite fault.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

from openquake.hazardlib.contexts import simple_cmaker
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014

from benchmark.metrics import _haversine_km

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR      = os.path.join(OUTPUT_DIR, 'mislocation_impact')    # figures are saved in this subfolder
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CASE_STUDY = 'MTJ_2024_M7'
EVENT_ID   = 'nc75095651'
USGS_LAT, USGS_LON, USGS_MAG = 40.3740, -125.0217, 7.0
N_TRIGGERS = 4
MAX_TRIGS  = 10

# Actual EPIC replay logs (location AND magnitude per version). KDE_Seismicity <- control run,
# ETAS <- sigma_and_prior run. There is no pure-Uniform EPIC run, so Uniform keeps the benchmark
# location with the magnitude assumed correct (= USGS).
LOG_DIR       = os.path.join(PROJECT_ROOT, 'log_files', 'MTJ-selected')
LOG_RUN       = {'KDE_Seismicity': 'MTJ_control', 'ETAS': 'MTJ_sigma_and_prior'}
USGS_TIME     = pd.Timestamp('2024-12-05T18:44:21.11')   # nc75095651 origin
MAX_ORIGIN_DIFF_S = 30          # match log event to USGS origin time within this
LOG_ALERTS_ONLY = False         # True: only versions where an alert was actually sent (Ast == 1)

# Drop-worst-station version of the log estimates (log_scripts/epic_log_param_search.py, run with PRIOR=ETAS
# and PRIOR=KDE): locate EPIC's own triggers with the benchmark locator + the matching prior after dropping the
# station with the largest travel-time residual, and adjust the magnitude to the new location
# Magnitude for the drop-worst estimates AND for Uniform (UNIFORM_PD_MAG) comes from EPIC's own scaling
# relation (Kuyuk & Allen 2013; Williamson et al. 2023): per station M = 1.23*log10(Pd) + 1.39*log10(R) + 5.39,
# using the Pd values EPIC logged (log_pd) and R = distance from the estimate's location to the station;
# event magnitude = weighted average over the triggers EPIC used (Williamson et al. 2023; A. Williamson, pers. comm.:
# stations with < 4 s of waveform are down-weighted linearly). The window is the log's `buftime` column and the
# weight is clip(buftime / 4, W_FLOOR, 1); the floor (0.3) is fitted, not documented: with it this rule reproduces
# the logged event magnitudes to rms 0.04 (median of stations: 0.07).  Uniform keeps its benchmark location.
# This departs from the logged location. Needs >= 5 stations. False = use the logged values.
DROP_WORST   = True
DROP_N       = 5                # station count used when DROP_WORST
UNIFORM_PD_MAG = True           # Uniform magnitude from the Pd formula (False = assumed correct, = USGS)
PD_A, PD_B, PD_C = 1.23, 1.39, 5.39
PD_WINDOW_S, W_FLOOR = 4.0, 0.3
DROP_SETTING = (0.22, 1.0)      # (sigma_s, PRIOR_ALPHA) row of the drop-station CSV
DROP_CSV     = {'KDE_Seismicity': 'epic_log_drop_station_KDE.csv', 'ETAS': 'epic_log_drop_station_ETAS.csv'}
if DROP_WORST:
    N_TRIGGERS = DROP_N         # Uniform (benchmark CSV) is taken at the same station count
AST_COL = 42                    # index of the Ast column in the E:I: line

# USGS ShakeMap product (event page: https://earthquake.usgs.gov/earthquakes/eventpage/nc75095651/shake-map)
SHAKEMAP_URL = ('https://earthquake.usgs.gov/product/shakemap/75095651/nc/'
                '1733537069669/download/grid.xml')
CS_DATA      = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', CASE_STUDY)
SHAKEMAP_XML = os.path.join(CS_DATA, 'shakemap_nc75095651_grid.xml')

CS_OUT = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', CASE_STUDY, 'output')
STATIC_DIR  = os.path.join(CS_OUT, 'time_independent', f'max_trigs_{MAX_TRIGS}')
DYNAMIC_DIR = os.path.join(CS_OUT, 'time_dependent',   f'max_trigs_{MAX_TRIGS}')

# Prior name -> results csv
PRIOR_CSV = {'Uniform':        os.path.join(STATIC_DIR,  'uniform_benchmark_results.csv'),
             'KDE_Seismicity': os.path.join(STATIC_DIR,  'kde_seismicity_benchmark_results.csv'),
             'ETAS':           os.path.join(DYNAMIC_DIR, 'etas_dynamic_benchmark_results.csv')}

# One figure per entry: prior -> style. Colors match the other paper figures.
# Second figure: static priors faded to gray so ETAS is the only thing that pops.
BG_ALPHA = 0.45
# Map label positions relative to each prior's dot (deg lon, deg lat), joined to the dot by a thin line
LABEL_OFFSET = {'ETAS': (0.6, 0.95), 'KDE_Seismicity': (0.2, -1.0), 'Uniform': (-1.3, -0.7)}
FIGURES = {
    'mislocation_impact': {
        'Uniform':        dict(color='gray',     alpha=1, marker='o', ms=1.0, bar='gray'),
        'KDE_Seismicity': dict(color='darkblue', alpha=1, marker='o', ms=1.0, bar='darkblue')},
    'mislocation_impact_ETAS': {
        'Uniform':        dict(color='gray',     alpha=BG_ALPHA, marker='o', ms=1.0, bar='#8c8c8c'),
        'KDE_Seismicity': dict(color='gray',     alpha=BG_ALPHA, marker='s', ms=1.0, bar='#cfcfcf'),
        'ETAS':           dict(color='red',      alpha=1,        marker='o', ms=1.2, bar='red')},
}

# Towns (name, lat, lon)
TOWNS = [('Crescent City', 41.755, -124.200), ('Arcata',  40.867, -124.083),
         ('Eureka',        40.802, -124.164), ('Ferndale', 40.576, -124.263),
         ('Petrolia',      40.319, -124.289), ('Garberville', 40.101, -123.797),
         ('Ukiah',         39.150, -123.207), ('Santa Rosa',  38.440, -122.714)]

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   12,
    'figure.dpi':       150,
    'savefig.transparent': True,
})
proj = ccrs.PlateCarree()

#%%
# ── Load ShakeMap, bEPIC estimates ───────────────────────────────────────────
if not os.path.exists(SHAKEMAP_XML):
    print('Downloading ShakeMap grid ...')
    urllib.request.urlretrieve(SHAKEMAP_URL, SHAKEMAP_XML)

grid = pd.read_csv(SHAKEMAP_XML, sep=' ', header=None, comment='<', skiprows=14,
                   names=['lon', 'lat', 'mmi', 'pga', 'pgv', 'psa03', 'psa10', 'psa30', 'vs30']).dropna()
lons = np.sort(grid['lon'].unique())
lats = np.sort(grid['lat'].unique())
mmi_2d = grid.pivot(index='lat', columns='lon', values='mmi').reindex(index=lats, columns=lons).values

est = {}   # prior -> (lat, lon, error_km, magnitude)

# --- previous version: every prior's location from the benchmark CSVs, magnitude assumed = USGS ---
# for name, csv in PRIOR_CSV.items():
#     d = pd.read_csv(csv)
#     r = d[(d.event_id == EVENT_ID) & (d.n_trigs == N_TRIGGERS)].iloc[0]
#     est[name] = (r.posterior_lat, r.posterior_lon, r.map_err_km, USGS_MAG)
#     print(f'{name}: ({r.posterior_lat:.3f}, {r.posterior_lon:.3f})  error {r.map_err_km:.0f} km')
# ---------------------------------------------------------------------------------------------------

def log_estimate(run):
    """(lat, lon, error_km, mag) of the first EPIC version with >= N_TRIGGERS stations (nS)."""
    rows = []
    for line in open(os.path.join(LOG_DIR, run, 'epic', f'{EVENT_ID}.log'), errors='replace'):
        _, _, msg = line.partition('| INFO | E:I:')
        if msg.startswith('F:'):           # 'E:I:F:' = the event's first alert version (same columns)
            msg = msg[2:]
        fl = msg.split()
        if len(fl) < 17 or not fl[0].lstrip('-').isdigit() or int(fl[0]) <= 0:
            continue
        if LOG_ALERTS_ONLY and (len(fl) <= AST_COL or fl[AST_COL] != '1'):
            continue
        rows.append((int(fl[1]), float(fl[2]), float(fl[3]), float(fl[5]), pd.Timestamp(fl[6]), int(fl[16])))
    d = pd.DataFrame(rows, columns=['ver', 'lat', 'lon', 'mag', 'time', 'nS'])
    d = d[(d.time - USGS_TIME).abs() <= pd.Timedelta(seconds=MAX_ORIGIN_DIFF_S)]
    row = d[d.nS >= N_TRIGGERS].iloc[0]
    err = float(_haversine_km(USGS_LAT, USGS_LON, row.lat, row.lon))
    print(f'  {run}: version {row.ver}, nS={row.nS}, M{row.mag:.2f}')
    return row.lat, row.lon, err, row.mag

def pd_magnitude(run, lat, lon, n):
    """Weighted-average station magnitude at (lat, lon) from the Pd values in an EPIC log (first version with n stations)."""
    hdr, rows = None, []
    for line in open(os.path.join(LOG_DIR, run, 'epic', f'{EVENT_ID}.log'), errors='replace'):
        _, _, msg = line.partition('| INFO | E:I:T:')
        fl = msg.split()
        if fl and fl[0] == 'H:':
            hdr = hdr or fl[1:]
        elif hdr and len(fl) >= len(hdr) and fl[0].lstrip('-').isdigit():
            rows.append(fl[:len(hdr)])
    t = pd.DataFrame(rows, columns=hdr)
    for c in ('eventid', 'ver', 'lat', 'lon', 'log_pd', 'updm', 'buftime'):
        t[c] = pd.to_numeric(t[c])
    t = t[t.eventid > 0]
    ver = next(v for v, g in t.groupby('ver') if g.sta.nunique() >= n)
    u = t[(t.ver == ver) & (t.updm == 1)]                       # triggers EPIC used for its Pd magnitude
    R = _haversine_km(lat, lon, u.lat.values, u.lon.values)     # epicentral, as in the log
    w = np.clip(u.buftime.values / PD_WINDOW_S, W_FLOOR, 1)      # linear down-weight below 4 s of waveform
    return float(np.average(PD_A * u.log_pd.values + PD_B * np.log10(R) + PD_C, weights=w))

def drop_estimate(name):
    """(lat, lon, error_km, mag) from the drop-worst-station CSV written by epic_log_param_search.py."""
    d = pd.read_csv(os.path.join(PROJECT_ROOT, 'log_scripts', DROP_CSV[name]))
    r = d[(d.sigma_s == DROP_SETTING[0]) & (d.alpha == DROP_SETTING[1]) & (d.n == DROP_N)].iloc[0]
    mag = pd_magnitude(LOG_RUN[name], r.drop_lat, r.drop_lon, DROP_N)
    print(f'  {name}: dropped {r.worst_station}; logged M{r.mag_log:.2f} -> Pd-formula M{mag:.2f}')
    return r.drop_lat, r.drop_lon, r.drop_worst_resid, mag

for name, csv in PRIOR_CSV.items():
    if name in LOG_RUN:
        est[name] = drop_estimate(name) if DROP_WORST else log_estimate(LOG_RUN[name])
    else:                                  # Uniform: benchmark location, assumed magnitude
        d = pd.read_csv(csv)
        r = d[(d.event_id == EVENT_ID) & (d.n_trigs == N_TRIGGERS)].iloc[0]
        mag = pd_magnitude(LOG_RUN['ETAS'], r.posterior_lat, r.posterior_lon, N_TRIGGERS) if UNIFORM_PD_MAG else USGS_MAG
        est[name] = (r.posterior_lat, r.posterior_lon, r.map_err_km, mag)
    la, lo, err, mg = est[name]
    print(f'{name}: ({la:.3f}, {lo:.3f})  error {err:.0f} km  M{mg:.2f}')

#%%
# ── Shaking model: anchor on ShakeMap, shift by GMPE-based f(R, M) ──────────────────────
def shakemap_mmi(lat, lon):
    i = np.argmin(np.hypot(grid['lat'].values - lat, (grid['lon'].values - lon) * np.cos(np.radians(lat))))
    return float(grid['mmi'].values[i])

GMPES  = [AbrahamsonEtAl2014(), BooreEtAl2014(), CampbellBozorgnia2014(), ChiouYoungs2014()]
CMAKER = simple_cmaker(GMPES, ['PGV', 'PGA'])
GM_IMT = 'PGA'      # intensity measure for the residual panel: 'PGV' or 'PGA'
DEPTH_KM = 10.0
VS30     = 400.0

def log_gm_model(R, M, imt='PGV'):
    """Median log10 PGV (cm/s) or PGA (g) at epicentral distance R (km, any shape) for magnitude M."""
    R = np.asarray(R, dtype=float)
    r = R.ravel()
    ctx = CMAKER.new_ctx(len(r))
    ctx.mag, ctx.rake, ctx.dip, ctx.ztor, ctx.hypo_depth = M, 0.0, 90.0, 0.0, DEPTH_KM
    ctx.width = 10 ** (-0.76 + 0.27 * M)                 # Wells & Coppersmith strike-slip
    ctx.rjb, ctx.rx, ctx.ry0 = r, r, np.zeros(len(r))
    ctx.rrup = np.sqrt(r ** 2 + DEPTH_KM ** 2)
    ctx.vs30, ctx.vs30measured = np.full(len(r), VS30), np.zeros(len(r), bool)
    ctx.z1pt0, ctx.z2pt5, ctx.sids = np.full(len(r), 0.1), np.full(len(r), 1.0), np.arange(len(r))
    mean = CMAKER.get_mean_stds([ctx])[0].mean(axis=0)   # (imt, site): ln(cm/s) for PGV, mean of the 4 models
    return (mean[['PGV', 'PGA'].index(imt)] / np.log(10)).reshape(R.shape)

def f(R, M):
    """Median MMI at epicentral distance R (km, any shape) for magnitude M."""
    log_pgv = log_gm_model(R, M, 'PGV')
    # WGRW12 PGV -> MMI (coefficients from the paper's table / shakelib): lower, middle and upper segments in
    # log10(PGV), plus the magnitude residual term C5 + C6*log10(D) + C7*M (C6 = 0 for PGV, so D is not needed).
    mmi = np.where(log_pgv < -1.21, 4.62 + 2.17 * log_pgv,
                   np.where(log_pgv < 0.53, 3.78 + 1.47 * log_pgv, 2.89 + 3.16 * log_pgv))
    mmi = mmi + 0.90 - 0.18 * np.clip(M, 3.0, 7.3)       # not clipped to [1, 10] here: only differences are used
    return mmi

def predicted_mmi(town_lat, town_lon, src_lat, src_lon, M):
    R_true = _haversine_km(USGS_LAT, USGS_LON, town_lat, town_lon)
    R_src  = _haversine_km(src_lat, src_lon, town_lat, town_lon)
    return float(np.clip(shakemap_mmi(town_lat, town_lon) + float(f(R_src, M) - f(R_true, USGS_MAG)), 1, 10))

def shakemap_pgv(lat, lon):
    i = np.argmin(np.hypot(grid['lat'].values - lat, (grid['lon'].values - lon) * np.cos(np.radians(lat))))
    return float(grid[GM_IMT.lower()].values[i])

def predicted_pgv(town_lat, town_lon, src_lat, src_lon, M):
    R_true = _haversine_km(USGS_LAT, USGS_LON, town_lat, town_lon)
    R_src  = _haversine_km(src_lat, src_lon, town_lat, town_lon)
    return shakemap_pgv(town_lat, town_lon) * 10 ** float(log_gm_model(R_src, M, GM_IMT) - log_gm_model(R_true, USGS_MAG, GM_IMT))

towns = pd.DataFrame(TOWNS, columns=['town', 'lat', 'lon'])
towns['obs'] = [shakemap_mmi(a, b) for a, b in zip(towns.lat, towns.lon)]
towns['obs_pgv'] = [shakemap_pgv(a, b) for a, b in zip(towns.lat, towns.lon)]
towns = towns.sort_values('obs').reset_index(drop=True)      # weakest shaking at the bottom
for name in PRIOR_CSV:
    la, lo, _, mg = est[name]
    towns[name] = [predicted_mmi(a, b, la, lo, mg) for a, b in zip(towns.lat, towns.lon)]
    towns[name + '_res'] = towns['obs'] - towns[name]          # USGS ShakeMap minus predicted (MMI)
    towns[name + '_pgv'] = [predicted_pgv(a, b, la, lo, mg) for a, b in zip(towns.lat, towns.lon)]
    towns[name + '_pgv_res'] = np.log10(towns['obs_pgv'] / towns[name + '_pgv'])   # log10(ShakeMap / predicted) PGV; >0 = under-predicted
print(towns.round(1).to_string())
print('mean abs residual:', {n: round(float(towns[n + '_res'].abs().mean()), 2) for n in PRIOR_CSV})

#%%
# ── Figure ──────────────────────────────────────────────────────────────────
AX_POS = {
    'map':  [0.03, 0.08, 0.44, 0.84],
    'cbar': [0.03, 0.045, 0.44, 0.015],
    'town': [0.60, 0.08, 0.37, 0.84],
}

def make_figure(styles, outname):
    """styles: prior name -> dict(color, alpha, marker, ms, bar) — drawn in order, later on top."""
    fig = plt.figure(figsize=(15, 8.5))
    label = lambda name: name.replace('_', ' ')

    # -- map -----------------------------------------------------------------
    ax = fig.add_axes(AX_POS['map'], projection=proj)
    ax.set_extent([-127.2, -122.3, 38.0, 42.4], crs=proj)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightcyan', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=3)

    mmi_levels = np.arange(3, 10)
    cmap = plt.get_cmap('Reds')
    ax.contourf(lons, lats, mmi_2d, levels=mmi_levels, cmap=cmap, extend='max',
                alpha=0.8, transform=proj, zorder=2)
    ax.contour(lons, lats, mmi_2d, levels=mmi_levels, colors='k', linewidths=0.4,
               alpha=0.5, transform=proj, zorder=2)
    cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(mmi_levels, cmap.N, extend='max'), cmap=cmap),
                      cax=fig.add_axes(AX_POS['cbar']), orientation='horizontal', extend='max')
    cb.set_label('MMI')

    geo = proj._as_mpl_transform(ax)
    for i, (name, st) in enumerate(styles.items()):
        la, lo, err, mg = est[name]
        z = 5 + 0.1 * i
        ax.annotate('', xy=(lo, la), xytext=(USGS_LON, USGS_LAT), transform=proj, zorder=z,
                    arrowprops=dict(arrowstyle='-|>', color=to_rgba(st['color'], st['alpha']),
                                    lw=2, shrinkA=8, shrinkB=4))
        ax.plot(lo, la, st['marker'], ms=11 * st['ms'], color=st['color'], alpha=st['alpha'],
                markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=z + 0.05)
        dx, dy = LABEL_OFFSET[name]                          # label position relative to its dot (deg lon, deg lat)
        lab_color = to_rgba(st['color'], 1)
        # name + distance error next to the dot, magnitude (slightly larger) on the far side of the anchor
        up = dy > 0                                          # label above the dot: name below the anchor, magnitude above
        box = dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5)
        ax.annotate(f'{label(name)}\n{err:.0f} km', xy=(lo, la), xycoords=geo,
                    xytext=(lo + dx, la + dy), textcoords=geo, color=lab_color, fontsize=11, fontweight='bold',
                    ha='center', va='top' if up else 'bottom', zorder=7, bbox=box,
                    arrowprops=dict(arrowstyle='-', lw=0.8, color=to_rgba(st['color'], max(st['alpha'], 0.7)),
                                    shrinkA=0, shrinkB=6))
        ax.annotate(f'M{mg:.1f}', xy=(lo + dx, la + dy), xycoords=geo, xytext=(0, 4 if up else -4),
                    textcoords='offset points', color=lab_color, fontsize=14, fontweight='bold',
                    ha='center', va='bottom' if up else 'top', zorder=7, bbox=box)
    ax.plot(USGS_LON, USGS_LAT, '*', ms=20, color='gold', markeredgecolor='black',
            markeredgewidth=0.8, transform=proj, zorder=8)
    # the mainshock itself: labeled out to the left, in the ocean
    ax.annotate('2024 M7.0\nMainshock', xy=(USGS_LON, USGS_LAT), xycoords=geo, xytext=(-126.35, 40.95),
                textcoords=geo, fontsize=13, fontweight='bold', ha='center', va='center', zorder=9,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black', shrinkA=2, shrinkB=10))

    ax.scatter(towns.lon, towns.lat, marker='s', s=28, color='black', transform=proj, zorder=6)
    for _, t in towns.iterrows():
        ax.text(t.lon + 0.08, t.lat + 0.03, t.town, fontsize=9, transform=proj, zorder=7,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)       # coordinate labels only, no grid lines
    gl.top_labels = gl.right_labels = False
    ax.set_title('USGS ShakeMap')

    # ---- previous panel: MMI residual (uncomment to restore; comment out the PGV panel below) ----
    # ax = fig.add_axes(AX_POS['town'])
    # n = len(styles)
    # h = 0.26                                              # bar thickness
    # y_towns = np.arange(len(towns))
    # y_mean = len(towns) + 0.6                             # summary group above the towns, with a gap
    # for k, (name, st) in enumerate(styles.items()):
    #     off = ((n - 1) / 2 - k) * h                       # first prior on top within each group
    #     ax.barh(y_towns + off, towns[name + '_res'], height=h * 0.9, color=st['bar'])
    #     m = float(towns[name + '_res'].abs().mean())
    #     ax.barh(y_mean + off, m, height=h * 0.9, color=st['bar'])
    #     ax.text(m + 0.05, y_mean + off, f'{label(name)}  {m:.1f}', va='center', ha='left', fontsize=12,
    #             fontweight='bold', color={'#8c8c8c': 'dimgray', '#cfcfcf': 'gray'}.get(st['bar'], st['bar']))
    # ax.set_yticks(list(y_towns) + [y_mean])
    # ax.set_yticklabels(list(towns.town) + ['Mean abs.'])
    # ax.get_yticklabels()[-1].set_fontweight('bold')
    # ax.tick_params(axis='y', length=0)
    # ax.set_xlabel('USGS ShakeMap − predicted MMI')
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.set_xlim(0, 2.6)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])

    # ---------------------------------------------------------------------------------------
    # -- town-by-town shaking residual: |USGS ShakeMap - predicted| PGV (cm/s) --------
    ax = fig.add_axes(AX_POS['town'])
    n = len(styles)
    h = 0.26
    y_towns = np.arange(len(towns))
    y_mean = len(towns) + 0.6
    for k, (name, st) in enumerate(styles.items()):
        off = ((n - 1) / 2 - k) * h
        ax.barh(y_towns + off, towns[name + '_pgv_res'], height=h * 0.9, color=st['bar'])
        m = float(towns[name + '_pgv_res'].mean())   # signed mean = bias
        ax.barh(y_mean + off, m, height=h * 0.9, color=st['bar'])
        ax.text(m + 0.03 * np.sign(m or 1), y_mean + off, f'{label(name)}  {m:+.2f}', va='center', ha='left' if m >= 0 else 'right', fontsize=12,
                fontweight='bold', color={'#8c8c8c': 'dimgray', '#cfcfcf': 'gray'}.get(st['bar'], st['bar']))
    ax.set_yticks(list(y_towns) + [y_mean])
    ax.set_yticklabels(list(towns.town) + ['Mean'])
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('log$_{10}$(USGS ShakeMap ' + GM_IMT + ' / predicted ' + GM_IMT + ')', labelpad=22)
    for txt, ha in (('← over-predicted ', 'right'), (' under-predicted →', 'left')):   # centered about 0
        ax.annotate(txt, xy=(0, 0), xycoords=('data', 'axes fraction'), xytext=(0, -30), textcoords='offset points',
                    ha=ha, va='top', fontsize=11, color='dimgray')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-0.2, 2.0)
    ax.set_xticks([-0.2, 0, 0.5, 1, 1.5, 2])

    fig.savefig(os.path.join(FIG_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.pdf'), bbox_inches='tight')
    plt.show()

for outname, styles in FIGURES.items():
    make_figure(styles, outname)
