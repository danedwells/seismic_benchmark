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
#   (a) map: USGS ShakeMap MMI, USGS epicenter, each prior's 4-trigger estimate
#   (b) town-by-town predicted MMI if the system believed each estimate, vs the
#       ShakeMap. Whiskers show an additional +/-DM magnitude error on top.
#   (c) sensitivity at one town: MMI error vs (distance to assumed epicenter,
#       magnitude error) — location and magnitude errors can compound or cancel.
#
# Hypothetical shaking model: the real ShakeMap is the anchor, and a ground-motion
# model supplies only the *change* in MMI from moving the source / changing the
# magnitude:
#
#     MMI_scenario(town) = MMI_ShakeMap(town) + f(R_assumed, M_assumed) - f(R_true, M_true)
#
# f(R, M) = mean of four NGA-West2 GMPEs (Abrahamson+2014, Boore+2014,
# Campbell & Bozorgnia 2014, Chiou & Youngs 2014 — the ones the ShakeMap itself
# used), evaluated for PGA and PGV at a point source (strike-slip, 10 km deep,
# Vs30 = 400 m/s), converted to MMI with Worden et al. (2012) and averaged.
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
FIGURES = {
    'mislocation_impact': {
        'Uniform':        dict(color='gray',     alpha=1, marker='o', ms=1.0),
        'KDE_Seismicity': dict(color='darkblue', alpha=1, marker='o', ms=1.0)},
    'mislocation_impact_ETAS': {
        'Uniform':        dict(color='gray',     alpha=BG_ALPHA, marker='o', ms=1.0),
        'KDE_Seismicity': dict(color='gray',     alpha=BG_ALPHA, marker='s', ms=1.0),
        'ETAS':           dict(color='red',      alpha=1,        marker='o', ms=1.2, above=True)},
}

# Towns (name, lat, lon)
TOWNS = [('Crescent City', 41.755, -124.200), ('Arcata',  40.867, -124.083),
         ('Eureka',        40.802, -124.164), ('Ferndale', 40.576, -124.263),
         ('Petrolia',      40.319, -124.289), ('Garberville', 40.101, -123.797),
         ('Ukiah',         39.150, -123.207), ('Santa Rosa',  38.440, -122.714)]
FOCUS_TOWN = 'Eureka'     # town used for the sensitivity panel (c)

ALERT_MMI = 4.0           # illustrative alerting threshold (MMI IV)
DM        = 0.5           # hypothetical magnitude error (+/-)
ERR_TOL   = 0.5           # +/- MMI band treated as "about right" in panel (c)

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
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

for name, csv in PRIOR_CSV.items():
    if name in LOG_RUN:
        est[name] = log_estimate(LOG_RUN[name])
    else:                                  # Uniform: benchmark location, assumed magnitude
        d = pd.read_csv(csv)
        r = d[(d.event_id == EVENT_ID) & (d.n_trigs == N_TRIGGERS)].iloc[0]
        est[name] = (r.posterior_lat, r.posterior_lon, r.map_err_km, USGS_MAG)
    la, lo, err, mg = est[name]
    print(f'{name}: ({la:.3f}, {lo:.3f})  error {err:.0f} km  M{mg:.2f}')

#%%
# ── Shaking model: anchor on ShakeMap, shift by GMPE-based f(R, M) ──────────────────────
def shakemap_mmi(lat, lon):
    i = np.argmin(np.hypot(grid['lat'].values - lat, (grid['lon'].values - lon) * np.cos(np.radians(lat))))
    return float(grid['mmi'].values[i])

GMPES  = [AbrahamsonEtAl2014(), BooreEtAl2014(), CampbellBozorgnia2014(), ChiouYoungs2014()]
CMAKER = simple_cmaker(GMPES, ['PGA', 'PGV'])
DEPTH_KM = 10.0
VS30     = 400.0

def f(R, M):
    """Median MMI at epicentral distance R (km, any shape) for magnitude M."""
    R = np.asarray(R, dtype=float)
    r = R.ravel()
    ctx = CMAKER.new_ctx(len(r))
    ctx.mag, ctx.rake, ctx.dip, ctx.ztor, ctx.hypo_depth = M, 0.0, 90.0, 0.0, DEPTH_KM
    ctx.width = 10 ** (-0.76 + 0.27 * M)                 # Wells & Coppersmith strike-slip
    ctx.rjb, ctx.rx, ctx.ry0 = r, r, np.zeros(len(r))
    ctx.rrup = np.sqrt(r ** 2 + DEPTH_KM ** 2)
    ctx.vs30, ctx.vs30measured = np.full(len(r), VS30), np.zeros(len(r), bool)
    ctx.z1pt0, ctx.z2pt5, ctx.sids = np.full(len(r), 0.1), np.full(len(r), 1.0), np.arange(len(r))
    mean = CMAKER.get_mean_stds([ctx])[0].mean(axis=0)   # (imt, site), ln(g) and ln(cm/s)
    log_pga = np.log10(np.exp(mean[0]) * 981.0)          # cm/s^2
    log_pgv = np.log10(np.exp(mean[1]))                  # cm/s
    # Worden et al. (2012) ground motion -> MMI
    mmi_pga = np.where(log_pga <= 1.57, 1.78 + 1.55 * log_pga, -1.60 + 3.70 * log_pga)
    mmi_pgv = np.where(log_pgv <= 0.53, 3.78 + 1.47 * log_pgv, 2.89 + 3.16 * log_pgv)
    return (0.5 * (mmi_pga + mmi_pgv)).reshape(R.shape)

def predicted_mmi(town_lat, town_lon, src_lat, src_lon, M):
    R_true = _haversine_km(USGS_LAT, USGS_LON, town_lat, town_lon)
    R_src  = _haversine_km(src_lat, src_lon, town_lat, town_lon)
    return shakemap_mmi(town_lat, town_lon) + float(f(R_src, M) - f(R_true, USGS_MAG))

towns = pd.DataFrame(TOWNS, columns=['town', 'lat', 'lon'])
towns['obs'] = [shakemap_mmi(a, b) for a, b in zip(towns.lat, towns.lon)]
towns = towns.sort_values('obs').reset_index(drop=True)      # weakest shaking at the bottom
for name in PRIOR_CSV:
    la, lo, _, mg = est[name]
    for tag, dM in (('', 0), ('_lo', -DM), ('_hi', DM)):
        towns[name + tag] = [predicted_mmi(a, b, la, lo, mg + dM)
                             for a, b in zip(towns.lat, towns.lon)]
print(towns.round(1).to_string())

#%%
# ── Figure ──────────────────────────────────────────────────────────────────
AX_POS = {
    'map':  [0.03, 0.08, 0.44, 0.84],
    'cbar': [0.03, 0.045, 0.44, 0.015],
    'town': [0.58, 0.56, 0.39, 0.36],
    'sens': [0.58, 0.08, 0.34, 0.34],
    'scb':  [0.935, 0.08, 0.012, 0.34],
}

def make_figure(styles, outname):
    """styles: prior name -> dict(color, alpha, marker, ms) — drawn in order, later on top."""
    fig = plt.figure(figsize=(15, 8.5))
    label = lambda name: name.replace('_', ' ')

    # -- (a) map -------------------------------------------------------------
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
    cb.set_label('USGS ShakeMap MMI')

    for i, (name, st) in enumerate(styles.items()):
        la, lo, err, mg = est[name]
        z = 5 + 0.1 * i
        ax.annotate('', xy=(lo, la), xytext=(USGS_LON, USGS_LAT), transform=proj, zorder=z,
                    arrowprops=dict(arrowstyle='-|>', color=to_rgba(st['color'], st['alpha']),
                                    lw=2, shrinkA=8, shrinkB=4))
        ax.plot(lo, la, st['marker'], ms=11 * st['ms'], color=st['color'], alpha=st['alpha'],
                markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=z + 0.05)
        above = st.get('above', False)   # label above the marker instead of below
        ax.text(lo, la + (0.12 if above else -0.12), f'{label(name)}\n{err:.0f} km, M{mg:.1f}' + (' (assumed)' if name not in LOG_RUN else ''), color=st['color'],
                fontsize=10, fontweight='bold', ha='center', va='bottom' if above else 'top', transform=proj, zorder=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
    ax.plot(USGS_LON, USGS_LAT, '*', ms=20, color='gold', markeredgecolor='black',
            markeredgewidth=0.8, transform=proj, zorder=8)

    ax.scatter(towns.lon, towns.lat, marker='s', s=28, color='black', transform=proj, zorder=6)
    for _, t in towns.iterrows():
        ax.text(t.lon + 0.08, t.lat + 0.03, t.town, fontsize=8.5, transform=proj, zorder=7,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_title(f'(a)  M7.0 mainshock: USGS ShakeMap and first estimate with ≥{N_TRIGGERS} stations')
    ax.legend(handles=[
        Line2D([], [], marker='*', ls='', ms=15, color='gold', markeredgecolor='black', label='USGS location'),
        Line2D([], [], marker='s', ls='', ms=6, color='black', label='town')],
        loc='lower left', fontsize=9, framealpha=0.9)

    # -- (b) town-by-town predicted MMI ---------------------------------------
    ax = fig.add_axes(AX_POS['town'])
    y = np.arange(len(towns))
    ax.axvspan(0, ALERT_MMI, color='gray', alpha=0.12, zorder=0)
    ax.axvline(ALERT_MMI, color='k', ls='--', lw=1, zorder=1)
    ax.text(ALERT_MMI - 0.1, 6.0, f'below alert\nthreshold\n(MMI {ALERT_MMI:.0f})', fontsize=8.5,
            va='center', ha='right', color='dimgray')
    n = len(styles)
    for k, (name, st) in enumerate(styles.items()):
        yy = y + (k - (n - 1) / 2) * 0.25
        ax.hlines(yy, towns[name + '_lo'], towns[name + '_hi'], color=st['color'], lw=2,
                  alpha=0.5 * st['alpha'], zorder=2 + 0.1 * k)
        ax.plot(towns[name], yy, st['marker'], color=st['color'], alpha=st['alpha'], ms=8 * st['ms'],
                markeredgecolor='black', zorder=3 + 0.1 * k, label=f'{label(name)}-located')
    ax.plot(towns['obs'], y, 'D', color='gold', ms=9, markeredgecolor='black', zorder=4,
            label='USGS ShakeMap')
    ax.set_yticks(y)
    ax.set_yticklabels(towns.town)
    lo_cols = [c for c in towns.columns if c.endswith('_lo')]
    ax.set_xlim(min(1.5, towns[lo_cols].min().min() - 0.3), 10.5)
    ax.set_xlabel('MMI')
    ax.set_title(f'(b)  Predicted shaking if each estimate were believed\n(whiskers: additional ±{DM} magnitude error)')
    ax.grid(axis='x', lw=0.3, alpha=0.5)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # -- (c) sensitivity at one town -------------------------------------------
    ft = towns[towns.town == FOCUS_TOWN].iloc[0]
    R_true = float(_haversine_km(USGS_LAT, USGS_LON, ft.lat, ft.lon))
    R_axis = np.linspace(20, 260, 100)
    dM_axis = np.linspace(-3, 1, 100)
    RR, DD = np.meshgrid(R_axis, dM_axis)
    err_mmi = np.array([f(R_axis, USGS_MAG + d) for d in dM_axis]) - f(R_true, USGS_MAG)

    ax = fig.add_axes(AX_POS['sens'])
    lim = 4
    cs = ax.contourf(RR, DD, err_mmi, levels=np.linspace(-lim, lim, 25), cmap='RdBu_r', extend='both')
    ax.contour(RR, DD, err_mmi, levels=[-ERR_TOL, ERR_TOL], colors='k', linewidths=1.5, linestyles='solid')
    ax.contour(RR, DD, err_mmi, levels=[0], colors='k', linewidths=0.8, linestyles='--')
    fig.colorbar(cs, cax=fig.add_axes(AX_POS['scb']), ticks=np.arange(-lim, lim + 1)).set_label(
        f'Predicted − ShakeMap MMI at {FOCUS_TOWN}')
    ax.plot(R_true, 0, '*', ms=18, color='gold', markeredgecolor='black', zorder=5)
    for i, (name, st) in enumerate(styles.items()):
        R_est = float(_haversine_km(est[name][0], est[name][1], ft.lat, ft.lon))
        ax.plot(R_est, est[name][3] - USGS_MAG, st['marker'], ms=10 * st['ms'], color=st['color'], alpha=st['alpha'],
                markeredgecolor='black', zorder=5 + 0.1 * i)
    ax.set_xlabel(f'Distance from assumed epicenter to {FOCUS_TOWN} (km)')
    ax.set_ylabel('Magnitude error  ΔM')
    ax.set_title(f'(c)  Location vs magnitude error at {FOCUS_TOWN}  (black: ±{ERR_TOL} MMI)')

    fig.savefig(os.path.join(OUTPUT_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{outname}.pdf'), bbox_inches='tight')
    plt.show()

for outname, styles in FIGURES.items():
    make_figure(styles, outname)
