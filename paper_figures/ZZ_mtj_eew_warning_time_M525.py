#%%
# =============================================================================
# mtj_eew_warning_time_M525.py — same as mtj_eew_warning_time.py, for the M5.25 foreshock-region event (nc75103356, 2024-12-15)
# near the Mendocino Triple Junction; see mislocation_impact_M525.py
# =============================================================================
#   alert time  = (arrival of the N_ALERT_TRIGS-th station's P pick - origin time)
#                 + ALERT_LATENCY_S      (telemetry + trigger + processing)
#   warning     = (hypocentral distance / Vs) - alert time      at a given site
#                 (positive = seconds of warning before S-waves arrive;
#                  negative = site is inside the "blind zone")
#
# Same method as Magna/eew_alert_map.py (Figure A). Picks come from the .run
# file for the mainshock; epicenter/origin time are the USGS/NCSN catalog values.
#
# Usage: python paper_figures/mtj_eew_warning_time_M525.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from openquake.hazardlib.contexts import simple_cmaker
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR      = os.path.join(OUTPUT_DIR, 'figures', 'eew_warning_time')    # figures are saved in this subfolder
os.makedirs(FIG_DIR, exist_ok=True)

CASE_STUDY   = 'MTJ_2024_M7'
EVENT_ID     = 'nc75103356'
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', CASE_STUDY)
CATALOG_PATH = os.path.join(DATA_DIR, 'mtj_2024_m7.csv')
RUN_PATH     = os.path.join(DATA_DIR, 'run_files', f'{EVENT_ID}.run')

# Hypothetical-mislocation figures (second half of the script)
SHIFT_KM     = 100.0     # mislocation distance, due west (offshore) or due east (inland)
ALERT_MMI    = 3.0      # alert is issued to sites whose *predicted* MMI is at least this (app threshold, as in mislocation_impact_M525.py)
PD_B         = 1.39      # Pd magnitude scaling: M = 1.23*log10(Pd) + 1.39*log10(R) + 5.39 (EPIC; Kuyuk & Allen 2013)
SCEN_EXTENT  = [-127.5, -121.5, 37.7, 42.0]    # ShakeMap grid starts at 37.67 N
SHAKEMAP_XML = os.path.join(DATA_DIR, 'shakemap_nc75095651_grid.xml')    # M7 ShakeMap grid: used only for its Vs30 column (covers the full map; the M5.25 grid is too small)

# ---------------------------------------------------------------------------
# Config (same values as the Magna figure)
# ---------------------------------------------------------------------------
N_ALERT_TRIGS   = 4      # alert when this many stations have reported a P pick
ALERT_LATENCY_S = 4.0    # seconds from the last needed P pick to the alert reaching users
VS_KMS          = 3.5    # S-wave speed (km/s)

# Map extent [lon0, lon1, lat0, lat1]
EXTENT = [-126.0, -121.5, 37.3, 42.0]

# name: (lat, lon, label offset in points (dx, dy), horizontal alignment)
CITIES = {
    'Crescent City': (41.7558, -124.2026, ( 8,  4), 'left'),
    'Eureka':        (40.8021, -124.1637, ( 8,  4), 'left'),
    'Redding':       (40.5865, -122.3917, ( 8,  4), 'left'),
    'Garberville':   (40.1007, -123.7945, ( 8,  4), 'left'),
    'Fort Bragg':    (39.4457, -123.8053, ( 8,  4), 'left'),
    'Ukiah':         (39.1502, -123.2078, ( 8,  4), 'left'),
    'Santa Rosa':    (38.4404, -122.7141, ( 8,  4), 'left'),
}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                     'axes.titlesize': 13, 'axes.titleweight': 'bold', 'figure.dpi': 150})

#%%
# ---------------------------------------------------------------------------
# Alert time
# ---------------------------------------------------------------------------
catalog = pd.read_csv(CATALOG_PATH)
ev = catalog[catalog['id'] == EVENT_ID].iloc[0]
origin = pd.Timestamp(ev['time'])
ev_lat, ev_lon, ev_mag = ev['latitude'], ev['longitude'], ev['mag']
depth_km = 10.0 if pd.isna(ev['depth']) else float(ev['depth'])

run = pd.read_csv(RUN_PATH)
version = run[run['version'] == N_ALERT_TRIGS - 1]        # version k holds k+1 stations
t_pick = version['trigger time'].max()                    # the Nth station's P pick (unix s)
alert_time_s = t_pick - origin.timestamp() + ALERT_LATENCY_S
blind_radius_km = np.sqrt(max((alert_time_s * VS_KMS) ** 2 - depth_km ** 2, 0.0))
print(f"M{ev_mag:g}: alert at {alert_time_s:.1f} s after origin; blind-zone radius {blind_radius_km:.1f} km")


def warning_s(lat, lon):
    """Seconds between the alert and S-wave arrival at (lat, lon) (negative = blind zone)."""
    dx = (np.asarray(lon) - ev_lon) * 111.32 * np.cos(np.radians(ev_lat))
    dy = (np.asarray(lat) - ev_lat) * 110.57
    return np.sqrt(dx ** 2 + dy ** 2 + depth_km ** 2) / VS_KMS - alert_time_s


print("\nWarning time at cities (s; negative = no warning):")
for name, (clat, clon, *_) in CITIES.items():
    print(f"  {name:15s} {warning_s(clat, clon):6.1f}")

#%%
# ---------------------------------------------------------------------------
# Figure — warning-time map
# ---------------------------------------------------------------------------
proj = ccrs.PlateCarree()
lon_g = np.linspace(EXTENT[0], EXTENT[1], 400)
lat_g = np.linspace(EXTENT[2], EXTENT[3], 400)
LON, LAT = np.meshgrid(lon_g, lat_g)
W = warning_s(LAT, LON)

fig = plt.figure(figsize=(8, 9))
ax = fig.add_axes([0.10, 0.07, 0.72, 0.85], projection=proj)
ax.set_extent(EXTENT, crs=proj)
ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=3)
ax.set_xticks(np.arange(-126, -121.4, 1.0), crs=proj)
ax.set_yticks(np.arange(38, 42.1, 1.0), crs=proj)
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))

levels = np.arange(-10, 51, 4)
norm = mcolors.TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])
cf = ax.contourf(LON, LAT, W, levels=levels, cmap='RdYlGn', norm=norm, extend='both',
                 transform=proj, alpha=0.75, zorder=1)
cs0 = ax.contour(LON, LAT, W, levels=[0], colors='k', linewidths=1.8, transform=proj, zorder=2)
ax.clabel(cs0, fmt=lambda v: 'blind zone', fontsize=10)
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.03)
cbar.set_label('Warning time before S-wave arrival (s)')

used = version.drop_duplicates(subset='station')
ax.scatter(used['longitude'], used['latitude'], marker='^', s=45, color='steelblue',
           edgecolors='k', linewidths=0.6, transform=proj, zorder=5,
           label=f'First {N_ALERT_TRIGS} stations')
ax.scatter(ev_lon, ev_lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1,
           transform=proj, zorder=6, label=f'M{ev_mag:g} epicenter')
for name, (clat, clon, (dx, dy), ha) in CITIES.items():
    w = warning_s(clat, clon)
    ax.scatter(clon, clat, s=40, c='k', transform=proj, zorder=5)
    ax.annotate(f'{name}\n{w:+.0f} s', (clon, clat), xycoords=proj._as_mpl_transform(ax),
                xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=10,
                fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_title(f'Hypothetical EEW warning time — {origin:%Y-%m-%d} Mendocino Triple Junction M{ev_mag:g}\n'
             f'alert {alert_time_s:.1f} s after origin '
             f'({N_ALERT_TRIGS} stations + {ALERT_LATENCY_S:g} s latency)', fontsize=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
fig.savefig(os.path.join(FIG_DIR, 'mtj_eew_M525_warning_time.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'mtj_eew_M525_warning_time.pdf'), bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Hypothetical mislocation: epicenter at its true location, SHIFT_KM further offshore, or SHIFT_KM inland
# =============================================================================
# Everything here uses only what a real-time system can compute (no ShakeMap, which arrives too late to alert on).
# The P picks do not change, so the alert is still ready ALERT_TIME s after origin. What changes:
#   * magnitude — EPIC's Pd scaling has M = 1.23*log10(Pd) + 1.39*log10(R) + 5.39 per station, so with
#     the observed Pd fixed, moving the epicenter changes each station's R and hence
#         M_hyp = M_usgs + 1.39 * mean_stations( log10(R_hyp / R_true) )
#     (equal station weights; the 4 stations of the alert version)
#   * who gets alerted — a site is alerted if the *predicted* MMI f(R_hyp, M_hyp) >= ALERT_MMI, where f is
#     the GMPE-mean/Worden-2012 shaking model of mislocation_impact.py, in one of two site models:
#         'gmpe' = flat Vs30 = 400 m/s;  'vs30' = Vs30 grid (taken from the USGS ShakeMap grid file)
#   * reference ("true") alert zone — the same model evaluated at the true epicenter and USGS magnitude.
#     False alarm = alerted but not in the reference zone; missed = in the reference zone but not alerted.
#     This isolates the effect of mislocation; it does NOT say whether the model matches observed shaking.
#   * the warning actually received — true S-wave arrival (true hypocenter) minus the alert time
GMPES  = [AbrahamsonEtAl2014(), BooreEtAl2014(), CampbellBozorgnia2014(), ChiouYoungs2014()]
CMAKER = simple_cmaker(GMPES, ['PGV'])


def f_mmi(R, M, depth=10.0, vs30=None):
    """Median MMI at epicentral distance R (km, any shape) for magnitude M (mean of 4 NGA-West2 GMPEs, WGRW12).
    vs30=None: Vs30 = 400 m/s with fixed z1.0 = 0.1 km, z2.5 = 1.0 km. Otherwise vs30 (scalar or array like R) with
    z1.0 (CY14 California relation) and z2.5 (CB14 relation) derived from it."""
    R = np.asarray(R, dtype=float)
    r = R.ravel()
    if vs30 is None:
        v, z1, z25 = np.full(len(r), 400.0), np.full(len(r), 0.1), np.full(len(r), 1.0)
    else:
        v = np.broadcast_to(vs30, R.shape).ravel().astype(float)
        z1 = np.exp(-7.15 / 4 * np.log((v ** 4 + 571 ** 4) / (1360 ** 4 + 571 ** 4))) / 1000.0
        z25 = np.exp(7.089 - 1.144 * np.log(v))
    ctx = CMAKER.new_ctx(len(r))
    ctx.mag, ctx.rake, ctx.dip, ctx.ztor, ctx.hypo_depth = M, 0.0, 90.0, 0.0, depth
    ctx.width = 10 ** (-0.76 + 0.27 * M)
    ctx.rjb, ctx.rx, ctx.ry0 = r, r, np.zeros(len(r))
    ctx.rrup = np.sqrt(r ** 2 + depth ** 2)
    ctx.vs30, ctx.vs30measured = v, np.zeros(len(r), bool)
    ctx.z1pt0, ctx.z2pt5, ctx.sids = z1, z25, np.arange(len(r))
    log_pgv = np.log10(np.exp(CMAKER.get_mean_stds([ctx])[0].mean(axis=0)[0]))   # cm/s
    mmi = np.where(log_pgv < -1.21, 4.62 + 2.17 * log_pgv,
                   np.where(log_pgv < 0.53, 3.78 + 1.47 * log_pgv, 2.89 + 3.16 * log_pgv))
    return (mmi + 0.90 - 0.18 * np.clip(M, 3.0, 7.3)).reshape(R.shape)


def epi_km(lat0, lon0, lat, lon):
    """Flat-earth epicentral distance (km)."""
    return np.hypot((np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)),
                    (np.asarray(lat) - lat0) * 110.57)



# Vs30 from the ShakeMap grid file (only the Vs30 column is used; offshore it is a constant 600 m/s)
sm = pd.read_csv(SHAKEMAP_XML, sep=' ', header=None, comment='<', skiprows=14,
                 names=['lon', 'lat', 'mmi', 'pga', 'pgv', 'psa03', 'psa10', 'psa30', 'vs30']).dropna()
sm_lons, sm_lats = np.sort(sm['lon'].unique()), np.sort(sm['lat'].unique())
vs30_interp = RegularGridInterpolator(
    (sm_lats, sm_lons), sm.pivot(index='lat', columns='lon', values='vs30').reindex(index=sm_lats, columns=sm_lons).values,
    bounds_error=False, fill_value=None)    # nearest value off-grid


def shakemap_vs30(lat, lon):
    return vs30_interp(np.column_stack([np.ravel(lat), np.ravel(lon)])).reshape(np.shape(lat))


def predict_mmi(lat0, lon0, M, lat, lon, site_model):
    """Predicted MMI at (lat, lon) for an event assumed at (lat0, lon0) with magnitude M."""
    vs30 = shakemap_vs30(lat, lon) if site_model == 'vs30' else None
    return f_mmi(epi_km(lat0, lon0, lat, lon), M, vs30=vs30)


stations = version.drop_duplicates(subset='station')


def scenario(shift_km):
    """Assumed epicenter and magnitude for an epicenter moved shift_km east (+) / west (-)."""
    lon_h = ev_lon + shift_km / (111.32 * np.cos(np.radians(ev_lat)))
    R_true = epi_km(ev_lat, ev_lon, stations['latitude'], stations['longitude'])
    R_hyp  = epi_km(ev_lat, lon_h,  stations['latitude'], stations['longitude'])
    return lon_h, ev_mag + PD_B * np.mean(np.log10(R_hyp / R_true))


lon_g = np.linspace(SCEN_EXTENT[0], SCEN_EXTENT[1], 260)
lat_g = np.linspace(SCEN_EXTENT[2], SCEN_EXTENT[3], 220)
LON, LAT = np.meshgrid(lon_g, lat_g)
W_true = warning_s(LAT, LON)                              # warning if alerted, from the TRUE hypocenter


def make_scenario_figure(shift_km, where, outname, site_model):
    lon_h, M_h = scenario(shift_km)
    A = predict_mmi(ev_lat, lon_h, M_h, LAT, LON, site_model) >= ALERT_MMI           # sites the system would alert
    A_ref = predict_mmi(ev_lat, ev_lon, ev_mag, LAT, LON, site_model) >= ALERT_MMI    # same model, correct location
    print(f"\n[{site_model}] {where}: assumed M{M_h:.2f} (true M{ev_mag:g}); "
          f"alert zone {A.sum() / A_ref.sum():.2f}x the reference; false alarm {(A & ~A_ref).sum()} cells, "
          f"missed {(~A & A_ref).sum()} cells (of {A_ref.sum()})")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.10, 0.07, 0.72, 0.85], projection=proj)
    ax.set_extent(SCEN_EXTENT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=3)
    ax.set_xticks(np.arange(-127, -121.4, 1.0), crs=proj)
    ax.set_yticks(np.arange(38, 42.1, 1.0), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))

    # alerted AND in the reference zone: colored by the warning actually received
    levels = np.arange(-10, 51, 4)
    norm = mcolors.TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])
    cf = ax.contourf(LON, LAT, np.where(A & A_ref, W_true, np.nan), levels=levels, cmap='RdYlGn', norm=norm,
                     extend='both', transform=proj, alpha=0.75, zorder=1)
    # false alarm: alerted, outside the reference zone;  missed: inside the reference zone, no alert
    for mask, color, alpha, hatch in [(A & ~A_ref, '#9467bd', 0.25, '///'), (~A & A_ref, '#7f7f7f', 0.55, 'xxx')]:
        if mask.any():
            ax.contourf(LON, LAT, mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha,
                        transform=proj, zorder=1.4)
            ax.contourf(LON, LAT, mask.astype(float), levels=[0.5, 1.5], colors='none', hatches=[hatch],
                        transform=proj, zorder=1.5)
    ax.contour(LON, LAT, A_ref.astype(float), levels=[0.5], colors='k', linewidths=1.2, linestyles='--',
               transform=proj, zorder=2)
    if shift_km:
        ax.contour(LON, LAT, A.astype(float), levels=[0.5], colors='#6a3d9a', linewidths=1.8, transform=proj, zorder=2)
    ax.contour(LON, LAT, W_true, levels=[0], colors='k', linewidths=1.2, transform=proj, zorder=2)
    fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.03).set_label(
        'Warning time received before S-wave arrival (s)')

    ax.scatter(ev_lon, ev_lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1, transform=proj, zorder=6)
    if shift_km:
        ax.scatter(lon_h, ev_lat, marker='*', s=450, c='white', edgecolors='#6a3d9a', linewidths=1.5,
                   transform=proj, zorder=6)
        ax.annotate('', xy=(lon_h, ev_lat), xytext=(ev_lon, ev_lat), transform=proj, zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color='#6a3d9a', lw=2, shrinkA=12, shrinkB=12))
    for name, (clat, clon, (dx, dy), ha) in CITIES.items():
        alerted = bool(predict_mmi(ev_lat, lon_h, M_h, clat, clon, site_model) >= ALERT_MMI)
        in_ref = bool(predict_mmi(ev_lat, ev_lon, ev_mag, clat, clon, site_model) >= ALERT_MMI)
        txt = (f'{warning_s(clat, clon):+.0f} s' if alerted and in_ref else
               'false alarm' if alerted else 'MISSED' if in_ref else 'No alert')
        print(f"  {name:15s} {txt}")
        ax.scatter(clon, clat, s=40, c='k', transform=proj, zorder=5)
        ax.annotate(f'{name}\n{txt}', (clon, clat), xycoords=proj._as_mpl_transform(ax), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, fontsize=10, fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

    ax.annotate(f'Assumed M{M_h:.2f}' if shift_km else f'M{M_h:.2f}', (lon_h, ev_lat), xycoords=proj._as_mpl_transform(ax),
                xytext=(0, 24), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))
    head_txt = ('Hypothetical: M{:.1f} at its true location'.format(ev_mag) if not shift_km else
                f'Hypothetical: M{ev_mag:g} mislocated {abs(shift_km):.0f} km {where}')
    ax.set_title(head_txt, fontsize=12, pad=10)
    ax.legend(handles=[
        Line2D([], [], marker='*', ls='', ms=16, mfc='yellow', mec='k', label='True epicenter'),
        Line2D([], [], marker='*', ls='', ms=16, mfc='white', mec='#6a3d9a', label='Assumed epicenter'),
        Line2D([], [], color='#6a3d9a', lw=1.8, label='Alert zone (assumed)'),
        Line2D([], [], color='k', ls='--', lw=1.2, label='Alert zone (correct location)'),
        Patch(fc='#9467bd', alpha=0.4, hatch='///', label='False alarm (outside correct-location zone)'),
        Patch(fc='#7f7f7f', alpha=0.6, hatch='xxx', label='Missed (inside correct-location zone, no alert)')],
        loc='lower left', fontsize=8.5, framealpha=0.9)
    outname = f'{outname}_assumedM{M_h:.2f}_alert{alert_time_s:.1f}s_MMI{ALERT_MMI:g}'
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.pdf'), bbox_inches='tight')
    plt.show()


for site_model in ['gmpe', 'vs30']:
    for shift, where, tag in [(0.0, 'true location', 'true'), (-SHIFT_KM, 'further offshore', 'offshore'),
                              (+SHIFT_KM, 'inland', 'inland')]:
        make_scenario_figure(shift, where, f'mtj_eew_M525_{site_model}_{tag}', site_model)
