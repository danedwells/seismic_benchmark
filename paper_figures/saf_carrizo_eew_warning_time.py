#%%
# =============================================================================
# saf_carrizo_eew_warning_time.py — hypothetical EEW warning time for a M7.0 on the Carrizo Plain / Cholame section of
# the San Andreas (the quietest stretch in small earthquakes; see data_examination_scripts/saf_quiet_zones.py).
# =============================================================================
# There is no real event, so the P picks are synthetic:
#   P arrival at a station = bEPIC's ak135 travel time for the hypocentral distance   (origin time = 0)
#   alert time  = (arrival at the N_ALERT_TRIGS-th closest station) + ALERT_LATENCY_S
#   warning     = (hypocentral distance / Vs) - alert time      at a given site
#                 (positive = seconds of warning before S-waves arrive;
#                  negative = site is inside the "blind zone")
# Stations are every station that triggered in the benchmark's real EPIC runs (data/california/run_files).
# Same method as mtj_eew_warning_time.py.
#
# Usage: python paper_figures/saf_carrizo_eew_warning_time.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from benchmark.runner import get_unique_stations
from shakealert_footprint import footprint

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR      = os.path.join(OUTPUT_DIR, 'figures', 'saf_carrizo')    # figures are saved in this subfolder
os.makedirs(FIG_DIR, exist_ok=True)
RUN_DIR      = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')

# Hypothetical event: Carrizo Plain near Wallace Creek, on the (approximate) fault trace
EV_LAT, EV_LON, EV_MAG, DEPTH_KM = 35.25, -119.80, 7.0, 8.0    # depth = bEPIC's search depth

# Hypothetical-mislocation figures
ALERT_MMI    = 3.0       # ShakeAlert alerts the area inside the MMI >= 3 contour (once M >= 4.5)
FOOTPRINT    = 'octagon'  # 'octagon' (what phone alerts use) or 'circle'
PD_B         = 1.39      # Pd magnitude scaling: M = 1.23*log10(Pd) + 1.39*log10(R) + 5.39 (EPIC; Kuyuk & Allen 2013)
SCEN_EXTENT  = [-127.0, -112.0, 30.0, 41.0]

N_ALERT_TRIGS   = 4      # alert when this many stations have reported a P pick
ALERT_LATENCY_S = 4.0    # seconds from the last needed P pick to the alert reaching users
VS_KMS          = 3.5    # S-wave speed (km/s)

# Map extent [lon0, lon1, lat0, lat1]
EXTENT = SCEN_EXTENT

# name: (lat, lon, label offset in points (dx, dy), horizontal alignment)
CITIES = {
    'Fresno':          (36.7378, -119.7871, ( 8,  4), 'left'),
    'Paso Robles':     (35.6269, -120.6910, (-8, 18), 'right'),
    'San Luis Obispo': (35.2828, -120.6596, (-8,  2), 'right'),
    'Santa Maria':     (34.9530, -120.4357, (-8, -14), 'right'),
    'Santa Barbara':   (34.4208, -119.6982, (-8, -34), 'right'),
    'Bakersfield':     (35.3733, -119.0187, ( 8,  4), 'left'),
    'Palmdale':        (34.5794, -118.1165, ( 8,  4), 'left'),
    'Los Angeles':     (34.0522, -118.2437, ( 8, -16), 'left'),
    'San Francisco':   (37.7749, -122.4194, ( 8,  4), 'left'),
    'San Diego':       (32.7157, -117.1611, ( 8, -16), 'left'),
}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                     'axes.titlesize': 13, 'axes.titleweight': 'bold', 'figure.dpi': 150})

#%%
# ---------------------------------------------------------------------------
# Synthetic picks and alert time
# ---------------------------------------------------------------------------
ev_lat, ev_lon, ev_mag, depth_km = EV_LAT, EV_LON, EV_MAG, DEPTH_KM
all_stations = get_unique_stations(RUN_DIR)
all_stations['hypo_km'] = np.sqrt(((all_stations['longitude'] - ev_lon) * 111.32 * np.cos(np.radians(ev_lat))) ** 2
                                  + ((all_stations['latitude'] - ev_lat) * 110.57) ** 2 + depth_km ** 2)
used = all_stations.nsmallest(N_ALERT_TRIGS, 'hypo_km')
print(used[['station', 'network', 'hypo_km']].to_string(index=False))
from bEPIC import EPIC_locate_prelim
alert_time_s = float(EPIC_locate_prelim._get_travel_time_function(DEPTH_KM)(used['hypo_km'].max())) + ALERT_LATENCY_S
blind_radius_km = np.sqrt(max((alert_time_s * VS_KMS) ** 2 - depth_km ** 2, 0.0))
print(f"M{ev_mag:.1f}: alert at {alert_time_s:.1f} s after origin; blind-zone radius {blind_radius_km:.1f} km")


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
ax.set_xticks(np.arange(-126, -111.9, 2.0), crs=proj)
ax.set_yticks(np.arange(30, 41.1, 2.0), crs=proj)
ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))

levels = np.arange(-10, 151, 10)
norm = mcolors.TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])
cf = ax.contourf(LON, LAT, W, levels=levels, cmap='RdYlGn', norm=norm, extend='both',
                 transform=proj, alpha=0.75, zorder=1)
cs0 = ax.contour(LON, LAT, W, levels=[0], colors='k', linewidths=1.8, transform=proj, zorder=2)
ax.clabel(cs0, fmt=lambda v: 'blind zone', fontsize=10)
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.03)
cbar.set_label('Warning time before S-wave arrival (s)')

ax.scatter(used['longitude'], used['latitude'], marker='^', s=45, color='steelblue',
           edgecolors='k', linewidths=0.6, transform=proj, zorder=5,
           label=f'First {N_ALERT_TRIGS} stations')
ax.scatter(ev_lon, ev_lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1,
           transform=proj, zorder=6, label=f'M{ev_mag:.1f} epicenter (hypothetical)')
for name, (clat, clon, (dx, dy), ha) in CITIES.items():
    w = warning_s(clat, clon)
    ax.scatter(clon, clat, s=40, c='k', transform=proj, zorder=5)
    ax.annotate(f'{name}\n{w:+.0f} s', (clon, clat), xycoords=proj._as_mpl_transform(ax),
                xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=10,
                fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_title(f'Hypothetical EEW warning time — Carrizo Plain M{ev_mag:.1f}\n'
             f'alert {alert_time_s:.1f} s after origin '
             f'({N_ALERT_TRIGS} stations + {ALERT_LATENCY_S:g} s latency)', fontsize=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
fig.savefig(os.path.join(FIG_DIR, 'saf_carrizo_eew_warning_time.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'saf_carrizo_eew_warning_time.pdf'), bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Hypothetical mislocation: where bEPIC places the event with each prior
# =============================================================================
# Alert footprint = ShakeAlert's method (A. Williamson): the BA08/Worden MMI >= ALERT_MMI contour distance for the
# estimated magnitude, drawn around the estimated epicenter as an octagon (see shakealert_footprint.py). Alerts only
# go out for M >= 4.5. The estimate comes from the first N_ALERT_TRIGS picks:
#   * epicenter — the bEPIC MAP location with each prior (saf_carrizo_bepic_location.py)
#   * magnitude — EPIC's Pd scaling has M = 1.23*log10(Pd) + 1.39*log10(R) + 5.39 per station, so with the observed Pd
#     fixed, a different epicenter changes each station's R and hence
#         M_hyp = M_true + 1.39 * mean_stations( log10(R_hyp / R_true) )
#     (equal station weights; the N_ALERT_TRIGS stations of the alert)
#   * reference alert zone — the footprint for the true epicenter and magnitude. False alarm = alerted but outside it;
#     missed = inside it but not alerted. The warning actually received = true S-wave arrival (true hypocenter) - alert time.
# A real event that keeps growing or moving would add a new footprint at each update (alerts cannot be rescinded), so
# the final alerted area is the union of the footprints; only the first one (at N_ALERT_TRIGS picks) is drawn here.
def epi_km(lat0, lon0, lat, lon):
    """Flat-earth epicentral distance (km)."""
    return np.hypot((np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)),
                    (np.asarray(lat) - lat0) * 110.57)


def inside(poly, lat, lon):
    """Boolean array (shape of lat): points inside the footprint outline (all False for no alert)."""
    if poly is None:
        return np.zeros(np.shape(lat), bool)
    return Path(poly).contains_points(np.column_stack([np.ravel(lon), np.ravel(lat)])).reshape(np.shape(lat))


stations = used


def scenario(lat_h, lon_h):
    """Assumed magnitude for an epicenter placed at (lat_h, lon_h)."""
    R_true = epi_km(ev_lat, ev_lon, stations['latitude'], stations['longitude'])
    R_hyp  = epi_km(lat_h, lon_h,  stations['latitude'], stations['longitude'])
    return ev_mag + PD_B * np.mean(np.log10(R_hyp / R_true))


lon_g = np.linspace(SCEN_EXTENT[0], SCEN_EXTENT[1], 600)
lat_g = np.linspace(SCEN_EXTENT[2], SCEN_EXTENT[3], 520)
LON, LAT = np.meshgrid(lon_g, lat_g)
W_true = warning_s(LAT, LON)                              # warning if alerted, from the TRUE hypocenter
poly_ref = footprint(ev_lon, ev_lat, ev_mag, ALERT_MMI, FOOTPRINT)
A_ref = inside(poly_ref, LAT, LON)


def make_scenario_figure(lat_h, lon_h, where, outname, label_offset=(-24, 0, 'right')):
    M_h = scenario(lat_h, lon_h)
    shift_km = float(epi_km(ev_lat, ev_lon, lat_h, lon_h))
    poly = footprint(lon_h, lat_h, M_h, ALERT_MMI, FOOTPRINT)
    A = inside(poly, LAT, LON)                            # sites the system would alert
    print(f"\n{where} ({shift_km:.0f} km): assumed M{M_h:.2f} (true M{ev_mag:.1f}); "
          f"alert zone {A.sum() / A_ref.sum():.2f}x the reference; false alarm {(A & ~A_ref).sum()} cells, "
          f"missed {(~A & A_ref).sum()} cells (of {A_ref.sum()})")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.10, 0.07, 0.72, 0.85], projection=proj)
    ax.set_extent(SCEN_EXTENT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=3)
    ax.set_xticks(np.arange(-126, -111.9, 2.0), crs=proj)
    ax.set_yticks(np.arange(30, 41.1, 2.0), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))

    # alerted AND in the reference zone: colored by the warning actually received
    levels = np.arange(-10, 151, 10)
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
    ax.plot(poly_ref[:, 0], poly_ref[:, 1], color='k', lw=1.2, ls='--', transform=proj, zorder=2)
    if shift_km and poly is not None:
        ax.plot(poly[:, 0], poly[:, 1], color='#6a3d9a', lw=1.8, transform=proj, zorder=2)
    ax.contour(LON, LAT, W_true, levels=[0], colors='k', linewidths=1.2, transform=proj, zorder=2)
    fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.03).set_label(
        'Warning time received before S-wave arrival (s)')

    ax.scatter(ev_lon, ev_lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1, transform=proj, zorder=6)
    if shift_km:
        ax.scatter(lon_h, lat_h, marker='*', s=450, c='white', edgecolors='#6a3d9a', linewidths=1.5,
                   transform=proj, zorder=6)
        ax.annotate('', xy=(lon_h, lat_h), xytext=(ev_lon, ev_lat), transform=proj, zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color='#6a3d9a', lw=2, shrinkA=12, shrinkB=12))
    for name, (clat, clon, (dx, dy), ha) in CITIES.items():
        alerted = bool(inside(poly, np.array([clat]), np.array([clon]))[0])
        in_ref = bool(inside(poly_ref, np.array([clat]), np.array([clon]))[0])
        txt = (f'{warning_s(clat, clon):+.0f} s' if alerted and in_ref else
               'false alarm' if alerted else 'MISSED' if in_ref else 'No alert')
        print(f"  {name:15s} {txt}")
        ax.scatter(clon, clat, s=40, c='k', transform=proj, zorder=5)
        ax.annotate(f'{name}\n{txt}', (clon, clat), xycoords=proj._as_mpl_transform(ax), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, fontsize=10, fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))
    # "Assumed M" label: next to the star for the true-location figure, otherwise out over the empty ocean with a thin
    # leader line (offset set per scenario in the calls below), so it never covers a city label
    ax.annotate(f'Assumed M{M_h:.1f}' if shift_km else f'M{M_h:.1f}', (lon_h, lat_h), xycoords=proj._as_mpl_transform(ax),
                xytext=label_offset[:2], textcoords='offset points', ha=label_offset[2], va='center', fontsize=10,
                fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='#6a3d9a', lw=1, shrinkA=0, shrinkB=12)
                if abs(label_offset[0]) + abs(label_offset[1]) > 40 else None)

    head_txt = (f'Hypothetical Carrizo Plain: M{ev_mag:.1f} at its true location' if not shift_km else
                f'Hypothetical Carrizo Plain: M{ev_mag:.1f} mislocated {shift_km:.0f} km {where}')
    ax.set_title(head_txt, fontsize=12, pad=10)
    ax.legend(handles=[
        Line2D([], [], marker='*', ls='', ms=16, mfc='yellow', mec='k', label='True epicenter'),
        Line2D([], [], marker='*', ls='', ms=16, mfc='white', mec='#6a3d9a', label='Assumed epicenter'),
        Line2D([], [], color='#6a3d9a', lw=1.8, label='Alert footprint (assumed)'),
        Line2D([], [], color='k', ls='--', lw=1.2, label='Alert footprint (correct location)'),
        Patch(fc='#9467bd', alpha=0.4, hatch='///', label='False alarm (outside correct-location footprint)'),
        Patch(fc='#7f7f7f', alpha=0.6, hatch='xxx', label='Missed (inside correct-location footprint, no alert)')],
        loc='lower left', fontsize=8.5, framealpha=0.9)
    outname = f'{outname}_assumedM{M_h:.1f}_alert{alert_time_s:.1f}s_{FOOTPRINT}_MMI{ALERT_MMI:g}'
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{outname}.pdf'), bbox_inches='tight')
    plt.show()


# Assumed epicenters: where bEPIC puts the event with the first N_ALERT_TRIGS synthetic picks and each prior
# (saf_carrizo_bepic_location.py, noise-free picks)
loc = pd.read_csv(os.path.join(FIG_DIR, 'saf_carrizo_bepic_locations.csv'))
loc = loc[loc['realization'] == 0].set_index('prior')
make_scenario_figure(ev_lat, ev_lon, 'true location', 'saf_carrizo_eew_true')
LABEL_OFFSETS = {'KDE_Seismicity': (-30, -110, 'center'), 'ETAS': (40, -110, 'center')}   # points (dx, dy, ha)
for prior, tag in [('KDE_Seismicity', 'kde'), ('ETAS', 'etas')]:
    make_scenario_figure(loc.loc[prior, 'lat'], loc.loc[prior, 'lon'],
                         f'by bEPIC with the {prior.replace("_Seismicity", "")} prior', f'saf_carrizo_eew_{tag}',
                         label_offset=LABEL_OFFSETS[prior])
